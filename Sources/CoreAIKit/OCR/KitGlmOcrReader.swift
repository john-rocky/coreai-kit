// KitGlmOcrReader.swift — GLM-OCR (zai-org, GLM-4.V small, MIT) document OCR on the VL rope-shift
// rider. A CogViT tower + GLM text decoder that rides `VLRuntime` + `VLArchitecture.glmOcr`: the
// page is letterboxed into a fixed 672×896 portrait canvas, CLIP-normalized, vision-encoded once
// into the decoder's static `image_embeds`, and the text decodes on top. Same rider contract as
// `KitMineruReader`; the difference is GLM's ChatML framing (`[gMASK]<sop><|user|>…<|assistant|>`
// with `<|begin_of_image|>`/`<|image|>`/`<|end_of_image|>`).
//
//   let reader = try await KitGlmOcrReader(catalog: "glm-ocr")
//   let text = try await reader.read(imageAt: documentURL)

import CoreAILanguageModels
import CoreAIKitVision
import CoreGraphics
import Foundation
import ImageIO
import Tokenizers

/// GLM-OCR document reader: one image → recognized text (single "Text Recognition:" pass).
public final class KitGlmOcrReader: @unchecked Sendable {
    private let runtime: VLRuntime
    private let arch = VLArchitecture.glmOcr

    /// Loads by catalog id (downloads the `vision/` + `decoder/` bundles; the decoder bundle
    /// carries its own tokenizer).
    public convenience init(
        catalog id: String = "glm-ocr",
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .ocr)
        let vision = try await store.download(
            ModelID(entry.repo, path: "vision"), progress: downloadProgress)
        let decoder = try await store.download(
            ModelID(entry.repo, path: "decoder"), progress: downloadProgress)
        try await self.init(visionDir: vision, decoderDir: decoder)
    }

    /// Loads local directories: the vision bundle dir (one `*.aimodel`/`*.aimodelc`) and the decoder
    /// LanguageBundle dir (its model + tokenizer + metadata).
    public init(visionDir: URL, decoderDir: URL) async throws {
        runtime = try await VLRuntime(
            decoderBundleAt: decoderDir, visionModelAt: visionDir, arch: .glmOcr)
    }

    /// OCR an image file (any format) → recognized text.
    public func read(imageAt url: URL, maxTokens: Int = 2048) async throws -> String {
        try await read(ImageFile.load(url).cgImage, maxTokens: maxTokens)
    }

    /// OCR one image. `prompt` selects the GLM-OCR mode (`"Text Recognition:"` default; also
    /// `"Table Recognition:"` / `"Formula Recognition:"`).
    public func read(
        _ image: CGImage,
        orientation: CGImagePropertyOrientation = .up,
        prompt: String = "Text Recognition:",
        maxTokens: Int = 2048
    ) async throws -> String {
        let tokenizer = runtime.tokenizer

        // GLM ChatML: [gMASK]<sop><|user|>\n<begin_of_image><image>×N<end_of_image>{prompt}<|assistant|>\n
        let text =
            "[gMASK]<sop><|user|>\n"
            + arch.visionStart
            + String(repeating: arch.imagePad, count: arch.mergedTokens)
            + arch.visionEnd
            + prompt
            + "<|assistant|>\n"

        var tokens = tokenizer.encode(text: text).map(Int32.init)

        // Rewrite the N consecutive <|image|> ids to extension ids `vocab + slot`.
        guard let padID = singleTokenID(arch.imagePad, tokenizer: tokenizer) else {
            throw KitVisionError.visionOutputMissing(arch.imagePad)
        }
        var imageStart = -1
        var slot: Int32 = 0
        for i in tokens.indices where tokens[i] == padID && slot < Int32(arch.mergedTokens) {
            if imageStart < 0 { imageStart = i }
            tokens[i] = arch.vocab + slot
            slot += 1
        }
        guard slot == Int32(arch.mergedTokens), imageStart >= 0 else {
            throw KitVisionError.visionOutputMissing(
                "\(arch.imagePad)×\(arch.mergedTokens) (found \(slot))")
        }

        try await runtime.attach(cgImage: image, orientation: orientation, segmentID: nil)
        runtime.setImageShift(imageStart: imageStart)
        try await runtime.engine.reset()

        let eos = tokenizer.eosTokenId
        let userTurn = tokenizer.convertTokenToId("<|user|>")  // next-turn marker also ends output
        var outputIDs: [Int] = []
        let stream = try await runtime.engine.generate(
            with: tokens, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        for try await output in stream {
            let id = Int(output.tokenId)
            if id == eos || id == userTurn { break }
            outputIDs.append(id)
        }
        return tokenizer.decode(tokens: outputIDs, skipSpecialTokens: false)
    }

    // MARK: - Helpers

    private func singleTokenID(_ token: String, tokenizer: any Tokenizer) -> Int32? {
        if let id = tokenizer.convertTokenToId(token) { return Int32(id) }
        let ids = tokenizer.encode(text: token)
        return ids.count == 1 ? Int32(ids[0]) : nil
    }
}
