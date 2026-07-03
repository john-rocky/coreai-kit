// KitDocReader.swift — Unlimited-OCR (baidu) document OCR behind one `read(_:)`. Image in,
// structured markdown out (tables survive as <table>/<tr>/<td>), fully on-device on the
// STOCK runtime — no engine patch:
//
//   CGImage --pad-to-640, x/127.5-1--> [1,3,640,640] fp16
//     --vision .aimodel (GraphModel)--> visual tokens [1,100,1280]
//     --arrange (10x10 rows + image_newline + view_seperator)--> [111,1280]
//     --scatter into embed_tokens(prompt_input_ids)--> prefix [1,115,1280]
//     --decoder prefill + greedy decode (KV cache via MutableViews)--> ids --detok--> markdown
//
// The host loop mirrors the app-verified pipeline (apps/CoreAIOCR, itself gated against the
// Python reference): greedy with no_repeat_ngram=35 + a consecutive-run cap — pure greedy
// derails into degenerate repeats on dense tables, and the oracle used the same guards.

import CoreAI
import CoreAIKitVision
import CoreGraphics
import Foundation
import Tokenizers

/// Unlimited-OCR document reader: one image → markdown.
public final class KitDocReader: @unchecked Sendable {
    // Locked spec from the verified conversion recipe (Base mode, 640px, 10×10 grid).
    private enum Spec {
        static let imageSide = 640
        static let patchGrid = 10           // 10x10 = 100 patches
        static let visualTokens = 111       // 100 patches + 10 row newlines + 1 seperator
        static let prefixLen = 115          // 111 visual + BOS + 3 prompt-text tokens
        static let hidden = 1280
        static let vocab = 129_280
        static let imageTokenID: Int32 = 128_815
        static let eos: Int32 = 1
    }

    private let vision: GraphModel
    private let decoder: DocDecoder
    private let embed: [Float16]            // [vocab*hidden] embed_tokens table (host gather)
    private let imageNewline: [Float16]     // [hidden]
    private let viewSeperator: [Float16]    // [hidden]
    private let promptIDs: [Int32]          // [prefixLen]
    private let tokenizer: any Tokenizer

    /// Loads a document-OCR model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let reader = try await KitDocReader(catalog: "unlimited-ocr")
    /// ```
    ///
    /// First use downloads the vision + decoder bundles, the host constant tables, and the
    /// tokenizer (cached afterwards).
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .ocr)
        let vision = try await store.download(
            ModelID(entry.repo, path: "vision"), progress: downloadProgress)
        let decoder = try await store.download(
            ModelID(entry.repo, path: "decoder"), progress: downloadProgress)
        let assets = try await store.download(
            ModelID(entry.repo, path: "assets"), progress: downloadProgress)
        let tokenizer = try await store.download(
            ModelID(entry.repo, path: "tokenizer"), progress: downloadProgress)
        try await self.init(
            visionDir: vision, decoderDir: decoder, assetsDir: assets, tokenizerDir: tokenizer)
    }

    /// Loads local directories: the vision / decoder bundle dirs (each holding one
    /// `*.aimodel`), the constant-table dir (`embed_tokens.f16`, `image_newline.f16`,
    /// `view_seperator.f16`, `prompt_input_ids.i32`), and the tokenizer dir.
    public init(
        visionDir: URL, decoderDir: URL, assetsDir: URL, tokenizerDir: URL
    ) async throws {
        vision = try await GraphModel(
            contentsOf: Self.aimodel(in: visionDir), computeUnits: .gpu)
        decoder = try await DocDecoder(contentsOf: Self.aimodel(in: decoderDir))
        embed = try Self.readF16(assetsDir.appendingPathComponent("embed_tokens.f16"))
        imageNewline = try Self.readF16(assetsDir.appendingPathComponent("image_newline.f16"))
        viewSeperator = try Self.readF16(assetsDir.appendingPathComponent("view_seperator.f16"))
        promptIDs = try Self.readI32(assetsDir.appendingPathComponent("prompt_input_ids.i32"))
        tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
    }

    /// OCR an image file (any format) → structured markdown.
    public func read(imageAt url: URL, maxTokens: Int = 1024) async throws -> String {
        try await read(ImageFile.load(url).cgImage, maxTokens: maxTokens)
    }

    /// OCR one image → structured markdown (special tokens kept so table markup survives).
    public func read(_ image: CGImage, maxTokens: Int = 1024) async throws -> String {
        // 1) preprocess: aspect-fit into 640², gray pad, x/127.5 - 1, CHW fp16
        let pixels = Self.padTo640CHW(image)

        // 2) vision encoder -> [1,100,1280]
        let visionOut = try await vision.run([
            "image": .float16(pixels, shape: [1, 3, Spec.imageSide, Spec.imageSide])
        ])
        guard let patches = visionOut["visual_tokens"]?.floats() else {
            throw VisionError.missingOutput("visual_tokens")
        }

        // 3) arrange: per row 10 patches then image_newline; then view_seperator
        let grid = Spec.patchGrid
        let hidden = Spec.hidden
        var visual = [Float16]()
        visual.reserveCapacity(Spec.visualTokens * hidden)
        for row in 0..<grid {
            for col in 0..<grid {
                let base = (row * grid + col) * hidden
                for i in 0..<hidden { visual.append(Float16(patches[base + i])) }
            }
            visual.append(contentsOf: imageNewline)
        }
        visual.append(contentsOf: viewSeperator)

        // 4) prefix = embed_tokens(promptIDs) with the 111 visual rows in the <image> slots
        var prefix = [Float16](repeating: 0, count: Spec.prefixLen * hidden)
        var slot = 0
        for (pos, id) in promptIDs.enumerated() {
            if id == Spec.imageTokenID {
                let src = slot * hidden
                for i in 0..<hidden { prefix[pos * hidden + i] = visual[src + i] }
                slot += 1
            } else {
                let src = Int(id) * hidden
                for i in 0..<hidden { prefix[pos * hidden + i] = embed[src + i] }
            }
        }

        // 5) prefill + greedy decode with the oracle's repeat guards
        decoder.resetState()
        var logits = try await decoder.prefill(
            prefixEmbeds: prefix, prefixLen: Spec.prefixLen, hidden: hidden, vocab: Spec.vocab)
        var generated = [Int32]()
        var token = pick(logits, generated)
        var position = Int32(Spec.prefixLen)
        for _ in 0..<maxTokens {
            generated.append(token)
            if token == Spec.eos { break }
            let tokenEmbed = Array(embed[Int(token) * hidden ..< (Int(token) + 1) * hidden])
            logits = try await decoder.decode(
                tokenEmbed: tokenEmbed, pos: position, hidden: hidden, vocab: Spec.vocab)
            token = pick(logits, generated)
            position += 1
        }

        // 6) detokenize, special tokens kept (<table>/<tr>/<td> structure survives)
        let ids = generated.filter { $0 != Spec.eos }.map { Int($0) }
        return tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    // MARK: - sampling guards (match the verified oracle: no_repeat_ngram=35, run cap 6)

    private let noRepeatN = 35
    private let maxRun = 6

    private func pick(_ logits: [Float], _ generated: [Int32]) -> Int32 {
        var banned = Set<Int32>()
        if generated.count >= maxRun, let last = generated.last,
            generated.suffix(maxRun).allSatisfy({ $0 == last })
        {
            banned.insert(last)
        }
        let m = noRepeatN - 1
        if generated.count >= m {
            let suffix = Array(generated.suffix(m))
            var i = 0
            while i + m < generated.count {
                if Array(generated[i..<i + m]) == suffix { banned.insert(generated[i + m]) }
                i += 1
            }
        }
        var best = -1
        var bestValue = -Float.greatestFiniteMagnitude
        for i in 0..<logits.count
        where logits[i] > bestValue && (banned.isEmpty || !banned.contains(Int32(i))) {
            bestValue = logits[i]
            best = i
        }
        return Int32(max(best, 0))
    }

    // MARK: - preprocessing (PIL ImageOps.pad(640, gray) + normalize(mean=std=0.5))

    private static func padTo640CHW(_ image: CGImage) -> [Float16] {
        let side = Spec.imageSide
        let scale = min(
            Double(side) / Double(image.width), Double(side) / Double(image.height))
        let dw = Int((Double(image.width) * scale).rounded())
        let dh = Int((Double(image.height) * scale).rounded())
        let ox = (side - dw) / 2
        let oy = (side - dh) / 2

        var rgba = [UInt8](repeating: 128, count: side * side * 4)  // mean*255 pad
        rgba.withUnsafeMutableBytes { buffer in
            guard
                let ctx = CGContext(
                    data: buffer.baseAddress, width: side, height: side, bitsPerComponent: 8,
                    bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return }
            ctx.interpolationQuality = .high
            // CG origin is bottom-left; oy from the top maps to (side-dh-oy) at the bottom.
            ctx.draw(image, in: CGRect(x: ox, y: side - dh - oy, width: dw, height: dh))
        }

        var out = [Float16](repeating: 0, count: 3 * side * side)
        let plane = side * side
        for y in 0..<side {
            for x in 0..<side {
                let p = (y * side + x) * 4
                let idx = y * side + x
                out[idx] = Float16(Float(rgba[p]) / 127.5 - 1)
                out[plane + idx] = Float16(Float(rgba[p + 1]) / 127.5 - 1)
                out[2 * plane + idx] = Float16(Float(rgba[p + 2]) / 127.5 - 1)
            }
        }
        return out
    }

    // MARK: - assets

    private static func aimodel(in dir: URL) throws -> URL {
        if dir.pathExtension == "aimodel" { return dir }
        let entries = try FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)
        guard let found = entries.first(where: { $0.pathExtension == "aimodel" }) else {
            throw VisionError.bundleLayout("no .aimodel found under \(dir.path)")
        }
        return found
    }

    private static func readF16(_ url: URL) throws -> [Float16] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
    }

    private static func readI32(_ url: URL) throws -> [Int32] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
}

// MARK: - the unified prefill/decode bundle (one AIModel, two functions, shared KV state)

/// `StatefulGraphModel` loads one function per instance; the OCR decoder is ONE bundle with
/// `prefill` + `decode` sharing the KV cache, so this wrapper holds both functions and the
/// two state buffers itself (loading the 3.4 GB bundle twice is not an option).
private final class DocDecoder: @unchecked Sendable {
    private let model: AIModel
    private let prefillFn: InferenceFunction
    private let decodeFn: InferenceFunction
    private let prefillDesc: InferenceFunctionDescriptor
    private let decodeDesc: InferenceFunctionDescriptor
    private let keyDescriptor: NDArrayDescriptor
    private let valueDescriptor: NDArrayDescriptor
    private var keyCache: NDArray
    private var valueCache: NDArray
    // Stable storage for the output view: `MutableViews` is non-escapable and borrows its
    // arrays up to `function.run`, so locals (or optional unwraps) cannot back it — same
    // pattern as `StatefulGraphModel`. Reallocated per call before the insert.
    private var logitsArray: NDArray

    init(contentsOf url: URL) async throws {
        model = try await AIModel(
            contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard
            let pd = model.functionDescriptor(for: "prefill"),
            let dd = model.functionDescriptor(for: "decode"),
            let pf = try model.loadFunction(named: "prefill"),
            let df = try model.loadFunction(named: "decode")
        else {
            throw VisionError.functionNotFound("prefill/decode")
        }
        prefillDesc = pd
        decodeDesc = dd
        prefillFn = pf
        decodeFn = df

        guard prefillDesc.stateNames.count == 2,
            case .ndArray(let kd) = prefillDesc.stateDescriptor(of: prefillDesc.stateNames[0]),
            case .ndArray(let vd) = prefillDesc.stateDescriptor(of: prefillDesc.stateNames[1])
        else {
            throw VisionError.statefulGraphUnsupported(prefillDesc.stateNames)
        }
        keyDescriptor = kd
        valueDescriptor = vd
        keyCache = NDArray(descriptor: keyDescriptor)
        valueCache = NDArray(descriptor: valueDescriptor)
        guard case .ndArray(let ld) = prefillDesc.outputDescriptor(of: prefillDesc.outputNames[0])
        else { throw VisionError.missingOutput(prefillDesc.outputNames[0]) }
        logitsArray = NDArray(descriptor: ld.resolvingDynamicDimensions(ld.shape.map { max($0, 1) }))
        resetState()
    }

    /// Zero the KV cache — call before each document.
    func resetState() {
        keyCache = NDArray(descriptor: keyDescriptor)
        valueCache = NDArray(descriptor: valueDescriptor)
        zeroF16(&keyCache)
        zeroF16(&valueCache)
    }

    /// Prefill the assembled prefix → last-token logits; seeds the KV cache.
    func prefill(
        prefixEmbeds: [Float16], prefixLen: Int, hidden: Int, vocab: Int
    ) async throws -> [Float] {
        let input = try ndArray(
            prefillDesc, "inputs_embeds", shape: [1, prefixLen, hidden]) {
            var view = $0.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: prefixEmbeds)
        }
        return try await runLogits(
            prefillFn, prefillDesc, inputs: ["inputs_embeds": input], vocab: vocab)
    }

    /// One decode step: the current token's embedding at absolute position `pos`.
    func decode(
        tokenEmbed: [Float16], pos: Int32, hidden: Int, vocab: Int
    ) async throws -> [Float] {
        let embeds = try ndArray(decodeDesc, "inputs_embeds", shape: [1, 1, hidden]) {
            var view = $0.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: tokenEmbed)
        }
        let position = try ndArray(decodeDesc, "pos", shape: [1]) {
            var view = $0.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: [pos])
        }
        return try await runLogits(
            decodeFn, decodeDesc, inputs: ["inputs_embeds": embeds, "pos": position],
            vocab: vocab)
    }

    private func ndArray(
        _ descriptor: InferenceFunctionDescriptor, _ name: String, shape: [Int],
        fill: (inout NDArray) -> Void
    ) throws -> NDArray {
        guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else {
            throw VisionError.unknownInput(name)
        }
        var array = NDArray(descriptor: d.resolvingDynamicDimensions(shape))
        fill(&array)
        return array
    }

    private func runLogits(
        _ function: InferenceFunction, _ descriptor: InferenceFunctionDescriptor,
        inputs: [String: NDArray], vocab: Int
    ) async throws -> [Float] {
        let logitsName = descriptor.outputNames[0]
        guard case .ndArray(let ld) = descriptor.outputDescriptor(of: logitsName) else {
            throw VisionError.missingOutput(logitsName)
        }
        logitsArray = NDArray(descriptor: ld.resolvingDynamicDimensions([1, 1, vocab]))
        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: descriptor.stateNames[0])
        states.insert(&valueCache, for: descriptor.stateNames[1])
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logitsArray, for: logitsName)
        _ = try await function.run(
            inputs: inputs, states: consume states, outputViews: consume outputViews)

        // Read back through a local binding — a scoped view chained off the stored
        // property trips the lifetime checker.
        let produced = logitsArray
        var out = [Float16](repeating: 0, count: vocab)
        produced.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
            out.withUnsafeMutableBufferPointer {
                $0.baseAddress!.update(from: pointer, count: vocab)
            }
        }
        return out.map { Float($0) }
    }
}

private func zeroF16(_ array: inout NDArray) {
    let count = array.shape.reduce(1, *)
    var view = array.mutableView(as: Float16.self)
    view.copyElements(fromContentsOf: [Float16](repeating: 0, count: count))
}
