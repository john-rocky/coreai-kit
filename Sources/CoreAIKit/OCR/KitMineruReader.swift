// KitMineruReader.swift — MinerU2.5-Pro whole-page document OCR: image → structured markdown
// (tables as <table>/<tr>/<td>, formulas as LaTeX), on the VL rope-shift rider. Unlike
// `KitDocReader` (Unlimited-OCR, single-pass MoE + host arrange), MinerU is a stock Qwen2-VL
// that rides `VLRuntime` + `VLArchitecture.mineru`: the page is letterboxed into a fixed
// 672×896 portrait canvas, CLIP-normalized, vision-encoded once into the decoder's static
// `image_embeds`, and the text decodes on top.
//
//   let reader = try await KitMineruReader(catalog: "mineru2.5-pro")
//   let markdown = try await reader.read(imageAt: documentURL)
//
// v1 is a single "Text Recognition:" pass over the whole page — correct for clean single-column
// documents. MinerU's 2-stage layout→region pipeline (multi-column / rotated / cross-page) is a
// follow-up (source of truth: opendatalab/mineru-vl-utils).

import CoreAILanguageModels
import CoreAIKitVision
import CoreGraphics
import Foundation
import ImageIO          // CGImagePropertyOrientation (macOS: not re-exported by CoreGraphics)
import Tokenizers

/// MinerU2.5-Pro document reader: one image → structured markdown.
public final class KitMineruReader: @unchecked Sendable {
    private let runtime: VLRuntime           // 768 portrait — single-pass + per-region recognition
    private let arch = VLArchitecture.mineru
    private let layoutRuntime: VLRuntime?     // 1036² square — 2-stage `Layout Detection` (optional)
    private let layoutArch = VLArchitecture.mineruLayout

    /// Loads by catalog id (downloads the `vision/` + `decoder/` bundles; the decoder bundle
    /// carries its own tokenizer). First use downloads; later launches load from cache. This
    /// loads the recognition bundle only — `readStructured` needs the layout bundle too (see the
    /// `visionDir:…layoutVisionDir:…` initializer).
    public convenience init(
        catalog id: String = "mineru2.5-pro",
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

    /// Loads the recognition (768) bundle only. `read` (single-pass) works; `readStructured`
    /// (2-stage) throws until a layout bundle is supplied.
    public convenience init(visionDir: URL, decoderDir: URL) async throws {
        try await self.init(
            visionDir: visionDir, decoderDir: decoderDir,
            layoutVisionDir: nil, layoutDecoderDir: nil)
    }

    /// Loads the recognition (768) bundle + an optional layout (1036²) bundle. Both are needed for
    /// the 2-stage `readStructured` (structured Markdown with `<table>` HTML).
    public init(
        visionDir: URL, decoderDir: URL,
        layoutVisionDir: URL?, layoutDecoderDir: URL?
    ) async throws {
        runtime = try await VLRuntime(
            decoderBundleAt: decoderDir, visionModelAt: visionDir, arch: .mineru)
        if let layoutVisionDir, let layoutDecoderDir {
            layoutRuntime = try await VLRuntime(
                decoderBundleAt: layoutDecoderDir,
                visionModelAt: layoutVisionDir,
                arch: .mineruLayout)
        } else {
            layoutRuntime = nil
        }
    }

    /// Whether the 2-stage `readStructured` is available (a layout bundle was loaded).
    public var supportsStructured: Bool { layoutRuntime != nil }

    /// OCR an image file (any format) → structured markdown.
    public func read(imageAt url: URL, maxTokens: Int = 2048) async throws -> String {
        try await read(ImageFile.load(url).cgImage, maxTokens: maxTokens)
    }

    /// OCR one image in a single whole-page pass. `prompt` selects the MinerU recognition mode
    /// (`"Text Recognition:"` default; also `"Table Recognition:"` / `"Formula Recognition:"`).
    /// Reads the page as plain text (reading order); use ``readStructured(_:)`` for the 2-stage
    /// layout→region pipeline that emits `<table>` HTML / LaTeX.
    public func read(
        _ image: CGImage,
        orientation: CGImagePropertyOrientation = .up,
        prompt: String = "Text Recognition:",
        maxTokens: Int = 2048
    ) async throws -> String {
        try await generate(
            runtime: runtime, arch: arch, cgImage: image, orientation: orientation,
            prompt: prompt, maxTokens: maxTokens, keepSpecialTokens: false)
    }

    /// Whole-page **2-stage** structuring → Markdown: a `Layout Detection` pass finds the blocks
    /// (title / text / table / formula …) + reading order, then each region is cropped and read
    /// with its type's prompt (tables → `<table>` HTML, formulas → LaTeX), assembled by `json2md`.
    /// Mirrors `opendatalab/mineru-vl-utils` `two_step_extract` + `post_process.json2md`.
    public func readStructured(imageAt url: URL, maxTokens: Int = 1024) async throws -> String {
        try await readStructured(ImageFile.load(url).cgImage, maxTokens: maxTokens)
    }

    /// See ``readStructured(imageAt:maxTokens:)``.
    public func readStructured(
        _ image: CGImage,
        orientation: CGImagePropertyOrientation = .up,
        maxTokens: Int = 1024
    ) async throws -> String {
        guard let layoutRuntime else {
            throw KitVisionError.bundleMissingMain  // needs the 1036² layout bundle
        }
        // 1) Layout (1036² square, stretch): the model emits `<|box_start|>…<|ref_start|>type…`.
        //    Keep special tokens so the box markers survive detokenization.
        let layoutText = try await generate(
            runtime: layoutRuntime, arch: layoutArch, cgImage: image, orientation: orientation,
            prompt: "\nLayout Detection:", maxTokens: 1024, keepSpecialTokens: true)
        let blocks = Self.parseLayout(layoutText)

        // The layout grid STRETCHES the page to a square, so boxes are 0–1 of the original page
        // directly (a stretch is linear — no letterbox inverse needed).

        // 2) Recognition (768): crop each region from the ORIGINAL image and read it by type.
        let skip: Set<String> = ["list", "equation_block", "image_block", "image", "chart", "abandon"]
        var pieces: [(content: String, type: String, mergePrev: Bool)] = []
        for block in blocks {
            if skip.contains(block.type) { continue }
            guard let crop = Self.crop(image, bbox: block.bbox) else { continue }
            let isTable = block.type == "table"
            var text = try await generate(
                runtime: runtime, arch: arch, cgImage: crop, orientation: .up,
                prompt: Self.recognitionPrompt(block.type),
                maxTokens: maxTokens, keepSpecialTokens: isTable)  // keep OTSL tokens for tables
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if isTable, !text.isEmpty { text = Self.otslToHTML(text) }
            if !text.isEmpty { pieces.append((text, block.type, block.mergePrev)) }
        }

        // 3) Assemble (json2md): join by reading order, folding text-continuation blocks.
        return Self.json2md(pieces)
    }

    // MARK: - Core generate

    /// Vision-encode `cgImage` on `runtime`/`arch` + rider prompt → generated text. With the pf64
    /// multifunction bundle the engine feeds the prompt in S=64 chunks (chunked prefill) then S=1.
    private func generate(
        runtime: VLRuntime, arch: VLArchitecture,
        cgImage: CGImage, orientation: CGImagePropertyOrientation,
        prompt: String, maxTokens: Int, keepSpecialTokens: Bool
    ) async throws -> String {
        let tokenizer = runtime.tokenizer
        let text =
            "\(arch.imStart)user\n"
            + arch.visionStart
            + String(repeating: arch.imagePad, count: arch.mergedTokens)
            + arch.visionEnd
            + prompt
            + "\(arch.imEnd)\n\(arch.imStart)assistant\n"

        var tokens = tokenizer.encode(text: text).map(Int32.init)
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

        try await runtime.attach(cgImage: cgImage, orientation: orientation, segmentID: nil)
        runtime.setImageShift(imageStart: imageStart)
        try await runtime.engine.reset()

        let eos = tokenizer.eosTokenId
        let imEnd = tokenizer.convertTokenToId(arch.imEnd)
        var outputIDs: [Int] = []
        let stream = try await runtime.engine.generate(
            with: tokens, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        for try await output in stream {
            let id = Int(output.tokenId)
            if id == eos || id == imEnd { break }
            outputIDs.append(id)
        }
        return tokenizer.decode(tokens: outputIDs, skipSpecialTokens: !keepSpecialTokens)
    }

    // MARK: - 2-stage helpers (ported from mineru-vl-utils)

    private struct LayoutBlock { let type: String; let bbox: [Double]; let mergePrev: Bool }

    /// `_layout_re` + `_convert_bbox` + `_parse_merge_prev` from `mineru_client.py`.
    private static func parseLayout(_ output: String) -> [LayoutBlock] {
        let pattern =
            #"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|>"#
            + #"<\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?"#
            + #"(.*?)(?=<\|box_start\|>|$)"#
        guard let re = try? NSRegularExpression(
            pattern: pattern, options: [.dotMatchesLineSeparators]) else { return [] }
        let ns = output as NSString
        var blocks: [LayoutBlock] = []
        re.enumerateMatches(in: output, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let m else { return }
            func group(_ i: Int) -> String {
                let r = m.range(at: i)
                return r.location == NSNotFound ? "" : ns.substring(with: r)
            }
            let coords = [group(1), group(2), group(3), group(4)].compactMap { Int($0) }
            guard coords.count == 4, !coords.contains(where: { $0 < 0 || $0 > 1000 }) else { return }
            var x1 = coords[0], y1 = coords[1], x2 = coords[2], y2 = coords[3]
            if x2 < x1 { swap(&x1, &x2) }
            if y2 < y1 { swap(&y1, &y2) }
            guard x1 != x2, y1 != y2 else { return }
            var type = group(5).lowercased()
            if type == "unknown" { type = "image" }
            if type == "inline_formula" { return }
            let tail = group(7)
            blocks.append(LayoutBlock(
                type: type,
                bbox: [Double(x1) / 1000, Double(y1) / 1000, Double(x2) / 1000, Double(y2) / 1000],
                mergePrev: tail.contains("txt_contd_tgt")))
        }
        return blocks
    }

    /// The recognition prompt for a block type (defaults to `Text Recognition:`).
    private static func recognitionPrompt(_ type: String) -> String {
        switch type {
        case "table": return "\nTable Recognition:"
        case "equation": return "\nFormula Recognition:"
        case "image", "chart": return "\nImage Analysis:"
        default: return "\nText Recognition:"
        }
    }

    /// Convert MinerU's OTSL table output (`<fcel>`=cell, `<ecel>`=empty, `<nl>`=new row) to HTML.
    /// A simplified port of `mineru_vl_utils`'s `convert_otsl_to_html` — row/col spans
    /// (`<lcel>`/`<ucel>`/`<xcel>`) are dropped (best-effort for spanned tables).
    private static func otslToHTML(_ otsl: String) -> String {
        let s = otsl.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.lowercased().hasPrefix("<table"), s.lowercased().hasSuffix("</table>") { return s }
        var html = "<table>"
        for rawRow in s.components(separatedBy: "<nl>") {
            let row = rawRow.trimmingCharacters(in: .whitespacesAndNewlines)
            if row.isEmpty { continue }
            var normalized = row.replacingOccurrences(of: "<ecel>", with: "<fcel>")
            for span in ["<lcel>", "<ucel>", "<xcel>"] {
                normalized = normalized.replacingOccurrences(of: span, with: "")
            }
            var cells = normalized.components(separatedBy: "<fcel>")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            if let first = cells.first, first.isEmpty { cells.removeFirst() }  // before 1st <fcel>
            html += "<tr>" + cells.map { "<td>\($0)</td>" }.joined() + "</tr>"
        }
        return html + "</table>"
    }

    /// Crop a 0–1 bbox out of the original image (top-left origin).
    private static func crop(_ image: CGImage, bbox: [Double]) -> CGImage? {
        let w = Double(image.width), h = Double(image.height)
        let x1 = max(0, Int((bbox[0] * w).rounded(.down)))
        let y1 = max(0, Int((bbox[1] * h).rounded(.down)))
        let x2 = min(image.width, Int((bbox[2] * w).rounded(.up)))
        let y2 = min(image.height, Int((bbox[3] * h).rounded(.up)))
        guard x2 - x1 >= 1, y2 - y1 >= 1 else { return nil }
        return image.cropping(to: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
    }

    /// `json2md`: join region contents by reading order, folding `merge_prev` text runs. Beyond the
    /// reference (which joins raw), `title` blocks get Markdown heading markers (first `#`, the rest
    /// `##`) so the output is real Markdown — the layout already tells us which blocks are titles.
    private static func json2md(_ pieces: [(content: String, type: String, mergePrev: Bool)]) -> String {
        var out: [String] = []
        var lastTextIdx = -1
        var titleCount = 0
        for piece in pieces where !piece.content.isEmpty {
            if piece.mergePrev, lastTextIdx >= 0 {
                let cjk = piece.content.range(of: #"\p{Han}"#, options: .regularExpression) != nil
                out[lastTextIdx] += (cjk ? "" : " ") + piece.content
            } else {
                var content = piece.content
                if piece.type == "title" {
                    titleCount += 1
                    content = (titleCount == 1 ? "# " : "## ") + content
                }
                out.append(content)
                if piece.type == "text" { lastTextIdx = out.count - 1 }
            }
        }
        return out.joined(separator: "\n\n")
    }

    // MARK: - Helpers

    private func singleTokenID(_ token: String, tokenizer: any Tokenizer) -> Int32? {
        if let id = tokenizer.convertTokenToId(token) { return Int32(id) }
        let ids = tokenizer.encode(text: token)
        return ids.count == 1 ? Int32(ids[0]) : nil
    }
}
