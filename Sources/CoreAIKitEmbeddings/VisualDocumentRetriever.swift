// VisualDocumentRetriever.swift — on-device visual document retrieval (ColModernVBERT export),
// the OCR-free, late-interaction (ColBERT / MaxSim) leg of the retrieval stack. A text query and
// a page *image* are each encoded into per-token L2-normalized 128-d multi-vectors by two static
// graphs (a ModernBERT text encoder + a SigLIP2 vision encoder, both with a custom_text_proj
// head baked in-graph), then scored with MaxSim on the host: `score = Σ_q max_d ⟨E_q, E_d⟩`.
//
// Pairs with `TextEmbedder` (text→text dense) and `Reranker` (cross-encoder): this matches
// queries against whole page *images* — tables, charts, scans, complex layouts — with no OCR.
// `embed/visual-retrieve -> rerank -> generate`, all on device.

@_exported import CoreAIKitCore
@_exported import CoreAIKitVision

import CoreGraphics
import Foundation
import Tokenizers

public final class VisualDocumentRetriever: @unchecked Sendable {
    /// A page's per-token multi-vector (`tokens[i]` is a 128-d unit vector). Encode your corpus
    /// once, keep these, and score many queries against them cheaply.
    public struct PageEmbedding: Sendable {
        public let tokens: [[Float]]
        public init(tokens: [[Float]]) { self.tokens = tokens }
    }

    /// A page encoded as an N×M grid of tiles, each tile a real sub-region run through the doc
    /// encoder independently. Tiles ARE spatially disjoint inputs, so the best-matching tile
    /// reliably localizes *where* the query matched (unlike within-tile patch tokens, whose
    /// spatial locality is destroyed by the bidirectional LLM + pixel-shuffle). Higher-res than
    /// a single global tile, so retrieval is sharper too.
    public struct TiledPageEmbedding: Sendable {
        public struct Tile: Sendable {
            public let rect: CGRect            // normalized (0...1) region within the page
            public let embedding: PageEmbedding
        }
        public let tiles: [Tile]
        public init(tiles: [Tile]) { self.tiles = tiles }
    }

    private let queryGraph: GraphModel
    private let docGraph: GraphModel
    private let tokenizer: any Tokenizer
    private let preprocessor: ImagePreprocessor

    private let qIdsInput: String
    private let qMaskInput: String
    private let qOutput: String
    private let docPixelsInput: String
    private let docMaskInput: String
    private let docOutput: String

    private let queryGrid: Int        // padded query length (e.g. 32)
    private let docPixelShape: [Int]  // [1, 1, 3, 512, 512]
    private let docImageSize: Int     // 512
    private let docSeqLen: Int        // 89 (single-tile)
    private let dim: Int              // 128
    private let padToken: Int32

    /// Image patches per side in the document multi-vector. The single-tile doc bundle emits a
    /// `patchGridSide × patchGridSide` grid of page patches (8×8 = 64 tokens after pixel-shuffle
    /// ×4 of the 32×32 SigLIP grid); each cell covers a 1/8 × 1/8 region of the page. Used by
    /// `patchRelevance` to localize where a query matched.
    public let patchGridSide = 8
    // The 64 image tokens are a contiguous block after the leading template tokens (positions
    // 13..76 in the s89 single-tile sequence).
    private let imageTokenStart = 13

    /// Loads the two bundle directories (each a `*.aimodel`); the query bundle also holds the
    /// `tokenizer/` folder. `padToken` is ModernBERT's `[PAD]` (50283).
    public init(
        queryBundleAt queryURL: URL,
        docBundleAt docURL: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        padToken: Int32 = 50283
    ) async throws {
        func aimodel(in url: URL) throws -> URL {
            // Accept either the bundle parent dir (holding the `.aimodel` + `tokenizer/`) or the
            // `.aimodel` directory itself, mirroring DepthEstimator.
            if url.pathExtension == "aimodel" { return url }
            guard let m = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil
            ).first(where: { $0.pathExtension == "aimodel" }) else {
                throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
            }
            return m
        }

        let queryGraph = try await GraphModel(contentsOf: try aimodel(in: queryURL),
                                              computeUnits: computeUnits)
        let docGraph = try await GraphModel(contentsOf: try aimodel(in: docURL),
                                            computeUnits: computeUnits)
        self.queryGraph = queryGraph
        self.docGraph = docGraph
        self.tokenizer = try await AutoTokenizer.from(
            modelFolder: queryURL.appendingPathComponent("tokenizer"))
        self.padToken = padToken

        // Query graph contract: input_ids [1,S] + attention_mask [1,S] -> [1,S,128].
        guard
            let qIds = queryGraph.inputNames.first(where: { $0.contains("input_ids") }),
            let qMask = queryGraph.inputNames.first(where: { $0.contains("mask") }),
            let qIdsShape = queryGraph.shape(ofInput: qIds), qIdsShape.count == 2,
            let qOut = queryGraph.outputNames.first,
            let qOutShape = queryGraph.shape(ofOutput: qOut), qOutShape.count == 3
        else {
            throw VisionError.bundleLayout(
                "unexpected query graph: in \(queryGraph.inputNames) out \(queryGraph.outputNames)")
        }
        self.qIdsInput = qIds
        self.qMaskInput = qMask
        self.qOutput = qOut
        self.queryGrid = qIdsShape[1]
        self.dim = qOutShape[2]

        // Doc graph contract: pixel_values [1,1,3,R,R] + pixel_attention_mask -> [1,Sd,128].
        guard
            let dPix = docGraph.inputNames.first(where: { $0.contains("pixel_values") }),
            let dMask = docGraph.inputNames.first(where: { $0.contains("pixel_attention") })
                ?? docGraph.inputNames.first(where: { $0.contains("mask") }),
            let dPixShape = docGraph.shape(ofInput: dPix), dPixShape.count == 5,
            let dOut = docGraph.outputNames.first,
            let dOutShape = docGraph.shape(ofOutput: dOut), dOutShape.count == 3
        else {
            throw VisionError.bundleLayout(
                "unexpected doc graph: in \(docGraph.inputNames) out \(docGraph.outputNames)")
        }
        self.docPixelsInput = dPix
        self.docMaskInput = dMask
        self.docOutput = dOut
        self.docPixelShape = dPixShape
        self.docImageSize = dPixShape[dPixShape.count - 1]
        self.docSeqLen = dOutShape[1]
        // ColModernVBERT / Idefics3 image normalization: (x/255 - 0.5) / 0.5.
        self.preprocessor = ImagePreprocessor(
            size: dPixShape[dPixShape.count - 1],
            mean: SIMD3(0.5, 0.5, 0.5), std: SIMD3(0.5, 0.5, 0.5))
    }

    /// Resolves both bundles via the `ModelStore` (downloading from the Hub if absent, or using a
    /// sideloaded copy already present at the store path), then loads them. Defaults to the fp16
    /// ColModernVBERT query/doc encoders.
    public convenience init(
        queryModel: ModelID = .colModernVBERTQuery,
        docModel: ModelID = .colModernVBERTDoc,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        padToken: Int32 = 50283,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let queryURL = try await store.download(queryModel, progress: downloadProgress)
        let docURL = try await store.download(docModel, progress: downloadProgress)
        try await self.init(
            queryBundleAt: queryURL, docBundleAt: docURL,
            computeUnits: computeUnits, padToken: padToken)
    }

    // MARK: - Encoding

    /// Encodes a text query into its per-token multi-vector (real tokens only). Pad tokens are
    /// dropped so MaxSim sums only over meaningful query terms.
    public func encode(query: String) async throws -> [[Float]] {
        var ids = tokenizer.encode(text: query).map(Int32.init)
        if ids.count > queryGrid { ids = Array(ids.prefix(queryGrid)) }
        let realCount = ids.count
        var mask = [Int32](repeating: 1, count: ids.count)
        while ids.count < queryGrid { ids.append(padToken); mask.append(0) }

        let outputs = try await queryGraph.run([
            qIdsInput: .int32(ids, shape: [1, queryGrid]),
            qMaskInput: .int32(mask, shape: [1, queryGrid]),
        ])
        guard let out = outputs[qOutput] else { throw VisionError.missingOutput(qOutput) }
        return rows(out.floats(), count: realCount)
    }

    /// Encodes a page image into its per-token multi-vector. The page is square-resized to the
    /// graph's tile size (single-tile "global image" view) and normalized in `ImagePreprocessor`.
    public func encode(page image: CGImage) async throws -> PageEmbedding {
        let chw = try preprocessor.chw(from: image)               // [3 * R * R], (x/255-0.5)/0.5
        let masked = docImageSize * docImageSize
        let pam = [Int32](repeating: 1, count: masked)            // full tile -> all valid
        let outputs = try await docGraph.run([
            docPixelsInput: .float32(chw, shape: docPixelShape),  // auto-cast to fp16 if needed
            docMaskInput: .int32(pam, shape: maskShape()),
        ])
        guard let out = outputs[docOutput] else { throw VisionError.missingOutput(docOutput) }
        return PageEmbedding(tokens: rows(out.floats(), count: docSeqLen))
    }

    // MARK: - Scoring

    /// ColBERT late interaction (MaxSim): for each query token, the best-matching doc token, summed.
    /// Vectors are L2-normalized in-graph, so the dot product is cosine.
    public static func score(query: [[Float]], page: PageEmbedding) -> Float {
        var total: Float = 0
        for q in query {
            var best = -Float.greatestFiniteMagnitude
            for d in page.tokens {
                var dot: Float = 0
                for i in 0..<q.count { dot += q[i] * d[i] }
                if dot > best { best = dot }
            }
            total += best
        }
        return total
    }

    /// Per-page-patch relevance to the query (ColPali-style similarity map / visual grounding):
    /// for each of the `patchGridSide × patchGridSide` page patches, the max similarity to any
    /// query token. Returns a row-major grid normalized to 0...1 — overlay it on the page to
    /// highlight *where* the query matched. Returns an all-zero grid if the page wasn't encoded
    /// by this retriever (token layout mismatch).
    public func patchRelevance(query: [[Float]], page: PageEmbedding) -> [Float] {
        let n = patchGridSide * patchGridSide
        let zeros = [Float](repeating: 0, count: n)
        guard page.tokens.count >= imageTokenStart + n, query.count >= 1 else { return zeros }
        let patches = (0..<n).map { page.tokens[imageTokenStart + $0] }

        // Per-query-token relevance, baselined PER TOKEN. A token like "the"/"in" (or the [CLS]/
        // [SEP] specials) matches every patch ~equally, so subtracting that token's own mean
        // over patches collapses it to ~0 everywhere; a token like "revenue" that spikes on one
        // region stays strongly positive there. Taking the max over CONTENT tokens of these
        // de-meaned scores yields a sharp, query-specific map instead of a mottled one. Drop the
        // first/last query tokens ([CLS]/[SEP]) — they are global aggregators that match broadly.
        let lo = query.count > 2 ? 1 : 0
        let hiTok = query.count > 2 ? query.count - 1 : query.count
        var rel = [Float](repeating: -Float.greatestFiniteMagnitude, count: n)
        for t in lo..<hiTok {
            let qt = query[t]
            var sims = [Float](repeating: 0, count: n)
            var sum: Float = 0
            for k in 0..<n {
                var dot: Float = 0
                for i in 0..<qt.count { dot += qt[i] * patches[k][i] }
                sims[k] = dot
                sum += dot
            }
            let mean = sum / Float(n)
            for k in 0..<n {
                let v = sims[k] - mean
                if v > rel[k] { rel[k] = v }
            }
        }
        for k in 0..<n { rel[k] = max(0, rel[k]) }
        let peak = rel.max() ?? 0
        guard peak > 1e-6 else { return zeros }
        var norm = rel.map { $0 / peak }

        // Single-location highlight: keep only the connected blob containing the peak patch
        // (4-connected cells ≥ 0.6 of the peak), and zero every other scattered hit — so the
        // overlay marks ONE region instead of several.
        let side = patchGridSide
        guard let peakIdx = norm.indices.max(by: { norm[$0] < norm[$1] }) else { return norm }
        let keepRatio: Float = 0.6
        var keep = Set<Int>()
        var stack = [peakIdx]
        while let c = stack.popLast() {
            if keep.contains(c) || norm[c] < keepRatio { continue }
            keep.insert(c)
            let r = c / side, col = c % side
            if r > 0 { stack.append(c - side) }
            if r < side - 1 { stack.append(c + side) }
            if col > 0 { stack.append(c - 1) }
            if col < side - 1 { stack.append(c + 1) }
        }
        for k in 0..<n where !keep.contains(k) { norm[k] = 0 }
        return norm
    }

    /// Encodes the query once and ranks a pre-encoded corpus by MaxSim (descending).
    public func retrieve(
        query: String, over pages: [PageEmbedding], topK: Int = 5
    ) async throws -> [(index: Int, score: Float)] {
        let q = try await encode(query: query)
        let scored = pages.enumerated()
            .map { (index: $0.offset, score: Self.score(query: q, page: $0.element)) }
            .sorted { $0.score > $1.score }
        return Array(scored.prefix(topK))
    }

    // MARK: - Tiled (reliable spatial grounding)

    /// Encodes a page as a `rows × cols` grid of tiles — each tile is cropped from the page and
    /// run through the (single-tile) document encoder. Reuses the shipped doc bundle; no
    /// re-export. ~rows·cols encoder calls per page.
    public func encodeTiled(
        page image: CGImage, rows: Int = 3, cols: Int = 3
    ) async throws -> TiledPageEmbedding {
        let w = image.width, h = image.height
        var tiles: [TiledPageEmbedding.Tile] = []
        for r in 0..<rows {
            for c in 0..<cols {
                let x = c * w / cols, y = r * h / rows
                let cw = (c + 1) * w / cols - x, ch = (r + 1) * h / rows - y
                guard let crop = image.cropping(to: CGRect(x: x, y: y, width: cw, height: ch))
                else { continue }
                let emb = try await encode(page: crop)
                let rect = CGRect(x: Double(c) / Double(cols), y: Double(r) / Double(rows),
                                  width: 1.0 / Double(cols), height: 1.0 / Double(rows))
                tiles.append(.init(rect: rect, embedding: emb))
            }
        }
        return TiledPageEmbedding(tiles: tiles)
    }

    /// MaxSim over the union of all tiles' image patches — the page-level retrieval score.
    public func score(query: [[Float]], tiledPage: TiledPageEmbedding) -> Float {
        var total: Float = 0
        for q in query {
            var best = -Float.greatestFiniteMagnitude
            for tile in tiledPage.tiles {
                let s = bestPatchDot(q, in: tile.embedding)
                if s > best { best = s }
            }
            total += best
        }
        return total
    }

    /// The page region (normalized rect) that best matches the query — highlight this one tile.
    public func bestTile(query: [[Float]], tiledPage: TiledPageEmbedding) -> CGRect? {
        var bestRect: CGRect?
        var bestScore = -Float.greatestFiniteMagnitude
        for tile in tiledPage.tiles {
            let s = imageMaxSim(query: query, page: tile.embedding)
            if s > bestScore { bestScore = s; bestRect = tile.rect }
        }
        return bestRect
    }

    // MaxSim over a single tile's image patches only (template tokens excluded — they're constant
    // across tiles and would not differentiate them).
    private func imageMaxSim(query: [[Float]], page: PageEmbedding) -> Float {
        var total: Float = 0
        for q in query { total += max(0, bestPatchDot(q, in: page)) }
        return total
    }

    private func bestPatchDot(_ q: [Float], in page: PageEmbedding) -> Float {
        let upper = min(imageTokenStart + patchGridSide * patchGridSide, page.tokens.count)
        var best = -Float.greatestFiniteMagnitude
        var k = imageTokenStart
        while k < upper {
            let d = page.tokens[k]
            var dot: Float = 0
            for i in 0..<q.count { dot += q[i] * d[i] }
            if dot > best { best = dot }
            k += 1
        }
        return best
    }

    // MARK: - Helpers

    private func maskShape() -> [Int] {
        // [1, 1, R, R] mirroring the doc pixel layout's leading dims.
        [1, 1, docImageSize, docImageSize]
    }

    private func rows(_ flat: [Float], count: Int) -> [[Float]] {
        var out = [[Float]]()
        out.reserveCapacity(count)
        for t in 0..<count { out.append(Array(flat[(t * dim)..<((t + 1) * dim)])) }
        return out
    }
}
