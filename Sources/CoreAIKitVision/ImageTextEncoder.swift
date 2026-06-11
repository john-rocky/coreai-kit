// ImageTextEncoder.swift — typed CLIP-style joint image/text embedding pipeline.
//
// The exported CLIP graph is a single joint function (pixel_values, input_ids,
// attention_mask) → (…, image_embeds, text_embeds): encoding one modality feeds a dummy
// counterpart input and reads only the relevant output. Embeddings are L2-normalized
// in-graph, so cosine similarity is a plain dot product.

import CoreGraphics
import Foundation

public final class ImageTextEncoder: @unchecked Sendable {
    private let graph: GraphModel
    private let tokenizer: CLIPTokenizer
    private let preprocessor: ImagePreprocessor

    private let imageInput: String
    private let textInput: String
    private let maskInput: String?
    private let imageShape: [Int]  // [1, 3, H, W]
    private let textShape: [Int]  // [B, 77]
    private let imageEmbedsOutput = "image_embeds"
    private let textEmbedsOutput = "text_embeds"

    public let embeddingDimension: Int

    /// Loads a bundle directory holding one `*.aimodel` plus an HF `tokenizer.json`.
    public init(
        bundleAt url: URL, computeUnits: GraphModel.ComputeUnits = .neuralEngine
    ) async throws {
        let fm = FileManager.default
        let modelURL: URL
        let tokenizerRoot: URL
        if url.pathExtension == "aimodel" {
            modelURL = url
            tokenizerRoot = url.deletingLastPathComponent()
        } else {
            guard
                let found = try fm.contentsOfDirectory(
                    at: url, includingPropertiesForKeys: nil
                ).first(where: { $0.pathExtension == "aimodel" })
            else {
                throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
            }
            modelURL = found
            tokenizerRoot = url
        }

        let graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.graph = graph
        self.tokenizer = try CLIPTokenizer(bundleAt: tokenizerRoot)

        // Identify inputs by name with a shape-rank fallback, then adapt to the graph's
        // static dimensions instead of assuming them.
        func pick(_ namePart: String, rank: Int) throws -> String {
            if let byName = graph.inputNames.first(where: { $0.contains(namePart) }) {
                return byName
            }
            if let byRank = graph.inputNames.first(where: {
                graph.shape(ofInput: $0)?.count == rank
            }) {
                return byRank
            }
            throw VisionError.bundleLayout(
                "could not identify '\(namePart)' input among \(graph.inputNames)")
        }
        self.imageInput = try pick("pixel", rank: 4)
        self.textInput = try pick("input_ids", rank: 2)
        self.maskInput = graph.inputNames.first(where: { $0.contains("mask") })

        guard let imageShape = graph.shape(ofInput: imageInput), imageShape.count == 4,
            let textShape = graph.shape(ofInput: textInput), textShape.count == 2
        else {
            throw VisionError.bundleLayout("unexpected CLIP input shapes")
        }
        self.imageShape = imageShape
        self.textShape = textShape
        self.preprocessor = ImagePreprocessor(
            size: imageShape[2], mean: ImagePreprocessor.clip224.mean,
            std: ImagePreprocessor.clip224.std)

        guard let embedsShape = graph.shape(ofOutput: imageEmbedsOutput) else {
            throw VisionError.bundleLayout(
                "no '\(imageEmbedsOutput)' output among \(graph.outputNames)")
        }
        self.embeddingDimension = embedsShape[embedsShape.count - 1]
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .clipViTB32,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .neuralEngine,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    // MARK: - Encoding

    /// Embeds an image (preprocessing included). The vector is L2-normalized.
    public func encode(image: CGImage) async throws -> [Float] {
        let pixels = try preprocessor.chw(from: image)
        let outputs = try await graph.run(inputs(pixels: pixels, tokenRow: nil))
        return try embeddingRow(outputs, key: imageEmbedsOutput)
    }

    /// Embeds a text query. The vector is L2-normalized.
    public func encode(text: String) async throws -> [Float] {
        let ids = tokenizer.encode(text, contextLength: textShape[1])
        let outputs = try await graph.run(inputs(pixels: nil, tokenRow: ids))
        return try embeddingRow(outputs, key: textEmbedsOutput)
    }

    /// Dot product — inputs are already L2-normalized, so this is cosine similarity.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "dimension mismatch")
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot
    }

    // MARK: - Private

    private func inputs(
        pixels: [Float]?, tokenRow: [Int32]?
    ) -> [String: TensorValue] {
        var result: [String: TensorValue] = [:]

        result[imageInput] = .float32(
            pixels ?? [Float](repeating: 0, count: imageShape.reduce(1, *)),
            shape: imageShape)

        // The joint function always wants a full [B, L] text batch: row 0 carries the real
        // query (or a minimal [sot, eot] dummy), remaining rows replicate the dummy.
        let batch = textShape[0]
        let length = textShape[1]
        var dummy = [Int32](repeating: CLIPTokenizer.eotTokenId, count: length)
        dummy[0] = CLIPTokenizer.sotTokenId
        let firstRow = tokenRow ?? dummy
        var ids: [Int32] = []
        ids.reserveCapacity(batch * length)
        ids += firstRow
        for _ in 1..<max(batch, 1) { ids += dummy }
        result[textInput] = .int32(ids, shape: textShape)

        if let maskInput {
            // 1 through the first EOT (inclusive), 0 after — HF processor semantics.
            var mask = [Int32]()
            mask.reserveCapacity(batch * length)
            for row in 0..<batch {
                let rowIds = Array(ids[(row * length)..<((row + 1) * length)])
                let eotIndex = rowIds.firstIndex(of: CLIPTokenizer.eotTokenId) ?? (length - 1)
                for i in 0..<length {
                    mask.append(i <= eotIndex ? 1 : 0)
                }
            }
            result[maskInput] = .int32(mask, shape: textShape)
        }

        return result
    }

    private func embeddingRow(_ outputs: [String: TensorValue], key: String) throws -> [Float] {
        guard let value = outputs[key] else {
            throw VisionError.missingOutput(key)
        }
        let flat = value.floats()
        return Array(flat.prefix(embeddingDimension))
    }
}
