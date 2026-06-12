// TextEmbedder.swift — on-device text embeddings for retrieval/RAG, over a static
// embedding graph (EmbeddingGemma export: transformer + mean pooling + dense projection
// + L2 normalize, all in-graph).

@_exported import CoreAIKitCore
@_exported import CoreAIKitVision

import Foundation
import Tokenizers

public final class TextEmbedder: @unchecked Sendable {
    /// Task prefixes prepended before tokenization — embedding models are trained with
    /// asymmetric query/document prompts.
    public struct Prompts: Sendable {
        public var query: String
        public var document: String

        public init(query: String, document: String) {
            self.query = query
            self.document = document
        }

        /// EmbeddingGemma's retrieval prompts (from its sentence-transformers config).
        public static let embeddingGemma = Prompts(
            query: "task: search result | query: ",
            document: "title: none | text: ")
    }

    private let graph: GraphModel
    private let tokenizer: any Tokenizer
    private let prompts: Prompts
    private let idsInput: String
    private let maskInput: String
    private let outputName: String
    public let sequenceLength: Int
    public let dimension: Int

    /// Loads a bundle directory holding one `*.aimodel` plus a `tokenizer/` folder.
    public init(
        bundleAt url: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        prompts: Prompts = .embeddingGemma
    ) async throws {
        guard
            let modelURL = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil
            ).first(where: { $0.pathExtension == "aimodel" })
        else {
            throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
        }
        let graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.graph = graph
        self.prompts = prompts
        self.tokenizer = try await AutoTokenizer.from(
            modelFolder: url.appendingPathComponent("tokenizer"))

        guard
            let idsInput = graph.inputNames.first(where: { $0.contains("input_ids") }),
            let maskInput = graph.inputNames.first(where: { $0.contains("mask") }),
            let idsShape = graph.shape(ofInput: idsInput), idsShape.count == 2,
            let outputName = graph.outputNames.first,
            let outputShape = graph.shape(ofOutput: outputName)
        else {
            throw VisionError.bundleLayout(
                "unexpected embedding graph contract: inputs \(graph.inputNames), "
                    + "outputs \(graph.outputNames)")
        }
        self.idsInput = idsInput
        self.maskInput = maskInput
        self.sequenceLength = idsShape[1]
        self.outputName = outputName
        self.dimension = outputShape[outputShape.count - 1]
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .embeddingGemma300m,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    // MARK: - Embedding

    /// Embeds a search query (query prompt applied). L2-normalized.
    public func embed(query: String) async throws -> [Float] {
        try await embed(text: prompts.query + query)
    }

    /// Embeds a document/passage (document prompt applied). L2-normalized.
    public func embed(document: String) async throws -> [Float] {
        try await embed(text: prompts.document + document)
    }

    /// Embeds raw text with no prompt prefix. L2-normalized.
    public func embed(text: String) async throws -> [Float] {
        var ids = tokenizer.encode(text: text).map(Int32.init)
        if ids.count > sequenceLength {
            ids = Array(ids.prefix(sequenceLength))
        }
        var mask = [Int32](repeating: 1, count: ids.count)
        let pad = Int32(0)
        while ids.count < sequenceLength {
            ids.append(pad)
            mask.append(0)
        }
        let outputs = try await graph.run([
            idsInput: .int32(ids, shape: [1, sequenceLength]),
            maskInput: .int32(mask, shape: [1, sequenceLength]),
        ])
        guard let embedding = outputs[outputName] else {
            throw VisionError.missingOutput(outputName)
        }
        return Array(embedding.floats().prefix(dimension))
    }

    /// Dot product — embeddings are L2-normalized in-graph, so this is cosine.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "dimension mismatch")
        var dot: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i] }
        return dot
    }
}
