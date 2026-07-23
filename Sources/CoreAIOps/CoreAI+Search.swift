// CoreAI+Search.swift — anchored semantic search over local catalog embeddings.
//
// ```swift
// let hits = try await CoreAI.search("refund policy", in: paragraphs, topK: 3)
// ```
//
// The op embeds the query and every document per call — right for "rank these strings
// now", wasteful for a corpus queried repeatedly. A persistent index should hold a
// `TextEmbedder` (CoreAIKitEmbeddings) and store the document vectors once.

import CoreAIKit
import CoreAIKitEmbeddings
import Foundation

/// One ranked result from `CoreAI.search`.
public struct SearchHit: Sendable, Identifiable {
    /// Position of the document in the input array.
    public let index: Int
    /// The matched document, verbatim.
    public let document: String
    /// Cosine similarity to the query in [-1, 1] (embeddings are L2-normalized).
    public let score: Float

    public var id: Int { index }
}

extension CoreAI {
    /// Default embedding model (768-d, multilingual).
    public static let defaultEmbeddingModel = "embeddinggemma-300m"

    /// Query + strings → ranked matches, best first, by semantic similarity. First use
    /// downloads and loads the model (cached afterwards).
    public static func search(
        _ query: String, in documents: [String], topK: Int = 5,
        options: OpOptions = OpOptions()
    ) async throws -> [SearchHit] {
        let embedder = try await SearchOpModels.shared.embedder(
            catalog: options.model ?? defaultEmbeddingModel)
        let queryVector = try await embedder.embed(query: query)
        var hits: [SearchHit] = []
        hits.reserveCapacity(documents.count)
        for (index, document) in documents.enumerated() {
            let vector = try await embedder.embed(document: document)
            hits.append(
                SearchHit(
                    index: index, document: document,
                    score: TextEmbedder.cosineSimilarity(queryVector, vector)))
        }
        return Array(hits.sorted { $0.score > $1.score }.prefix(topK))
    }
}

/// Process-wide cache of loaded embedders, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached.
actor SearchOpModels {
    static let shared = SearchOpModels()

    private var embedderLoads: [String: Task<TextEmbedder, Error>] = [:]

    func embedder(catalog id: String) async throws -> TextEmbedder {
        if let load = embedderLoads[id] { return try await load.value }
        let load = Task<TextEmbedder, Error> {
            let entry = try await ModelCatalog.entry(forID: id, expecting: .textEmbedding)
            guard let model = entry.modelID else {
                throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
            }
            // Prompt prefixes are per-family; every catalog text-embedding entry today
            // is EmbeddingGemma-shaped, which is also `TextEmbedder`'s default.
            return try await TextEmbedder(
                model: model, downloadProgress: OpDownloads.forward)
        }
        embedderLoads[id] = load
        do {
            return try await load.value
        } catch {
            embedderLoads[id] = nil
            throw error
        }
    }
}
