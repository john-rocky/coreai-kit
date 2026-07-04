// QuickStart.swift — the take-home core of this runner: query + page images in → ranked pages
// out, one typed function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly
// this function; the GUI app drives the same `VisualDocumentRetriever(catalog:)` gesture with
// tiled per-page encoding for spatial highlights. Want visual document search in your own app?
// This file is the part you copy; the model card's 💻 snippet is the marked block below.

import CoreAIKitEmbeddings
import Foundation

/// Rank page images against a text query with the visual document retriever in the catalog
/// (`ModelCatalog.builtin.available(.retrieval)`: ColModernVBERT). Pages are matched as
/// pictures — tables, charts, scans — no OCR. First use downloads the two encoder bundles
/// (progress via `downloadProgress`), later runs load from the local cache. Encode a corpus
/// once and keep the embeddings; scoring a query is then just host-side MaxSim.
func search(
    query: String,
    pages: [URL],
    model id: String = "colmodernvbert",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> [(page: URL, score: Float)] {
    // CARD-SNIPPET-BEGIN
    let retriever = try await VisualDocumentRetriever(
        catalog: id, downloadProgress: downloadProgress)
    var corpus: [VisualDocumentRetriever.PageEmbedding] = []
    for url in pages {
        corpus.append(try await retriever.encode(page: ImageFile.load(url).cgImage))
    }
    let hits = try await retriever.retrieve(query: query, over: corpus, topK: pages.count)
    // CARD-SNIPPET-END
    return hits.map { (page: pages[$0.index], score: $0.score) }
}
