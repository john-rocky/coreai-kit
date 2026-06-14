import Foundation
import XCTest

@testable import CoreAIKitEmbeddings

/// Parity + ranking smoke for the Qwen3 retrieval stack (embedder + reranker) over real local
/// bundles. Opt-in:
///
///     KIT_QWEN3_EMBED_BUNDLE=/path/to/_qwen3emb_out \
///     KIT_QWEN3_RERANK_BUNDLE=/path/to/_qwen3rerank_out \
///     swift test --filter Qwen3RetrievalSmoke
///
/// Each bundle directory holds the *.aimodel, tokenizer/, and the reference.json the export
/// scripts wrote (torch-side embeddings / scores), so this checks Swift-runtime ≡ torch.
final class Qwen3RetrievalSmokeTests: XCTestCase {

    // MARK: Embedder

    private struct EmbedRef: Decodable {
        struct Entry: Decodable { let kind: String; let text: String }
        let texts: [String: Entry]
        let embeddings: [String: [Float]]
        let expected_topdoc: [String: String]
    }

    private func cos(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0
        for i in 0..<min(a.count, b.count) { d += a[i] * b[i] }
        return d
    }

    func testEmbeddingTorchParityAndRanking() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_QWEN3_EMBED_BUNDLE"] else {
            throw XCTSkip("Set KIT_QWEN3_EMBED_BUNDLE to the local embedding bundle directory.")
        }
        let bundle = URL(fileURLWithPath: path)
        let ref = try JSONDecoder().decode(
            EmbedRef.self,
            from: Data(contentsOf: bundle.appendingPathComponent("reference.json")))

        let embedder = try await TextEmbedder(
            bundleAt: bundle, computeUnits: .gpu, prompts: .qwen3Embedding)
        XCTAssertEqual(embedder.dimension, 1024)

        var vectors: [String: [Float]] = [:]
        for (key, entry) in ref.texts {
            let v = entry.kind == "query"
                ? try await embedder.embed(query: entry.text)
                : try await embedder.embed(document: entry.text)
            // Cross-runtime parity: Swift embedding ≡ the torch reference vector.
            let c = cos(v, ref.embeddings[key]!)
            print("PROBE embed \(key): swift-vs-torch cos=\(c)")
            XCTAssertGreaterThan(c, 0.999, key)
            vectors[key] = v
        }

        // Retrieval: each query's top-scoring document matches the expected pairing.
        let docKeys = vectors.keys.filter { $0.hasPrefix("d_") }
        for (qKey, expected) in ref.expected_topdoc {
            let top = docKeys.max { cos(vectors[qKey]!, vectors[$0]!) < cos(vectors[qKey]!, vectors[$1]!) }!
            print("PROBE rank \(qKey): swift top=\(top) expected=\(expected)")
            XCTAssertEqual(top, expected, qKey)
        }
    }

    /// Exercises the Hub download of a flat (root-layout) repo end to end. Opt-in & networked:
    ///     KIT_QWEN3_DOWNLOAD=1 swift test --filter testHubDownloadFlatRepo
    func testHubDownloadFlatRepo() async throws {
        guard ProcessInfo.processInfo.environment["KIT_QWEN3_DOWNLOAD"] == "1" else {
            throw XCTSkip("Set KIT_QWEN3_DOWNLOAD=1 to download from the Hub (networked, ~1.1 GB).")
        }
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("kit-qwen3emb-\(UUID().uuidString)")
        let store = ModelStore(directory: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let embedder = try await TextEmbedder(
            model: .qwen3Embedding0_6B, store: store, prompts: .qwen3Embedding)
        let q = try await embedder.embed(query: "What is the capital of Japan?")
        let d = try await embedder.embed(document: "Tokyo is the capital of Japan.")
        let far = try await embedder.embed(document: "Python is a programming language.")
        XCTAssertEqual(embedder.dimension, 1024)
        XCTAssertGreaterThan(
            TextEmbedder.cosineSimilarity(q, d), TextEmbedder.cosineSimilarity(q, far))
    }

    // MARK: Reranker

    private struct RerankRef: Decodable {
        struct Pair: Decodable { let relevant: Bool; let query: String; let doc: String }
        let pairs: [String: Pair]
        let scores: [String: Float]
        let rank_groups: [String: [String]]
    }

    func testRerankerTorchParityAndRanking() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_QWEN3_RERANK_BUNDLE"] else {
            throw XCTSkip("Set KIT_QWEN3_RERANK_BUNDLE to the local reranker bundle directory.")
        }
        let bundle = URL(fileURLWithPath: path)
        let ref = try JSONDecoder().decode(
            RerankRef.self,
            from: Data(contentsOf: bundle.appendingPathComponent("reference.json")))

        let reranker = try await Reranker(bundleAt: bundle, computeUnits: .gpu)

        var scored: [String: Float] = [:]
        for (key, pair) in ref.pairs {
            let s = try await reranker.score(query: pair.query, document: pair.doc)
            print("PROBE rerank \(key): swift=\(s) torch=\(ref.scores[key]!)")
            // Cross-runtime parity (fp16 GPU vs torch); ranking is huge-margin so the tol is loose.
            XCTAssertEqual(s, ref.scores[key]!, accuracy: 0.05, key)
            scored[key] = s
        }

        // Each relevant document must outrank the irrelevant one sharing its query.
        for (topic, group) in ref.rank_groups {
            XCTAssertGreaterThan(scored[group[0]]!, scored[group[1]]!, topic)
        }
    }
}
