import Foundation
import XCTest

@testable import CoreAIKitEmbeddings

/// Parity + ranking smoke over a real embedding bundle. Opt-in:
///
///     KIT_EMBED_BUNDLE=/path/to/dir swift test --filter TextEmbedderSmoke
///
/// The bundle directory holds the *.aimodel, tokenizer/, and the reference.json the
/// export script wrote (torch-side cosines for fixed text pairs).
final class TextEmbedderSmokeTests: XCTestCase {
    private struct Reference: Decodable {
        struct Entry: Decodable {
            let kind: String
            let text: String
        }
        let texts: [String: Entry]
        let cosines: [String: Double]
    }

    private func norm(_ v: [Float]) -> Float {
        var sum: Float = 0
        for x in v { sum += x * x }
        return sum.squareRoot()
    }

    func testTorchParityAndSemanticRanking() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_EMBED_BUNDLE"] else {
            throw XCTSkip("Set KIT_EMBED_BUNDLE to a local embedding bundle directory.")
        }
        let bundle = URL(fileURLWithPath: path)
        let ref = try JSONDecoder().decode(
            Reference.self,
            from: Data(contentsOf: bundle.appendingPathComponent("reference.json")))

        let embedder = try await TextEmbedder(bundleAt: bundle, computeUnits: .gpu)
        XCTAssertEqual(embedder.dimension, 768)
        XCTAssertEqual(embedder.sequenceLength, 256)

        var vectors: [String: [Float]] = [:]
        for (key, entry) in ref.texts {
            let vector =
                entry.kind == "query"
                ? try await embedder.embed(query: entry.text)
                : try await embedder.embed(document: entry.text)
            XCTAssertEqual(norm(vector), 1.0, accuracy: 0.05, key)
            vectors[key] = vector
        }

        // Cross-runtime parity: Swift cosines must match the torch export's.
        for (pair, expected) in ref.cosines {
            let parts = pair.split(separator: "|").map(String.init)
            let got = TextEmbedder.cosineSimilarity(vectors[parts[0]]!, vectors[parts[1]]!)
            print("PROBE cos \(pair): swift=\(got) torch=\(expected)")
            XCTAssertEqual(Double(got), expected, accuracy: 0.05, pair)
        }

        // Retrieval semantics: each query must rank its matching document first.
        let queryBike = vectors["query_bike"]!
        let queryCapital = vectors["query_capital"]!
        let docBike = vectors["doc_bike"]!
        let docTokyo = vectors["doc_tokyo"]!
        XCTAssertGreaterThan(
            TextEmbedder.cosineSimilarity(queryBike, docBike),
            TextEmbedder.cosineSimilarity(queryBike, docTokyo))
        XCTAssertGreaterThan(
            TextEmbedder.cosineSimilarity(queryCapital, docTokyo),
            TextEmbedder.cosineSimilarity(queryCapital, docBike))
    }
}
