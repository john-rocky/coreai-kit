import Foundation
import Tokenizers
import XCTest

@testable import CoreAIKit

final class KitTextNormalizerTests: XCTestCase {

    // MARK: - Chunking

    /// Everything shorter than the budget is one chunk — the common case must not be split.
    func testShortTranscriptIsOneChunk() {
        XCTAssertEqual(
            KitTextNormalizer.chunkRanges(count: 120, budget: 450) { _ in true },
            [0..<120])
        XCTAssertEqual(
            KitTextNormalizer.chunkRanges(count: 450, budget: 450) { _ in true },
            [0..<450])
    }

    /// The pieces are evened out rather than packed. Packing 500 tokens at a 450 budget leaves
    /// a 50-token tail the model rewrites with nothing around it, and it reads like one.
    func testChunksAreEvenedOutNotPacked() {
        let ranges = KitTextNormalizer.chunkRanges(count: 500, budget: 450) { _ in true }
        XCTAssertEqual(ranges, [0..<250, 250..<500])
    }

    /// Every chunk stays inside the budget, and the pieces tile the input exactly — no token
    /// dropped, none rewritten twice.
    func testChunksTileTheInputWithinBudget() {
        for count in [451, 900, 901, 2_048, 4_097] {
            let ranges = KitTextNormalizer.chunkRanges(count: count, budget: 450) { _ in true }
            XCTAssertEqual(ranges.first?.lowerBound, 0, "count=\(count)")
            XCTAssertEqual(ranges.last?.upperBound, count, "count=\(count)")
            for (a, b) in zip(ranges, ranges.dropFirst()) {
                XCTAssertEqual(a.upperBound, b.lowerBound, "gap or overlap at count=\(count)")
            }
            for range in ranges {
                XCTAssertLessThanOrEqual(range.count, 450, "over budget at count=\(count)")
                XCTAssertGreaterThan(range.count, 0, "empty chunk at count=\(count)")
            }
        }
    }

    /// A cut lands in front of a word, never inside one.
    func testChunksCutAtWordBoundaries() {
        // Word starts every 7 tokens.
        let ranges = KitTextNormalizer.chunkRanges(count: 1_000, budget: 450) { $0 % 7 == 0 }
        for range in ranges.dropLast() {
            XCTAssertEqual(range.upperBound % 7, 0, "cut inside a word at \(range.upperBound)")
        }
        XCTAssertEqual(ranges.last?.upperBound, 1_000)
    }

    /// One unbreakable run longer than the budget is cut through rather than looping forever.
    func testUnbreakableRunStillTerminates() {
        let ranges = KitTextNormalizer.chunkRanges(count: 1_000, budget: 450) { _ in false }
        XCTAssertEqual(ranges.last?.upperBound, 1_000)
        XCTAssertFalse(ranges.isEmpty)
        for range in ranges { XCTAssertLessThanOrEqual(range.count, 450) }
    }

    func testEmptyInputHasNoChunks() {
        XCTAssertEqual(KitTextNormalizer.chunkRanges(count: 0, budget: 450) { _ in true }, [])
    }

    // MARK: - Stitching

    func testJoinerMatchesTheStructureTheModelWasAskedFor() {
        XCTAssertEqual(KitTextNormalizer.joiner(structure: .prose, context: .general), " ")
        XCTAssertEqual(KitTextNormalizer.joiner(structure: .prose, context: .email), "\n\n")
        XCTAssertEqual(KitTextNormalizer.joiner(structure: .lists, context: .general), "\n")
        XCTAssertEqual(KitTextNormalizer.joiner(structure: .lists, context: .email), "\n")
    }

    // MARK: - Prompt format

    /// The token sequence the device gate actually ran, transcribed from
    /// `ondevice/PipelinedBench` `ModelSpec(dirName: "s1_mini_decode_int8lin")` — the currency
    /// and date case, the one int4 corrupted. If the Swift rendering ever stops matching this,
    /// the model is being driven outside its trained input format.
    private static let oraclePrompt: [Int32] = [
        151644, 8948, 198, 2610, 525, 264, 1467, 4622, 3135, 369, 8806, 4686,
        9345, 60312, 13, 576, 1946, 12033, 448, 264, 2524, 1555, 37838, 279,
        41328, 11, 5944, 11, 323, 2266, 5003, 26, 4240, 279, 35715, 311, 2432,
        1846, 5003, 323, 2550, 1172, 279, 27722, 1467, 13, 151645, 198, 151644,
        872, 198, 58, 623, 98607, 25, 18267, 8460, 278, 60, 508, 22952, 25,
        60701, 60, 508, 1972, 25, 4586, 921, 1782, 24615, 3697, 311, 17073,
        2326, 16183, 3040, 7739, 323, 32417, 11192, 323, 432, 594, 4152, 389,
        15217, 4843, 17073, 17073, 4743, 151645, 198, 151644, 77091, 198,
        151667, 271, 151668, 271,
    ]

    private static let oracleTranscript =
        "the invoice came to twenty three thousand four hundred and fifty dollars and it's "
        + "due on march third twenty twenty six"

    /// Renders the prompt against the shipped tokenizer and compares it token for token.
    ///
    ///     KIT_S1_TOKENIZER=<bundle>/tokenizer swift test --filter KitTextNormalizer
    func testPromptMatchesTheGatedTokenSequence() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_S1_TOKENIZER"] else {
            throw XCTSkip("Set KIT_S1_TOKENIZER to the S1-mini bundle's tokenizer/ directory.")
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: path))
        let rendered = try KitTextNormalizer.promptTokens(
            for: Self.oracleTranscript, styling: .semiFormal, structure: .prose,
            context: .general, tokenizer: tokenizer)
        XCTAssertEqual(rendered, Self.oraclePrompt)
    }

    /// The thinking block must arrive already closed. With thinking left on, S1-mini emits an
    /// empty `<think>` block and stops, and every call returns "" — a pipeline that looks like
    /// it works and produces nothing.
    func testPromptEndsWithAClosedThinkingBlock() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_S1_TOKENIZER"] else {
            throw XCTSkip("Set KIT_S1_TOKENIZER to the S1-mini bundle's tokenizer/ directory.")
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: path))
        for styling in TranscriptStyling.allCases {
            for structure in TranscriptStructure.allCases {
                for context in TranscriptContext.allCases {
                    let ids = try KitTextNormalizer.promptTokens(
                        for: "um so anyway", styling: styling, structure: structure,
                        context: context, tokenizer: tokenizer)
                    XCTAssertEqual(
                        Array(ids.suffix(4)), [151667, 271, 151668, 271],
                        "\(styling.rawValue)/\(structure.rawValue)/\(context.rawValue)")
                }
            }
        }
    }
}
