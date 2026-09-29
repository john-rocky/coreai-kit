// DeciderVisionTests.swift — the parts of decider-2b-vision's read-out that are checkable without weights: the chunk
// order, the author's slot rule, how a typed question is laid out, the patch layout, the catalog entry, and — with
// the bundle's tokenizer and the zoo's fixture — every row the kit renders against the author's processor ids:
//
//     KIT_DECIDERVISION_FIXTURE=<zoo>/models/decider-2b-vision/fixtures-decider-2b-vision.json \
//     [KIT_DECIDERVISION_TOKENIZER=<decoder bundle>/tokenizer] swift test --filter DeciderVision
//
// Without KIT_DECIDERVISION_TOKENIZER the tokenizer is read from the ModelStore cache, where a first
// `KitVisionDecider(catalog: "decider-2b-vision")` puts it.

import Foundation
import Testing
import Tokenizers

@testable import CoreAIKit

struct DeciderVisionTests {
    // MARK: - The decoder's chunk order

    /// One slot at 104 (the fixture's r01 at g256): six 16-token prefill calls, then `main` from 96 through the
    /// slot — the 15 calls the zoo's Swift run made for that row.
    @available(macOS 27, iOS 27, *)
    @Test func aSlotIsReadAfterWholeChunksThenSingleTokens() {
        let steps = DeciderVisionDecoder.schedule(slots: [104], chunk: 16)
        #expect(steps.filter(\.prefill).map(\.start) == [0, 16, 32, 48, 64, 80])
        #expect(steps.filter { !$0.prefill }.map(\.start) == Array(96...104))
        #expect(steps.compactMap(\.reads) == [104])
        #expect(steps.last == .init(prefill: false, start: 104, count: 1, reads: 104))
    }

    /// A prefill call answers a slot when the slot is its last token; the next slot starts from where it ended.
    @available(macOS 27, iOS 27, *)
    @Test func aChunkEndingAtASlotIsThatSlotsRead() {
        let steps = DeciderVisionDecoder.schedule(slots: [15, 70], chunk: 16)
        #expect(steps[0] == .init(prefill: true, start: 0, count: 16, reads: 15))
        // 16 ..< 64 in chunks (70 − 64 + 1 < 16), then 64 … 70 one at a time.
        #expect(steps.dropFirst().filter(\.prefill).map(\.start) == [16, 32, 48])
        #expect(steps.filter { !$0.prefill }.map(\.start) == Array(64...70))
        #expect(steps.compactMap(\.reads) == [15, 70])
        // Every id is fed exactly once, in order.
        #expect(steps.flatMap { Array($0.start..<($0.start + $0.count)) } == Array(0...70))
    }

    @available(macOS 27, iOS 27, *)
    @Test func withoutAPrefillFunctionEveryCallIsOneToken() {
        let steps = DeciderVisionDecoder.schedule(slots: [3, 9], chunk: nil)
        #expect(steps.allSatisfy { !$0.prefill && $0.count == 1 })
        #expect(steps.map(\.start) == Array(0...9))
        #expect(steps.compactMap(\.reads) == [3, 9])
    }

    // MARK: - The author's slot rule and question layout

    @Test func theSlotIsSpaceParenAfterAColonWithAnswerJustBefore() {
        let r = DeciderVisionPromptRenderer.self
        // "…Answer: (" and "…Answer 2: (" (the number is one more token).
        #expect(r.findSlots([7, 7, r.answer, r.colon, r.slotToken]) == [4])
        #expect(r.findSlots([7, r.answer, 220, 17, r.colon, r.slotToken, 9, r.answer, r.colon, r.slotToken]) == [5, 9])
        // " (" without the colon, or with "Answer" more than five tokens back, is not a slot.
        #expect(r.findSlots([7, r.answer, 9, r.slotToken]).isEmpty)
        #expect(r.findSlots([r.answer, 9, 9, 9, 9, 9, r.colon, r.slotToken]).isEmpty)
        #expect(r.processorIDs([r.visionStart, r.vocab, r.vocab + 63, r.visionEnd, 5]) == [
            r.visionStart, r.imagePad, r.imagePad, r.visionEnd, 5,
        ])
    }

    /// The author's `decide_json`: a yes/no is the choice no / yes, a score lists "k: level", a choice its options
    /// (with `name: criterion` when an option carries a description, the decider form's rule).
    @Test func typedQuestionsListTheAuthorsOptions() {
        let r = DeciderVisionPromptRenderer.self
        #expect(r.options(for: .noul("Is the ball visible?")) == ["no", "yes"])
        #expect(r.options(for: .noul("Urgent?", yes: "today", no: "can wait")) == ["no: can wait", "yes: today"])
        #expect(r.options(for: .score("How close?", levels: ["far", "near", "touching"])) == [
            "0: far", "1: near", "2: touching",
        ])
        #expect(r.options(for: .choice("Which?", ["up", "down", "stay"])) == ["up", "down", "stay"])
        #expect(
            r.options(for: .choice("Which?", options: [.init(id: "up", description: "move the paddle up"), .init("stay")]))
                == ["up: move the paddle up", "stay"])
    }

    /// The answer folds the slot's probabilities the author's way: P(yes) for a yes/no, Σ k · p(k) for a score.
    @Test func answersFoldTheLetterProbabilities() throws {
        let timing = Decision.Timing(promptTokens: 10, reusedTokens: 0, seconds: 0.1, imageSeconds: 0.02, decoderSeconds: 0.07)
        let noul = DecisionPrompt.answer(for: .noul("Visible?"), probabilities: [0.25, 0.75], timing: timing)
        #expect(noul.noul == 0.75)
        let score = DecisionPrompt.answer(
            for: .score("How close?", levels: ["far", "near", "touching"]), probabilities: [0.2, 0.3, 0.5],
            timing: timing)
        #expect(abs(try #require(score.score) - 1.3) < 1e-12)
        #expect(score.timing.imageSeconds == 0.02 && score.timing.decoderSeconds == 0.07)
        // The text path's timing keeps its three fields and says nothing about an image.
        let text = Decision.Timing(promptTokens: 10, reusedTokens: 4, seconds: 0.1)
        #expect(text.imageSeconds == nil && text.decoderSeconds == nil && text.processedTokens == 6)
    }

    // MARK: - The image host

    @available(macOS 27, iOS 27, *)
    @Test func gridsNameTheirTiles() {
        #expect(KitVisionDecider.Grid.g256.tile == 256 && KitVisionDecider.Grid.g256.imageTokens == 64)
        #expect(KitVisionDecider.Grid.g448.tile == 448 && KitVisionDecider.Grid.g448.imageTokens == 196)
        #expect(KitVisionDecider.Grid(tile: 448) == .g448 && KitVisionDecider.Grid(tile: 300) == nil)
    }

    /// Patches run block by block (2 × 2 merge), each vector (C, T, 16, 16) with the frame at both T, normalised
    /// (x / 255 − 0.5) / 0.5.
    @Test func patchesAreMergeBlockMajorWithTheFrameTwice() throws {
        // A 32×32 tile: one merge block of four 16×16 patches; the pixel value encodes its column.
        var pixels = [UInt8](repeating: 0, count: 32 * 32 * 3)
        for y in 0..<32 {
            for x in 0..<32 {
                pixels[(y * 32 + x) * 3] = UInt8(x)  // R = column
                pixels[(y * 32 + x) * 3 + 1] = UInt8(y)  // G = row
                pixels[(y * 32 + x) * 3 + 2] = 255
            }
        }
        let tile = DeciderVisionPreprocessor.RGB8(width: 32, height: 32, pixels: pixels, decodePath: "test")
        let p = try DeciderVisionPreprocessor.patches(tile)
        let v = DeciderVisionPreprocessor.patchVector
        #expect(p.count == 4 * v)
        func norm(_ x: Int) -> Float { Float((Double(x) / 255.0 - 0.5) / 0.5) }
        // Patch 1 is the top-right one: its R plane starts at column 16; both temporal copies are equal.
        #expect(p[1 * v] == norm(16))
        #expect(p[1 * v + 256] == norm(16))
        // Patch 2 is the bottom-left one: its G plane (channel 1, T 0) starts at row 16.
        #expect(p[2 * v + 2 * 256] == norm(16))
        #expect(p[3 * v + 4 * 256] == norm(255))
        // A tile that is not a square multiple of 32 is refused.
        let odd = DeciderVisionPreprocessor.RGB8(width: 32, height: 16, pixels: [UInt8](repeating: 0, count: 32 * 16 * 3), decodePath: "test")
        #expect(throws: DeciderVisionError.self) { try DeciderVisionPreprocessor.patches(odd) }
    }

    /// Pillow's bicubic weights for a reduction sum to one and span twice the reduction factor either side.
    @Test func resizeWeightsAreNormalisedAndPillowsWidth() {
        let (starts, weights) = DeciderVisionPreprocessor.filterWeights(inSize: 640, outSize: 256)
        #expect(starts.count == 256 && weights.count == 256)
        for w in weights { #expect(abs(w.reduce(0, +) - 1) < 1e-12) }
        // Output 128 of a 2.5× reduction: centre 321.25, support 5, inputs 316 … 326; the last one lies 2.1
        // scaled units out, past the kernel's reach, and weighs 0.
        #expect(starts[128] == 316 && weights[128].count == 11 && weights[128].last == 0)
        // An image already at the tile's size is not resampled.
        let rgb = DeciderVisionPreprocessor.RGB8(width: 2, height: 2, pixels: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], decodePath: "test")
        #expect(DeciderVisionPreprocessor.resizeBicubic(rgb, width: 2, height: 2).pixels == rgb.pixels)
    }

    // MARK: - The catalog entry

    @available(macOS 27, iOS 27, *)
    @Test func theCatalogEntryPairsWithThePreset() throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "decider-2b-vision"))
        let preset = DeciderVisionModelID.decider2bVision
        #expect(entry.kind == .visionDecision)
        #expect(entry.revision == preset.decoder.revision)
        #expect(preset.towers.values.allSatisfy { $0.revision == preset.decoder.revision && $0.repo == entry.repo })
        for key in ["macos", "ios"] { #expect(entry.variants[key]?.path == preset.decoder.path) }
        #expect(Set(preset.towers.keys) == Set(KitVisionDecider.Grid.allCases))
        // Not a text decision model: `TypedDecisions`, `systemone models` and the MCP listing leave it out.
        #expect(!TypedDecisions.supports(entry))
        #expect(DeciderVisionModelID.byCatalogID["decider-2b-vision"] == preset)
        #expect(preset.pinned("abc").towers[.g448]?.revision == "abc")
    }

    // MARK: - Every fixture row, against the author's processor ids (tokenizer only)

    struct Fixture: Decodable {
        struct Request: Decodable {
            struct Question: Decodable {
                let text: String
                let options: [String]
            }
            let id: String
            let image: String?
            let context: String
            let questions: [Question]
        }
        struct Row: Decodable {
            let id: String
            let arm: String
            let request_id: String
            let ids: [Int]
            let slots: [Int]
            let nopts: [Int]
            let rope_shift: Int?
        }
        let schema: String
        let requests: [Request]
        let rows: [Row]
    }

    static let environment = ProcessInfo.processInfo.environment

    /// KIT_DECIDERVISION_TOKENIZER, else the tokenizer a catalog download left in the ModelStore cache.
    static var tokenizerFolder: URL? {
        if let path = environment["KIT_DECIDERVISION_TOKENIZER"] { return URL(fileURLWithPath: path) }
        guard #available(macOS 27, iOS 27, *),
            let bundle = ModelStore.default.localURL(for: DeciderVisionModelID.decider2bVision.decoder)
        else { return nil }
        return bundle.appendingPathComponent("tokenizer")
    }

    static var fixtureEnabled: Bool { environment["KIT_DECIDERVISION_FIXTURE"] != nil && tokenizerFolder != nil }

    /// The fixture's g256, g448 and text runs (76): the kit's row, turned back into the processor's form, equals the
    /// ids the author's `prepare()` fed, with the same answer slots and option counts.
    @Test(.enabled(if: fixtureEnabled, "set KIT_DECIDERVISION_FIXTURE (and the tokenizer) to check the 76 runs"))
    func everyFixtureRowRendersTheAuthorsIDs() async throws {
        let fixture = try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: URL(fileURLWithPath: Self.environment["KIT_DECIDERVISION_FIXTURE"]!)))
        #expect(fixture.schema == "coreai-decider-vision-fixtures/1")
        let tokenizer = try await AutoTokenizer.from(modelFolder: try #require(Self.tokenizerFolder))
        let renderer = DeciderVisionPromptRenderer(tokenizer: tokenizer)
        let requests = Dictionary(uniqueKeysWithValues: fixture.requests.map { ($0.id, $0) })
        var runs = 0, equal = 0, slots = 0
        for row in fixture.rows where ["g256", "g448", "text"].contains(row.arm) {
            let request = try #require(requests[row.request_id])
            let grid: Int? = row.arm == "g256" ? 8 : row.arm == "g448" ? 14 : nil
            let built = try renderer.row(
                state: request.context,
                questions: request.questions.map { .choice($0.text, $0.options) }, grid: grid)
            runs += 1
            slots += built.slots.count
            let same = DeciderVisionPromptRenderer.processorIDs(built.ids) == row.ids && built.slots == row.slots
                && built.optionCounts == row.nopts
            #expect(same, "\(row.id): ids or slots differ from the author's")
            if let shift = row.rope_shift, grid != nil {
                #expect(Int(built.ropeShiftAmount) == shift, "\(row.id): rope shift")
            }
            if same { equal += 1 }
        }
        print("DeciderVision prompt rows: \(equal)/\(runs) runs equal the author's ids, \(slots) slots")
        #expect(runs == 76 && equal == 76 && slots == 108)
    }
}
