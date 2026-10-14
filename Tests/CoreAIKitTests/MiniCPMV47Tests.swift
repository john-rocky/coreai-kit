// MiniCPMV47Tests.swift — MiniCPM-V 4.7's prompt and rope shift, pinned to the ids of the reference
// processor (transformers 5.19.0, `enable_thinking=False`, one 448x448 thumbnail). The shift, the
// geometry and the catalog entry need nothing; the prompts need the bundle's tokenizer:
//
//     KIT_MINICPMV47_TOKENIZER=<decoder bundle>/tokenizer swift test --filter MiniCPMV47
//
// Without KIT_MINICPMV47_TOKENIZER the tokenizer is read from the ModelStore cache, where a first
// `KitVisionModel(catalog: "minicpm-v-4.7-1b")` puts it.

import CoreGraphics
import Foundation
import FoundationModels
import Testing
import Tokenizers

@testable import CoreAIKit

struct MiniCPMV47Tests {
    // MARK: - The rope shift and the geometry

    /// P1 holds `<image>` at 6, the pads at 7…70, `</image>` at 71 and its newline at 72: the reference
    /// shifts from 73 by 58, so the text after the frame resumes at 6 + 8 + 1 = 15 on the canvas.
    @Test func theShiftStartsAfterTheImageEndAndItsNewline() {
        let arch = VLArchitecture.miniCPMV47
        #expect(arch.ropeShiftStart(imageStart: 7) == 73)
        #expect(arch.ropeShiftAmount == 58)
        // P4 opens with a system turn: `<image>` at 17.
        #expect(arch.ropeShiftStart(imageStart: 18) == 84)
        // The Qwen-style shift keeps its numbers: start right after the pads, amount mergedTokens − grid.
        #expect(VLArchitecture.qwen3VL2B.ropeShiftStart(imageStart: 5) == 5 + 196)
        #expect(VLArchitecture.qwen3VL2B.ropeShiftAmount == 196 - 14)
        #expect(VLArchitecture.mineru.ropeShiftAmount == 768 - 32)
    }

    @Test func theArchitectureIsFourSixsTowerWithTheProcessorsBlock() {
        let new = VLArchitecture.miniCPMV47, old = VLArchitecture.miniCPMV46
        #expect(new.vocab == old.vocab && new.mergedTokens == old.mergedTokens && new.grid == old.grid)
        #expect(new.hidden == old.hidden && new.imageSide == old.imageSide && new.patchSize == old.patchSize)
        #expect(new.visionInput == old.visionInput && new.visionOutput == old.visionOutput)
        #expect(new.visionStart == "<image_id>0</image_id><image>" && new.visionEnd == "</image>\n")
        #expect(new.ropeShifted && new.ropeShiftFrameTail == 2)
        #expect(new.assistantPrefix == "<think>\n\n</think>\n\n")
        // 4.6 keeps plain 1D positions and its prompt.
        #expect(!old.ropeShifted && old.ropeShiftFrameTail == 0 && old.assistantPrefix.isEmpty)
    }

    @Test func theCatalogEntryPairsWithThePreset() throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "minicpm-v-4.7-1b"))
        let preset = try #require(VLModelID.byCatalogID["minicpm-v-4.7-1b"])
        #expect(entry.kind == .vlm)
        #expect(preset == .miniCPMV47 && preset.arch == .miniCPMV47)
        #expect(entry.repo == preset.decoder.repo && entry.repo == preset.vision.repo)
        for key in ["macos", "ios"] { #expect(entry.variants[key]?.path == preset.decoder.path) }
        let pinned = preset.pinned(try #require(entry.revision))
        #expect(pinned.decoder.revision == entry.revision && pinned.vision.revision == entry.revision)
    }

    // MARK: - Every reference prompt (the bundle's tokenizer)

    static let pad: Int32 = 248_056
    static let think: [Int32] = [248_068, 271, 248_069, 271]  // <think> "\n\n" </think> "\n\n"
    static let userHead: [Int32] = [248_045, 846, 198]  // <|im_start|> user "\n"
    static let block: [Int32] = [248_090, 15, 248_091, 248_078]  // <image_id> 0 </image_id> <image>
    static let assistantHead: [Int32] = [248_046, 198, 248_045, 74_455, 198] + think

    /// The processor's ids: the question's own tokens around the shared turn markers.
    static func imagePrompt(system: [Int32] = [], question: [Int32]) -> [Int32] {
        system + userHead + block + Array(repeating: pad, count: 64) + [248_079, 198] + question
            + assistantHead
    }

    static let p0: [Int32] = userHead + [7734, 799, 11316, 883, 279, 17415, 13] + assistantHead
    static let p1 = imagePrompt(question: [3710, 369, 303, 411, 2099, 30])
    static let p2 = imagePrompt(question: [72240, 411, 2099, 303, 799, 11316, 13])
    static let p3 = imagePrompt(question: [3710, 1414, 369, 5158, 303, 279, 2099, 30])
    static let p4 = imagePrompt(
        system: [248_045, 8678, 198, 2523, 513, 264, 61446, 17313, 13, 248_046, 198],
        question: [3710, 7736, 635, 488, 1436, 30])
    /// The greedy answer to P1 (32 tokens, not stopped), which turn 2 carries as history.
    static let a1Text =
        "This image shows **Earth** — specifically, a vivid, close-up view of the planet from space.\n\nKey features visible:\n- The blue oceans covering"
    static let a1: [Int32] = [
        1919, 2099, 4774, 2972, 42373, 332, 1892, 11352, 11, 264, 40630, 11, 3160, 5072, 1593, 314, 279, 11247, 494,
        3433, 13, 271, 1536, 4275, 9155, 25, 198, 12, 561, 6105, 51393, 17661,
    ]
    /// The template's render of [P1, A1, "What colors do you see?"].
    static let turn2: [Int32] = p1 + a1 + [248_046, 198] + userHead + [3710, 7736, 635, 488, 1436, 30]
        + assistantHead

    static let environment = ProcessInfo.processInfo.environment

    /// KIT_MINICPMV47_TOKENIZER, else the tokenizer a catalog download left in the ModelStore cache.
    static var tokenizerFolder: URL? {
        if let path = environment["KIT_MINICPMV47_TOKENIZER"] { return URL(fileURLWithPath: path) }
        guard #available(macOS 27, iOS 27, *),
            let revision = ModelCatalog.builtin.entry(id: "minicpm-v-4.7-1b")?.revision,
            let bundle = ModelStore.default.localURL(for: VLModelID.miniCPMV47.pinned(revision).decoder)
        else { return nil }
        return bundle.appendingPathComponent("tokenizer")
    }

    static var tokenizerAvailable: Bool { tokenizerFolder != nil }

    static func pixel() throws -> CGImage {
        let context = try #require(
            CGContext(
                data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        return try #require(context.makeImage())
    }

    /// A user turn as the reference has it: the image first, then the question.
    @available(macOS 27, iOS 27, *)
    static func user(_ question: String, image: CGImage? = nil) -> Transcript.Entry {
        var segments: [Transcript.Segment] = []
        if let image { segments.append(.attachment(.init(content: .image(.init(image))))) }
        segments.append(.text(.init(content: question)))
        return .prompt(.init(segments: segments))
    }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: tokenizerAvailable, "set KIT_MINICPMV47_TOKENIZER (or download the model once)"))
    func everyReferencePromptRendersTheProcessorsIDs() async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: try #require(Self.tokenizerFolder))
        let arch = VLArchitecture.miniCPMV47
        let image = try Self.pixel()
        let cases: [(name: String, entries: [Transcript.Entry], ids: [Int32], imageStart: Int?)] = [
            ("P0", [Self.user("Write one sentence about the ocean.")], Self.p0, nil),
            ("P1", [Self.user("What is in this image?", image: image)], Self.p1, 7),
            ("P2", [Self.user("Describe this image in one sentence.", image: image)], Self.p2, 7),
            ("P3", [Self.user("What text is written in the image?", image: image)], Self.p3, 7),
            (
                "P4",
                [
                    .instructions(
                        .init(segments: [.text(.init(content: "You are a concise assistant."))], toolDefinitions: [])),
                    Self.user("What colors do you see?", image: image),
                ], Self.p4, 18
            ),
            (
                "turn2",
                [
                    Self.user("What is in this image?", image: image),
                    .response(.init(segments: [.text(.init(content: Self.a1Text))])),
                    Self.user("What colors do you see?"),
                ], Self.turn2, 7
            ),
        ]
        for c in cases {
            let rendered = try VLPromptRenderer.render(
                transcript: Transcript(entries: c.entries), tokenizer: tokenizer, arch: arch)
            // Back to the processor's form: the 64 slots `vocab + k` are its pads, in order.
            let processor = rendered.tokens.map { $0 >= arch.vocab ? Self.pad : $0 }
            let slots = rendered.tokens.filter { $0 >= arch.vocab }.map { $0 - arch.vocab }
            #expect(processor == c.ids, "\(c.name): the ids differ from the processor's")
            #expect(rendered.imageStart == c.imageStart, "\(c.name): first image token")
            if c.imageStart != nil { #expect(slots == Array(0..<64), "\(c.name): slot order") }
            let shift = rendered.imageStart.map { "\(arch.ropeShiftStart(imageStart: $0))/\(arch.ropeShiftAmount)" }
            print(
                "MiniCPMV47 render \(c.name): \(rendered.tokens.count) ids, equal \(processor == c.ids), "
                    + "imageStart \(rendered.imageStart.map(String.init) ?? "nil"), shift \(shift ?? "none"), "
                    + "ids \(processor)")
        }
        // The comparison can fail: 4.6's prompt for the same turn is not 4.7's.
        let old = try VLPromptRenderer.render(
            transcript: Transcript(entries: [Self.user("What is in this image?", image: image)]),
            tokenizer: tokenizer, arch: .miniCPMV46)
        #expect(old.tokens.map { $0 >= arch.vocab ? Self.pad : $0 } != Self.p1)
    }
}
