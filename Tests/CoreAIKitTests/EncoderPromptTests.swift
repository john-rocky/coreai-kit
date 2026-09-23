// EncoderPromptTests.swift — the encoder form (laya) without weights: the publisher's row layout
// over a stub tokenizer (option rendering, budget, squeeze, markers, state room), the readout
// against the host contract's worked rows, the bundle declaration, and — when the publisher's
// tokenizer and fixture are on this machine — its 201 rows at both windows, token for token.

import Foundation
import Testing

@testable import CoreAIKit
import CoreAIKitVision

struct EncoderPromptTests {
    /// One token per Unicode scalar: every count below is a character count.
    static func stub(_ text: String) -> [Int32] { text.unicodeScalars.map { Int32($0.value) + 1000 } }

    static func layout(window: Int, head: Int) -> EncoderPrompt.Layout {
        EncoderPrompt.Layout(window: window, headMaxLength: head, clsTokenID: 2, sepTokenID: 1, padTokenID: 0, maskTokenID: 4)
    }

    static func build(
        _ question: Decision.Question, state: String = "st", window: Int = 64, head: Int = 32
    ) throws -> EncoderPrompt.Rendered {
        try EncoderPrompt.build(
            stateTokens: stub(state), question: question, layout: layout(window: window, head: head),
            maskText: "<mask>", encode: stub)
    }

    static let short = Decision.Question.choice("Q?", options: [.init("a"), .init(id: "b", description: "x")])

    @Test func optionsRenderThePublishersWay() {
        let choice = Decision.Question.choice(
            "Which?",
            options: [
                .init("billing"), .init(id: "technical", description: "bugs, outages"),
                .init(id: "sales", description: "sales: pricing"),  // the wire codec's form
                .init(id: "other", description: ""),
            ])
        #expect(EncoderPrompt.optionTexts(for: choice) == ["billing", "technical: bugs, outages", "sales: pricing", "other"])
        // Scalar for scalar: a description opening with a combining mark keeps its own prefix,
        // and a canonically equivalent description is still a description.
        let marks = Decision.Question.choice("Which?", options: [.init(id: "l", description: "l: \u{301}x"), .init(id: "\u{e9}", description: "e\u{301}")])
        #expect(EncoderPrompt.optionTexts(for: marks).map { Array($0.unicodeScalars) }
            == [Array("l: \u{301}x".unicodeScalars), Array("\u{e9}: e\u{301}".unicodeScalars)])
        #expect(EncoderPrompt.optionTexts(for: .score("How urgent?", levels: ["not urgent", "soon"]))
            == ["level 0: not urgent", "level 1: soon"])
        #expect(EncoderPrompt.optionTexts(for: .noul("Refund?"))
            == ["false: no, the statement does not hold", "true: yes, the statement holds"])
        #expect(EncoderPrompt.optionTexts(for: .noul("Refund?", yes: "money back is asked for", no: ""))
            == ["false: no, the statement does not hold", "true: money back is asked for"])
    }

    @Test func aRowIsHeadOptionsStateEachClosedBySep() throws {
        let r = try Self.build(Self.short)
        let head = Self.stub("choice question: Q?")
        #expect(r.tokens == [2] + head + [1] + [4] + Self.stub(" a") + [4] + Self.stub(" b: x") + [1] + Self.stub("st") + [1])
        #expect(r.markers == [Int32(head.count + 2), Int32(head.count + 5)])
        #expect(r.qtype == 0 && r.stateTokens == 2)
        let noul = try Self.build(.noul("Ok?"), window: 128, head: 96)
        #expect(noul.qtype == 2 && Array(noul.tokens[1...16]) == Self.stub("noul question: Ok?").prefix(16).map { $0 })
        #expect(try Self.build(.score("How?", levels: ["a", "b"])).qtype == 1)
    }

    @Test func anOptionKeepsItsMarkerAndFortyEightTextTokens() throws {
        let long = String(repeating: "y", count: 60)
        let r = try Self.build(.choice("Q?", options: [.init("a"), .init(long)]), window: 256, head: 200)
        #expect(r.markers.count == 2)
        let second = Int(r.markers[1])
        #expect(Array(r.tokens[second..<second + 49]) == [4] + Self.stub(" " + long).prefix(48))
        #expect(r.tokens[second + 49] == 1)  // the options' closing SEP follows at once
    }

    @Test func optionsAreSqueezedWhenTheHeadWouldKeepFewerThanSixteen() throws {
        // Three 31-token options (32 with the marker) take 96 of 64: each is cut to
        // max(4, (64 − 16) / 3) = 16, which leaves the head exactly 16.
        let text = String(repeating: "o", count: 30)
        let instructions = String(repeating: "i", count: 40)
        let r = try Self.build(
            .choice(instructions, options: [.init(text), .init(text + "!"), .init(text + "?")]), window: 256, head: 64)
        #expect(r.markers == [18, 34, 50])
        #expect(Array(r.tokens[1..<17]) == Self.stub("choice question: " + instructions).prefix(16).map { $0 })
        #expect(Array(r.tokens[18..<34]) == [4] + Self.stub(" " + text).prefix(15))
        #expect(r.tokens[66] == 1)
    }

    @Test func theHeadKeepsAtLeastEightTokens() throws {
        // Seven options: each squeezed to max(4, 16 / 7) = 4, which leaves the head 4 → 8.
        let options = (0..<7).map { Decision.Option("opt\($0)") }
        let r = try Self.build(.choice("A long instruction", options: options), window: 128, head: 32)
        #expect(r.markers.first == 10)
        #expect(zip(r.markers, r.markers.dropFirst()).allSatisfy { $1 - $0 == 4 })
    }

    @Test func theStateFillsTheRoomAndIsCutFromItsEnd() throws {
        // Head and options take 31 tokens with their SEPs: a 40-token window leaves the state 8.
        let state = String(repeating: "s", count: 20)
        let r = try Self.build(Self.short, state: state, window: 40)
        #expect(r.tokens.count == 40 && r.stateTokens == 8)
        #expect(Array(r.tokens[31..<39]) == Self.stub(state).prefix(8).map { $0 } && r.tokens[39] == 1)
        let empty = try Self.build(Self.short, state: "", window: 40)
        #expect(empty.stateTokens == 0 && empty.tokens.count == 32 && empty.tokens.suffix(2) == [1, 1])
    }

    @Test func aRowPastTheWindowIsCutAndAMarkerItDropsRefusesTheQuestion() throws {
        // 31 tokens before the state: a 30-token window keeps both markers (21, 24) and loses
        // the closing SEPs, which are not put back.
        let cut = try Self.build(Self.short, window: 30)
        #expect(cut.tokens.count == 30 && cut.markers == [21, 24] && cut.stateTokens == 0 && cut.tokens.last != 1)
        #expect(throws: DecisionError.promptTooLong(tokens: 32, max: 24)) {
            try Self.build(Self.short, window: 24)
        }
    }

    @Test func maskTextBecomesASpaceBeforeTokenizing() throws {
        var seen: [String] = []
        _ = try EncoderPrompt.build(
            stateTokens: [], question: .choice("Is <mask> here?", options: [.init("a<mask>"), .init("b")]),
            layout: Self.layout(window: 64, head: 32), maskText: "<mask>", encode: { seen.append($0); return Self.stub($0) })
        #expect(seen == ["choice question: Is   here?", " a ", " b"])
        #expect(EncoderPrompt.unmasked("x<mask><mask>y", "<mask>") == "x  y")
        // Code units, as Python's str.replace: a combining mark after the token does not shield it.
        #expect(EncoderPrompt.unmasked("<mask>\u{0301}", "<mask>") == " \u{0301}")
    }

    @Test func formatRoundTripsAndAnEncoderEntryCanDecide() {
        #expect(Decision.Format(rawValue: "encoder") == .encoder)
        #expect(EncoderPrompt.maxOptions == 20)
        let entry = CatalogEntry(
            id: "laya-multilingual", name: "Laya Multilingual", repo: "example/laya", kind: .decision,
            variants: ["macos": .init(path: "macos/fp16-s256"), "ios": .init(path: "ios/fp16-s256")], format: "encoder")
        #expect(TypedDecisions.supports(entry))
    }
}

/// The question cache: a question's part of the row is made once and reused on every state, keyed
/// by what the tokenizer reads of the question, byte for byte, the least recently asked dropped first.
struct EncoderQuestionCacheTests {
    static let layout = EncoderPromptTests.layout(window: 64, head: 32)

    static func part(_ question: Decision.Question) -> EncoderPrompt.QuestionPart {
        EncoderPrompt.questionPart(question, layout: layout, maskText: "<mask>", encode: EncoderPromptTests.stub)
    }

    @Test func aHitRendersExactlyWhatAFreshBuildRenders() throws {
        var cache = EncoderPrompt.QuestionCache(capacity: 16)
        var made = 0
        let question = EncoderPromptTests.short
        let first = cache.part(for: question, maskText: "<mask>") { made += 1; return Self.part(question) }
        let second = cache.part(for: question, maskText: "<mask>") { made += 1; return Self.part(question) }
        #expect(!first.hit && second.hit && made == 1 && first.part == second.part)
        for state in ["st", "", String(repeating: "s", count: 40)] {
            let cached = try EncoderPrompt.assemble(second.part, stateTokens: EncoderPromptTests.stub(state), layout: Self.layout)
            let fresh = try EncoderPrompt.build(
                stateTokens: EncoderPromptTests.stub(state), question: question, layout: Self.layout, maskText: "<mask>",
                encode: EncoderPromptTests.stub)
            #expect(cached == fresh)
        }
    }

    @Test func theKeyIsTheQuestionsValueByteForByte() {
        var cache = EncoderPrompt.QuestionCache(capacity: 16)
        let question = Decision.Question.choice("Q?", options: [.init("a"), .init(id: "b", description: "x")])
        _ = cache.part(for: question, maskText: "<mask>") { Self.part(question) }
        // An equal value built again hits, and so does one the wire codec wrote the same way.
        #expect(cache.part(for: .choice("Q?", options: [.init("a"), .init(id: "b", description: "x")]), maskText: "<mask>") { Self.part(question) }.hit)
        #expect(cache.part(for: .choice("Q?", options: [.init("a"), .init(id: "b", description: "b: x")]), maskText: "<mask>") { Self.part(question) }.hit)
        // Another description, another type, another instruction: misses.
        #expect(!cache.part(for: .choice("Q?", options: [.init("a"), .init(id: "b", description: "y")]), maskText: "<mask>") { Self.part(question) }.hit)
        #expect(!cache.part(for: .score("Q?", levels: ["a", "b: x"]), maskText: "<mask>") { Self.part(question) }.hit)
        #expect(!cache.part(for: .choice("Q!", options: [.init("a"), .init(id: "b", description: "x")]), maskText: "<mask>") { Self.part(question) }.hit)
        // Canonically equivalent is not the same text to this tokenizer: é (U+00E9) and e + U+0301 miss each other.
        _ = cache.part(for: .noul("caf\u{e9}?"), maskText: "<mask>") { Self.part(.noul("caf\u{e9}?")) }
        #expect(!cache.part(for: .noul("cafe\u{301}?"), maskText: "<mask>") { Self.part(.noul("cafe\u{301}?")) }.hit)
        #expect(Decision.Question.noul("caf\u{e9}?") == Decision.Question.noul("cafe\u{301}?"))  // why the key is not `==`
    }

    @Test func theLeastRecentlyAskedIsDroppedPastSixteen() {
        var cache = EncoderPrompt.QuestionCache(capacity: 16)
        let questions = (0..<17).map { Decision.Question.noul("Question \($0)?") }
        for q in questions.prefix(16) { _ = cache.part(for: q, maskText: "<mask>") { Self.part(q) } }
        #expect(cache.count == 16)
        // Asking the oldest again makes it the newest; the 17th question then drops question 1.
        #expect(cache.part(for: questions[0], maskText: "<mask>") { Self.part(questions[0]) }.hit)
        _ = cache.part(for: questions[16], maskText: "<mask>") { Self.part(questions[16]) }
        #expect(cache.count == 16)
        #expect(cache.part(for: questions[0], maskText: "<mask>") { Self.part(questions[0]) }.hit)
        #expect(!cache.part(for: questions[1], maskText: "<mask>") { Self.part(questions[1]) }.hit)
        #expect(EncoderDecider.questionCacheCapacity == 16)
    }
}

/// The host decoder against HOST_CONTRACT §E: the publisher's worked rows, whose probabilities
/// and features the reference computed in float32.
struct EncoderReadoutTests {
    /// The shipped multilingual calibration (laya_ml_calibration.json).
    static let multilingual = EncoderReadout.Temperatures(
        byType: [2.042901414492613, 3.684653192472766, 3.629100511692587],
        byOptions: [
            "choice:2": 1.397546128996061, "choice:3-5": 1.3591985667953796, "choice:6-10": 1.0,
            "choice:11+": 2.445877143176841, "score:3-5": 3.684653192472766, "noul:2": 3.629100511692587,
        ])
    static let english = EncoderReadout.Temperatures(
        byType: [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
        byOptions: [
            "choice:3-5": 1.7601518630981445, "choice:6-10": 1.0000158548355103, "score:3-5": 1.2514300346374512,
            "noul:2": 1.983399510383606, "choice:11+": 0.10058280825614929, "choice:2": 1.9063563346862793,
        ])
    static let department = Decision.Question.choice(
        "Which department should handle this request?",
        options: [
            .init(id: "billing", description: "invoices, payments, refunds"),
            .init(id: "technical", description: "bugs, outages, system errors"),
            .init(id: "sales", description: "pricing, new contracts"),
            .init(id: "other", description: "everything else"),
        ])

    static func close(_ a: [Double], _ b: [Double], _ tolerance: Double) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= tolerance }
    }

    static func rounded(_ x: Double) -> Double { (x * 10000).rounded() / 10000 }

    @Test func multilingualA04AtTheSourceTemperatureAndCalibrated() {
        let logits: [Float] = [7.643063545227051, -4.5210771560668945, 0.46446093916893005, -3.5530107021331787]
        let raw = EncoderReadout.probabilities(logits: logits, temperature: 1)
        #expect(Self.close(raw, [0.9992189407348633, 5.210045401327079e-06, 0.0007621371187269688, 1.3717257388634607e-05], 1e-7))
        #expect(Self.rounded(DecisionPrompt.certainty(raw)) == 0.9953)
        let features = EncoderReadout.actFeatures(logits: logits).map(Double.init)
        #expect(Self.close(features, [0.9992189407348633, 0.9984567761421204, 0.004666685126721859, 0.01568627543747425], 1e-7))
        #expect(EncoderReadout.actProbability(actLogits: [1316.4495849609375, -1573.5196533203125]) == 1)

        let t = Self.multilingual.temperature(for: Self.department)
        #expect(t == 1.3591985667953796)
        let deployed = EncoderReadout.probabilities(logits: logits, temperature: t)
        #expect(Self.close(deployed, [0.9945505261421204, 0.00012909529323223978, 0.005057105794548988, 0.0002631658280733973], 2e-7))
        let answer = DecisionPrompt.answer(
            for: Self.department, probabilities: deployed, timing: .init(promptTokens: 77, reusedTokens: 0, seconds: 0))
        guard case .choice(let choice) = answer.value else {
            Issue.record("not a choice")
            return
        }
        #expect(choice.id == "billing")
        #expect(choice.options.map { Self.rounded(choice.probabilities[$0]!) } == [0.9946, 0.0001, 0.0051, 0.0003])
        #expect(Self.rounded(choice.certainty) == 0.9744)
    }

    @Test func englishA01AtItsChoiceBucket() {
        let logits: [Float] = [1.0703915357589722, -1.8068784475326538, -2.2441372871398926, -2.46530818939209]
        let t = Self.english.temperature(for: Self.department)
        #expect(t == 1.7601518630981445)
        let p = EncoderReadout.probabilities(logits: logits, temperature: t)
        #expect(Self.close(p, [0.6750863194465637, 0.13165292143821716, 0.10269340872764587, 0.09056732803583145], 2e-7))
        #expect(Self.rounded(DecisionPrompt.certainty(p)) == 0.2906)
        let features = EncoderReadout.actFeatures(logits: logits).map(Double.init)
        #expect(Self.close(features, [0.8914421796798706, 0.8412644863128662, 0.3307645320892334, 0.01568627543747425], 1e-7))
        #expect(EncoderReadout.actProbability(actLogits: [3710.220947265625, -3030.86669921875]) == 1)
    }

    @Test func bucketsAndTheirLookup() {
        typealias T = EncoderReadout.Temperatures
        #expect([2, 3, 5, 6, 10, 11, 20].map { T.bucket(qtype: 0, options: $0) }
            == ["choice:2", "choice:3-5", "choice:3-5", "choice:6-10", "choice:6-10", "choice:11+", "choice:11+"])
        #expect(T.bucket(qtype: 1, options: 4) == "score:3-5" && T.bucket(qtype: 2, options: 2) == "noul:2")
        // A bucket wins; a type without one falls back to its per-type value.
        #expect(Self.multilingual.temperature(qtype: 0, options: 7) == 1.0)
        #expect(Self.multilingual.temperature(qtype: 1, options: 2) == 3.684653192472766)
        #expect(Self.multilingual.temperature(for: .noul("x")) == 3.629100511692587)
        #expect(EncoderReadout.qtype(of: .score(levels: ["a", "b"])) == 1)
        #expect(T.one.temperature(qtype: 2, options: 2) == 1)
    }

    @Test func tinyTemperaturesAreFlooredAndTheActHeadIsATwoWaySoftmax() {
        #expect(EncoderReadout.probabilities(logits: [0.001, 0], temperature: 0)
            == EncoderReadout.probabilities(logits: [0.001, 0], temperature: 1e-3))
        #expect(EncoderReadout.actProbability(actLogits: [0, 0]) == 0.5)
        #expect(EncoderReadout.optionLogits(tokenLogits: [0, 1, 2, 3, 4], markers: [3, 1]) == [3, 1])
        // Two options: the entropy is over ln 2 and the size feature 2/255.
        let features = EncoderReadout.actFeatures(logits: [0, 0])
        #expect(features[0] == 0.5 && features[1] == 0 && abs(features[2] - 1) < 1e-6 && features[3] == Float(2.0 / 255))
    }
}

/// The bundle declaration, and that both initialisers recognise an encoder bundle before
/// reading it as a language bundle.
struct EncoderBundleTests {
    static let block =
        #"{"head": "encoder", "layout": "laya", "window": 256, "head_max_len": 256, "source_max_len": 1024, "#
        + #""option_text_tokens": 48, "cls_token_id": 2, "sep_token_id": 1, "pad_token_id": 0, "mask_token_id": 4, "#
        + #""functions": {"main": "main", "act": "act"}, "source_temperature": [1, 1, 1], "#
        + #""temperature": [2.042901414492613, 3.684653192472766, 3.629100511692587], "#
        + #""temperature_by_options": {"choice:2": 1.397546128996061, "choice:3-5": 1.3591985667953796, "choice:6-10": 1.0, "#
        + #""choice:11+": 2.445877143176841, "score:3-5": 3.684653192472766, "noul:2": 3.629100511692587}}"#

    static func bundle(_ metadata: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("encoder-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(metadata.utf8).write(to: dir.appendingPathComponent("metadata.json"))
        return dir
    }

    @Test func layoutComesFromTheBundleDeclaration() throws {
        let dir = try Self.bundle(#"{"metadata_version": "0.2", "kind": "encoder", "decision": "# + Self.block + "}")
        defer { try? FileManager.default.removeItem(at: dir) }
        let layout = try #require(try EncoderPrompt.Layout.read(bundleAt: dir))
        #expect(layout == EncoderPrompt.Layout(
            window: 256, headMaxLength: 256, clsTokenID: 2, sepTokenID: 1, padTokenID: 0, maskTokenID: 4,
            temperatures: EncoderReadoutTests.multilingual, sourceTemperatures: .one))
        #expect(layout.windowed(512).window == 512 && layout.windowed(512).temperatures == layout.temperatures)

        try Data(#"{"decision": {"head": "slot", "n_slots": 8}}"#.utf8).write(to: dir.appendingPathComponent("metadata.json"))
        #expect(try EncoderPrompt.Layout.read(bundleAt: dir) == nil)
        try Data(#"{"language": {"vocab_size": 256}}"#.utf8).write(to: dir.appendingPathComponent("metadata.json"))
        #expect(try EncoderPrompt.Layout.read(bundleAt: dir) == nil)
        #expect(try EncoderPrompt.Layout.read(bundleAt: dir.appendingPathComponent("missing")) == nil)

        for broken in [
            #"{"decision": {"head": "encoder", "head_max_len": 256, "cls_token_id": 2, "sep_token_id": 1, "pad_token_id": 0, "mask_token_id": 4}}"#,
            #"{"decision": {"head": "encoder", "window": 256, "head_max_len": 256, "cls_token_id": 2, "sep_token_id": 1, "pad_token_id": 0}}"#,
            #"{"decision": {"head": "encoder", "window": 256, "head_max_len": 256, "cls_token_id": 2, "sep_token_id": 1, "pad_token_id": 0, "mask_token_id": 4, "temperature": [1, 2]}}"#,
            #"{"decision": {"head": "encoder", "window": 256, "head_max_len": 256, "cls_token_id": 2, "sep_token_id": 1, "pad_token_id": 0, "mask_token_id": 4, "temperature": [1, 0, 1]}}"#,
        ] {
            try Data(broken.utf8).write(to: dir.appendingPathComponent("metadata.json"))
            #expect(throws: DecisionError.self) { try EncoderPrompt.Layout.read(bundleAt: dir) }
        }
    }

    @Test func anEncoderBundleIsNotReadAsALanguageBundle() async throws {
        // No graph in the directory: the encoder path fails on the missing graph — a language
        // bundle read would have failed on its metadata instead.
        let dir = try Self.bundle(#"{"metadata_version": "0.2", "kind": "encoder", "decision": "# + Self.block + "}")
        defer { try? FileManager.default.removeItem(at: dir) }
        await #expect(throws: KitBundleError.self) { try await TypedDecisions(bundleAt: dir) }
        var chat = TypedDecisions.Configuration()
        chat.format = .chat
        await #expect(throws: DecisionError.self) { try await TypedDecisions(bundleAt: dir, configuration: chat) }
    }

    @Test func theNeuralEngineIsRefusedBeforeAnythingLoads() async throws {
        // Refused from the metadata alone: with no graph in the directory, anything past the check
        // would fail on the missing graph (KitBundleError) instead.
        let dir = try Self.bundle(#"{"metadata_version": "0.2", "kind": "encoder", "decision": "# + Self.block + "}")
        defer { try? FileManager.default.removeItem(at: dir) }
        var ane = TypedDecisions.Configuration()
        ane.computeUnits = .neuralEngine
        do {
            _ = try await TypedDecisions(bundleAt: dir, configuration: ane)
            Issue.record("a Neural Engine preference loaded")
        } catch DecisionError.unsupportedModel(let id, let reason) {
            #expect(id == dir.lastPathComponent)
            #expect(reason.contains("Neural Engine") && reason.contains("1e-3"))
        }
        // The other units get as far as the graph.
        for units in [GraphModel.ComputeUnits.gpu, .cpuOnly] {
            var configuration = TypedDecisions.Configuration()
            configuration.computeUnits = units
            await #expect(throws: KitBundleError.self) { try await TypedDecisions(bundleAt: dir, configuration: configuration) }
        }
    }
}

/// The publisher's 201 multilingual rows at both windows, when its tokenizer and fixture are on
/// this machine:
///
///     LAYA_TOKENIZER_DIR=<snapshot>/multilingual/tokenizer \
///     LAYA_FIXTURES_DIR=<dir with ml_rows_s256.json, ml_rows_s512.json, ml_fixtures.json> \
///     swift test --filter EncoderFixtureTests
struct EncoderFixtureTests {
    static let environment = ProcessInfo.processInfo.environment
    static let enabled = environment["LAYA_TOKENIZER_DIR"] != nil && environment["LAYA_FIXTURES_DIR"] != nil

    /// A fixture question in the wire form, rebuilt the way the wire codec builds it.
    static func question(_ value: JSONValue) -> Decision.Question {
        let instructions = value["instructions"].map { $0.stringValue ?? $0.dumps() } ?? ""
        func text(_ value: JSONValue) -> String? {
            switch value {
            case .null: return nil
            case .string(let s): return s
            default: return value.dumps()
            }
        }
        switch value["type"]?.stringValue {
        case "choice":
            if let members = value["criteria"]?.members {
                return .choice(instructions, options: members.map { member in
                    let description = text(member.value).flatMap { $0.isEmpty ? nil : $0 }
                    return .init(id: member.key, description: description.map { "\(member.key): \($0)" } ?? member.key)
                })
            }
            return .choice(instructions, (value["criteria"]?.elements ?? []).compactMap(text))
        case "score":
            return .score(instructions, levels: (value["criteria"]?.elements ?? []).compactMap(text))
        default:
            let criteria = value["criteria"]
            return .noul(instructions, yes: criteria?["true"].flatMap(text), no: criteria?["false"].flatMap(text))
        }
    }

    static func ints(_ value: JSONValue?) -> [Int32] {
        (value?.elements ?? []).compactMap { $0.doubleValue.map { Int32($0) } }
    }

    @Test(.enabled(if: enabled, "set LAYA_TOKENIZER_DIR and LAYA_FIXTURES_DIR to check the publisher's 201 rows"))
    func everyPublisherRowAtBothWindows() async throws {
        let fixtures = URL(fileURLWithPath: Self.environment["LAYA_FIXTURES_DIR"]!)
        let tokenizer = URL(fileURLWithPath: Self.environment["LAYA_TOKENIZER_DIR"]!)
        var states: [String: String] = [:]
        for fixture in try JSONValue.parse(Data(contentsOf: fixtures.appendingPathComponent("ml_fixtures.json"))).elements ?? [] {
            guard let id = fixture["id"]?.stringValue, let state = fixture["state"] else { continue }
            states[id] = state.stringValue ?? state.dumps()
        }
        let prompt = try await EncoderPrompt(tokenizerFolder: tokenizer, window: 256, headMaxLength: 256)
        #expect(prompt.layout.clsTokenID == 2 && prompt.layout.sepTokenID == 1 && prompt.layout.padTokenID == 0)
        #expect(prompt.layout.maskTokenID == 4 && prompt.maskText == "<mask>")
        // The reference tokenizer gives an empty string no tokens; the Swift one would give "▁".
        #expect(prompt.contextTokens(state: "") == [] && prompt.encode("") == [])
        for window in [256, 512] {
            let rows = try JSONValue.parse(Data(contentsOf: fixtures.appendingPathComponent("ml_rows_s\(window).json"))).elements ?? []
            let windowed = prompt.windowed(window)
            var tokensExact = 0, markersExact = 0
            for row in rows {
                let id = row["row_id"]?.stringValue ?? "?"
                guard let fixture = row["fixture_id"]?.stringValue, let state = states[fixture], let q = row["question"] else {
                    Issue.record("\(id): no state or question")
                    continue
                }
                let rendered = try windowed.render(state: state, question: Self.question(q))
                let ids = Self.ints(row["sequence_ids"])
                let markers = Self.ints(row["marker_positions"])
                if rendered.tokens == ids { tokensExact += 1 } else {
                    let first = zip(rendered.tokens, ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                    print("\(id) s\(window): tokens kit \(rendered.tokens.count) ref \(ids.count), first diff \(first.map(String.init) ?? "-")")
                }
                if rendered.markers == markers { markersExact += 1 } else {
                    print("\(id) s\(window): markers kit \(rendered.markers) ref \(markers)")
                }
                #expect(rendered.qtype == Int(row["qtype"]?.doubleValue ?? -1))
            }
            print("s\(window): tokens identical \(tokensExact)/\(rows.count), markers identical \(markersExact)/\(rows.count)")
            #expect(rows.count == 201 && tokensExact == rows.count && markersExact == rows.count)
        }
    }
}
