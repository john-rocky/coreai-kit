// TypedDecisionsTests.swift — the parts of a typed decision that are checkable without
// weights: the request rendering (byte-for-byte the reference form), the answer-shape
// validation, and the readout arithmetic from logits to an answer.

import Foundation
import Testing

@testable import CoreAIKit
@testable import CoreAIOps

struct DecisionPromptTests {
    /// The user turn must be the reference rendering exactly — Python's `json.dumps` with
    /// `ensure_ascii=False` — or the published fixture numbers are not comparable.
    @Test func jsonStringsEscapeLikeTheReference() {
        let cases: [(String, String)] = [
            ("plain", "\"plain\""),
            ("say \"hi\"", "\"say \\\"hi\\\"\""),
            ("back\\slash", "\"back\\\\slash\""),
            ("line\nbreak\ttab\r", "\"line\\nbreak\\ttab\\r\""),
            ("\u{01}ctl\u{1f}", "\"\\u0001ctl\\u001f\""),
            ("日本語 と 絵文字 🙂", "\"日本語 と 絵文字 🙂\""),
            ("slash/ok", "\"slash/ok\""),
            ("\u{7f}del", "\"\u{7f}del\""),
        ]
        for (input, expected) in cases {
            #expect(DecisionPrompt.jsonString(input) == expected)
        }
    }

    @Test func userPayloadIsTheReferenceObject() {
        let payload = DecisionPrompt.userPayload(
            state: "The optician ordered replacement lenses.",
            criterion: "Assess the claim.",
            options: ["The evidence establishes the claim", "It doesn't"])
        #expect(
            payload
                == "{\"evidence\": \"The optician ordered replacement lenses.\", "
                + "\"criterion\": \"Assess the claim.\", "
                + "\"options\": [{\"letter\": \"A\", \"description\": \"The evidence establishes the claim\"}, "
                + "{\"letter\": \"B\", \"description\": \"It doesn't\"}]}")
    }

    @Test func questionShapesRenderTheirOptions() {
        let noul = Decision.Question.noul("Is it urgent?", yes: "needs action today", no: "can wait")
        #expect(noul.optionDescriptions == ["no: can wait", "yes: needs action today"])
        #expect(noul.optionIDs == ["no", "yes"])
        #expect(Decision.Question.noul("Is it urgent?").optionDescriptions == ["no", "yes"])

        let score = Decision.Question.score("How angry?", levels: ["calm", "annoyed", "furious"])
        #expect(score.optionDescriptions == ["0: calm", "1: annoyed", "2: furious"])
        #expect(score.optionIDs == ["0", "1", "2"])

        let choice = Decision.Question.choice(
            "Which?", options: [.init(id: "a", description: "first"), .init("second")])
        #expect(choice.optionDescriptions == ["first", "second"])
        #expect(choice.optionIDs == ["a", "second"])
    }

    @Test func validationRefusesTheWrongShapes() {
        #expect(throws: DecisionError.emptyInstructions) {
            try DecisionPrompt.validate(.choice("  ", ["a", "b"]))
        }
        #expect(throws: DecisionError.tooFewOptions(count: 1)) {
            try DecisionPrompt.validate(.choice("Which?", ["only"]))
        }
        let seventeen = (0..<17).map { "option \($0)" }
        #expect(throws: DecisionError.tooManyOptions(count: 17, max: 16)) {
            try DecisionPrompt.validate(.choice("Which?", seventeen))
        }
        let eleven = (0..<11).map { "level \($0)" }
        #expect(throws: DecisionError.tooManyOptions(count: 11, max: 10)) {
            try DecisionPrompt.validate(.score("How much?", levels: eleven))
        }
        #expect(throws: Never.self) {
            try DecisionPrompt.validate(.choice("Which?", (0..<16).map { "o\($0)" }))
            try DecisionPrompt.validate(.noul("Yes?"))
            // A slot head addresses its options by control token: its ceiling is its slot count.
            try DecisionPrompt.validate(.choice("Which?", (0..<255).map { "o\($0)" }), maxOptions: 255)
        }
        #expect(throws: DecisionError.tooManyOptions(count: 256, max: 255)) {
            try DecisionPrompt.validate(.choice("Which?", (0..<256).map { "o\($0)" }), maxOptions: 255)
        }
    }

    @Test func commonPrefix() {
        #expect(DecisionPrompt.commonPrefixLength([1, 2, 3], [1, 2, 4]) == 2)
        #expect(DecisionPrompt.commonPrefixLength([1, 2], [1, 2, 3]) == 2)
        #expect(DecisionPrompt.commonPrefixLength([], [1]) == 0)
    }
}

struct DecisionReadoutTests {
    @Test func softmaxIsADistribution() {
        let p = DecisionPrompt.probabilities(logits: [2, 1, 0], temperature: 1)
        #expect(abs(p.reduce(0, +) - 1) < 1e-12)
        #expect(p[0] > p[1] && p[1] > p[2])
        // A higher temperature flattens; a lower one sharpens.
        let flat = DecisionPrompt.probabilities(logits: [2, 1, 0], temperature: 10)
        let sharp = DecisionPrompt.probabilities(logits: [2, 1, 0], temperature: 0.1)
        #expect(flat[0] < p[0] && sharp[0] > p[0])
        // Large logits do not overflow.
        let big = DecisionPrompt.probabilities(logits: [1000, 999], temperature: 1)
        #expect(big[0].isFinite && abs(big.reduce(0, +) - 1) < 1e-12)
    }

    @Test func certaintyIsOneMinusNormalisedEntropy() {
        #expect(DecisionPrompt.certainty([1, 0, 0]) == 1)
        #expect(abs(DecisionPrompt.certainty([0.5, 0.5])) < 1e-12)
        #expect(abs(DecisionPrompt.certainty([1.0 / 3, 1.0 / 3, 1.0 / 3])) < 1e-12)
        #expect(DecisionPrompt.certainty([1]) == 1)
    }

    private let timing = Decision.Timing(promptTokens: 10, reusedTokens: 4, seconds: 0.5)

    @Test func choiceAnswerRanksTheOptions() {
        let question = Decision.Question.choice("Which?", ["billing", "delivery", "other"])
        let answer = DecisionPrompt.answer(for: question, probabilities: [0.2, 0.7, 0.1], timing: timing)
        guard case .choice(let choice) = answer.value else {
            Issue.record("expected a choice")
            return
        }
        #expect(choice.id == "delivery")
        #expect(choice.confidence == 0.7)
        #expect(choice.ranking == ["delivery", "billing", "other"])
        #expect(choice.options == ["billing", "delivery", "other"])
        #expect(choice.probabilities["billing"] == 0.2)
        #expect(answer.choice == "delivery")
        #expect(answer.probabilities == [0.2, 0.7, 0.1])
        #expect(answer.timing.processedTokens == 6)
        #expect(answer.timing.milliseconds == 500)
    }

    @Test func scoreAnswerIsTheExpectedLevel() {
        let question = Decision.Question.score("How angry?", levels: ["calm", "annoyed", "furious"])
        let answer = DecisionPrompt.answer(for: question, probabilities: [0.1, 0.3, 0.6], timing: timing)
        guard case .score(let score) = answer.value else {
            Issue.record("expected a score")
            return
        }
        #expect(abs(score.value - 1.5) < 1e-12)
        #expect(score.level == 2)
        #expect(score.confidence == 0.6)
        #expect(answer.score == 1.5)
    }

    @Test func noulAnswerIsPYes() {
        let answer = DecisionPrompt.answer(for: .noul("Reply?"), probabilities: [0.25, 0.75], timing: timing)
        #expect(answer.noul == 0.75)
        #expect(answer.confidence == 0.75)
        #expect(answer.choice == nil && answer.score == nil)
        let no = DecisionPrompt.answer(for: .noul("Reply?"), probabilities: [0.9, 0.1], timing: timing)
        #expect(no.confidence == 0.9)
    }

    /// The op enum walks `capability`, `prepare` and the doctor; the new op must name a
    /// catalog model there like every other model-backed op.
    @Test func decideOpIsRegistered() {
        #expect(CoreAI.Op.allCases.contains(.decide))
        #expect(CoreAI.Op.decide.defaultModelID == CoreAI.defaultDecisionModel)
        #expect(ModelCatalog.builtin.entry(id: CoreAI.defaultDecisionModel)?.kind == .chat)
        #expect(!CoreAI.Op.decide.summary.isEmpty)
    }
}

/// The decider form: row planning and the narrow text, checkable without a tokenizer.
struct DeciderPromptTests {
    @Test func choiceAndNoulAreOneRowEach() {
        let choice = Decision.Question.choice(
            "Which team should handle the request?",
            options: [
                .init(id: "repair", description: "Replace damaged parts"),
                .init(id: "billing", description: "Handle payments"),
                .init("delivery"),
            ])
        let rows = DeciderPrompt.rows(for: choice)
        #expect(rows.count == 1)
        #expect(rows[0].question == "Which team should handle the request?")
        #expect(rows[0].options == ["repair: Replace damaged parts", "billing: Handle payments", "delivery"])

        let noul = Decision.Question.noul(
            "Does the sender request a refund?",
            yes: "Money back is requested", no: "No money back is requested")
        #expect(
            DeciderPrompt.rows(for: noul)
                == [.init(question: "Does the sender request a refund?",
                          options: ["no: No money back is requested", "yes: Money back is requested"])])
        #expect(DeciderPrompt.rows(for: .noul("Urgent?"))[0].options == ["no", "yes"])
    }

    @Test func scoreIsOneYesNoRowPerLevel() {
        let score = Decision.Question.score("How full is the tank?", levels: ["empty", "one quarter full", "full"])
        let rows = DeciderPrompt.rows(for: score)
        #expect(rows.count == 3)
        #expect(rows[1].question == "How full is the tank?\nProposed answer: one quarter full\nDoes the proposed answer fit?")
        #expect(rows.allSatisfy { $0.options == ["no", "yes"] })
    }

    @Test func narrowTextIsTheTrainedForm() {
        let row = DeciderPrompt.Row(question: "Which?", options: ["a", "b", "c"])
        #expect(DeciderPrompt.narrowText(row) == "\n\nQuestion: Which?\nOptions:\n(A) a\n(B) b\n(C) c\nAnswer: (")
    }

    @Test func combineNormalisesFitAndSurvivesZeroMass() {
        let p = DeciderPrompt.combine(fit: [0.2, 0.6, 0.2])
        #expect(abs(p[1] - 0.6) < 1e-12 && abs(p.reduce(0, +) - 1) < 1e-12)
        #expect(DeciderPrompt.combine(fit: [0, 0]) == [0, 0])
    }
}

/// The `Shared state:` + JSON task form (APUS-OpenJev-v1): the user turn byte for byte the
/// author's `render_prompt`, the fixed yes/no criteria, and the readout order.
struct SharedStatePromptTests {
    @Test func userTurnIsTheAuthorsRenderPrompt() {
        let question = Decision.Question.choice(
            "Select the appropriate next workflow action.",
            options: [
                .init(id: "close", description: "Close the resolved support ticket."),
                .init(id: "refund", description: "refund: Refund an undelivered order."),  // the wire codec's form
            ])
        let row = SharedStatePrompt.row(for: question)
        #expect(row == .init(primitive: "choice", instructions: "Select the appropriate next workflow action.",
                             options: ["Close the resolved support ticket.", "Refund an undelivered order."]))
        #expect(
            SharedStatePrompt.userContent(state: "Order 731 has been delivered. The customer's message says thank you.", row: row)
                == "Shared state:\nOrder 731 has been delivered. The customer's message says thank you.\n\n"
                + "{\"criteria\": [{\"description\": \"Close the resolved support ticket.\", \"label\": \"A\"}, "
                + "{\"description\": \"Refund an undelivered order.\", \"label\": \"B\"}], "
                + "\"instructions\": \"Select the appropriate next workflow action.\", \"primitive\": \"choice\"}"
                + "\nReturn only the selected letter: A, B.\nAnswer:")
        let sixteen = SharedStatePrompt.row(for: .choice("Which?", (1...16).map { "Click the Page \($0) button." }))
        #expect(SharedStatePrompt.userContent(state: "s", row: sixteen).hasSuffix(
            "\nReturn only the selected letter: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P.\nAnswer:"))
    }

    @Test func noulUsesTheFixedCriteriaYesFirstAndTheKitReportsNoThenYes() {
        let noul = Decision.Question.noul("The customer thanked the agent.")
        let row = SharedStatePrompt.row(for: noul)
        #expect(row.primitive == "noul")
        #expect(row.descriptions == ["The stated proposition is true.", "The stated proposition is false."])
        #expect(SharedStatePrompt.probabilities(kitOrder: [0.8, 0.2], for: noul) == [0.2, 0.8])
        let described = SharedStatePrompt.row(for: .noul("Urgent?", yes: "needs action today", no: "can wait"))
        #expect(described.instructions == "Urgent?\nyes: needs action today\nno: can wait")
        let score = SharedStatePrompt.row(for: .score("How upset?", levels: ["calm", "annoyed", "furious"]))
        #expect(score.primitive == "choice" && score.descriptions == ["calm", "annoyed", "furious"])
        #expect(SharedStatePrompt.probabilities(kitOrder: [0.1, 0.2, 0.7], for: .score("x", levels: ["a", "b", "c"])) == [0.1, 0.2, 0.7])
        #expect(Decision.Format(rawValue: "sharedState") == .sharedState)
    }
}

/// The plain-text decision-function form (Jev-Style-Qwen3.5-2B-Decision): the prompt byte
/// for byte the author's `build_prompt`, the three shapes as one choice each, the readout order.
struct DecisionFunctionPromptTests {
    @Test func promptIsTheAuthorsBuildPrompt() {
        let question = Decision.Question.choice(
            "Which news section does this article belong to?",
            options: [.init("World"), .init("Sports"), .init(id: "biz", description: "Business"),
                      .init(id: "sci", description: "sci: Science/Technology")])
        let row = DecisionFunctionPrompt.row(for: question)
        #expect(row.options == ["World", "Sports", "Business", "Science/Technology"])
        #expect(
            DecisionFunctionPrompt.text(state: "Shares of the chipmaker jumped 8% after it raised its revenue forecast.", row: row)
                == "You are a decision function. Read the state, then answer the question by choosing exactly one option.\n\n"
                + "[State]\nShares of the chipmaker jumped 8% after it raised its revenue forecast.\n\n"
                + "[Question]\nWhich news section does this article belong to?\n\n"
                + "[Options]\nA. World\nB. Sports\nC. Business\nD. Science/Technology\n\nAnswer:")
    }

    @Test func boolAndScoreAreChoices() {
        let bool = DecisionFunctionPrompt.row(for: .noul("Does the premise entail the hypothesis?"))
        #expect(bool.options == ["yes", "no"])
        #expect(DecisionFunctionPrompt.probabilities(kitOrder: [0.976, 0.024], for: .noul("x")) == [0.024, 0.976])
        let score = DecisionFunctionPrompt.row(for: .score("Rate the sentiment.", levels: ["very negative", "negative", "neutral", "positive", "very positive"]))
        #expect(score.options.count == 5 && score.options[3] == "positive")
        #expect(DecisionFunctionPrompt.maxOptions == 26)
        #expect(Decision.Format(rawValue: "decisionFunction") == .decisionFunction)
    }
}

/// The lettered option list (OpenJev): the helper's text, the score and yes/no rows, the
/// calibration arithmetic and the bundle declaration, checkable without a tokenizer or weights.
struct LetterListPromptTests {
    @Test func userTurnIsTheHelpersPrompt() {
        let question = Decision.Question.choice(
            "Which destination is printed on the sorting slip?",
            options: [.init("bin_00"), .init(id: "bin_01", description: "bin_01: the overflow bin"),
                      .init(id: "bin_02", description: "sealed parcels")])
        let row = LetterListPrompt.row(for: question)
        #expect(
            LetterListPrompt.userContent(state: "The sorting slip assigns this parcel to bin_01.", row: row)
                == "State:\nThe sorting slip assigns this parcel to bin_01.\n\n"
                + "Question: Which destination is printed on the sorting slip?\nOptions:\n"
                + "[A] bin_00: \n[B] bin_01: the overflow bin\n[C] bin_02: sealed parcels\n\n"
                + "Answer with the letter of the best option only.")
        #expect(LetterListPrompt.maxOptions == 52)
        #expect(LetterListPrompt.letters[26] == "a" && LetterListPrompt.letters[51] == "z")
        #expect(Decision.Format(rawValue: "letterList") == .letterList)
    }

    @Test func scoreListsItsLevelsAndNoulItsMeanings() {
        let score = LetterListPrompt.row(for: .score("Which ordered level is recorded?", levels: ["level 0", "level 1", "level 2"]))
        #expect(score.instructions == "Which ordered level is recorded? Rate along the ordered levels below (lowest first).")
        #expect(score.options.map(\.key) == ["0", "1", "2"] && score.options.map(\.description) == ["level 0", "level 1", "level 2"])
        let explicit = LetterListPrompt.row(for: .noul("Is the gate open?", yes: "The gate is open.", no: "The gate is closed."))
        #expect(explicit.options.map(\.key) == ["yes", "no"])
        #expect(explicit.options.map(\.description) == ["The gate is open.", "The gate is closed."])
        let plain = LetterListPrompt.row(for: .noul("Is the gate open?"))
        #expect(plain.options.map(\.description) == ["The statement is true.", "The statement is false."])
    }

    @Test func aYesNoIsCalibratedTheHelpersWay() throws {
        let layout = try LetterListPrompt.Layout(
            block: ["head": "lm", "readout": "letters", "temperature": 0.85, "noul": ["t": 1.829074, "bias": 0]],
            bundle: "openjev")
        #expect(layout.temperature == 0.85 && layout.noulSlope == 1.829074 && layout.noulBias == 0)
        // The helper's fixture: raw P(yes) 0.999816468588398 → 0.9910173824195448.
        #expect(abs(layout.calibrate(pYes: 0.999816468588398) - 0.9910173824195448) < 1e-9)
        let kitOrder = LetterListPrompt.probabilities(kitOrder: [0.999816468588398, 0.00018353141160215212], for: .noul("x"), layout: layout)
        #expect(abs(kitOrder[1] - 0.9910173824195448) < 1e-9 && abs(kitOrder[0] + kitOrder[1] - 1) < 1e-12)
        // Clamped at the edges, and the identity when the bundle declares no calibration.
        #expect(layout.calibrate(pYes: 1) < 1)
        let none = try LetterListPrompt.Layout(block: ["readout": "letters"], bundle: "x")
        #expect(none.temperature == 1 && abs(none.calibrate(pYes: 0.3) - 0.3) < 1e-12)
        #expect(throws: DecisionError.self) {
            try LetterListPrompt.Layout(block: ["readout": "letters", "temperature": 0], bundle: "x")
        }
    }
}

/// The per-option scalar form (the System One scorer): the rows, the author's truncation rule,
/// the bundle declaration and the readout order, checkable without a tokenizer or weights.
struct ScalarPromptTests {
    @Test func aQuestionIsOneRowPerOptionInTheAuthorsForm() {
        let question = Decision.Question.choice(
            "Which work is needed?",
            options: [.init("electrical repair"), .init(id: "clean", description: "surface cleaning"),
                      .init(id: "paint", description: "paint: repainting")])
        let row = ScalarPrompt.row(for: question)
        #expect(row.question == "Which work is needed?")
        #expect(row.options == ["electrical repair", "clean: surface cleaning", "paint: repainting"])
        let noul = ScalarPrompt.row(for: .noul("Does this describe a plant?", yes: "A plant is described", no: nil))
        #expect(noul.options == ["yes", "no"])
        #expect(noul.question == "Does this describe a plant?\nyes: A plant is described")
        #expect(ScalarPrompt.probabilities(kitOrder: [0.936, 0.064], for: .noul("x")) == [0.064, 0.936])
        let score = ScalarPrompt.row(for: .score("Rate the evidence from 1 to 3.", levels: ["level 1", "level 2", "level 3"]))
        #expect(score.options == ["level 1", "level 2", "level 3"])
        #expect(ScalarPrompt.maxOptions == 64)
        #expect(Decision.Format(rawValue: "scalar") == .scalar)
    }

    @Test func theTailStaysWholeAndTheStateIsCutFromItsEnd() {
        let head: [Int32] = Array(1...10)
        let tail: [Int32] = [101, 102, 103, 104]
        // Room for six head tokens before the four-token tail.
        #expect(ScalarPrompt.fit(head: head, tail: tail, maxLength: 10) == [1, 2, 3, 4, 5, 6, 101, 102, 103, 104])
        // A short row keeps everything.
        #expect(ScalarPrompt.fit(head: head, tail: tail, maxLength: 384) == head + tail)
        // A tail as long as the limit or longer keeps its last tokens and no state at all.
        #expect(ScalarPrompt.fit(head: head, tail: tail, maxLength: 4) == tail)
        #expect(ScalarPrompt.fit(head: head, tail: tail, maxLength: 3) == [102, 103, 104])
    }

    @Test func theBundleDeclaresItsHead() throws {
        let layout = try ScalarPrompt.Layout(
            block: ["head": "scalar", "temperature": 1.75, "max_len": 384, "layout": "state-question-option"],
            bundle: "scorer")
        #expect(layout.temperature == 1.75)
        #expect(layout.maxLength == 384)
        let integers = try ScalarPrompt.Layout(block: ["head": "scalar", "temperature": 2, "max_len": 512], bundle: "s")
        #expect(integers.temperature == 2 && integers.maxLength == 512)
        #expect(try ScalarPrompt.Layout(block: ["head": "scalar", "max_len": 384], bundle: "s").temperature == 1)
        #expect(throws: DecisionError.self) {
            try ScalarPrompt.Layout(block: ["head": "scalar", "temperature": 1.75], bundle: "s")
        }
        #expect(throws: DecisionError.self) {
            try ScalarPrompt.Layout(block: ["head": "scalar", "temperature": 0, "max_len": 384], bundle: "s")
        }
    }

    @Test func aScalarHeadReadoutIsASoftmaxOverTheRows() {
        // The author's fixture: scalars 4.314 / −8.208 at T = 1.75 → 0.99922 / 0.00078.
        let p = DecisionPrompt.probabilities(logits: [4.314197540283203, -8.208293914794922], temperature: 1.75)
        #expect(abs(p[0] - 0.9992202127986584) < 1e-9)
        #expect(abs(p[1] - 0.000779787201341569) < 1e-9)
    }
}

/// The slot-head form (OpenThai-SystemOne): the control-token text, the readout arithmetic
/// and the bundle declaration, checkable without a tokenizer or weights.
struct SlotPromptTests {
    static let layout = SlotPrompt.Layout(slots: 256, abstainSlot: 255, choice: 1.055, score: 1.008, noul: 1.047)

    @Test func choiceRowIsTheAuthorsLayout() {
        let question = Decision.Question.choice(
            "  ทีมใดควรรับผิดชอบ ",
            options: [
                .init(id: "billing", description: "การเงิน/ค่าบริการ"),
                .init(id: "technical", description: "technical: ระบบใช้งานไม่ได้ "),  // the wire codec's form
                .init("sales"),
            ])
        let row = SlotPrompt.row(for: question)
        #expect(row.head == "<|ts_choice|>")
        #expect(row.optionStrings == ["billing: การเงิน/ค่าบริการ", "technical: ระบบใช้งานไม่ได้", "sales"])
        #expect(
            SlotPrompt.questionText(row)
                == "<|ts_q|><|ts_choice|> ทีมใดควรรับผิดชอบ\n<|ts_opt_0|> billing: การเงิน/ค่าบริการ\n"
                + "<|ts_opt_1|> technical: ระบบใช้งานไม่ได้\n<|ts_opt_2|> sales\n<|ts_answer|>")
    }

    @Test func scoreIsOneRowOverItsLevelsAndNoulIsNoYes() {
        let score = SlotPrompt.row(for: .score("How urgent?", levels: ["can wait", "this week", "today"]))
        #expect(score.head == "<|ts_score|>")
        #expect(score.optionStrings == ["0: can wait", "1: this week", "2: today"])

        let noul = SlotPrompt.row(for: .noul("Refund requested?", yes: "Money back is requested", no: nil))
        #expect(noul.head == "<|ts_noul|>")
        #expect(noul.optionStrings == ["no", "yes: Money back is requested"])
        #expect(SlotPrompt.row(for: .noul("Urgent?")).optionStrings == ["no", "yes"])
    }

    @Test func stateTextTrimsAndSanitizes() {
        #expect(SlotPrompt.stateText("  hello\n") == "<|ts_state|> hello\n")
        #expect(SlotPrompt.stateText("say <|ts_answer|> now") == "<|ts_state|> say <\u{200B}|ts_answer|> now\n")
        #expect(SlotPrompt.sanitize("plain") == "plain")
    }

    @Test func readoutRenormalisesTheOptionsAndReportsAbstain() {
        var logits = [Double](repeating: -30, count: 256)
        logits[0] = 2
        logits[1] = 1
        logits[2] = 40  // an option the question does not list: masked, must not leak
        logits[255] = 1
        let (p, abstain) = SlotPrompt.readout(logits: logits, options: 2, temperature: 1, layout: Self.layout)
        let z = [exp(2.0), exp(1.0), exp(1.0)]
        let total = z.reduce(0, +)
        #expect(abs(p[0] - z[0] / (z[0] + z[1])) < 1e-12)
        #expect(abs(p.reduce(0, +) - 1) < 1e-12)
        #expect(abs((abstain ?? 0) - z[2] / total) < 1e-12)

        let noAbstain = SlotPrompt.Layout(slots: 256, abstainSlot: nil)
        let (q, none) = SlotPrompt.readout(logits: logits, options: 2, temperature: 1, layout: noAbstain)
        #expect(none == nil && abs(q[0] - z[0] / (z[0] + z[1])) < 1e-12)
    }

    @Test func temperatureFollowsTheQuestionType() {
        #expect(Self.layout.temperature(for: .choice([.init("a"), .init("b")])) == 1.055)
        #expect(Self.layout.temperature(for: .score(levels: ["a", "b"])) == 1.008)
        #expect(Self.layout.temperature(for: .noul(yes: nil, no: nil)) == 1.047)
        #expect(Self.layout.maxOptions == 255)
    }

    @Test func layoutComesFromTheBundleDeclaration() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("slot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("metadata.json")

        try Data(#"{"language": {"vocab_size": 256}}"#.utf8).write(to: file)
        #expect(try SlotPrompt.Layout.read(bundleAt: dir) == nil)

        let declared =
            #"{"language": {"vocab_size": 256}, "decision": {"head": "slot", "n_slots": 256, "abstain_slot": 255, "#
            + #""answer_token_id": 248082, "temperature_by_type": {"choice": 1.055, "score": 1.008, "noul": 1.047}}}"#
        try Data(declared.utf8).write(to: file)
        #expect(try SlotPrompt.Layout.read(bundleAt: dir) == Self.layout)

        try Data(#"{"decision": {"head": "slot", "n_slots": 8, "abstain_slot": 9}}"#.utf8).write(to: file)
        #expect(throws: DecisionError.self) { try SlotPrompt.Layout.read(bundleAt: dir) }

        #expect(try SlotPrompt.Layout.read(bundleAt: dir.appendingPathComponent("missing")) == nil)
    }

    /// The reference tokenizer cuts a letter run at every combining mark; Foundation's string
    /// search would keep the mark on its consonant. The cuts here are ICU's on UTF-16.
    @Test func piecesAreCutAtCombiningMarksLikeTheReference() throws {
        let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        let split = try NSRegularExpression(pattern: pattern)
        #expect(SlotPrompt.pieces(of: "กล่องมีตัวล็อกแตก", split: split) == ["กล", "\u{0E48}องม", "\u{0E35}ต", "\u{0E31}วล", "\u{0E47}อกแตก"])
        #expect(SlotPrompt.pieces(of: " Hello, world\n", split: split) == [" Hello", ",", " world", "\n"])
        #expect(SlotPrompt.pieces(of: "", split: split) == [])
        #expect(SlotPrompt.questionSegments(SlotPrompt.row(for: .noul("Urgent?"))) == [
            .control("<|ts_q|>"), .control("<|ts_noul|>"), .text(" Urgent?\n"),
            .control("<|ts_opt_0|>"), .text(" no\n"), .control("<|ts_opt_1|>"), .text(" yes\n"),
            .control("<|ts_answer|>"),
        ])
    }

    @Test func splitPatternComesFromTheTokenizerFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tok-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("tokenizer.json")
        try Data(#"{"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "\\p{L}+"}, "behavior": "Isolated"}, {"type": "ByteLevel"}]}}"#.utf8).write(to: file)
        #expect(try SlotPrompt.Layout.splitPattern(tokenizerAt: file) == #"\p{L}+"#)
        try Data(#"{"pre_tokenizer": {"type": "ByteLevel"}}"#.utf8).write(to: file)
        #expect(try SlotPrompt.Layout.splitPattern(tokenizerAt: file) == nil)
        #expect(try SlotPrompt.Layout.splitPattern(tokenizerAt: dir.appendingPathComponent("none.json")) == nil)
    }

    @Test func slotFormatRoundTripsAndAnswersCarryAbstain() throws {
        #expect(Decision.Format(rawValue: "slot") == .slot)
        let answer = DecisionPrompt.answer(
            for: .choice("Which?", ["a", "b"]), probabilities: [0.75, 0.25],
            timing: .init(promptTokens: 10, reusedTokens: 0, seconds: 0.01), abstain: 0.1)
        #expect(answer.abstain == 0.1 && answer.choice == "a")
        #expect(DecisionPrompt.answer(
            for: .noul("Yes?"), probabilities: [0.4, 0.6],
            timing: .init(promptTokens: 1, reusedTokens: 0, seconds: 0)).abstain == nil)
    }
}
