// DeciderVisionPromptRenderer.swift — a state and typed questions → the decider-2b-vision decoder's id row, the
// author's contract (the checkpoint's `decider/prompt.py build()` and `decider/vision.py prepare()` at revision
// 863e290). Ported from the model zoo's `apps/DeciderVision/Sources/DeciderVision/PromptBuilder.swift` (2307ecf):
//
//   text = encode("Context:\n" + state)[:1536]
//        + per question: encode("\n\nQuestion{ k}: {text}\nOptions:" + "\n({letter}) {option}"… + "\nAnswer{ k}: (")
//   text = encode(decode(text))                              prepare() hands the decoded string to the processor
//   ids  = [<|vision_start|>] + [V + k for k in 0 ..< H·W] + [<|vision_end|>] + text      (a row with an image)
//        = text                                                                           (a text-only row)
//
// " k" is the 1-based question number, written only when there are several questions. Every question of a call
// is one block of ONE row, read once: each answer slot sees the image, the state and every question, but no other
// question's answer (the author's multi-question contract; `TypedDecisions` reads one prompt per question). An id
// ≥ V (248,320) is row id − V of the tower's output. The slots follow the author's rule: " (" (318) right after
// ":" (25) with "Answer" (15666) among the 5 tokens before it, one per question. rope_shift_start = 1 + H·W (the
// <|vision_end|> index), amount = H·W − max(H, W); a text-only row gets start 2³⁰ and amount 0.
//
// A typed question reads the way the author's `decide_json` (`decider/infer.py`) lays it out: a choice lists its
// options (`name: criterion` when an option carries a description, as the decider form writes it); a yes/no is the
// choice `no` / `yes`, read as P(yes); a score lists its levels as `k: level` and answers Σ k · p(k). At most 10
// options, the letters A–J: the author samples a longer list down at training time keeping the gold option, which
// a host cannot do, so a longer one is refused. Option strings go in as written (`neutralize_none` is false in
// this checkpoint's decider_config.json).

import Foundation
import Tokenizers

struct DeciderVisionPromptRenderer: Sendable {
    static let vocab = 248_320
    static let visionStart = 248_053
    static let imagePad = 248_056
    static let visionEnd = 248_054
    static let slotToken = 318
    static let colon = 25
    static let answer = 15_666
    static let letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    static let maxOptions = letters.count
    static let maxContextTokens = 1536
    static let noShift: Int32 = 1 << 30

    /// One built row: what the decoder reads and where the answers are.
    struct Row: Sendable, Equatable {
        /// Decoder ids; the image block is V + k.
        let ids: [Int]
        /// Positions of the answer slots, ascending, one per question.
        let slots: [Int]
        let ropeShiftStart: Int32
        let ropeShiftAmount: Int32
        /// Merged grid side of the image block (8 or 14), nil for a text-only row.
        let grid: Int?
        /// Options per question.
        let optionCounts: [Int]
    }

    let tokenizer: any Tokenizer

    /// The option strings a typed question lists, in answer order.
    static func options(for question: Decision.Question) -> [String] {
        switch question.kind {
        case .choice(let options): return options.map(DeciderPrompt.optionText)
        case .score, .noul: return question.optionDescriptions
        }
    }

    private func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// The author's `build()` with no shuffling, before prepare()'s decode and re-encode.
    func textIDs(state: String, questions: [(text: String, options: [String])]) -> [Int] {
        var ids = Array(encode("Context:\n" + state).prefix(Self.maxContextTokens))
        let multi = questions.count > 1
        for (k, q) in questions.enumerated() {
            let number = multi ? " \(k + 1)" : ""
            var block = "\n\nQuestion\(number): \(q.text)\nOptions:"
            for (j, option) in q.options.enumerated() { block += "\n(\(Self.letters[j])) \(option)" }
            block += "\nAnswer\(number): ("
            ids += encode(block)
        }
        return ids
    }

    /// The decoder row. `grid` = merged grid side of the tower that made the image rows (8 / 14), nil for a
    /// text-only row. Every question is validated first (instructions, 2–10 options).
    func row(state: String, questions: [Decision.Question], grid: Int?) throws -> Row {
        guard !questions.isEmpty else { throw DeciderVisionError.prompt("no questions") }
        for question in questions {
            try DecisionPrompt.validate(question, maxOptions: Self.maxOptions)
        }
        let built = textIDs(
            state: state, questions: questions.map { ($0.instructions, Self.options(for: $0)) })
        let text = encode(tokenizer.decode(tokens: built, skipSpecialTokens: false))
        let ids: [Int]
        let start: Int32, amount: Int32
        if let g = grid {
            let n = g * g
            ids = [Self.visionStart] + (0..<n).map { Self.vocab + $0 } + [Self.visionEnd] + text
            start = Int32(1 + n)
            amount = Int32(n - g)
        } else {
            ids = text
            start = Self.noShift
            amount = 0
        }
        let slots = Self.findSlots(ids)
        guard slots.count == questions.count else {
            // The state or a question writes "Answer: (" itself, or a decode/re-encode merged a slot away.
            throw DeciderVisionError.prompt(
                "the author's slot rule finds \(slots.count) answer slots for \(questions.count) questions")
        }
        return Row(
            ids: ids, slots: slots, ropeShiftStart: start, ropeShiftAmount: amount, grid: grid,
            optionCounts: questions.map { Self.options(for: $0).count })
    }

    /// The author's slot rule on the final row.
    static func findSlots(_ ids: [Int]) -> [Int] {
        guard ids.count > 2 else { return [] }
        return (2..<ids.count).filter { i in
            ids[i] == slotToken && ids[i - 1] == colon && ids[max(0, i - 5)..<i].contains(answer)
        }
    }

    /// V + k back to <|image_pad|>: the processor's form of the same row.
    static func processorIDs(_ ids: [Int]) -> [Int] { ids.map { $0 >= vocab ? imagePad : $0 } }
}
