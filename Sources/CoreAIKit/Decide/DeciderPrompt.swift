// DeciderPrompt.swift — the prompt form a decision model is trained on (`Decision.Format.decider`).
//
// No chat template and no special tokens: the state under a `Context:` head, then one
// question block — `Question:`, `Options:` as lettered lines, `Answer: (` — whose last token
// is the answer slot. The option labels are single tokens, so the probability of each option
// is the softmax over their logits at that position. They are the author's label table
// (`LabelTable`, `.decider` rule): A–Z, then the two-letter names AA, AB, … the tokenizer
// writes as one token, 255 in all, as the author's builder labels a wide question. The
// context is encoded on its own and the question block on its own, so every question on a
// state shares the context's tokens exactly (the shared prefix `TypedDecisions.prefill` runs
// once).
//
// A score question is not one row: each level is judged alone, as a yes/no row that names the
// level without its number or its neighbours, and the levels' P(yes) are normalised into the
// distribution. That is how the model's own API answers a score, and how its fixtures are
// scored; a single row listing every level is a different question the model was not
// trained on.
//
// Static and tokenizer-in, like `DecisionPrompt`, so a rendering can be checked against the
// model's fixture token ids without loading weights.

import Foundation
import Tokenizers

enum DeciderPrompt {
    /// A row of at most this many options writes its labels as text (`(A) …`); a wider row
    /// puts each label's token id in by itself, the form the model's own builder switches to
    /// past ten options.
    static let narrowLimit = 10
    /// The calibration temperature on the model's card: what its API applies to the slot logits.
    static let defaultTemperature = 1.03
    static let contextHead = "Context:\n"

    /// One scored row: a question and its options as the model reads them.
    struct Row: Sendable, Equatable {
        let question: String
        let options: [String]
    }

    /// The rows one typed question becomes: one row for a choice or a yes/no; one yes/no row
    /// per level for a score.
    static func rows(for question: Decision.Question) -> [Row] {
        switch question.kind {
        case .choice(let options):
            return [Row(question: question.instructions, options: options.map(optionText))]
        case .score(let levels):
            return levels.map { level in
                Row(
                    question: "\(question.instructions)\nProposed answer: \(level)\nDoes the proposed answer fit?",
                    options: ["no", "yes"])
            }
        case .noul(let yes, let no):
            return [
                Row(
                    question: question.instructions,
                    options: [no.map { "no: \($0)" } ?? "no", yes.map { "yes: \($0)" } ?? "yes"])
            ]
        }
    }

    /// One option as the author's form writes it: `name: criterion`, or the name alone when
    /// there is no criterion. An option that came through the wire already carries that
    /// composed text as its description (`SystemOne.request(from:)` writes `key: description`,
    /// the text the chat form reads), so it is kept as it is rather than composed twice —
    /// the same rule as the slot, scalar and letter-list forms.
    static func optionText(_ option: Decision.Option) -> String {
        if option.description.isEmpty || option.description == option.id { return option.id }
        if option.description.hasPrefix(option.id + ": ") { return option.description }
        return "\(option.id): \(option.description)"
    }

    /// The tokens every row on `state` starts with.
    static func contextTokens(state: String, tokenizer: any Tokenizer) -> [Int32] {
        tokenizer.encode(text: contextHead + state, addSpecialTokens: false).map(Int32.init)
    }

    /// The question block of a row with at most `narrowLimit` options, as one string;
    /// `labels` holds the label of each option, in order.
    static func narrowText(_ row: Row, labels: [String]) -> String {
        var text = "\n\nQuestion: \(row.question)\nOptions:"
        for (index, option) in row.options.enumerated() {
            text += "\n(\(labels[index])) \(option)"
        }
        return text + "\nAnswer: ("
    }

    /// The prompt tokens of one row on one state, and the answer-slot token per option.
    static func render(
        state: String, row: Row, labels: LabelTable, tokenizer: any Tokenizer
    ) throws -> DecisionPrompt.Rendered {
        try render(state: state, row: row, labels: labels) {
            tokenizer.encode(text: $0, addSpecialTokens: false).map(Int32.init)
        }
    }

    /// The same, with the tokenizer's encoder (no special tokens) as a closure, so a row can
    /// be checked without a tokenizer.
    static func render(
        state: String, row: Row, labels: LabelTable, encode: (String) -> [Int32]
    ) throws -> DecisionPrompt.Rendered {
        let slots = try labels.slots(count: row.options.count)
        var tokens = encode(contextHead + state)
        if row.options.count <= narrowLimit {
            tokens += encode(narrowText(row, labels: labels.names))
        } else {
            tokens += encode("\n\nQuestion: \(row.question)\nOptions:")
            let open = encode("\n(")
            for (index, option) in row.options.enumerated() {
                tokens += open + [slots[index]] + encode(") \(option)")
            }
            tokens += encode("\nAnswer: (")
        }
        return DecisionPrompt.Rendered(tokens: tokens, slots: slots)
    }

    /// Per-level P(fits) → the distribution over levels, as the model's API combines them.
    static func combine(fit: [Double]) -> [Double] {
        let mass = fit.reduce(0, +)
        let total = mass > 0 ? mass : 1e-9
        return fit.map { $0 / total }
    }
}
