// DeciderPrompt.swift — the prompt form a decision model is trained on (`Decision.Format.decider`).
//
// No chat template and no special tokens: the state under a `Context:` head, then one
// question block — `Question:`, `Options:` as lettered lines, `Answer: (` — whose last token
// is the answer slot. The option letters are single tokens, so the probability of each option
// is the softmax over their logits at that position. The context is encoded on its own and
// the question block on its own, so every question on a state shares the context's tokens
// exactly (the shared prefix `TypedDecisions.prefill` runs once).
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
    /// Answer labels in option order. The first ten are written as text inside the row
    /// (`(A) …`); a wider row puts each label's token id in by itself, the form the model's
    /// own builder switches to past ten options.
    static let letters: [String] = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
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
            return [
                Row(
                    question: question.instructions,
                    options: options.map { $0.id == $0.description ? $0.id : "\($0.id): \($0.description)" })
            ]
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

    /// The tokens every row on `state` starts with.
    static func contextTokens(state: String, tokenizer: any Tokenizer) -> [Int32] {
        tokenizer.encode(text: contextHead + state, addSpecialTokens: false).map(Int32.init)
    }

    /// The question block of a row with at most `narrowLimit` options, as one string.
    static func narrowText(_ row: Row) -> String {
        var text = "\n\nQuestion: \(row.question)\nOptions:"
        for (index, option) in row.options.enumerated() {
            text += "\n(\(letters[index])) \(option)"
        }
        return text + "\nAnswer: ("
    }

    /// The prompt tokens of one row on one state, and the answer-slot token per option.
    static func render(state: String, row: Row, tokenizer: any Tokenizer) throws -> DecisionPrompt.Rendered {
        let labels = try labelTokens(count: row.options.count, tokenizer: tokenizer)
        func encode(_ text: String) -> [Int32] {
            tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        }
        var tokens = contextTokens(state: state, tokenizer: tokenizer)
        if row.options.count <= narrowLimit {
            tokens += encode(narrowText(row))
        } else {
            tokens += encode("\n\nQuestion: \(row.question)\nOptions:")
            let open = encode("\n(")
            for (index, option) in row.options.enumerated() {
                tokens += open + [labels[index]] + encode(") \(option)")
            }
            tokens += encode("\nAnswer: (")
        }
        return DecisionPrompt.Rendered(tokens: tokens, slots: labels)
    }

    /// The answer-slot token for each of the first `count` letters; each must be one token.
    static func labelTokens(count: Int, tokenizer: any Tokenizer) throws -> [Int32] {
        guard count <= letters.count else {
            throw DecisionError.tooManyOptions(count: count, max: letters.count)
        }
        return try letters.prefix(count).map { letter in
            let ids = tokenizer.encode(text: letter, addSpecialTokens: false)
            guard ids.count == 1 else { throw DecisionError.answerSlotNotSingleToken(letter: letter) }
            return Int32(ids[0])
        }
    }

    /// Per-level P(fits) → the distribution over levels, as the model's API combines them.
    static func combine(fit: [Double]) -> [Double] {
        let mass = fit.reduce(0, +)
        let total = mass > 0 ? mass : 1e-9
        return fit.map { $0 / total }
    }
}
