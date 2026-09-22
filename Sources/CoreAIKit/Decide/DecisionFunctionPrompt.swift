// DecisionFunctionPrompt.swift — the plain-text "decision function" form (`Decision.Format.decisionFunction`):
// chaoliangUNSW's Jev-Style-Qwen3.5-2B-Decision and models trained the same way.
//
// No chat template and no BOS: a fixed header, the state, the question and lettered options,
// then `Answer:`; the answer is the next token, one of the space-prefixed letters ` A`, ` B`, …
// (each a single token in the Qwen vocabulary). The model's calibration temperature is folded
// into its final norm, so the readout is a plain softmax over the listed letters' logits:
//
//   You are a decision function. Read the state, then answer the question by choosing exactly one option.
//
//   [State]
//   {state}
//
//   [Question]
//   {question}
//
//   [Options]
//   A. {option}
//   B. {option}
//
//   Answer:
//
// The author's three shapes are one prompt each: a choice over the options, a bool as the
// choice `yes` / `no` (A = yes), a score as the choice over its ordered levels. The kit reports
// a noul as `[no, yes]`, so the bool readout is swapped into that order. Up to 26 options, one
// letter each.

import Foundation
import Tokenizers

enum DecisionFunctionPrompt {
    static let header =
        "You are a decision function. Read the state, then answer the question by choosing exactly one option.\n\n"
    static let letters: [String] = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
    static let maxOptions = letters.count

    /// One rendered question: its text and its options in letter order.
    struct Row: Sendable, Equatable {
        let question: String
        let options: [String]
    }

    static func row(for question: Decision.Question) -> Row {
        switch question.kind {
        case .choice(let options):
            return Row(question: question.instructions, options: options.map(optionText))
        case .score(let levels):
            return Row(question: question.instructions, options: levels)
        case .noul(let yes, let no):
            var text = question.instructions
            if let yes { text += "\nyes: \(yes)" }
            if let no { text += "\nno: \(no)" }
            return Row(question: text, options: ["yes", "no"])
        }
    }

    /// An option as the author writes it after `A. `: the description, or the id alone.
    static func optionText(_ option: Decision.Option) -> String {
        if option.description == option.id { return option.id }
        if option.description.hasPrefix(option.id + ": ") {
            return String(option.description.dropFirst(option.id.count + 2))
        }
        return option.description
    }

    /// The whole prompt, byte for byte the author's `build_prompt`.
    static func text(state: String, row: Row) -> String {
        let lines = row.options.enumerated().map { "\(letters[$0.offset]). \($0.element)" }.joined(separator: "\n")
        return header + "[State]\n\(state)\n\n[Question]\n\(row.question)\n\n[Options]\n\(lines)\n\nAnswer:"
    }

    static func encode(_ text: String, tokenizer: any Tokenizer) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// The prompt tokens of one question on one state, and the letter token per option.
    static func render(state: String, question: Decision.Question, tokenizer: any Tokenizer) throws -> DecisionPrompt.Rendered {
        let row = row(for: question)
        return DecisionPrompt.Rendered(
            tokens: encode(text(state: state, row: row), tokenizer: tokenizer),
            slots: try labelTokens(count: row.options.count, tokenizer: tokenizer))
    }

    /// The longest token prefix every question on `state` shares.
    static func statePrefix(state: String, tokenizer: any Tokenizer) -> [Int32] {
        let a = encode(text(state: state, row: Row(question: "A", options: ["a", "b"])), tokenizer: tokenizer)
        let b = encode(text(state: state, row: Row(question: "B", options: ["b", "a"])), tokenizer: tokenizer)
        return Array(a.prefix(DecisionPrompt.commonPrefixLength(a, b)))
    }

    /// The space-prefixed letter token per option; each must be one token that round-trips.
    static func labelTokens(count: Int, tokenizer: any Tokenizer) throws -> [Int32] {
        guard count <= letters.count else { throw DecisionError.tooManyOptions(count: count, max: letters.count) }
        return try letters.prefix(count).map { letter in
            let ids = tokenizer.encode(text: " " + letter, addSpecialTokens: false)
            guard ids.count == 1, tokenizer.decode(tokens: ids, skipSpecialTokens: false) == " " + letter else {
                throw DecisionError.answerSlotNotSingleToken(letter: " " + letter)
            }
            return Int32(ids[0])
        }
    }

    /// The readout in the kit's option order: a noul's letters are yes then no.
    static func probabilities(kitOrder p: [Double], for question: Decision.Question) -> [Double] {
        if case .noul = question.kind, p.count == 2 { return [p[1], p[0]] }
        return p
    }
}
