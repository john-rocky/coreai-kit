// SharedStatePrompt.swift — the prompt form of APUS-OpenJev-v1 (`Decision.Format.sharedState`),
// the author's `jev.dynamic.prompt.v2`.
//
// The model keeps the ordinary LM head; what it was trained on is one user turn under the chat
// template (thinking closed), read at the option letters A–P as the next token:
//
//   Shared state:
//   {state}
//
//   {"criteria": [{"description": "…", "label": "A"}, {"description": "…", "label": "B"}], "instructions": "…", "primitive": "choice"}
//   Return only the selected letter: A, B.
//   Answer:
//
// The JSON is Python's `json.dumps(…, ensure_ascii=False, sort_keys=True)` — keys in sorted
// order, `, ` and `: ` separators, non-ASCII verbatim — which `OrderedJSON` writes byte for
// byte. The author's primitives are `choice` (2–16 criteria, each with a description the model
// reads; the criterion's id is not shown) and a yes/no on a proposition (`noul`, whose two
// criteria are fixed: "The stated proposition is true." as A, "The stated proposition is
// false." as B). There is no ordered-scale primitive, so a score question is rendered as a
// choice over its levels. Probabilities are the softmax over the letter logits, temperature 1:
// the author's runtime applies no calibration and says so (`calibrated: false`).
//
// Static and tokenizer-in, like the other renderings; `decide-cli parity` checks the token
// rows against the author's compiled fixture.

import Foundation
import Tokenizers

enum SharedStatePrompt {
    static let letters = DecisionPrompt.letters
    static let yesDescription = "The stated proposition is true."
    static let noDescription = "The stated proposition is false."

    /// One rendered question: the author's primitive name, the instructions, and the
    /// criteria descriptions in label order.
    struct Row: Sendable, Equatable {
        let primitive: String
        let instructions: String
        let descriptions: [String]
    }

    /// The row a typed question becomes. A noul's criteria are the author's fixed pair, yes
    /// first — the kit reports `[no, yes]`, so `probabilities(kitOrder:)` swaps the readout.
    static func row(for question: Decision.Question) -> Row {
        switch question.kind {
        case .choice(let options):
            return Row(primitive: "choice", instructions: question.instructions, options: options.map(description))
        case .score(let levels):
            return Row(primitive: "choice", instructions: question.instructions, options: levels)
        case .noul(let yes, let no):
            var instructions = question.instructions
            if let yes { instructions += "\nyes: \(yes)" }
            if let no { instructions += "\nno: \(no)" }
            return Row(primitive: "noul", instructions: instructions, options: [yesDescription, noDescription])
        }
    }

    /// The text the model reads for a choice option: its description, or its id when there
    /// is none. The wire codec's composed `id: description` is split back to the description,
    /// since the author's form shows descriptions alone.
    static func description(_ option: Decision.Option) -> String {
        if option.description == option.id { return option.id }
        if option.description.hasPrefix(option.id + ": ") {
            return String(option.description.dropFirst(option.id.count + 2))
        }
        return option.description
    }

    /// The user turn, byte for byte the author's `render_prompt`.
    static func userContent(state: String, row: Row) -> String {
        let criteria = JSONValue.array(
            row.descriptions.enumerated().map { index, description in
                .object([.init("description", .string(description)), .init("label", .string(letters[index]))])
            })
        let task = JSONValue.object([
            .init("criteria", criteria),
            .init("instructions", .string(row.instructions)),
            .init("primitive", .string(row.primitive)),
        ])
        let listed = letters.prefix(row.descriptions.count).joined(separator: ", ")
        return "Shared state:\n\(state)\n\n" + task.dumps() + "\nReturn only the selected letter: \(listed).\nAnswer:"
    }

    static func messages(state: String, row: Row) -> [[String: any Sendable]] {
        [["role": "user", "content": userContent(state: state, row: row)]]
    }

    /// The prompt tokens for one question on one state, and the letter token per criterion.
    static func render(state: String, question: Decision.Question, tokenizer: any Tokenizer) throws -> DecisionPrompt.Rendered {
        let row = row(for: question)
        let tokens = try DecisionPrompt.tokens(messages: messages(state: state, row: row), tokenizer: tokenizer)
        let slots = try DecisionPrompt.slotTokens(count: row.descriptions.count, tokenizer: tokenizer)
        return DecisionPrompt.Rendered(tokens: tokens, slots: slots)
    }

    /// The longest token prefix every question on `state` shares: two renderings that differ
    /// from the first character after the state.
    static func statePrefix(state: String, tokenizer: any Tokenizer) throws -> [Int32] {
        let a = try DecisionPrompt.tokens(
            messages: messages(state: state, row: Row(primitive: "choice", instructions: "A", options: ["a", "b"])),
            tokenizer: tokenizer)
        let b = try DecisionPrompt.tokens(
            messages: messages(state: state, row: Row(primitive: "noul", instructions: "B", options: ["b", "a"])),
            tokenizer: tokenizer)
        return Array(a.prefix(DecisionPrompt.commonPrefixLength(a, b)))
    }

    /// The readout in the kit's option order: a noul's letters are yes then no, the kit
    /// reports no then yes.
    static func probabilities(kitOrder p: [Double], for question: Decision.Question) -> [Double] {
        if case .noul = question.kind, p.count == 2 { return [p[1], p[0]] }
        return p
    }
}

extension SharedStatePrompt.Row {
    init(primitive: String, instructions: String, options: [String]) {
        self.init(primitive: primitive, instructions: instructions, descriptions: options)
    }
}
