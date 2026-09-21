// DecisionPrompt.swift — how a state and a typed question become the token sequence whose
// last position is scored.
//
// The rendering is the one an instruction-tuned chat model answers best zero-shot, and the
// one whose published numbers the kit can be checked against: a fixed system line, then one
// user turn holding a JSON object — the state as `evidence`, the instructions as
// `criterion`, and the options as lettered descriptions — then the assistant turn opened
// with its thinking block already closed. The answer slot is the next token; the letters
// A–P are single tokens in every catalog chat tokenizer, so the probability of each option
// is the softmax over those letter logits at that one position.
//
// Static and tokenizer-in so the rendering can be checked against a reference token
// sequence without loading weights.

import Foundation
import Tokenizers

enum DecisionPrompt {
    /// One answer slot per option, in this order. Sixteen is the ceiling of the shape: a
    /// longer option list is a retrieval problem, not a decision.
    static let letters: [String] = Array("ABCDEFGHIJKLMNOP").map(String.init)
    static let maxOptions = letters.count
    static let maxScoreLevels = 10

    /// The system line. Fixed: it is what makes the letter at the answer slot the whole
    /// answer, and it is shared verbatim by every decision so its tokens are prefilled once.
    static let system =
        "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
        + "Respond with only its uppercase letter, with no explanation or reasoning."

    struct Rendered: Sendable, Equatable {
        /// The full prompt, ending at the answer slot.
        let tokens: [Int32]
        /// The answer-slot token per option, in option order.
        let slots: [Int32]
    }

    /// Validates the question's shape against the letter table.
    static func validate(_ question: Decision.Question) throws {
        guard !question.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DecisionError.emptyInstructions
        }
        let count = question.optionDescriptions.count
        guard count >= 2 else { throw DecisionError.tooFewOptions(count: count) }
        let limit: Int
        if case .score = question.kind { limit = maxScoreLevels } else { limit = maxOptions }
        guard count <= limit else { throw DecisionError.tooManyOptions(count: count, max: limit) }
    }

    /// The user turn: the request as one JSON object, with Python's default `json.dumps`
    /// spacing — the reference rendering the published numbers were produced with.
    static func userPayload(state: String, criterion: String, options: [String]) -> String {
        let rendered = options.enumerated().map { index, description in
            "{\"letter\": \(jsonString(letters[index])), \"description\": \(jsonString(description))}"
        }
        return "{\"evidence\": \(jsonString(state)), \"criterion\": \(jsonString(criterion)), "
            + "\"options\": [\(rendered.joined(separator: ", "))]}"
    }

    static func messages(state: String, criterion: String, options: [String]) -> [[String: any Sendable]] {
        [
            ["role": "system", "content": system],
            ["role": "user", "content": userPayload(state: state, criterion: criterion, options: options)],
        ]
    }

    /// The prompt tokens for one question on one state.
    static func render(
        state: String, question: Decision.Question, tokenizer: any Tokenizer
    ) throws -> Rendered {
        try validate(question)
        let options = question.optionDescriptions
        let tokens = try tokens(
            messages: messages(state: state, criterion: question.instructions, options: options),
            tokenizer: tokenizer)
        let slots = try slotTokens(count: options.count, tokenizer: tokenizer)
        return Rendered(tokens: tokens, slots: slots)
    }

    /// The longest token prefix every question on `state` shares: the system turn and the
    /// user turn up to the end of the state. Prefilling it once is what makes the second
    /// decision on the same state cheap.
    static func statePrefix(state: String, tokenizer: any Tokenizer) throws -> [Int32] {
        // Two renderings that differ from the first character after the state; their
        // common prefix is exactly the tokens that do not depend on the question.
        let a = try tokens(
            messages: messages(state: state, criterion: "A", options: ["a", "b"]), tokenizer: tokenizer)
        let b = try tokens(
            messages: messages(state: state, criterion: "B", options: ["b", "a"]), tokenizer: tokenizer)
        return Array(a.prefix(commonPrefixLength(a, b)))
    }

    /// Chat-template rendering with thinking off. A template that ignores the flag would let
    /// the model open its own think block at the answer slot, so the closed block is appended
    /// when the rendering does not already end in one. The check is on the decoded tail, not
    /// on token ids: a template may spell `<think>` in pieces that differ from the vocabulary's
    /// own token (MiniCPM5 does), and an id comparison then appends a second closed block —
    /// six tokens the reference rendering does not have.
    static func tokens(messages: [[String: any Sendable]], tokenizer: any Tokenizer) throws -> [Int32] {
        var ids = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: true,
            truncation: false, maxLength: nil, tools: nil,
            additionalContext: ["enable_thinking": false]
        ).map(Int32.init)
        let tail = KitTextNormalizer.closedThink(tokenizer)
        if !tail.isEmpty {
            let ending = tokenizer.decode(
                tokens: ids.suffix(tail.count + 4).map(Int.init), skipSpecialTokens: false)
            if !ending.contains("</think>") { ids += tail }
        }
        return ids
    }

    /// The answer-slot token for each of the first `count` letters. Each must be one token
    /// that round-trips, and must not merge with the newline that precedes the slot.
    static func slotTokens(count: Int, tokenizer: any Tokenizer) throws -> [Int32] {
        let newline = tokenizer.encode(text: "\n\n", addSpecialTokens: false)
        return try letters.prefix(count).map { letter in
            let ids = tokenizer.encode(text: letter, addSpecialTokens: false)
            guard ids.count == 1, tokenizer.decode(tokens: ids, skipSpecialTokens: false) == letter,
                tokenizer.encode(text: "\n\n" + letter, addSpecialTokens: false) == newline + ids
            else { throw DecisionError.answerSlotNotSingleToken(letter: letter) }
            return Int32(ids[0])
        }
    }

    static func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    /// A JSON string literal the way Python's `json.dumps(s, ensure_ascii=False)` writes it:
    /// quotes, backslashes and the named control characters escaped, other control
    /// characters as `\u00XX`, everything else (including non-ASCII) verbatim.
    static func jsonString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}

// MARK: - Readout

extension DecisionPrompt {
    /// Softmax over the answer-slot logits, in option order. `temperature` scales the logits
    /// first (1 = the raw distribution).
    static func probabilities(
        logits: [Double], temperature: Double
    ) -> [Double] {
        let scaled = logits.map { $0 / temperature }
        guard let peak = scaled.max() else { return [] }
        let weights = scaled.map { exp($0 - peak) }
        let total = weights.reduce(0, +)
        return weights.map { $0 / total }
    }

    /// 1 − normalised entropy: 1 when all the mass is on one option, 0 when flat.
    static func certainty(_ probabilities: [Double]) -> Double {
        guard probabilities.count > 1 else { return 1 }
        let entropy = -probabilities.reduce(0.0) { $0 + ($1 > 0 ? $1 * log($1) : 0) }
        return max(0, 1 - entropy / log(Double(probabilities.count)))
    }

    /// Folds a probability vector into the question's answer shape.
    static func answer(
        for question: Decision.Question, probabilities p: [Double], timing: Decision.Timing
    ) -> Decision.Answer {
        let value: Decision.Answer.Value
        switch question.kind {
        case .choice(let options):
            let ids = options.map(\.id)
            let ranking = ids.indices.sorted { p[$0] > p[$1] }.map { ids[$0] }
            let best = p.indices.max { p[$0] < p[$1] } ?? 0
            value = .choice(
                Decision.Choice(
                    id: ids[best], confidence: p[best], certainty: certainty(p),
                    probabilities: Dictionary(uniqueKeysWithValues: zip(ids, p)),
                    ranking: ranking, options: ids))
        case .score:
            let best = p.indices.max { p[$0] < p[$1] } ?? 0
            let expected = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
            value = .score(
                Decision.Score(
                    value: expected, level: best, confidence: p[best], certainty: certainty(p),
                    probabilities: p))
        case .noul:
            value = .noul(p[1])
        }
        return Decision.Answer(value: value, timing: timing)
    }
}
