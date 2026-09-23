// LetterListPrompt.swift — the lettered option list under the chat template
// (`Decision.Format.letterList`): OpenJev, and models served through its helper.
//
// The model keeps the ordinary LM head; what its helper sends is one user turn under the chat
// template (thinking closed), read at the BARE letters A–Z then a–z as the next token:
//
//   State:
//   {state}
//
//   Question: {instructions}
//   Options:
//   [A] key: description
//   [B] key:
//
//   Answer with the letter of the best option only.
//
// An option without a description is written `key: ` (the helper's own rendering). The
// letters' logits are divided by the temperature the bundle declares (`decision.temperature`,
// 0.85 for OpenJev, fitted by its author) and softmaxed over the listed letters. A score
// appends " Rate along the ordered levels below (lowest first)." to the question and lists
// the levels as `[A] 0: level`; its answer is the expected index. A yes/no lists
// `[A] yes: <what yes means>` and `[B] no: <what no means>` — the helper's defaults "The
// statement is true." / "The statement is false." when the question gives none — and its
// P(yes) is calibrated as the helper does: sigmoid(logit(p_yes) / t + bias) with the bundle's
// `decision.noul` (t 1.829074, bias 0 for OpenJev). Up to 52 options, one letter each.
//
// Static and tokenizer-in, like the other renderings; `decide-cli parity` checks the token
// rows against the author's fixture.

import Foundation
import Tokenizers

enum LetterListPrompt {
    static let letters: [String] = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz").map(String.init)
    static let maxOptions = letters.count
    static let yesDescription = "The statement is true."
    static let noDescription = "The statement is false."
    static let scoreSuffix = " Rate along the ordered levels below (lowest first)."
    static let closing = "Answer with the letter of the best option only."

    /// What a letter-list bundle declares in its metadata.json `decision` block.
    struct Layout: Sendable, Equatable {
        /// The letters' logits are divided by this before the softmax.
        let temperature: Double
        /// The yes/no calibration: P(yes) → sigmoid(logit(P(yes)) / slope + bias).
        let noulSlope: Double
        let noulBias: Double

        init(temperature: Double, noulSlope: Double = 1, noulBias: Double = 0) {
            self.temperature = temperature
            self.noulSlope = noulSlope
            self.noulBias = noulBias
        }

        /// Reads the bundle's `decision` block when it declares a letter readout; nil otherwise.
        static func read(bundleAt url: URL) throws -> Layout? {
            let file = url.appendingPathComponent("metadata.json")
            guard FileManager.default.fileExists(atPath: file.path) else { return nil }
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
            guard let block = root?["decision"] as? [String: Any], (block["readout"] as? String) == "letters" else {
                return nil
            }
            return try Layout(block: block, bundle: url.lastPathComponent)
        }

        init(block: [String: Any], bundle: String) throws {
            func number(_ value: Any?) -> Double? {
                if let value = value as? Double { return value }
                if let value = value as? Int { return Double(value) }
                return nil
            }
            let temperature = number(block["temperature"]) ?? 1
            let noul = block["noul"] as? [String: Any] ?? [:]
            let slope = number(noul["t"]) ?? 1
            let bias = number(noul["bias"]) ?? 0
            guard temperature > 0, slope > 0 else {
                throw DecisionError.unsupportedModel(
                    id: bundle, reason: "its metadata.json declares a letter readout with a non-positive temperature or yes/no slope")
            }
            self.init(temperature: temperature, noulSlope: slope, noulBias: bias)
        }

        /// The helper's yes/no calibration of a raw P(yes).
        func calibrate(pYes: Double) -> Double {
            let p = min(max(pYes, 1e-4), 1 - 1e-4)
            let z = log(p / (1 - p)) / noulSlope + noulBias
            return 1 / (1 + exp(-z))
        }
    }

    /// One rendered question: the question line and the `key: description` pairs in letter order.
    struct Row: Sendable, Equatable {
        let instructions: String
        let options: [(key: String, description: String)]

        static func == (a: Row, b: Row) -> Bool {
            a.instructions == b.instructions && a.options.count == b.options.count
                && zip(a.options, b.options).allSatisfy { $0.key == $1.key && $0.description == $1.description }
        }
    }

    static func row(for question: Decision.Question) -> Row {
        switch question.kind {
        case .choice(let options):
            return Row(instructions: question.instructions, options: options.map { ($0.id, description($0)) })
        case .score(let levels):
            return Row(
                instructions: question.instructions + scoreSuffix,
                options: levels.enumerated().map { (String($0.offset), $0.element) })
        case .noul(let yes, let no):
            return Row(
                instructions: question.instructions,
                options: [("yes", yes ?? yesDescription), ("no", no ?? noDescription)])
        }
    }

    /// What follows `key: ` for a choice option: nothing when the option is only a name, the
    /// description otherwise (the wire codec's composed `id: description` split back).
    static func description(_ option: Decision.Option) -> String {
        if option.description == option.id || option.description.isEmpty { return "" }
        if option.description.hasPrefix(option.id + ": ") {
            return String(option.description.dropFirst(option.id.count + 2))
        }
        return option.description
    }

    /// The user turn, byte for byte the helper's text-mode prompt.
    static func userContent(state: String, row: Row) -> String {
        let lines = row.options.enumerated().map { "[\(letters[$0.offset])] \($0.element.key): \($0.element.description)" }
        return "State:\n\(state)\n\nQuestion: \(row.instructions)\nOptions:\n\(lines.joined(separator: "\n"))\n\n\(closing)"
    }

    static func messages(state: String, row: Row) -> [[String: any Sendable]] {
        [["role": "user", "content": userContent(state: state, row: row)]]
    }

    /// The prompt tokens for one question on one state, and the letter token per option.
    static func render(state: String, question: Decision.Question, tokenizer: any Tokenizer) throws -> DecisionPrompt.Rendered {
        let row = row(for: question)
        let tokens = try DecisionPrompt.tokens(messages: messages(state: state, row: row), tokenizer: tokenizer)
        let slots = try labelTokens(count: row.options.count, tokenizer: tokenizer)
        return DecisionPrompt.Rendered(tokens: tokens, slots: slots)
    }

    /// The longest token prefix every question on `state` shares: two renderings that differ
    /// from the first character after the state.
    static func statePrefix(state: String, tokenizer: any Tokenizer) throws -> [Int32] {
        let a = try DecisionPrompt.tokens(
            messages: messages(state: state, row: Row(instructions: "A", options: [("a", ""), ("b", "")])),
            tokenizer: tokenizer)
        let b = try DecisionPrompt.tokens(
            messages: messages(state: state, row: Row(instructions: "B", options: [("b", ""), ("a", "")])),
            tokenizer: tokenizer)
        return Array(a.prefix(DecisionPrompt.commonPrefixLength(a, b)))
    }

    /// The bare letter token per option; each must be one token that round-trips.
    static func labelTokens(count: Int, tokenizer: any Tokenizer) throws -> [Int32] {
        guard count <= letters.count else { throw DecisionError.tooManyOptions(count: count, max: letters.count) }
        return try letters.prefix(count).map { letter in
            let ids = tokenizer.encode(text: letter, addSpecialTokens: false)
            guard ids.count == 1, tokenizer.decode(tokens: ids, skipSpecialTokens: false) == letter else {
                throw DecisionError.answerSlotNotSingleToken(letter: letter)
            }
            return Int32(ids[0])
        }
    }

    /// The readout in the kit's option order: a noul's letters are yes then no, calibrated
    /// the helper's way and reported as [no, yes]; a choice or score is the distribution as is.
    static func probabilities(kitOrder p: [Double], for question: Decision.Question, layout: Layout) -> [Double] {
        if case .noul = question.kind, p.count == 2 {
            let yes = layout.calibrate(pYes: p[0])
            return [1 - yes, yes]
        }
        return p
    }
}
