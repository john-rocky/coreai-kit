// ScalarPrompt.swift — the per-option scalar form (`Decision.Format.scalar`): pngwn's
// System One scorer (system-one-qwen3.5-4b-scorer) and models trained the same way.
//
// The LM head is replaced by a scalar scoring head (a linear layer to one number), so the
// bundle's logits are one value wide (`language.vocab_size` 1) and a question is one row per
// option, each read at its last token:
//
//   State:
//   {state}
//
//   Question:
//   {question}
//
//   Option:
//   {option}
//
// No chat template and no special tokens. The rows' scalars are softmaxed together at the
// calibration temperature the bundle declares in its metadata.json (`decision.temperature`,
// 1.75 for the scorer, fitted by its author on a validation split). The author's `encode`
// keeps the tail (question + option) whole and cuts the state from its end so the row fits
// `decision.max_len` (384 for the scorer): the head pools the last token, so the option must
// stay legible at the end of the sequence. A tail that alone exceeds the limit keeps its last
// tokens.
//
// A yes/no is the two rows `yes` / `no` (the kit reports [no, yes], so the readout is
// swapped); a score is one row per level, lowest first. An option with a description is read
// as `id: description`, the wire codec's composed form; an option that is only a name is read
// as that name.

import Foundation
import Tokenizers

enum ScalarPrompt {
    /// What a scalar-head bundle declares in its metadata.json `decision` block.
    struct Layout: Sendable, Equatable {
        /// Softmax temperature over a question's row scalars.
        let temperature: Double
        /// The longest row in tokens; the state is cut to fit.
        let maxLength: Int

        init(temperature: Double, maxLength: Int) {
            self.temperature = temperature
            self.maxLength = maxLength
        }

        /// Reads the bundle's `decision` block when it declares a scalar head; nil otherwise.
        static func read(bundleAt url: URL) throws -> Layout? {
            let file = url.appendingPathComponent("metadata.json")
            guard FileManager.default.fileExists(atPath: file.path) else { return nil }
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
            guard let block = root?["decision"] as? [String: Any], (block["head"] as? String) == "scalar" else {
                return nil
            }
            return try Layout(block: block, bundle: url.lastPathComponent)
        }

        init(block: [String: Any], bundle: String) throws {
            func number(_ key: String) -> Double? {
                if let value = block[key] as? Double { return value }
                if let value = block[key] as? Int { return Double(value) }
                return nil
            }
            guard let maxLength = number("max_len"), maxLength >= 16 else {
                throw DecisionError.unsupportedModel(
                    id: bundle, reason: "its metadata.json declares a scalar head without a usable 'max_len'")
            }
            let temperature = number("temperature") ?? 1
            guard temperature > 0 else {
                throw DecisionError.unsupportedModel(
                    id: bundle, reason: "its metadata.json declares a scalar head with a non-positive 'temperature'")
            }
            self.init(temperature: temperature, maxLength: Int(maxLength))
        }
    }

    /// Options a choice may list. The head scores any number, one row each, so this bounds the
    /// cost of one decision rather than the model (the scorer's author trained with option
    /// sets capped at 16 and evaluated up to 77).
    static let maxOptions = 64

    /// One question as its rows read it: the question text and one option string per row.
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

    /// An option as the scorer reads it after `Option:`: the name alone, or `id: description`
    /// when a description was given (once, even when the description already carries it).
    static func optionText(_ option: Decision.Option) -> String {
        if option.description.isEmpty || option.description == option.id { return option.id }
        if option.description.hasPrefix(option.id + ": ") { return option.description }
        return "\(option.id): \(option.description)"
    }

    static func encode(_ text: String, tokenizer: any Tokenizer) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// The head of every row on `state`: `State:` and the state.
    static func headTokens(state: String, tokenizer: any Tokenizer) -> [Int32] {
        encode("State:\n" + state, tokenizer: tokenizer)
    }

    /// The tail of one row: the question and one option, kept whole.
    static func tailTokens(question: String, option: String, tokenizer: any Tokenizer) -> [Int32] {
        encode("\n\nQuestion:\n\(question)\n\nOption:\n\(option)", tokenizer: tokenizer)
    }

    /// The author's `encode`: the tail whole, the head cut from its end so the row fits
    /// `maxLength`; a tail longer than that on its own keeps its last `maxLength` tokens.
    static func fit(head: [Int32], tail: [Int32], maxLength: Int) -> [Int32] {
        if tail.count >= maxLength { return Array(tail.suffix(maxLength)) }
        return Array(head.prefix(maxLength - tail.count)) + tail
    }

    /// One token row per option, in option order.
    static func render(
        state: String, question: Decision.Question, layout: Layout, tokenizer: any Tokenizer
    ) -> [[Int32]] {
        let row = row(for: question)
        let head = headTokens(state: state, tokenizer: tokenizer)
        return row.options.map { option in
            fit(head: head, tail: tailTokens(question: row.question, option: option, tokenizer: tokenizer),
                maxLength: layout.maxLength)
        }
    }

    /// The tokens every row on `state` shares: the head, as far as any row could keep it.
    static func statePrefix(state: String, layout: Layout, tokenizer: any Tokenizer) -> [Int32] {
        Array(headTokens(state: state, tokenizer: tokenizer).prefix(max(0, layout.maxLength - 1)))
    }

    /// The readout in the kit's option order: a noul's rows are yes then no.
    static func probabilities(kitOrder p: [Double], for question: Decision.Question) -> [Double] {
        if case .noul = question.kind, p.count == 2 { return [p[1], p[0]] }
        return p
    }
}
