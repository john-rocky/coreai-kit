// SlotPrompt.swift — the prompt form of a slot-head decision model (`Decision.Format.slot`):
// OpenThai-SystemOne, and any model built the same way.
//
// The model's LM head is replaced by a small "slot head": the hidden state at a
// `<|ts_answer|>` control token is projected to N slot logits (256 for OpenThai), slot i
// meaning "the option introduced by `<|ts_opt_i|>`", the last slot meaning "none of them"
// (abstain). No chat template, no BOS and no letters: the options are addressed by control
// tokens, so the answer needs no single-token labels. One causal sequence holds the state and
// one question:
//
//   <|ts_state|> {state}\n
//   <|ts_q|><|ts_choice|> {instructions}\n
//   <|ts_opt_0|> {name}: {description}\n
//   <|ts_opt_1|> {name}\n
//   <|ts_answer|>                                ← the position that is read
//
// `<|ts_score|>` and `<|ts_noul|>` head the other two shapes: a score is ONE row whose options
// are its levels as `i: {level}`, a noul is `no` / `yes` with their descriptions. The author's
// encoder writes a newline after `<|ts_answer|>` and, for a request with several questions,
// lays the questions out one after another in the same sequence. The model is causal, so what
// is read at an answer token does not depend on anything after it; the kit renders one
// question per row, ending at that token — the author's single-question encoding minus its
// trailing newline. The state part is encoded on its own (it is what every question on a
// state shares and what `TypedDecisions.prefill` runs once) and the question part on its own,
// exactly as the author's encoder concatenates them.
//
// The readout is the author's too: the slot logits divided by the temperature the bundle
// carries for the question's type, every slot past the option count masked except the abstain
// slot, softmax, the first k renormalised as the option probabilities and the abstain mass
// reported beside them (`Decision.Answer.abstain`). All of that is declared by the bundle, not
// by a catalog id: `metadata.json` carries a `decision` block (`head: "slot"`, `n_slots`,
// `abstain_slot`, `answer_token_id`, `temperature_by_type`) and `Layout.read` turns it into
// the numbers this file needs.
//
// ## Tokenizing the text between the control tokens
//
// The reference tokenizer (Hugging Face `tokenizers`) pre-tokenizes with the regex in
// tokenizer.json — a run of letters `\p{L}+` ends at a combining mark, so Thai, and every
// script that writes vowels and tones as marks, is cut at each mark before BPE. The Swift
// tokenizer applies the same regex through Foundation's string search, whose match ranges
// snap to grapheme clusters: the mark stays attached to its consonant, the run continues, and
// the BPE merges differ (the first Thai fixture row came out 67 tokens where the reference
// has 113). `Encoder` therefore cuts each text segment with the same regex through
// `NSRegularExpression` on UTF-16 (ICU, code-point semantics — the reference's cuts) and
// encodes the pieces one by one; the control tokens are placed by id. Latin text is cut the
// same way by both engines, so this changes nothing there.
//
// Static and tokenizer-in, like the other two renderings, so the exact token rows can be
// checked against the author's fixture without loading weights (`decide-cli parity`).

import Foundation
import Tokenizers

enum SlotPrompt {
    static let stateToken = "<|ts_state|>"
    static let questionToken = "<|ts_q|>"
    static let choiceToken = "<|ts_choice|>"
    static let scoreToken = "<|ts_score|>"
    static let noulToken = "<|ts_noul|>"
    static let answerToken = "<|ts_answer|>"
    static func optionToken(_ index: Int) -> String { "<|ts_opt_\(index)|>" }

    /// What the bundle declares about its head, from the `decision` block of `metadata.json`,
    /// plus the pre-tokenizer regex of its tokenizer.
    struct Layout: Sendable, Equatable {
        /// Width of the slot head.
        let slots: Int
        /// The slot that means "none of the options"; nil when the head has no such slot.
        let abstainSlot: Int?
        /// Softmax temperature per question type, as the model's own API applies it.
        let choiceTemperature: Double
        let scoreTemperature: Double
        let noulTemperature: Double
        /// The `Split` regex of the tokenizer's pre-tokenizer, when it has one.
        let splitPattern: String?

        init(
            slots: Int, abstainSlot: Int?, choice: Double = 1, score: Double = 1, noul: Double = 1,
            splitPattern: String? = nil
        ) {
            self.slots = slots
            self.abstainSlot = abstainSlot
            self.choiceTemperature = choice
            self.scoreTemperature = score
            self.noulTemperature = noul
            self.splitPattern = splitPattern
        }

        /// Options a question may list: every slot but the abstain one.
        var maxOptions: Int { slots - (abstainSlot == nil ? 0 : 1) }

        func temperature(for kind: Decision.Question.Kind) -> Double {
            switch kind {
            case .choice: return choiceTemperature
            case .score: return scoreTemperature
            case .noul: return noulTemperature
            }
        }

        /// The layout a bundle directory declares, or nil when its `metadata.json` has no
        /// `decision` block naming a slot head (an ordinary language bundle).
        static func read(bundleAt url: URL) throws -> Layout? {
            let file = url.appendingPathComponent("metadata.json")
            guard FileManager.default.fileExists(atPath: file.path) else { return nil }
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
            guard let block = root?["decision"] as? [String: Any], (block["head"] as? String) == "slot" else {
                return nil
            }
            return try Layout(
                block: block, bundle: url.lastPathComponent,
                splitPattern: try splitPattern(tokenizerAt: url.appendingPathComponent("tokenizer/tokenizer.json")))
        }

        init(block: [String: Any], bundle: String, splitPattern: String? = nil) throws {
            guard let slots = block["n_slots"] as? Int, slots >= 2 else {
                throw DecisionError.unsupportedModel(
                    id: bundle, reason: "its metadata.json declares a slot head without a usable 'n_slots'")
            }
            let abstain = block["abstain_slot"] as? Int
            if let abstain, !(0..<slots).contains(abstain) {
                throw DecisionError.unsupportedModel(
                    id: bundle, reason: "its metadata.json puts 'abstain_slot' \(abstain) outside the \(slots) slots")
            }
            let temperatures = block["temperature_by_type"] as? [String: Any] ?? [:]
            func temperature(_ key: String) -> Double {
                if let value = temperatures[key] as? Double, value > 0 { return value }
                if let value = temperatures[key] as? Int, value > 0 { return Double(value) }
                return 1
            }
            self.init(
                slots: slots, abstainSlot: abstain,
                choice: temperature("choice"), score: temperature("score"), noul: temperature("noul"),
                splitPattern: splitPattern)
        }

        /// The regex of the first `Split` pre-tokenizer in a tokenizer.json, or nil.
        static func splitPattern(tokenizerAt file: URL) throws -> String? {
            guard FileManager.default.fileExists(atPath: file.path) else { return nil }
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
            guard let pre = root?["pre_tokenizer"] as? [String: Any] else { return nil }
            let steps = (pre["pretokenizers"] as? [[String: Any]]) ?? [pre]
            for step in steps where (step["type"] as? String) == "Split" {
                if let pattern = (step["pattern"] as? [String: Any])?["Regex"] as? String { return pattern }
            }
            return nil
        }
    }

    /// One option as the author's encoder names it: the name the answer reports and, when
    /// there is one, the description the model reads after it.
    struct Slot: Sendable, Equatable {
        let name: String
        let description: String?
    }

    /// One rendered question: its head token, its instructions and its options.
    struct Row: Sendable, Equatable {
        let head: String
        let question: String
        let options: [Slot]

        /// The option strings exactly as they follow their control tokens in the row — the
        /// form a fixture records.
        var optionStrings: [String] { options.map(SlotPrompt.optionText) }
    }

    /// The row a typed question becomes — always one.
    static func row(for question: Decision.Question) -> Row {
        switch question.kind {
        case .choice(let options):
            return Row(head: choiceToken, question: question.instructions, options: options.map(slot))
        case .score(let levels):
            return Row(
                head: scoreToken, question: question.instructions,
                options: levels.enumerated().map { Slot(name: String($0.offset), description: $0.element) })
        case .noul(let yes, let no):
            return Row(
                head: noulToken, question: question.instructions,
                options: [Slot(name: "no", description: no), Slot(name: "yes", description: yes)])
        }
    }

    /// A choice option's name and description. An `Option` made from one string has no
    /// description; one the wire codec made already carries `name: description` as its
    /// description, which is split back so the name is not written twice.
    static func slot(_ option: Decision.Option) -> Slot {
        if option.description == option.id { return Slot(name: option.id, description: nil) }
        if option.description.hasPrefix(option.id + ": ") {
            return Slot(name: option.id, description: String(option.description.dropFirst(option.id.count + 2)))
        }
        return Slot(name: option.id, description: option.description)
    }

    /// User text cannot smuggle a control token into the sequence.
    static func sanitize(_ text: String) -> String {
        text.contains("<|ts_") ? text.replacingOccurrences(of: "<|ts_", with: "<\u{200B}|ts_") : text
    }

    private static func clean(_ text: String) -> String {
        sanitize(text).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// `<|ts_opt_i|>` is followed by `name: description`, or by the name alone.
    static func optionText(_ slot: Slot) -> String {
        let name = clean(slot.name)
        guard let description = slot.description.map(clean), !description.isEmpty else { return name }
        return "\(name): \(description)"
    }

    // MARK: - Text

    /// The state part of every row on `state`: the state token, a space, the trimmed state,
    /// a newline. Structured states arrive already serialised the reference way.
    static func stateText(_ state: String) -> String {
        stateToken + stateSegment(state)
    }

    /// The question part of a row, ending at the answer token.
    static func questionText(_ row: Row) -> String {
        questionSegments(row).map(\.text).joined()
    }

    /// One piece of a row: a control token, placed by id, or text between two of them,
    /// tokenized on its own — the reference tokenizer's own segmentation.
    enum Segment: Equatable {
        case control(String)
        case text(String)

        var text: String {
            switch self {
            case .control(let token): return token
            case .text(let text): return text
            }
        }
    }

    static func stateSegment(_ state: String) -> String { " " + clean(state) + "\n" }

    static func questionSegments(_ row: Row) -> [Segment] {
        var segments: [Segment] = [
            .control(questionToken), .control(row.head), .text(" " + clean(row.question) + "\n"),
        ]
        for (index, slot) in row.options.enumerated() {
            segments.append(.control(optionToken(index)))
            segments.append(.text(" " + optionText(slot) + "\n"))
        }
        segments.append(.control(answerToken))
        return segments
    }

    // MARK: - Tokens

    /// Turns segments into token ids the way the reference tokenizer does: control tokens by
    /// id, text cut by the tokenizer's own pre-tokenizer regex with ICU's code-point
    /// semantics (see the header), each piece encoded on its own.
    struct Encoder {
        let tokenizer: any Tokenizer
        let split: NSRegularExpression?
        private let controls: [String: Int32]

        /// Fails when a control token is not one token in this tokenizer.
        init(tokenizer: any Tokenizer, layout: Layout) throws {
            self.tokenizer = tokenizer
            self.split = try layout.splitPattern.map { try NSRegularExpression(pattern: $0) }
            var controls = try SlotPrompt.controlTokenIDs(tokenizer: tokenizer)
            for index in 0..<layout.slots {
                let token = optionToken(index)
                if controls[token] == nil {
                    let ids = tokenizer.encode(text: token, addSpecialTokens: false)
                    guard ids.count == 1 else { throw DecisionError.controlTokenNotSingleToken(token: token) }
                    controls[token] = Int32(ids[0])
                }
            }
            self.controls = controls
        }

        func encode(_ segments: [Segment]) throws -> [Int32] {
            var ids: [Int32] = []
            for segment in segments {
                switch segment {
                case .control(let token):
                    guard let id = controls[token] else { throw DecisionError.controlTokenNotSingleToken(token: token) }
                    ids.append(id)
                case .text(let text):
                    ids += encode(text: text)
                }
            }
            return ids
        }

        /// The ids of one text segment: NFC (the tokenizer's normalizer, applied to the whole
        /// segment first so no cut falls inside a composed sequence), the reference cuts, BPE
        /// per piece.
        func encode(text: String) -> [Int32] {
            let normalized = text.precomposedStringWithCanonicalMapping
            guard let split else { return tokenizer.encode(text: normalized, addSpecialTokens: false).map(Int32.init) }
            return SlotPrompt.pieces(of: normalized, split: split).flatMap { piece in
                tokenizer.encode(text: piece, addSpecialTokens: false).map(Int32.init)
            }
        }
    }

    /// The pre-tokenizer's cuts of `text`, taken on UTF-16 with ICU — every match of the
    /// regex, in order. Characters no alternative matches (none, for the reference regex) are
    /// dropped, as the reference drops them.
    static func pieces(of text: String, split: NSRegularExpression) -> [String] {
        let ns = text as NSString
        return split.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
    }

    /// The tokens every row on `state` starts with.
    static func contextTokens(state: String, encoder: Encoder) throws -> [Int32] {
        try encoder.encode([.control(stateToken), .text(stateSegment(state))])
    }

    /// The prompt tokens of one question on one state, and the slot per option.
    static func render(state: String, question: Decision.Question, encoder: Encoder) throws -> DecisionPrompt.Rendered {
        let row = row(for: question)
        let tokens = try contextTokens(state: state, encoder: encoder) + encoder.encode(questionSegments(row))
        return DecisionPrompt.Rendered(tokens: tokens, slots: (0..<row.options.count).map(Int32.init))
    }

    /// The control tokens a slot-head bundle's tokenizer must know, each as one token. Checked
    /// at load so a bundle that is not what its metadata says fails there, not on a decision.
    @discardableResult
    static func controlTokenIDs(tokenizer: any Tokenizer) throws -> [String: Int32] {
        var ids: [String: Int32] = [:]
        for token in [stateToken, questionToken, choiceToken, scoreToken, noulToken, answerToken, optionToken(0), optionToken(1)] {
            let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
            guard encoded.count == 1 else { throw DecisionError.controlTokenNotSingleToken(token: token) }
            ids[token] = Int32(encoded[0])
        }
        return ids
    }

    // MARK: - Readout

    /// The author's readout over the head's logits: temperature, mask, softmax, the option
    /// mass renormalised, the abstain mass reported apart.
    static func readout(
        logits: [Double], options count: Int, temperature: Double, layout: Layout
    ) -> (probabilities: [Double], abstain: Double?) {
        var indices = Array(0..<count)
        if let abstain = layout.abstainSlot { indices.append(abstain) }
        let p = DecisionPrompt.probabilities(logits: indices.map { logits[$0] }, temperature: temperature)
        let mass = p.prefix(count).reduce(0, +)
        let options = p.prefix(count).map { $0 / max(mass, 1e-12) }
        return (Array(options), layout.abstainSlot == nil ? nil : p[count])
    }
}
