// EncoderPrompt.swift — the input of an encoder-type decision model (`Decision.Format.encoder`):
// laya (convaiinnovations/laya), and any model built the same way.
//
// An encoder reads a whole question in one forward pass and has no answer slot to reach: each
// option is introduced by a mask token, the model gives every position a logit, and the
// options are read at their mask positions (`EncoderReadout.swift`). One row per question:
//
//   [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
//
// The row is the publisher's `build_sequence` (laya 0.3.4 `common.py`), step for step:
//
//   - the head is `choice question: …`, `score question: …` or `noul question: …`;
//   - a choice option is its label, or `label: description`; a score level is `level i: …`;
//     a noul is always `false: …` then `true: …`, with the publisher's wording when the
//     question leaves a side undescribed;
//   - each option is the mask id followed by the first 48 tokens of " " + its text;
//   - the options and the head share `head_max_len` tokens: when fewer than 16 are left for
//     the head, every option is cut to an equal share first, then the head is cut to what is
//     left (never below 8);
//   - the state fills the rest of the window, cut from its end, and a SEP closes the row;
//   - the row is cut to the window, and a question whose marker that cut drops is refused.
//
// Literal mask-token text in the instructions, the options or the state becomes a space first,
// so text cannot plant a marker. Every piece of text is tokenized on its own without special
// tokens; the special ids are placed by id. An empty piece is no tokens: the reference
// tokenizer emits nothing for it, where the Swift tokenizer's Metaspace step would emit a lone
// "▁".
//
// The state is the part every question on it shares, and what `TypedDecisions.prefill`
// tokenizes once. A structured state arrives as the text the wire codec wrote the reference
// way (`JSONValue.dumps`, Python's `json.dumps(state, ensure_ascii=False)`).
//
// Tokenizer-in, like the other renderings, so the exact rows can be checked against the
// publisher's fixture without weights (`decide-cli parity --tokens-only`).

import Foundation
import Tokenizers

/// Low level: the stable API is `TypedDecisions`, which builds its rows with this.
public struct EncoderPrompt: Sendable {
    /// Options a choice may list: the publisher's multilingual contract was checked up to 20.
    public static let maxOptions = 20

    /// What the bundle declares about the model's input and its calibration, from the
    /// `decision` block of its metadata.json (`head: "encoder"`).
    public struct Layout: Sendable, Equatable {
        /// The static sequence length the graph was exported for.
        public let window: Int
        /// Tokens the head (instructions and options, markers included) may take.
        public let headMaxLength: Int
        /// Text tokens kept per option, after its marker.
        public let optionTextTokens: Int
        public let clsTokenID: Int32
        public let sepTokenID: Int32
        public let padTokenID: Int32
        public let maskTokenID: Int32
        /// The temperatures a question is read at: the calibration the bundle ships.
        public let temperatures: EncoderReadout.Temperatures
        /// The publisher's own temperatures, the ones its reference outputs were made at.
        public let sourceTemperatures: EncoderReadout.Temperatures

        public init(
            window: Int, headMaxLength: Int, optionTextTokens: Int = 48,
            clsTokenID: Int32, sepTokenID: Int32, padTokenID: Int32, maskTokenID: Int32,
            temperatures: EncoderReadout.Temperatures = .one,
            sourceTemperatures: EncoderReadout.Temperatures = .one
        ) {
            self.window = window
            self.headMaxLength = headMaxLength
            self.optionTextTokens = optionTextTokens
            self.clsTokenID = clsTokenID
            self.sepTokenID = sepTokenID
            self.padTokenID = padTokenID
            self.maskTokenID = maskTokenID
            self.temperatures = temperatures
            self.sourceTemperatures = sourceTemperatures
        }

        /// The same layout at another window (the model's other static shape).
        public func windowed(_ window: Int) -> Layout {
            Layout(
                window: window, headMaxLength: headMaxLength, optionTextTokens: optionTextTokens,
                clsTokenID: clsTokenID, sepTokenID: sepTokenID, padTokenID: padTokenID, maskTokenID: maskTokenID,
                temperatures: temperatures, sourceTemperatures: sourceTemperatures)
        }

        /// The layout a bundle directory declares, or nil when its metadata.json has no
        /// `decision` block naming an encoder head (a language bundle, or no metadata at all).
        public static func read(bundleAt url: URL) throws -> Layout? {
            let file = url.appendingPathComponent("metadata.json")
            guard FileManager.default.fileExists(atPath: file.path) else { return nil }
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
            guard let block = root?["decision"] as? [String: Any], (block["head"] as? String) == "encoder" else {
                return nil
            }
            return try Layout(block: block, bundle: url.lastPathComponent)
        }

        init(block: [String: Any], bundle: String) throws {
            func refuse(_ reason: String) -> DecisionError {
                .unsupportedModel(id: bundle, reason: "its metadata.json declares an encoder head " + reason)
            }
            func integer(_ key: String) -> Int? {
                if let value = block[key] as? Int { return value }
                if let value = block[key] as? Double, value == value.rounded() { return Int(value) }
                return nil
            }
            func number(_ value: Any?) -> Double? {
                if let value = value as? Double { return value }
                if let value = value as? Int { return Double(value) }
                return nil
            }
            func temperatures(_ typeKey: String, _ bucketKey: String?) throws -> EncoderReadout.Temperatures? {
                guard let raw = block[typeKey] else { return nil }
                guard let list = raw as? [Any], list.count == EncoderReadout.questionTypes.count else {
                    throw refuse("whose '\(typeKey)' is not one temperature per question type")
                }
                let byType = list.compactMap(number)
                var byOptions: [String: Double] = [:]
                if let bucketKey, let buckets = block[bucketKey] as? [String: Any] {
                    for (bucket, value) in buckets {
                        guard let t = number(value) else { throw refuse("whose '\(bucketKey)' holds a non-number") }
                        byOptions[bucket] = t
                    }
                }
                guard byType.count == list.count, (byType + byOptions.values).allSatisfy({ $0.isFinite && $0 > 0 }) else {
                    throw refuse("whose '\(typeKey)' holds a temperature that is not a positive number")
                }
                return EncoderReadout.Temperatures(byType: byType, byOptions: byOptions)
            }
            guard let window = integer("window"), window > 0 else { throw refuse("without a usable 'window'") }
            guard let head = integer("head_max_len"), head > 0 else { throw refuse("without a usable 'head_max_len'") }
            let optionTokens = integer("option_text_tokens") ?? 48
            guard optionTokens > 0 else { throw refuse("with 'option_text_tokens' \(optionTokens)") }
            var ids: [Int32] = []
            for key in ["cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id"] {
                guard let id = integer(key), id >= 0 else { throw refuse("without a usable '\(key)'") }
                ids.append(Int32(id))
            }
            let source = try temperatures("source_temperature", nil) ?? .one
            self.init(
                window: window, headMaxLength: head, optionTextTokens: optionTokens,
                clsTokenID: ids[0], sepTokenID: ids[1], padTokenID: ids[2], maskTokenID: ids[3],
                temperatures: try temperatures("temperature", "temperature_by_options") ?? source,
                sourceTemperatures: source)
        }
    }

    /// One question on one state, as the model reads it.
    public struct Rendered: Sendable, Equatable {
        /// The row, unpadded: at most `window` tokens.
        public let tokens: [Int32]
        /// The zero-based position of each option's marker, in option order.
        public let markers: [Int32]
        /// The question's type, in `EncoderReadout.questionTypes` order.
        public let qtype: Int
        /// State tokens the row kept.
        public let stateTokens: Int
    }

    public let layout: Layout
    let tokenizer: any Tokenizer
    /// The mask token's text, replaced by a space wherever the instructions, an option or the
    /// state spells it.
    let maskText: String

    /// A builder over the tokenizer in `tokenizerFolder` (tokenizer.json, tokenizer_config.json).
    public init(tokenizerFolder: URL, layout: Layout) async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        try self.init(tokenizer: tokenizer, layout: layout)
    }

    /// A builder for a model whose metadata is not at hand: the window and the head budget as
    /// given, the special tokens as the tokenizer's own config names them (`cls_token`,
    /// `sep_token`, `pad_token`, `mask_token`) — the ids the reference builder takes from its
    /// tokenizer — and temperature 1.
    public init(tokenizerFolder: URL, window: Int, headMaxLength: Int, optionTextTokens: Int = 48) async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        let config = try JSONSerialization.jsonObject(
            with: Data(contentsOf: tokenizerFolder.appendingPathComponent("tokenizer_config.json"))) as? [String: Any]
        var ids: [Int32] = []
        for key in ["cls_token", "sep_token", "pad_token", "mask_token"] {
            let value = config?[key]
            let text = value as? String ?? (value as? [String: Any])?["content"] as? String
            guard let text, let id = tokenizer.convertTokenToId(text) else {
                throw DecisionError.unsupportedModel(
                    id: tokenizerFolder.lastPathComponent, reason: "its tokenizer config names no usable '\(key)'")
            }
            ids.append(Int32(id))
        }
        try self.init(
            tokenizer: tokenizer,
            layout: Layout(
                window: window, headMaxLength: headMaxLength, optionTextTokens: optionTextTokens,
                clsTokenID: ids[0], sepTokenID: ids[1], padTokenID: ids[2], maskTokenID: ids[3]))
    }

    init(tokenizer: any Tokenizer, layout: Layout) throws {
        guard let mask = tokenizer.convertIdToToken(Int(layout.maskTokenID)) else {
            throw DecisionError.unsupportedModel(
                id: "encoder", reason: "its tokenizer has no token for mask id \(layout.maskTokenID)")
        }
        self.tokenizer = tokenizer
        self.layout = layout
        self.maskText = mask
    }

    private init(tokenizer: any Tokenizer, layout: Layout, maskText: String) {
        self.tokenizer = tokenizer
        self.layout = layout
        self.maskText = maskText
    }

    /// The same builder at another window: the model's other static shape, the same tokenizer.
    public func windowed(_ window: Int) -> EncoderPrompt {
        EncoderPrompt(tokenizer: tokenizer, layout: layout.windowed(window), maskText: maskText)
    }

    // MARK: - Rows

    /// The state's tokens: the part every question on it shares.
    public func contextTokens(state: String) -> [Int32] {
        encode(Self.unmasked(state, maskText))
    }

    /// The row of one question on one state, and the position of each option's marker.
    public func render(state: String, question: Decision.Question) throws -> Rendered {
        try render(stateTokens: contextTokens(state: state), question: question)
    }

    /// The row over state tokens already made by `contextTokens(state:)`.
    func render(stateTokens: [Int32], question: Decision.Question) throws -> Rendered {
        try render(stateTokens: stateTokens, part: questionPart(question))
    }

    /// The row over state tokens and a question part already made (`questionPart(_:)`).
    func render(stateTokens: [Int32], part: QuestionPart) throws -> Rendered {
        try Self.assemble(part, stateTokens: stateTokens, layout: layout)
    }

    /// The part of every row of `question` that no state changes.
    func questionPart(_ question: Decision.Question) -> QuestionPart {
        Self.questionPart(question, layout: layout, maskText: maskText, encode: encode)
    }

    /// One piece of text, without special tokens; nothing for an empty piece (see the header).
    func encode(_ text: String) -> [Int32] {
        guard !text.isEmpty else { return [] }
        return tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    // MARK: - The publisher's build_sequence

    /// Every occurrence of the mask token's text replaced by one space, code unit for code unit
    /// (Python's `str.replace`: no grapheme or canonical matching).
    static func unmasked(_ text: String, _ maskText: String) -> String {
        text.replacingOccurrences(of: maskText, with: " ", options: .literal)
    }

    /// The part of a row that depends on the question alone: `[CLS] head [SEP] options [SEP]` after
    /// the head budget and the squeeze, and where each option's marker sits in it. A state only
    /// fills the room the window leaves after it, so the part is the same on every state.
    struct QuestionPart: Sendable, Equatable {
        let ids: [Int32]
        let markers: [Int32]
        let qtype: Int
    }

    /// The row of `question` over `stateTokens`, each text piece encoded by `encode`.
    static func build(
        stateTokens: [Int32], question: Decision.Question, layout: Layout, maskText: String,
        encode: (String) -> [Int32]
    ) throws -> Rendered {
        try assemble(
            questionPart(question, layout: layout, maskText: maskText, encode: encode), stateTokens: stateTokens,
            layout: layout)
    }

    static func questionPart(
        _ question: Decision.Question, layout: Layout, maskText: String, encode: (String) -> [Int32]
    ) -> QuestionPart {
        let qtype = EncoderReadout.qtype(of: question.kind)
        let (head, options) = texts(of: question, maskText: maskText)
        let headIDs = encode(head)
        var optionRows = options.map { [layout.maskTokenID] + encode($0).prefix(layout.optionTextTokens) }
        var budget = layout.headMaxLength - optionRows.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let share = max(4, (layout.headMaxLength - 16) / max(1, optionRows.count))
            optionRows = optionRows.map { Array($0.prefix(share)) }
            budget = layout.headMaxLength - optionRows.reduce(0) { $0 + $1.count }
        }
        var ids: [Int32] = [layout.clsTokenID] + headIDs.prefix(max(8, budget)) + [layout.sepTokenID]
        var markers: [Int32] = []
        for row in optionRows {
            markers.append(Int32(ids.count))
            ids += row
        }
        ids.append(layout.sepTokenID)
        return QuestionPart(ids: ids, markers: markers, qtype: qtype)
    }

    /// A row: the question's part, as much of the state as the window leaves room for (cut from its
    /// end), a closing SEP, the whole cut to the window.
    static func assemble(_ part: QuestionPart, stateTokens: [Int32], layout: Layout) throws -> Rendered {
        let room = max(0, layout.window - part.ids.count - 1)
        let state = stateTokens.prefix(room)
        let ids = part.ids + state + [layout.sepTokenID]
        let kept = part.markers.filter { $0 < layout.window }
        // The publisher refuses a question whose options no longer fit (`Agent.system_one`).
        guard kept.count == part.markers.count else {
            throw DecisionError.promptTooLong(tokens: part.ids.count + 1, max: layout.window)
        }
        return Rendered(tokens: Array(ids.prefix(layout.window)), markers: kept, qtype: part.qtype, stateTokens: state.count)
    }

    /// What the tokenizer reads of a question: the head (`<type> question: <instructions>`) and each
    /// option as " " + its text, mask-token text already turned into spaces.
    static func texts(of question: Decision.Question, maskText: String) -> (head: String, options: [String]) {
        let qtype = EncoderReadout.qtype(of: question.kind)
        return (
            EncoderReadout.questionTypes[qtype] + " question: " + unmasked(question.instructions, maskText),
            optionTexts(for: question).map { " " + unmasked($0, maskText) }
        )
    }

    /// The question parts of the last `capacity` questions, the least recently asked dropped first:
    /// a gate asks the same few questions of every new state, and tokenizing the head and every
    /// option again is most of a row's host time. The key is what the tokenizer reads of the
    /// question (`texts(of:maskText:)`), byte for byte: `Decision.Question`'s own `==` also matches
    /// canonically equivalent text, which this tokenizer (no Unicode normalizer) cuts differently.
    struct QuestionCache: Sendable {
        let capacity: Int
        private var entries: [(key: [UInt8], part: QuestionPart)] = []

        init(capacity: Int) {
            self.capacity = capacity
        }

        var count: Int { entries.count }

        /// The cached part of `question`, or `make()`'s, remembered.
        mutating func part(
            for question: Decision.Question, maskText: String, make: () -> QuestionPart
        ) -> (part: QuestionPart, hit: Bool) {
            let key = Self.key(question, maskText: maskText)
            if let index = entries.firstIndex(where: { $0.key == key }) {
                let entry = entries.remove(at: index)
                entries.append(entry)
                return (entry.part, true)
            }
            let part = make()
            guard capacity > 0 else { return (part, false) }
            entries.append((key, part))
            if entries.count > capacity { entries.removeFirst(entries.count - capacity) }
            return (part, false)
        }

        /// The head and the option texts, each length-prefixed so no two questions share a key.
        static func key(_ question: Decision.Question, maskText: String) -> [UInt8] {
            let (head, options) = EncoderPrompt.texts(of: question, maskText: maskText)
            var bytes: [UInt8] = []
            for text in [head] + options {
                let utf8 = Array(text.utf8)
                withUnsafeBytes(of: UInt32(utf8.count).littleEndian) { bytes += $0 }
                bytes += utf8
            }
            return bytes
        }
    }

    /// The option texts the model reads, in answer order (`render_options`).
    static func optionTexts(for question: Decision.Question) -> [String] {
        switch question.kind {
        case .choice(let options):
            return options.map(optionText)
        case .score(let levels):
            return levels.enumerated().map { "level \($0.offset): \($0.element)" }
        case .noul(let yes, let no):
            return [
                "false: " + (described(no) ?? "no, the statement does not hold"),
                "true: " + (described(yes) ?? "yes, the statement holds"),
            ]
        }
    }

    /// A choice option: its label alone when it has no description, else `label: description`.
    /// An `Option` made from one string has none; one the wire codec made already carries
    /// `label: description` as its description. Compared scalar for scalar: String's own `==`
    /// and `hasPrefix` match canonically equivalent text and whole grapheme clusters, so a
    /// description opening with a combining mark ("x: \u{301}…") would miss its own prefix.
    static func optionText(_ option: Decision.Option) -> String {
        let description = option.description.unicodeScalars
        if description.isEmpty || description.elementsEqual(option.id.unicodeScalars) { return option.id }
        if description.starts(with: (option.id + ": ").unicodeScalars) { return option.description }
        return option.id + ": " + option.description
    }

    private static func described(_ text: String?) -> String? {
        guard let text, !text.isEmpty else { return nil }
        return text
    }
}
