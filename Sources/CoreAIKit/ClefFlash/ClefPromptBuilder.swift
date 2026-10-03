// From the model zoo's apps/ClefFlash/Sources/ClefFlash/PromptBuilder.swift (5ef2247, sha256 7d5fa87d532d, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// ClefPromptBuilder — a SystemOne request -> the decoder row, the head's spans and the decoder's static inputs, the
// author's `encode_record()` (conversion/clef_flash/host.py `build_ids` / `static_inputs`). Every piece is tokenized on
// its own without special tokens, and the pieces are concatenated:
//
//   prefix   "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"         36 ids
//   [image]  <|vision_start|> (248053), N x <|image_pad|> (248056), <|vision_end|> (248054), "\n" (198)   N + 3 ids
//   state    render(state)            (cut so the row fits the author's max_length 16384)
//   schema   "\n\nSCHEMA FIELDS:\n", then per question i (1-based)
//            "\nFIELD {i}\nID: {id}\nTYPE: {type}\nINSTRUCTION: " + [render(instructions or id)]
//            + "\nALLOWED OPTIONS:\n" + per option j "OPTION {j}: " + [render({"option_id", "description"})] + "\n"
//            + "END FIELD\n"                                             [..] = the head's question / option spans
//   suffix   "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"   18 ids
//
// Static inputs: the N pads go to the graph as ids V + k (V = 248320, k row-major over the merged grid),
// image_rc[k] = (k / W, k % W), rope_shift_start = 37 + N (the <|vision_end|> index), rope_shift_amount =
// N - max(H, W); a text-only row gets start 1 << 30, amount 0, image_rc zero.

import Foundation
import Tokenizers

@available(macOS 27, iOS 27, *)
struct ClefMergedGrid: Sendable, Equatable, CustomStringConvertible {
    let h: Int
    let w: Int

    init(h: Int, w: Int) {
        self.h = h
        self.w = w
    }

    var count: Int { h * w }
    var description: String { "\(h)x\(w)" }
}

/// One question's place in the row: spans are [start, end) over the processor-form ids.
@available(macOS 27, iOS 27, *)
struct ClefQuestionLayout: Sendable, Equatable {
    let questionID: String
    let type: String
    let typeID: Int
    let questionSpan: [Int]
    let optionSpans: [[Int]]
    let optionIDs: [String]
}

@available(macOS 27, iOS 27, *)
struct ClefPromptRow: Sendable {
    /// The processor-form ids (N x <|image_pad|>): what the spans index and the lexical gather reads.
    let ids: [Int]
    /// The decoder's input_ids: the image pads mapped to V + k.
    let decoderIDs: [Int32]
    let questions: [ClefQuestionLayout]
    /// Index of <|vision_start|> (36) for an image row.
    let tokenOffset: Int?
    let grid: ClefMergedGrid?
    /// [n_image_max * 2] row-major (k / W, k % W) for k < N, zero after.
    let imageRC: [Int32]
    let ropeShiftStart: Int32
    let ropeShiftAmount: Int32
    let stateTokens: Int
    let stateTokensKept: Int
}

@available(macOS 27, iOS 27, *)
struct ClefPromptBuilder: Sendable {
    static let vocab = 248_320
    static let visionStart = 248_053
    static let visionEnd = 248_054
    static let imagePad = 248_056
    static let padID = 248_044           // <|endoftext|>: fills the last decoder chunk
    static let imEnd = 248_046
    static let newline = 198
    static let noShift: Int32 = 1 << 30
    static let maxLength = 16_384         // the author's encode_record / systemone default

    static let systemPrompt = "Read the complete state and schema. Decide every field jointly. Each answer "
        + "must be exactly one of that field's allowed options."
    static let prefix = "<|im_start|>system\n\(systemPrompt)<|im_end|>\n<|im_start|>user\nSTATE:\n"
    static let suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
    static let schemaHeader = "\n\nSCHEMA FIELDS:\n"

    let tokenizer: any Tokenizer
    let nImageMax: Int
    let prefixIDs: [Int]
    let suffixIDs: [Int]
    let newlineIDs: [Int]

    /// Loads the tokenizer and checks the ids the contract names (`metadata.json` decision.prompt and the special
    /// tokens).
    init(tokenizer: any Tokenizer, nImageMax: Int, prefixTokens: Int = 36, suffixTokens: Int = 18) throws {
        self.tokenizer = tokenizer
        self.nImageMax = nImageMax
        prefixIDs = tokenizer.encode(text: Self.prefix, addSpecialTokens: false)
        suffixIDs = tokenizer.encode(text: Self.suffix, addSpecialTokens: false)
        newlineIDs = tokenizer.encode(text: "\n", addSpecialTokens: false)
        var bad: [String] = []
        if prefixIDs.count != prefixTokens { bad.append("prefix encodes to \(prefixIDs.count) ids, the contract says \(prefixTokens)") }
        if suffixIDs.count != suffixTokens { bad.append("suffix encodes to \(suffixIDs.count) ids, the contract says \(suffixTokens)") }
        if newlineIDs != [Self.newline] { bad.append("\"\\n\" encodes to \(newlineIDs), not [198]") }
        for (text, id) in [("<|endoftext|>", Self.padID), ("<|vision_start|>", Self.visionStart),
                           ("<|vision_end|>", Self.visionEnd), ("<|image_pad|>", Self.imagePad), ("<|im_end|>", Self.imEnd)] {
            let got = tokenizer.encode(text: text, addSpecialTokens: false)
            if got != [id] { bad.append("\(text) encodes to \(got), not [\(id)]") }
        }
        if prefixIDs.last.map({ $0 == Self.newline }) != true || suffixIDs.last != 25 {
            bad.append("prefix / suffix ends \(prefixIDs.suffix(2)) / \(suffixIDs.suffix(2)) (want ...198 / ...25)")
        }
        if !bad.isEmpty { throw ClefFlashError.contract("tokenizer: \(bad.joined(separator: "; "))") }
    }

    static func load(tokenizerFolder: URL, nImageMax: Int, prefixTokens: Int = 36, suffixTokens: Int = 18)
        async throws -> ClefPromptBuilder
    {
        try ClefPromptBuilder(tokenizer: try await AutoTokenizer.from(modelFolder: tokenizerFolder), nImageMax: nImageMax,
                          prefixTokens: prefixTokens, suffixTokens: suffixTokens)
    }

    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// `host.build_ids`. `grid` = the merged grid of the tower that makes the image rows, nil for a text-only row.
    func build(_ request: ClefRequest, grid: ClefMergedGrid?) throws -> ClefPromptRow {
        var schema = encode(Self.schemaHeader)
        var raw: [(ClefQuestionLayout, Int)] = []
        for (qi, q) in request.questions.enumerated() {
            schema += encode("\nFIELD \(qi + 1)\nID: \(q.id)\nTYPE: \(q.type)\nINSTRUCTION: ")
            let q0 = schema.count
            var instructions = q.instructions ?? .null
            if instructions.isNull || instructions == .string("") { instructions = .string(q.id) }
            schema += encode(ClefPythonJSON.render(instructions))
            let q1 = schema.count
            schema += encode("\nALLOWED OPTIONS:\n")
            var spans: [[Int]] = []
            var oids: [String] = []
            for (oi, o) in try q.options().enumerated() {
                schema += encode("OPTION \(oi + 1): ")
                let o0 = schema.count
                var sem = [ClefJSONMember("option_id", .string(o.id))]
                if let d = o.description { sem.append(ClefJSONMember("description", d)) }
                schema += encode(ClefPythonJSON.render(.object(sem)))
                spans.append([o0, schema.count])
                oids.append(o.id)
                schema += newlineIDs
            }
            schema += encode("END FIELD\n")
            raw.append((ClefQuestionLayout(questionID: q.id, type: q.type, typeID: ClefRequest.questionTypes[q.type]!,
                                       questionSpan: [q0, q1], optionSpans: spans, optionIDs: oids), 0))
        }
        var prefix = prefixIDs
        var tokenOffset: Int? = nil
        if let g = grid {
            guard g.count <= nImageMax else {
                throw ClefFlashError.prompt("\(g) = \(g.count) image tokens > n_image_max \(nImageMax)")
            }
            tokenOffset = prefix.count
            prefix += [Self.visionStart] + Array(repeating: Self.imagePad, count: g.count) + [Self.visionEnd] + newlineIDs
        }
        let stateAll = encode(ClefPythonJSON.render(request.state))
        let fixed = prefix.count + schema.count + suffixIDs.count
        guard fixed <= Self.maxLength else {
            throw ClefFlashError.prompt("schema requires \(fixed) tokens before state; maximum is \(Self.maxLength)")
        }
        let state = Array(stateAll.prefix(Self.maxLength - fixed))
        let shift = prefix.count + state.count
        let questions = raw.map { q, _ in
            ClefQuestionLayout(questionID: q.questionID, type: q.type, typeID: q.typeID,
                           questionSpan: q.questionSpan.map { $0 + shift },
                           optionSpans: q.optionSpans.map { $0.map { $0 + shift } }, optionIDs: q.optionIDs)
        }
        let ids = prefix + state + schema + suffixIDs
        var mapped = ids.map { Int32($0) }
        var rc = [Int32](repeating: 0, count: nImageMax * 2)
        var start = Self.noShift
        var amount: Int32 = 0
        if let g = grid, let i0 = tokenOffset {
            for k in 0..<g.count {
                mapped[i0 + 1 + k] = Int32(Self.vocab + k)
                rc[2 * k] = Int32(k / g.w)
                rc[2 * k + 1] = Int32(k % g.w)
            }
            start = Int32(i0 + 1 + g.count)
            amount = Int32(g.count - max(g.h, g.w))
        }
        return ClefPromptRow(ids: ids, decoderIDs: mapped, questions: questions, tokenOffset: tokenOffset, grid: grid,
                         imageRC: rc, ropeShiftStart: start, ropeShiftAmount: amount, stateTokens: stateAll.count,
                         stateTokensKept: state.count)
    }
}
