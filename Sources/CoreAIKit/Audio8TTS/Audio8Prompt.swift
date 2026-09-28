// Audio8Prompt.swift — the Audio8-TTS prompt from the tokenizer alone, packed the way the slow AR takes it.
//
// `processing_arktts.py` encodes SEGMENTS one at a time with add_special_tokens=false and concatenates the ids,
// so the BPE never merges across a segment boundary. The Swift host does the same, with the same segments
// (zoo `conversion/audio8_tts/prompt.py` is the spec and is asserted id for id against the publisher's
// processor on the 18 oracle prompts; the smoke test asserts this file against that dump).
//
//   no reference:  <|im_start|>system\n · "convert the provided text to speech" · <|im_end|>\n ·
//                  <|im_start|>user\n · text · <|im_end|>\n · <|im_start|>assistant\n<|voice|>
//   voice clone:   <|im_start|>system\n · "convert the provided text to speech reference to the following:\n\nText:\n" ·
//                  <|speaker:0|> + reference text · "\n\nSpeech:\n" · [reference codebook-0 codes as semantic ids] ·
//                  <|im_end|>\n · <|im_start|>user\n · text · <|im_end|>\n · <|im_start|>assistant\n<|voice|>
//
// The packed prompt is [11, P]: row 0 the ids above, rows 1..10 zero except under the reference codes, where they
// hold the reference's ten codebooks (row 1 = codebook 0, the code the semantic id also encodes).

import Foundation
import Tokenizers

/// A reference voice: the codec codes of a 0.5–30 s recording (`[10][T]`, codebook 0 in 0..<4096, the rest in
/// 0..<1024) and its exact transcript.
public struct Audio8Voice: Sendable, Codable {
    public var referenceText: String
    public var codes: [[Int32]]

    public init(referenceText: String, codes: [[Int32]]) {
        self.referenceText = referenceText
        self.codes = codes
    }

    /// `voice.json` as written by the zoo's `register_voice.py` / `dump_swift_ref.py`.
    public init(contentsOf url: URL) throws {
        self = try JSONDecoder().decode(Audio8Voice.self, from: Data(contentsOf: url))
    }

    var frames: Int { codes.first?.count ?? 0 }
}

struct Audio8Prompt {
    static let systemPlain = "convert the provided text to speech"
    static let systemReference = "convert the provided text to speech reference to the following:\n\nText:\n"
    static let speechTag = "\n\nSpeech:\n"

    let tokenizer: Tokenizer

    static func clean(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).joined(separator: " ")
    }

    private func encode(_ s: String) -> [Int32] {
        tokenizer.encode(text: s, addSpecialTokens: false).map { Int32($0) }
    }

    /// Row 0 ids and the packed [11, P] prompt (row-major, `codes[row * P + col]`).
    func build(text: String, voice: Audio8Voice?) throws -> (rows: [Int32], length: Int) {
        let target = Self.clean(text)
        guard !target.isEmpty else { throw Audio8Error.message("text must not be empty") }
        let prefix: [String]
        let suffix: [String]
        var refCodes: [[Int32]] = []
        if let voice {
            var ref = Self.clean(voice.referenceText)
            guard !ref.isEmpty else { throw Audio8Error.message("reference text must not be empty") }
            if !ref.contains("<|speaker:") { ref = "<|speaker:0|>" + ref }
            prefix = ["<|im_start|>system\n", Self.systemReference, ref, Self.speechTag]
            suffix = ["<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
            guard voice.codes.count == Audio8Sampling.numCodebooks, voice.frames > 0 else {
                throw Audio8Error.message("reference codes must be [10][T>0]")
            }
            refCodes = voice.codes
        } else {
            prefix = ["<|im_start|>system\n", Self.systemPlain, "<|im_end|>\n", "<|im_start|>user\n", target, "<|im_end|>\n",
                      "<|im_start|>assistant\n<|voice|>"]
            suffix = []
        }
        var pre: [Int32] = []
        for s in prefix { pre.append(contentsOf: encode(s)) }
        var suf: [Int32] = []
        for s in suffix { suf.append(contentsOf: encode(s)) }
        let T = refCodes.first?.count ?? 0
        let P = pre.count + T + suf.count
        var rows = [Int32](repeating: 0, count: (Audio8Sampling.numCodebooks + 1) * P)
        for (i, v) in pre.enumerated() { rows[i] = v }
        for t in 0..<T {
            rows[pre.count + t] = refCodes[0][t] + Audio8Sampling.semanticBegin
            for cb in 0..<Audio8Sampling.numCodebooks { rows[(cb + 1) * P + pre.count + t] = refCodes[cb][t] }
        }
        for (i, v) in suf.enumerated() { rows[pre.count + T + i] = v }
        return (rows, P)
    }
}

enum Audio8Error: Error, CustomStringConvertible {
    case message(String)
    var description: String { switch self { case .message(let m): return m } }
}
