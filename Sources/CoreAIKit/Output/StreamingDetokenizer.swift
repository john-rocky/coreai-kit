// StreamingDetokenizer.swift — incremental UTF-8-safe detokenization shared by the
// kit's streaming decode loops (FM text/vision/audio/Gemma executors, guided loop).

/// Turns a token stream into text deltas without emitting a partial UTF-8 character —
/// and without stalling on one that will never complete.
///
/// Byte-fallback tokenizers decode an in-progress multi-byte character to U+FFFD until
/// its bytes are complete, so text is held back only while the decode *ends* in a
/// replacement character. A replacement character with clean text after it is
/// permanent — a stray raw-byte token no later token can repair — and is emitted
/// as-is, matching `VanillaDecodingStrategy.safeUTF8Prefix` in coreai-models. Holding
/// on *any* U+FFFD (the previous behaviour of the loops this type replaces) turned one
/// stray byte into a starved turn: every later delta was withheld and a FoundationModels
/// turn ended with "Session ended without producing a response" — deterministic on long
/// CJK output.
///
/// After each fully clean decode one token of context is retained: SentencePiece-style
/// tokenizers need a predecessor to infer leading spaces, and one token bounds the
/// re-decode to O(1) per step.
struct StreamingDetokenizer {
    private let decode: ([Int]) -> String
    private var pendingTokens: [Int] = []
    /// Prefix of the current window's decode already accounted for — emitted text,
    /// or the retained context token's standalone decode.
    private var accountedText = ""

    init(decode: @escaping ([Int]) -> String) {
        self.decode = decode
    }

    /// Feeds one token and returns the text that newly decodes cleanly — empty while
    /// the decode still ends in an in-progress character.
    mutating func consume(_ tokenId: Int) -> String {
        pendingTokens.append(tokenId)
        let decoded = decode(pendingTokens)
        guard decoded.hasSuffix("\u{FFFD}") else {
            let common = decoded.commonPrefix(with: accountedText)
            let delta = String(decoded.dropFirst(common.count))
            if let last = pendingTokens.last {
                pendingTokens = [last]
                accountedText = decode(pendingTokens)
            }
            return delta
        }
        // The trailing replacement character may still resolve; everything before it
        // is settled, so emit the clean growth and keep the window.
        let emittable = Self.droppingTrailingReplacements(decoded)
        let common = emittable.commonPrefix(with: accountedText)
        let delta = String(emittable.dropFirst(common.count))
        if !delta.isEmpty { accountedText = emittable }
        return delta
    }

    private static func droppingTrailingReplacements(_ text: String) -> String {
        var end = text.endIndex
        while end > text.startIndex {
            let previous = text.index(before: end)
            guard text[previous] == "\u{FFFD}" else { break }
            end = previous
        }
        return String(text[..<end])
    }
}
