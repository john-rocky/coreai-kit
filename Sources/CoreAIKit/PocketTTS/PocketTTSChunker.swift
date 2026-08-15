import Foundation

/// The text front end: upstream's `prepare_text_prompt` + `split_into_best_sentences`,
/// ported statement-for-statement, plus the one thing upstream does NOT do — a hard
/// enforcement of the KV-cache invariant.
///
/// The invariant (NOTES.md §16.1–16.3): the flow-LM has no context window, so a chunk
/// occupies `voice_pos + n_text + max_gen_len` cache slots and nothing ever rolls out.
/// The graphs are exported at S_MAX = 512. Upstream's chunker overshoots its own
/// 50-token cap when a long sentence has no comma/colon to split on (measured: 121
/// tokens, cache high-water 494/512) and merely logs a warning; past the budget the
/// utterance is cut off mid-word with no EOS and no per-graph gate can see it. So this
/// port derives the real per-voice token budget from the invariant, and hard-splits on
/// whitespace as a last resort instead of trusting the sentence/clause boundaries.
struct PocketTTSChunker {
    let sp: PocketTTSSentencePiece
    let sentenceBoundaryIDs: Set<Int>
    let clauseBoundaryIDs: Set<Int>

    init(sp: PocketTTSSentencePiece) {
        self.sp = sp
        // Upstream derives the boundary ids by tokenizing ".!...?" / ",;:" and dropping
        // the first token (the dummy-prefix-attached one). Same derivation here, so the
        // ids track the tokenizer instead of being hardcoded.
        self.sentenceBoundaryIDs = Set(sp.encode(".!...?").dropFirst())
        self.clauseBoundaryIDs = Set(sp.encode(",;:").dropFirst())
    }

    // MARK: upstream ports

    /// `prepare_text_prompt` (tts_model.py:913). Returns the cleaned text and the
    /// frames-after-EOS guess (3 for <=4 words, else 1 — the caller adds 2).
    static func prepareTextPrompt(_ raw: String) throws -> (text: String, framesGuess: Int) {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw PocketTTSError.message("text prompt cannot be empty") }
        // Python: replace("\n", " ").replace("\r", " ").replace("  ", " ") — note the
        // double-space replace is a SINGLE left-to-right pass, not a fixpoint. Replicated
        // exactly (replacingOccurrences has the same non-overlapping single-pass semantics)
        // because the tokenizer sees whatever this leaves behind.
        text = text.replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
        let words = text.split(separator: " ", omittingEmptySubsequences: true).count
        let framesGuess = words <= 4 ? 3 : 1
        let first = text.first!
        if !first.isUppercase, first.isLetter {
            text = String(first).uppercased() + text.dropFirst()
        }
        if let last = text.last, last.isLetter || last.isNumber {
            text += "."
        }
        // pad_with_spaces_for_short_inputs is False for the english config; not ported.
        return (text, framesGuess)
    }

    /// `_find_boundary_indices`: positions splitting the token list after each run of
    /// boundary tokens. Always starts with 0 and ends with count.
    static func boundaryIndices(_ tokens: [Int], boundaries: Set<Int>) -> [Int] {
        var indices = [0]
        var previousWasBoundary = false
        for (idx, token) in tokens.enumerated() {
            if boundaries.contains(token) {
                previousWasBoundary = true
            } else {
                if previousWasBoundary { indices.append(idx) }
                previousWasBoundary = false
            }
        }
        indices.append(tokens.count)
        return indices
    }

    /// `_segments_from_boundaries`: decode each token span back to text.
    func segments(_ tokens: [Int], _ indices: [Int]) -> [(count: Int, text: String)] {
        var out: [(Int, String)] = []
        for i in 0..<(indices.count - 1) {
            let span = Array(tokens[indices[i]..<indices[i + 1]])
            out.append((span.count, sp.decode(span)))
        }
        return out
    }

    /// `split_into_best_sentences`, verbatim: sentence boundaries first, clause
    /// sub-splits for oversized sentences, then greedy merging up to `maxTokens`.
    func upstreamSplit(_ rawText: String, maxTokens: Int) throws -> [String] {
        let text = try Self.prepareTextPrompt(rawText).text
            .trimmingCharacters(in: .whitespaces)
        let tokens = sp.encode(text)

        let sentenceIdx = Self.boundaryIndices(tokens, boundaries: sentenceBoundaryIDs)
        let sentences = segments(tokens, sentenceIdx)

        var refined: [(count: Int, text: String)] = []
        for seg in sentences {
            if seg.count <= maxTokens {
                refined.append(seg)
            } else {
                let sub = sp.encode(seg.text.trimmingCharacters(in: .whitespaces))
                let subIdx = Self.boundaryIndices(sub, boundaries: clauseBoundaryIDs)
                let subSegs = segments(sub, subIdx)
                if subSegs.count > 1 { refined.append(contentsOf: subSegs) }
                else { refined.append(seg) }
            }
        }

        var chunks: [String] = []
        var current = ""
        var currentTokens = 0
        for (count, sentence) in refined {
            if current.isEmpty {
                current = sentence
                currentTokens = count
                continue
            }
            if currentTokens + count > maxTokens {
                chunks.append(current.trimmingCharacters(in: .whitespaces))
                current = sentence
                currentTokens = count
            } else {
                current += " " + sentence
                currentTokens += count
            }
        }
        if !current.isEmpty { chunks.append(current.trimmingCharacters(in: .whitespaces)) }
        return chunks
    }

    // MARK: the invariant

    /// Largest text-token count that keeps `voicePos + n + maxGenLen(n)` within `sMax`,
    /// additionally capped by upstream's own MAX_TOKEN_PER_CHUNK. For the shipped voices
    /// (126–162 conditioning positions) the upstream 50 cap is the binding one; the
    /// derivation is here so a longer voice can never silently break the budget.
    ///
    /// Throws rather than clamping when the voice leaves no room at all: flooring at one
    /// token would hand the pipeline a budget the invariant does not support, and the
    /// overflow `precondition` in the AR loop would then trap on chunker output that this
    /// function promised was safe. Unreachable for the shipped voices, reachable for a
    /// custom embedding dropped into `voicesDir`, which the API invites.
    static func tokenBudget(voicePos: Int, sMax: Int = PocketTTSModel.sMax) throws -> Int {
        var n = PocketTTSModel.maxTokensPerChunk
        while n > 0, voicePos + n + PocketTTSModel.maxGenLen(tokens: n) > sMax { n -= 1 }
        guard n > 0 else {
            let onePos = 1 + PocketTTSModel.maxGenLen(tokens: 1)
            throw PocketTTSError.message(
                "voice conditioning is too long: \(voicePos) positions leave no room for a "
                + "single text token within S_MAX \(sMax), which needs \(onePos) more. "
                + "The limit is \(sMax - onePos) positions; the shipped voices use 126-162.")
        }
        return n
    }

    /// The production chunker: upstream's split, then hard enforcement of the budget.
    /// Every returned chunk is guaranteed to re-tokenize to <= `budget` tokens — which is
    /// what the pipeline actually prefills, so the runtime overflow assert can never fire
    /// on chunker output.
    func chunk(_ rawText: String, voicePos: Int) throws -> [String] {
        let budget = try Self.tokenBudget(voicePos: voicePos)
        var out: [String] = []
        for c in try upstreamSplit(rawText, maxTokens: budget) {
            if sp.encode(c).count <= budget {
                out.append(c)
            } else {
                out.append(contentsOf: hardSplit(c, budget: budget))
            }
        }
        return out
    }

    /// Last-resort split of a single over-budget chunk: greedy accumulation of
    /// whitespace-separated words, re-tokenizing the candidate each time (token counts
    /// are not additive across a join, so the candidate is measured, not estimated).
    /// A single word that alone exceeds the budget is bisected by characters.
    func hardSplit(_ text: String, budget: Int) -> [String] {
        var out: [String] = []
        var current = ""
        func flush() {
            let t = current.trimmingCharacters(in: .whitespaces)
            if !t.isEmpty { out.append(t) }
            current = ""
        }
        for word in text.split(separator: " ", omittingEmptySubsequences: true) {
            let candidate = current.isEmpty ? String(word) : current + " " + word
            if sp.encode(candidate).count <= budget {
                current = candidate
            } else if current.isEmpty {
                // one pathological word over the whole budget: split it by characters
                var piece = ""
                for ch in word {
                    if sp.encode(piece + String(ch)).count > budget, !piece.isEmpty {
                        out.append(piece)
                        piece = ""
                    }
                    piece.append(ch)
                }
                if !piece.isEmpty { out.append(piece) }
            } else {
                flush()
                current = String(word)
            }
        }
        flush()
        return out
    }
}
