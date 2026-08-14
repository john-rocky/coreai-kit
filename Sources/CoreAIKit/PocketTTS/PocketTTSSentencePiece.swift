import Foundation

/// Native SentencePiece **unigram** encoder/decoder, driven straight off the shipped
/// `tokenizer.model` protobuf. This is the encode side the port needs (parakeet-swift
/// only ever decodes, and from a `tokenizer.json` — no prior art to reuse for encoding).
///
/// Scope is deliberately exact-for-this-model rather than general:
///   * The model's normalizer is **`identity`** with an empty precompiled charsmap
///     (verified against the shipped file; asserted at load). Normalization is therefore
///     only: optional whitespace cleanup (off for this model), the dummy-prefix space,
///     and escaping U+0020 to "▁" (U+2581).
///   * Unigram Viterbi over the normalized text, matching pieces via a byte trie.
///     Mirrors `sentencepiece`'s `PocketTTSModel::Encode`: at each codepoint boundary all trie
///     matches become lattice nodes; if none of them covers exactly one codepoint, an
///     `<unk>` node covering one codepoint is added with `min_score - 10.0`.
///   * `byte_fallback` is on in this model: an `<unk>` piece in the best path is emitted
///     as its UTF-8 bytes' `<0xXX>` piece ids, exactly like upstream.
///
/// Scores accumulate in Float, matching the C++ (`float score`) so tie behaviour cannot
/// drift. Verified against the Python `sentencepiece` binding on a punctuation / number /
/// apostrophe / unicode corpus before anything downstream was wired (see NOTES.md §18).
struct PocketTTSSentencePiece {
    enum PieceType: Int { case normal = 1, unknown = 2, control = 3, userDefined = 4, unused = 5, byte = 6 }

    struct Piece {
        let text: String
        let score: Float
        let type: PieceType
    }

    let pieces: [Piece]
    let unkID: Int
    let addDummyPrefix: Bool
    let removeExtraWhitespaces: Bool
    let escapeWhitespaces: Bool
    private let unkScore: Float
    private let byteToID: [Int]          // 256 entries, -1 when absent
    private let trie: PocketTTSByteTrie

    var vocabularySize: Int { pieces.count }

    // MARK: protobuf load

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let root = try PocketTTSProtoReader(Array(data))

        var pieces: [Piece] = []
        var unkID = 0
        var addDummyPrefix = true
        var removeExtraWhitespaces = true
        var escapeWhitespaces = true
        var charsmapBytes = 0
        var normalizerName = ""

        while let f = try root.next() {
            switch f.number {
            case 1:   // repeated SentencePiece
                let p = try PocketTTSProtoReader(f.bytes())
                var text = "", score: Float = 0, type = 1
                while let pf = try p.next() {
                    switch pf.number {
                    case 1: text = String(decoding: pf.bytes(), as: UTF8.self)
                    case 2: score = pf.float()
                    case 3: type = Int(pf.varint())
                    default: break
                    }
                }
                pieces.append(Piece(text: text, score: score, type: PieceType(rawValue: type) ?? .normal))
            case 2:   // TrainerSpec
                let t = try PocketTTSProtoReader(f.bytes())
                while let tf = try t.next() {
                    if tf.number == 40 { unkID = Int(tf.varint()) }
                }
            case 3:   // NormalizerSpec
                let n = try PocketTTSProtoReader(f.bytes())
                while let nf = try n.next() {
                    switch nf.number {
                    case 1: normalizerName = String(decoding: nf.bytes(), as: UTF8.self)
                    case 2: charsmapBytes = nf.bytes().count
                    case 3: addDummyPrefix = nf.varint() != 0
                    case 4: removeExtraWhitespaces = nf.varint() != 0
                    case 5: escapeWhitespaces = nf.varint() != 0
                    default: break
                    }
                }
            default:
                break
            }
        }

        // This implementation only handles the identity normalizer. A charsmap would mean
        // NFKC-style rewriting that we have deliberately not implemented — fail loudly
        // rather than tokenize subtly differently from the Python reference.
        guard charsmapBytes == 0 else {
            throw PocketTTSError.message(
                "tokenizer.model uses normalizer '\(normalizerName)' with a \(charsmapBytes)-byte "
                + "charsmap; only the identity normalizer is supported")
        }

        self.pieces = pieces
        self.unkID = unkID
        self.addDummyPrefix = addDummyPrefix
        self.removeExtraWhitespaces = removeExtraWhitespaces
        self.escapeWhitespaces = escapeWhitespaces

        var minScore = Float.greatestFiniteMagnitude
        var byteMap = [Int](repeating: -1, count: 256)
        let trie = PocketTTSByteTrie()
        for (id, p) in pieces.enumerated() {
            switch p.type {
            case .normal, .userDefined:
                minScore = min(minScore, p.score)
                trie.insert(Array(p.text.utf8), id: id)
            case .byte:
                // "<0xAB>" -> 0xAB
                if p.text.count == 6, p.text.hasPrefix("<0x"), p.text.hasSuffix(">") {
                    let hex = p.text.dropFirst(3).dropLast()
                    if let v = Int(hex, radix: 16), v >= 0, v < 256 { byteMap[v] = id }
                }
            default:
                break
            }
        }
        self.unkScore = minScore - 10.0   // kUnkPenalty, sentencepiece unigram default
        self.byteToID = byteMap
        self.trie = trie
    }

    // MARK: encode

    func normalize(_ text: String) -> [UInt8] {
        var t = text
        if removeExtraWhitespaces {          // off for this model, implemented for completeness
            while t.hasPrefix(" ") { t.removeFirst() }
            while t.hasSuffix(" ") { t.removeLast() }
            while t.contains("  ") { t = t.replacingOccurrences(of: "  ", with: " ") }
        }
        if addDummyPrefix { t = " " + t }
        if escapeWhitespaces { t = t.replacingOccurrences(of: " ", with: "\u{2581}") }
        return Array(t.utf8)
    }

    /// Text → token ids, `sp.encode(text, out_type=int)` semantics.
    func encode(_ text: String) -> [Int] {
        let bytes = normalize(text)
        let n = bytes.count
        if n == 0 { return [] }

        // Codepoint step table: cpLen[i] = UTF-8 length of the codepoint starting at i
        // (only queried at codepoint boundaries).
        var cpLen = [Int](repeating: 1, count: n)
        var i = 0
        while i < n {
            let b = bytes[i]
            let len = b < 0x80 ? 1 : (b < 0xE0 ? 2 : (b < 0xF0 ? 3 : 4))
            cpLen[i] = min(len, n - i)
            i += cpLen[i]
        }

        // Viterbi. dp in Float to match sentencepiece's float lattice scores.
        let negInf = -Float.greatestFiniteMagnitude
        var dp = [Float](repeating: negInf, count: n + 1)
        var backStart = [Int](repeating: -1, count: n + 1)
        var backPiece = [Int](repeating: -1, count: n + 1)   // -1 = unreached, -2 = unk
        dp[0] = 0

        var pos = 0
        while pos < n {
            if dp[pos] != negInf {
                let oneCp = pos + cpLen[pos]
                var sawSingleCp = false
                trie.match(bytes, from: pos) { end, id in
                    if end == oneCp { sawSingleCp = true }
                    let s = dp[pos] + pieces[id].score
                    if s > dp[end] {
                        dp[end] = s; backStart[end] = pos; backPiece[end] = id
                    }
                }
                if !sawSingleCp {
                    let s = dp[pos] + unkScore
                    if s > dp[oneCp] {
                        dp[oneCp] = s; backStart[oneCp] = pos; backPiece[oneCp] = -2
                    }
                }
            }
            pos += cpLen[pos]
        }

        // Walk back, expanding unk pieces to byte-fallback ids.
        var revs: [[Int]] = []
        var at = n
        while at > 0 {
            let start = backStart[at]
            let pieceID = backPiece[at]
            precondition(start >= 0, "sentencepiece: unreachable lattice position \(at)")
            if pieceID == -2 {
                var ids: [Int] = []
                for b in bytes[start..<at] {
                    let bid = byteToID[Int(b)]
                    ids.append(bid >= 0 ? bid : unkID)
                }
                revs.append(ids)
            } else {
                revs.append([pieceID])
            }
            at = start
        }
        return revs.reversed().flatMap { $0 }
    }

    // MARK: decode

    /// Token ids → text, `sp.decode(ids)` semantics: control pieces skipped, byte pieces
    /// merged back into UTF-8, `<unk>` becomes the default `unk_surface` " ⁇ ", "▁" becomes
    /// a space, and the dummy-prefix leading space is dropped.
    func decode(_ ids: [Int]) -> String {
        var out = ""
        var byteRun: [UInt8] = []
        func flushBytes() {
            if !byteRun.isEmpty { out += String(decoding: byteRun, as: UTF8.self); byteRun = [] }
        }
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            let p = pieces[id]
            switch p.type {
            case .control, .unused:
                flushBytes()
            case .unknown:
                flushBytes()
                out += " \u{2047} "    // default unk_surface
            case .byte:
                if p.text.count == 6, let v = Int(p.text.dropFirst(3).dropLast(), radix: 16) {
                    byteRun.append(UInt8(v))
                }
            case .normal, .userDefined:
                flushBytes()
                out += p.text
            }
        }
        flushBytes()
        out = out.replacingOccurrences(of: "\u{2581}", with: " ")
        if addDummyPrefix, out.hasPrefix(" ") { out.removeFirst() }
        return out
    }
}

// MARK: - byte trie

/// A plain byte-wise trie over the vocabulary. 3740 pieces — dictionary children are
/// plenty fast for a per-chunk tokenize (the AR loop dwarfs this by orders of magnitude).
final class PocketTTSByteTrie {
    private struct Node {
        var children: [UInt8: Int] = [:]
        var pieceID: Int = -1
    }
    private var nodes: [Node] = [Node()]

    func insert(_ bytes: [UInt8], id: Int) {
        var at = 0
        for b in bytes {
            if let next = nodes[at].children[b] {
                at = next
            } else {
                nodes.append(Node())
                nodes[at].children[b] = nodes.count - 1
                at = nodes.count - 1
            }
        }
        nodes[at].pieceID = id
    }

    /// Invoke `found(endIndex, pieceID)` for every piece matching `bytes[from...]`.
    func match(_ bytes: [UInt8], from: Int, _ found: (Int, Int) -> Void) {
        var at = 0
        var i = from
        while i < bytes.count, let next = nodes[at].children[bytes[i]] {
            at = next
            i += 1
            if nodes[at].pieceID >= 0 { found(i, nodes[at].pieceID) }
        }
    }
}

// MARK: - protobuf wire reader

/// Just enough protobuf: varint, fixed32, and length-delimited fields, iterated in order.
final class PocketTTSProtoReader {
    struct Field {
        let number: Int
        let wire: Int
        let intValue: UInt64
        let byteValue: ArraySlice<UInt8>

        func varint() -> UInt64 { intValue }
        func bytes() -> ArraySlice<UInt8> { byteValue }
        func float() -> Float {
            var v: UInt32 = 0
            for (i, b) in byteValue.enumerated() { v |= UInt32(b) << (8 * i) }
            return Float(bitPattern: v)
        }
    }

    private let buf: ArraySlice<UInt8>
    private var i: Int

    init(_ bytes: [UInt8]) throws { self.buf = bytes[...]; self.i = buf.startIndex }
    init(_ bytes: ArraySlice<UInt8>) throws { self.buf = bytes; self.i = bytes.startIndex }

    private func varint() throws -> UInt64 {
        var r: UInt64 = 0, s: UInt64 = 0
        while true {
            guard i < buf.endIndex else { throw PocketTTSError.message("protobuf: truncated varint") }
            let b = buf[i]; i += 1
            r |= UInt64(b & 0x7F) << s
            if b & 0x80 == 0 { return r }
            s += 7
        }
    }

    func next() throws -> Field? {
        guard i < buf.endIndex else { return nil }
        let tag = try varint()
        let number = Int(tag >> 3), wire = Int(tag & 7)
        switch wire {
        case 0:
            return Field(number: number, wire: wire, intValue: try varint(), byteValue: [])
        case 1, 5:
            let len = wire == 1 ? 8 : 4
            guard i + len <= buf.endIndex else { throw PocketTTSError.message("protobuf: truncated fixed") }
            defer { i += len }
            return Field(number: number, wire: wire, intValue: 0, byteValue: buf[i..<(i + len)])
        case 2:
            let len = Int(try varint())
            guard i + len <= buf.endIndex else { throw PocketTTSError.message("protobuf: truncated bytes") }
            defer { i += len }
            return Field(number: number, wire: wire, intValue: 0, byteValue: buf[i..<(i + len)])
        default:
            throw PocketTTSError.message("protobuf: unsupported wire type \(wire)")
        }
    }
}
