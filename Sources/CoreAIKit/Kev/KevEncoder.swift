// From the model zoo's apps/Kev/Sources/Kev/Encoder.swift (9e06b5a, sha256 d6d9224aee03), identifiers prefixed Kev for the kit.
// Encoder — the author's packed encoding and its rows (conversion/kev/host.py §3–4; kev/model.py `user_tokens`,
// `encode`, `admit`, `rows_of` at tag kev-1.0), on swift-transformers' tokenizer of the bundle's tokenizer/:
//
//   user_tokens(text) = tokenize(text with every <|name|> (name in [A-Za-z0-9_]+) rewritten as <¦name¦>), no
//                       special tokens ("" -> no tokens): caller text can never produce a delimiter
//   S_ids    = [<state>] + user_tokens(render(state))                                   Ls = len(S_ids)
//   branch_k = [<q>] + user_tokens(instr_k) + Σ_options ([<opt>] + user_tokens(option) + [</opt>]) + [<decide>]
//   packed   = S_ids + branch_1 + ... + branch_Q                                         usage.input_tokens
//   limits (serving, strict, nothing is cut): Ls <= 65,536 and Ls + len(branch_k) <= 73,728, else ContextOverflow
//   row_k    = S_ids + branch_k at positions 0 ..< L_k; decide = L_k - 1; opts[j] = the index of option j's </opt>
//   plan     the graph's calls (`KevGraphShape`, host.py `graph_shape` / `plan`): pieces of L ids (the call max), the
//            last padded up to a multiple of q (q = 1: no pad, a 1-id remainder folded into the piece before it); a
//            static-S bundle is L = q = S
//   graph    the padded end ceil(L_k / q) * q <= max_context_length - 1 (the position input's upper bound; q = 16:
//            L_k <= 4,080)
//   shared   the first k = floor(Ls / q) * q row tokens (whole multiples of q of state tokens) are every question's:
//            run once
//
// The delimiter ids are looked up by token text and must equal the bundle's metadata.json (`decision.row`).

import Foundation
import Tokenizers

struct KevDelimiters: Sendable, Equatable {
    let state: Int
    let q: Int
    let opt: Int
    let optEnd: Int
    let decide: Int
    let pad: Int

    static let tokens: [(name: String, token: String)] = [
        ("state", "<|fim_prefix|>"), ("q", "<|fim_middle|>"), ("opt", "<|box_start|>"), ("opt_end", "<|box_end|>"),
        ("decide", "<|fim_suffix|>"), ("pad", "<|endoftext|>"),
    ]

    init(state: Int, q: Int, opt: Int, optEnd: Int, decide: Int, pad: Int) {
        self.state = state
        self.q = q
        self.opt = opt
        self.optEnd = optEnd
        self.decide = decide
        self.pad = pad
    }

    var byName: [String: Int] {
        ["state": state, "q": q, "opt": opt, "opt_end": optEnd, "decide": decide, "pad": pad]
    }
}

/// The graph's call lengths, from the bundle's metadata.json (conversion/kev/host.py `graph_shape`): a static-S bundle
/// (`language.prefill_chunk` S) is graphMax = cap = q = qmin = S; round 14's dynamic-S bundle (`language.query_len_range`
/// [qmin, graphMax]) has cap = L = `language.query_len_call_max` (absent: graphMax) and q = `language.query_len_multiple`
/// (absent: 1).
struct KevGraphShape: Sendable, Equatable {
    let dynamic: Bool
    /// the longest call the graph takes (S of a static-S bundle)
    let graphMax: Int
    /// the longest call the host makes, L (<= graphMax; S of a static-S bundle)
    let cap: Int
    /// every call's length is a multiple of q
    let q: Int
    /// the graph's shortest call
    let qmin: Int

    init(dynamic: Bool, graphMax: Int, cap: Int, q: Int, qmin: Int) throws {
        guard qmin >= 1, qmin <= cap, cap <= graphMax, q >= 1, cap % q == 0, q == 1 || q >= qmin else {
            throw KevError.bundle("graph shape: range \(qmin)...\(graphMax), call max \(cap), multiple \(q) (want qmin <= call "
                + "max <= the range's max, the call max a multiple of q, q = 1 or >= qmin)")
        }
        self.dynamic = dynamic
        self.graphMax = graphMax
        self.cap = cap
        self.q = q
        self.qmin = qmin
    }

    /// A static-S graph: every call S ids.
    static func fixed(_ s: Int) throws -> KevGraphShape {
        try KevGraphShape(dynamic: false, graphMax: s, cap: s, q: s, qmin: s)
    }

    /// `host.plan`: an n-id run as calls -> (call length, real ids in it): pieces of cap ids, the remainder last, the last
    /// piece padded up to the next multiple of q (q = 1: no padding, a remainder below qmin takes ids from the piece
    /// before it). Only the last call holds a pad.
    func plan(_ n: Int) throws -> [(length: Int, real: Int)] {
        guard n >= 1 else { throw KevError.graphLimit("a run of \(n) ids") }
        let full = n / cap
        let rem = n % cap
        if q == 1 {
            var sizes = [Int](repeating: cap, count: full)
            if rem > 0 { sizes.append(rem) }
            if sizes[sizes.count - 1] < qmin {
                guard sizes.count > 1 else {
                    throw KevError.graphLimit("a run of \(n) ids is shorter than the graph's smallest call (\(qmin))")
                }
                sizes[sizes.count - 2] -= qmin - sizes[sizes.count - 1]
                sizes[sizes.count - 1] = qmin
            }
            return sizes.map { ($0, $0) }
        }
        var out = [(length: Int, real: Int)](repeating: (cap, cap), count: full)
        if rem > 0 { out.append(((rem + q - 1) / q * q, rem)) }
        return out
    }

    /// `host.call_lengths`: every length `plan` can produce (q, 2q, ..., cap; at q = 1 qmin ... cap).
    var callLengths: [Int] {
        q == 1 ? Array(max(qmin, 1)...cap) : Array(stride(from: q, through: cap, by: q))
    }

    /// `host.padded_end`: the last position + 1 an n-id row writes, its pad included.
    func paddedEnd(_ n: Int) throws -> Int { try plan(n).reduce(0) { $0 + $1.length } }

    /// `host.shared_prefix_plan`: k = floor(Ls / q) whole multiples of q of state tokens, 0 below the shortest call.
    func sharedPrefix(stateLength: Int) -> (k: Int, tokens: Int) {
        let k = stateLength / q
        return k * q < qmin ? (0, 0) : (k, k * q)
    }
}

/// One question's row: the state ids then the question's branch; `decide` / `opts` index `ids`.
struct KevRow: Sendable {
    let qid: String
    let type: String
    let keys: [String]
    let ids: [Int]
    let decide: Int
    let opts: [Int]
    let legend: [(String, String)]?
}

/// A request's encoding: the packed ids (the author's form), the state length and one row per question.
struct KevRows: Sendable {
    let model: String
    let packed: [Int]
    let stateLength: Int
    let rows: [KevRow]
    let meta: [KevQuestionMeta]
    var inputTokens: Int { packed.count }
}

struct KevTokenizer: Sendable {
    static let serveMaxState = 65_536                  // kev.model.SERVE_MAX_STATE (the <state> token included)
    static let serveMaxBranch = serveMaxState + 8_192  // kev.model.SERVE_MAX_BRANCH

    let tokenizer: any Tokenizer
    let delimiters: KevDelimiters

    /// The tokenizer of `folder` (tokenizer.json + tokenizer_config.json); its delimiter ids must equal `expected`
    /// (the bundle's metadata.json) when given.
    static func load(folder: URL, expected: KevDelimiters?) async throws -> KevTokenizer {
        try KevTokenizer(tokenizer: try await AutoTokenizer.from(modelFolder: folder), expected: expected)
    }

    init(tokenizer: any Tokenizer, expected: KevDelimiters?) throws {
        self.tokenizer = tokenizer
        var ids: [String: Int] = [:]
        var bad: [String] = []
        for (name, token) in KevDelimiters.tokens {
            guard let id = tokenizer.convertTokenToId(token) else {
                bad.append("\(token) is not in the tokenizer")
                continue
            }
            let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
            if encoded != [id] { bad.append("\(token) encodes to \(encoded), not [\(id)]") }
            ids[name] = id
        }
        guard bad.isEmpty, let s = ids["state"], let q = ids["q"], let o = ids["opt"], let oe = ids["opt_end"],
              let d = ids["decide"], let p = ids["pad"]
        else { throw KevError.contract("tokenizer: \(bad.joined(separator: "; "))") }
        delimiters = KevDelimiters(state: s, q: q, opt: o, optEnd: oe, decide: d, pad: p)
        if let expected, expected != delimiters {
            throw KevError.contract("tokenizer delimiter ids \(delimiters.byName) != metadata \(expected.byName)")
        }
    }

    /// `<|name|>` -> `<¦name¦>` (Python's `re.sub(r"<\|([A-Za-z0-9_]+)\|>", ...)`, on code points).
    static func rewriteDelimiterText(_ text: String) -> String {
        let s = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        func isWord(_ c: Unicode.Scalar) -> Bool {
            switch c.value {
            case 0x30...0x39, 0x41...0x5A, 0x61...0x7A, 0x5F: return true
            default: return false
            }
        }
        var i = 0
        while i < s.count {
            if s[i] == "<", i + 1 < s.count, s[i + 1] == "|" {
                var j = i + 2
                while j < s.count, isWord(s[j]) { j += 1 }
                if j > i + 2, j + 1 < s.count, s[j] == "|", s[j + 1] == ">" {
                    out.append("<")
                    out.append("\u{00A6}")
                    out.append(contentsOf: s[(i + 2)..<j])
                    out.append("\u{00A6}")
                    out.append(">")
                    i = j + 2
                    continue
                }
            }
            out.append(s[i])
            i += 1
        }
        return String(out)
    }

    /// Ids of `text` without special tokens (the plain tokenizer: `output_tokens` counts json.dumps(answers) this way).
    func plainTokens(_ text: String) -> [Int] {
        text.isEmpty ? [] : tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// `kev.model.user_tokens`.
    func userTokens(_ text: String) -> [Int] { plainTokens(Self.rewriteDelimiterText(text)) }

    /// The state's ids alone ([<state>] + user_tokens(render(state)), the first serving limit and its message): the
    /// prepared state's prefix (round 15). They equal the first `stateLength` ids of every row `rows` builds for it.
    func stateIDs(_ state: KevJSON) throws -> [Int] {
        let stateTokens = userTokens(KevText.render(state))
        if stateTokens.count + 1 > Self.serveMaxState {
            let n = stateTokens.count + 1
            throw KevError.contextOverflow("state is \(Self.grouped(n)) tokens, over the \(Self.grouped(Self.serveMaxState))"
                + "-token limit (the <state> token included): shorten the document or split it across requests")
        }
        return [delimiters.state] + stateTokens
    }

    /// `host.build_rows`: request -> record -> the packed encoding within the serving limits -> one row per question.
    func rows(_ request: KevRequest) throws -> KevRows {
        let (rec, meta) = KevText.record(request)
        let d = delimiters
        let stateTokens = userTokens(rec.state)
        if stateTokens.count + 1 > Self.serveMaxState {
            let n = stateTokens.count + 1
            throw KevError.contextOverflow("state is \(Self.grouped(n)) tokens, over the \(Self.grouped(Self.serveMaxState))"
                + "-token limit (the <state> token included): shorten the document or split it across requests")
        }
        let S = [d.state] + stateTokens
        var packed = S
        var rows: [KevRow] = []
        for (q, m) in zip(rec.questions, meta) {
            let instr = [d.q] + userTokens(q.instr)
            var branch = instr
            var ends: [Int] = []
            for o in q.options {
                branch += [d.opt] + userTokens(o) + [d.optEnd]
                ends.append(branch.count - 1)
            }
            branch.append(d.decide)
            if branch.count > Self.serveMaxBranch - S.count {
                throw KevError.contextOverflow("branch too long: \(branch.count) tokens with a \(S.count)-token state "
                    + "(row limit \(Self.serveMaxBranch))")
            }
            packed += branch
            let ids = S + branch
            rows.append(KevRow(qid: m.id, type: m.type, keys: m.keys, ids: ids, decide: ids.count - 1,
                               opts: ends.map { S.count + $0 }, legend: m.legend))
        }
        return KevRows(model: request.model, packed: packed, stateLength: S.count, rows: rows, meta: meta)
    }

    /// Python's f"{n:,}".
    static func grouped(_ n: Int) -> String {
        let s = String(n)
        var out = ""
        for (k, c) in s.enumerated() {
            if k > 0 && (s.count - k) % 3 == 0 { out += "," }
            out.append(c)
        }
        return out
    }

    /// `host.graph_context_check`: every row fits the exported graph (its padded end <= max_context_length - 1).
    static func graphCheck(_ rows: [KevRow], maxContext: Int, shape: KevGraphShape) throws {
        for r in rows {
            let padded = try shape.paddedEnd(r.ids.count)
            if padded > maxContext - 1 {
                throw KevError.graphLimit("question '\(r.qid)': a row of \(r.ids.count) tokens runs \(padded) padded "
                    + "positions, over the graph's \(maxContext - 1) (rows of at most \((maxContext - 1) / shape.q * shape.q) tokens)")
            }
        }
    }

    /// The static-S form of `graphCheck` (S = chunk).
    static func graphCheck(_ rows: [KevRow], maxContext: Int, chunk: Int) throws {
        try graphCheck(rows, maxContext: maxContext, shape: try KevGraphShape.fixed(chunk))
    }

    /// `host.shared_prefix_plan` of a static-S graph: k = Ls / S whole chunks of state tokens.
    static func sharedPrefix(stateLength: Int, chunk: Int) -> (k: Int, tokens: Int) {
        let k = stateLength / chunk
        return (k, k * chunk)
    }
}
