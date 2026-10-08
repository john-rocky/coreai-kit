// From the model zoo's apps/D1Omni/Sources/D1Omni/Prompt.swift (f9e0e09, sha256 ef320aa87334), identifiers prefixed D1 for the kit.
// D1Prompt — one request -> one token row per question, as the publisher builds it (conversion/d1_omni/host.py §1: the
// copy of prompt.py — Question / as_question, serialize, _criterion, render_options, encode, temperature_key — and
// modeling_d1's per-mode dispatch, request_rows; plus the media prefix lengths of vision.py / audio.py, which set a
// media row's text room):
//
//   mode   max_len (text positions)        noul default            option text after          state None
//   text   16384                            none (criteria)         -                          ""
//   image  min(896, 16384 - P)              {false: no, true: yes}                             ""
//   audio  min(15360, 16384 - P)            {false: no, true: yes}  option_000: ...            {} (JSON)
//   (P = the media prefix length; max_len below 64 is refused)
//
//   ids     = [<bos>] + [<state>] + enc(serialize(state))[:room] + question, cut to max_len
//   question= ([<q>] + enc(instructions))[:max(16, budget)] + per option [<opt>, <mask>] + enc(" " + text)[:per]
//             + [</opt>], then [<decide>]
//   budget  = max(96, min(K * 24 + 32, max_len // 2)), per = max(2, (budget - 3K) // K),
//   room    = max(0, max_len - len(question) - 2); markers = the <mask> positions (text-relative)
//
// Every comparison of caller text is by code points (Python's), never Swift's canonical-equivalence `==`.

import Foundation

@available(macOS 27, iOS 27, *)
enum D1QuestionType: String, Sendable, CaseIterable {
    case choice, score, noul

    /// prompt.py QTYPES: the qtype_onehot column.
    var index: Int {
        switch self {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }
}

/// prompt.py `Question` after `as_question`: the type, str(instructions) and the criteria as sent (nil = None).
@available(macOS 27, iOS 27, *)
struct D1Question: Sendable {
    let type: D1QuestionType
    let instructions: String
    /// a choice's {name: description}, a score's [level descriptions], a noul's optional {true / false: ...}
    let criteria: D1JSONValue?

    /// `as_question(q)` then `Question.__post_init__`, with the publisher's messages.
    init(json q: D1JSONValue) throws {
        guard case .object = q, q.has("type"), q.has("instructions") else {
            throw D1OmniError.request("a question is a dict with `type`, `instructions` and, for choice and score, `criteria`")
        }
        let rawType = q["type"]!
        guard case .string(let t) = rawType, let type = D1QuestionType.allCases.first(where: {
            $0.rawValue.unicodeScalars.elementsEqual(t.unicodeScalars)
        }) else {
            throw D1OmniError.request("question type must be one of ['choice', 'noul', 'score'], got \(D1PythonFormat.pyRepr(rawType))")
        }
        let crit: D1JSONValue? = (q["criteria"].map { $0.isNull ? nil : $0 }) ?? nil
        switch type {
        case .choice:
            guard let m = crit?.members, m.count >= 2 else {
                throw D1OmniError.request("a choice needs criteria {name: description} with at least two options")
            }
        case .score:
            guard let a = crit?.array, (2...10).contains(a.count) else {
                throw D1OmniError.request("a score needs criteria: a list of 2 to 10 level descriptions, lowest first")
            }
        case .noul:
            if let c = crit, c.members == nil {
                throw D1OmniError.request("noul criteria are optional: {\"true\": \"...\", \"false\": \"...\"} (or \"yes\", \"no\")")
            }
        }
        self.type = type
        self.instructions = D1PythonFormat.pyStr(q["instructions"]!)
        self.criteria = crit
    }

    /// The number of options (2 for a noul).
    var options: Int {
        switch type {
        case .noul: return 2
        case .choice: return criteria?.members?.count ?? 0
        case .score: return criteria?.array?.count ?? 0
        }
    }

    /// A choice's option names in order (`list(q.criteria)`).
    var names: [String] { criteria?.members?.map(\.key) ?? [] }
}

@available(macOS 27, iOS 27, *)
enum D1Mode: String, Sendable, CaseIterable {
    case text, image, audio
}

/// The values of config.json the host reads (metadata.json's decision block).
@available(macOS 27, iOS 27, *)
struct D1Config: Sendable {
    let maxLength: Int
    let imageTextLength: Int
    let audioTextLength: Int
    let minTextPositions: Int
    /// "<type>:<2|3-5|6-10|11+>" and "<type>" -> T
    let temperatures: [String: Double]

    static let publisher = D1Config(
        maxLength: 16384, imageTextLength: 896, audioTextLength: 15360, minTextPositions: 64,
        temperatures: ["choice:11+": 1.372515082359314, "choice:2": 1.7465145587921143, "choice:3-5": 1.3998981714248657,
                       "choice:6-10": 1.1751071214675903, "noul:2": 1.6663223505020142, "score:3-5": 1.7301132678985596,
                       "score:6-10": 1.0, "choice": 1.0, "score": 1.0, "noul": 1.0])

    init(maxLength: Int, imageTextLength: Int, audioTextLength: Int, minTextPositions: Int,
                temperatures: [String: Double]) {
        self.maxLength = maxLength
        self.imageTextLength = imageTextLength
        self.audioTextLength = audioTextLength
        self.minTextPositions = minTextPositions
        self.temperatures = temperatures
    }
}

/// One question's row (host.py `Row`).
@available(macOS 27, iOS 27, *)
struct D1Row: Sendable {
    let qid: String
    let question: D1Question
    let ids: [Int]
    /// the <mask> positions, text-relative (as `encode` returns them)
    let markers: [Int]
    /// text rows take the temperature
    let calibrate: Bool
    /// P, the media prefix length (0 for text)
    let prefixLength: Int
    let mode: D1Mode
    /// the text positions this row was built against
    let maxLen: Int

    /// P + len(ids): the positions the row takes in the graph (`usage.input_tokens` sums these)
    var positions: Int { prefixLength + ids.count }
}

/// What `encode` measured on the way (encode_rows.py's record of a row).
@available(macOS 27, iOS 27, *)
struct D1EncodeDetail: Sendable {
    let stateTokens: Int
    let stateRoom: Int
    let instructionsTokens: Int
    let budget: Int
    let perOption: Int
    let optionTokens: [Int]
    let qDelimPos: Int
}

@available(macOS 27, iOS 27, *)
enum D1Prompt {
    static let yesNo: [(String, String)] = [("false", "no"), ("true", "yes")]   // modeling_d1 YES_NO

    /// prompt.py `serialize`: a str as itself, anything else json.dumps(state, ensure_ascii=False).
    static func serialize(_ state: D1JSONValue) -> String {
        if case .string(let s) = state { return s }
        return D1PythonFormat.dumps(state, asciiOnly: false)
    }

    /// prompt.py `_criterion`: a str as itself, anything else json.dumps(value, ensure_ascii=False,
    /// separators=(", ", ": ")).
    static func criterion(_ value: D1JSONValue) -> String {
        if case .string(let s) = value { return s }
        return D1PythonFormat.dumps(value, asciiOnly: false, separators: (", ", ": "))
    }

    /// Python's `v is None or v == ""` for a member value.
    static func noneOrEmpty(_ v: D1JSONValue?) -> Bool {
        guard let v else { return true }
        if case .null = v { return true }
        if case .string(let s) = v { return s.unicodeScalars.isEmpty }
        return false
    }

    /// prompt.py `render_options`: the option texts in the model's order (a noul as [false, true]).
    static func renderOptions(_ q: D1Question, noulDefault: [(String, String)]?, audio: Bool) -> [String] {
        switch q.type {
        case .choice:
            let members = q.criteria?.members ?? []
            if audio {
                return members.enumerated().map { i, m in
                    let index = String(i)
                    let padded = String(repeating: "0", count: max(0, 3 - index.count)) + index
                    return "option_\(padded): " + criterion(noneOrEmpty(m.value) ? .string(m.key) : m.value)
                }
            }
            return members.map { noneOrEmpty($0.value) ? $0.key : "\($0.key): " + criterion($0.value) }
        case .score:
            return (q.criteria?.array ?? []).enumerated().map { "level \($0.offset): " + criterion($0.element) }
        case .noul:
            if audio { return ["false: no", "true: yes"] }
            // crit = q.criteria or noul_default or {}  (an empty dict is falsy)
            let crit: D1JSONValue
            if let c = q.criteria, let m = c.members, !m.isEmpty {
                crit = c
            } else if let d = noulDefault, !d.isEmpty {
                crit = .object(d.map { D1JSONMember($0.0, .string($0.1)) })
            } else {
                crit = .object([])
            }
            // crit.get("false", crit.get("no")): a present key wins even when its value is None
            let f: D1JSONValue? = crit.has("false") ? crit["false"] : crit["no"]
            let t: D1JSONValue? = crit.has("true") ? crit["true"] : crit["yes"]
            return ["false: " + (noneOrEmpty(f) ? "no, the statement does not hold" : criterion(f!)),
                    "true: " + (noneOrEmpty(t) ? "yes, the statement holds" : criterion(t!))]
        }
    }

    /// prompt.py `encode`: the ids of one question over one state and the position of each option's marker.
    /// `stateTokens`: `enc(serialize(state))` when the caller has it (kit: the request's rows tokenize the state once;
    /// swift-transformers' BPE takes 0.4 s on a 3.4k-token state), nil to tokenize it here.
    static func encode(_ tok: D1Tokenizer, state: D1JSONValue, question q: D1Question, maxLen: Int,
                              noulDefault: [(String, String)]?, audio: Bool, perOption: Int = 24,
                              stateTokens: [Int]? = nil) throws
        -> (ids: [Int], markers: [Int], detail: D1EncodeDetail)
    {
        let d = tok.ids
        let opts = renderOptions(q, noulDefault: noulDefault, audio: audio)
        let k = opts.count
        let budget = max(96, min(k * perOption + 32, d1FloorDiv(maxLen, 2)))
        let per = max(2, d1FloorDiv(budget - 3 * k, k))
        let instructions = tok.enc(q.instructions)
        var question = Array(([d.q] + instructions).prefix(max(16, budget)))
        var markers: [Int] = []
        var optionTokens: [Int] = []
        for text in opts {
            markers.append(question.count + 1)
            let o = tok.enc(" " + text)
            optionTokens.append(o.count)
            question += [d.opt, d.mask] + o.prefix(per) + [d.optEnd]
        }
        question.append(d.decide)
        let room = max(0, maxLen - question.count - 2)
        let stateFull = stateTokens ?? tok.enc(serialize(state))
        let stateIDs = [d.state] + stateFull.prefix(room)
        let ids = Array(([d.bos] + stateIDs + question).prefix(maxLen))
        markers = markers.map { $0 + 1 + stateIDs.count }
        if let last = markers.last, last >= maxLen {
            throw D1OmniError.request("the options do not fit in the context")
        }
        return (ids, markers, D1EncodeDetail(stateTokens: stateFull.count, stateRoom: room, instructionsTokens: instructions.count,
                                           budget: budget, perOption: per, optionTokens: optionTokens,
                                           qDelimPos: 1 + stateIDs.count))
    }

    /// host.py `mode_settings`: max_len / noul default / the temperature / the audio option form for one request.
    static func modeSettings(_ mode: D1Mode, prefixLength: Int, config: D1Config = .publisher) throws
        -> (maxLen: Int, noulDefault: [(String, String)]?, calibrate: Bool, audio: Bool)
    {
        var maxLen: Int
        let noul: [(String, String)]?
        let calibrate: Bool
        let spoken: Bool
        switch mode {
        case .image:
            (maxLen, noul, calibrate, spoken) = (config.imageTextLength, yesNo, false, false)
        case .audio:
            (maxLen, noul, calibrate, spoken) = (config.audioTextLength, yesNo, false, true)
        case .text:
            if prefixLength != 0 { throw D1OmniError.request("a text request has no media prefix") }
            (maxLen, noul, calibrate, spoken) = (config.maxLength, nil, true, false)
        }
        maxLen = min(maxLen, config.maxLength - prefixLength)
        if maxLen < config.minTextPositions {
            throw D1OmniError.request("the media take \(prefixLength) of the \(config.maxLength) positions; send fewer images")
        }
        return (maxLen, noul, calibrate, spoken)
    }

    /// host.py `request_rows`: a request (its state, its questions as a {name: question} object, the media's prefix
    /// length) -> one Row per question, in order. `state` nil or null is Python's None.
    static func rows(_ tok: D1Tokenizer, state: D1JSONValue?, questions: D1JSONValue, mode: D1Mode, prefixLength: Int = 0,
                            config: D1Config = .publisher) throws -> [D1Row]
    {
        try rowsWithDetail(tok, state: state, questions: questions, mode: mode, prefixLength: prefixLength, config: config)
            .map(\.row)
    }

    static func rowsWithDetail(_ tok: D1Tokenizer, state: D1JSONValue?, questions: D1JSONValue, mode: D1Mode,
                                      prefixLength: Int = 0, config: D1Config = .publisher) throws
        -> [(row: D1Row, detail: D1EncodeDetail)]
    {
        let named: [(String, D1JSONValue)]
        if let m = questions.members {
            named = m.map { ($0.key, $0.value) }
        } else if let a = questions.array {
            named = a.map { ("", $0) }
        } else {
            throw D1OmniError.request("questions: a {name: question} object or a list")
        }
        let qs = try named.map { ($0.0, try D1Question(json: $0.1)) }
        let settings = try modeSettings(mode, prefixLength: prefixLength, config: config)
        var s: D1JSONValue = state ?? .null
        if mode == .audio, s.isNull { s = .object([]) }   // how the audio questions were trained
        if s.isNull { s = .string("") }
        // kit: every question of the request reads the same state, tokenized once here
        let stateTokens = qs.isEmpty ? [] : tok.enc(serialize(s))
        return try qs.map { name, q in
            let (ids, markers, detail) = try encode(tok, state: s, question: q, maxLen: settings.maxLen,
                                                    noulDefault: settings.noulDefault, audio: settings.audio,
                                                    stateTokens: stateTokens)
            return (D1Row(qid: name, question: q, ids: ids, markers: markers, calibrate: settings.calibrate,
                        prefixLength: prefixLength, mode: mode, maxLen: settings.maxLen), detail)
        }
    }

    /// prompt.py `temperature_key`.
    static func temperatureKey(_ q: D1Question) -> String {
        let k = q.options
        return "\(q.type.rawValue):" + (k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+")
    }

    /// host.py `temperature`: temperatures[temperature_key(q)] if present, else temperatures[q.type], else 1.0.
    static func temperature(_ q: D1Question, config: D1Config = .publisher) -> Double {
        config.temperatures[temperatureKey(q)] ?? config.temperatures[q.type.rawValue] ?? 1.0
    }
}

/// The media prefix lengths (host.py: vision.py `layout()` copied verbatim and `prefix_length()`; audio.py's
/// waveform() cut / pad and the 8x subsampling's lengths). Round 8 uses them for a media row's max_len; the media
/// graphs' inputs are round 9's.
@available(macOS 27, iOS 27, *)
enum D1MediaLength {
    static let tile = 512, patch = 16, maxPatches = 1024
    static let sampleRate = 16000, minSamples = 8000, maxSeconds = 30, hop = 160

    /// vision.py `layout`: (grid (columns, rows), thumbnail (h, w), tiled).
    static func layout(width: Int, height: Int) throws -> (grid: (Int, Int), thumbnail: (Int, Int), tiled: Bool) {
        if min(width, height) < 1 { throw D1OmniError.request("empty image") }
        let factor = 32, maximum = 256 * 1024, minimum = 64 * 1024
        func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }
        var h = max(factor, pyRound(Double(height) / Double(factor)) * factor)
        var w = max(factor, pyRound(Double(width) / Double(factor)) * factor)
        if h * w > maximum {
            let beta = (Double(height * width) / Double(maximum)).squareRoot()
            h = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if h * w < minimum {
            let beta = (Double(minimum) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            w = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        let large = max(16, pyRound(Double(height) / Double(factor)) * factor)
            * max(16, pyRound(Double(width) / Double(factor)) * factor) > maximum * 2
        var grid = (1, 1)
        if large {
            var best = Double.infinity
            for r in ratios {
                let diff = abs(Double(width) / Double(height) - Double(r.0) / Double(r.1))
                if diff < best || (diff == best && Double(width * height) > 0.5 * Double(tile * tile * r.0 * r.1)) {
                    grid = r
                    best = diff
                }
            }
        }
        return (grid, (h, w), large)
    }

    /// `sorted({(x, y) ...: 2 <= x * y <= 10}, key=lambda r: r[0] * r[1])` in CPython's order: the sort is stable over
    /// the set's iteration order (fixed by the tuples' hashes), and a tie on `diff` within one product keeps the later
    /// ratio when the area test holds, so the order is copied, not re-derived (CPython 3.11.13 and 3.12.11 agree).
    static let ratios: [(Int, Int)] = [
        (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1), (3, 2), (2, 3), (7, 1),
        (1, 7), (4, 2), (2, 4), (1, 8), (8, 1), (1, 9), (3, 3), (9, 1), (2, 5), (5, 2), (10, 1), (1, 10),
    ]

    /// host.py `image_crops`: (height, width) of every crop in order: the tiles, then the thumbnail.
    static func imageCrops(width: Int, height: Int) throws -> [(Int, Int)] {
        let plan = try layout(width: width, height: height)
        var crops: [(Int, Int)] = []
        if plan.tiled { crops = Array(repeating: (tile, tile), count: plan.grid.0 * plan.grid.1) }
        return crops + [plan.thumbnail]
    }

    /// host.py `image_prefix_length`: sum over the images' crops of (h / 16)(w / 16) / 4. `sizes` = [(width, height)].
    static func imagePrefixLength(_ sizes: [(Int, Int)]) throws -> Int {
        try sizes.reduce(0) { acc, s in
            try acc + imageCrops(width: s.0, height: s.1).reduce(0) { $0 + ($1.0 / patch) * ($1.1 / patch) / 4 }
        }
    }

    static func subsample(_ length: Int) -> Int { d1FloorDiv(length + 2 - 3, 2) + 1 }

    /// host.py `audio_prefix_length`: the valid frames of n samples (cut to 30 s, padded to 0.5 s) through the 8x
    /// subsampling.
    static func audioPrefixLength(samples n: Int) -> Int {
        let s = max(min(n, maxSeconds * sampleRate), minSamples)
        var length = d1FloorDiv(s + 512 / 2 * 2 - 512, hop)
        for _ in 0..<3 { length = subsample(length) }
        return length
    }
}
