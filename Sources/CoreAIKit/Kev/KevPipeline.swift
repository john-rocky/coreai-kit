// From the model zoo's apps/Kev/Sources/Kev/KevDecider.swift (9e06b5a, sha256 83d1a5da86bf), identifiers prefixed Kev for the kit.
// KevPipeline — a SystemOne request -> the SystemOne response, in the order conversion/kev/decide.py runs it:
//
//   let kev = try await KevPipeline(bundle: bundleDir, asset: aimodelcURL)     // asset nil = the bundle's .aimodel (JIT)
//   let response = try await kev.decide(requestJSON: data)                     // shared: true = the shared prefix
//
//   request ──KevRequest (the author's checks)──> KevText.record (render, option texts, keys)
//           ──KevTokenizer.rows──> one row per question: the state ids + its branch, <decide> / </opt> indices
//   graph: per row the calls of KevGraphShape.plan(T) from zero states (direct), or the state's first floor(Ls / q) * q
//          tokens once and every question's rest from a copy of their states (shared) -> hidden [T, d] fp16
//   head (float64) at <decide> and every </opt> -> p (Float) -> KevAnswers.answers -> the response body
//
// Everything model-specific comes from the bundle: metadata.json (the graph's call lengths — `prefill_chunk` S, or round
// 14's `query_len_range` with round 15's `query_len_call_max` L and `query_len_multiple` q — max_context_length, the
// delimiter and pad ids, the head files and scale), head/ (weights, temperature, hidden size), tokenizer/. `callMax` /
// `multiple` at init override L and q within the graph's range (a host's choice; nil = the metadata's). An asset or a bundle that differs from the
// contract fails at load, not in a probability. `latency_ms` is the graph calls and the head (state resets and copies
// included; tokenizing and the answers not), as decide.py and the author's server time it. `warmUp()` runs every call
// length of the plan once (each length's first call in a process pays a one-time specialization).
//
// The prepared state (round 15): `prepare(state:)` runs the state's first k = floor(Ls / q) * q tokens once and keeps the
// four states (one copy: about 35 MB for Kev-0.8B, 160 MB for Kev-4B at a 4,096 context); `decide(prepared:
// questionsJSON:)` answers questions on it later, each from a copy of the kept states, only the row's tokens from k on.
// It is the shared prefix split in two (the same calls), so its hidden rows and p equal a `shared: true` request's bit
// for bit. The caller holds the prepared value; nothing is cached behind its back.

import CoreAI
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitKevDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 27, iOS 27, *)
final class KevPipeline: @unchecked Sendable {
    /// What the bundle's metadata.json says.
    struct Metadata: Sendable {
        let name: String
        /// assets.main, the `.aimodel` inside the bundle
        let asset: String
        let vocab: Int
        let maxContext: Int
        /// the graph's call lengths: `language.prefill_chunk` (static S) or `query_len_range` + `query_len_call_max` +
        /// `query_len_multiple`
        let shape: KevGraphShape
        /// the longest call the host makes (S of a static-S bundle)
        var chunk: Int { shape.cap }
        let delimiters: KevDelimiters
        let headFiles: [String]
        let headScale: Double

        /// `callMax` / `multiple`: override the metadata's `query_len_call_max` / `query_len_multiple` (a dynamic-S
        /// bundle only; the result is checked against the graph's range).
        init(bundle: URL, callMax: Int? = nil, multiple: Int? = nil) throws {
            let url = bundle.appendingPathComponent("metadata.json")
            let j = try KevJSONParser.parse(Data(contentsOf: url))
            guard let lang = j["language"], let asset = j["assets"]?["main"]?.string,
                  let vocab = lang["vocab_size"]?.intValue, let ctx = lang["max_context_length"]?.intValue,
                  let row = j["decision"]?["row"],
                  let head = j["decision"]?["head"], let files = head["files"]?.array?.compactMap(\.string),
                  let scale = head["scale"]?.double, let pad = row["pad"]?["id"]?.intValue
            else { throw KevError.bundle("\(url.path): no language / assets / decision.row / decision.head block") }
            if let range = lang["query_len_range"]?.array {   // round 14's dynamic-S graph
                guard range.count == 2, let lo = range[0].intValue, let hi = range[1].intValue else {
                    throw KevError.bundle("\(url.path): language.query_len_range is not [min, max]")
                }
                shape = try KevGraphShape(dynamic: true, graphMax: hi, cap: callMax ?? lang["query_len_call_max"]?.intValue ?? hi,
                                          q: multiple ?? lang["query_len_multiple"]?.intValue ?? 1, qmin: lo)
            } else if let s = lang["prefill_chunk"]?.intValue {
                guard callMax == nil, multiple == nil else {
                    throw KevError.bundle("callMax / multiple: a dynamic-S bundle's (query_len_range) only")
                }
                shape = try KevGraphShape.fixed(s)
            } else {
                throw KevError.bundle("\(url.path): language has neither prefill_chunk nor query_len_range")
            }
            func delim(_ n: String) throws -> Int {
                guard let i = row["delimiters"]?[n]?["id"]?.intValue else {
                    throw KevError.bundle("\(url.path): decision.row.delimiters.\(n).id missing")
                }
                return i
            }
            name = j["name"]?.string ?? bundle.lastPathComponent
            self.asset = asset
            self.vocab = vocab
            maxContext = ctx
            delimiters = KevDelimiters(state: try delim("state"), q: try delim("q"), opt: try delim("opt"),
                                       optEnd: try delim("opt_end"), decide: try delim("decide"), pad: pad)
            headFiles = files
            headScale = scale
        }
    }

    /// A state run once and kept (round 15): the request's `state` value, its ids, the plan (k multiples of q, k * q
    /// tokens run; k = 0 keeps nothing and every question runs whole), the kept rows and the four states after them.
    final class Prepared: @unchecked Sendable {
        let state: KevJSON
        let stateIDs: [Int]
        let plan: (k: Int, tokens: Int)
        /// the kept tokens' hidden rows [tokens * d] fp16
        let hidden: [Float16]
        let callSeconds: [Double]
        let callLengths: [Int]
        /// the prepare's wall seconds (state ids, calls, state copy)
        let seconds: Double
        let states: [String: NDArray]

        init(state: KevJSON, stateIDs: [Int], plan: (k: Int, tokens: Int), hidden: [Float16], callSeconds: [Double],
             callLengths: [Int], seconds: Double, states: [String: NDArray]) {
            self.state = state
            self.stateIDs = stateIDs
            self.plan = plan
            self.hidden = hidden
            self.callSeconds = callSeconds
            self.callLengths = callLengths
            self.seconds = seconds
            self.states = states
        }
    }

    /// Everything one decision did, for gates and timing.
    struct Trace: Sendable {
        let rows: KevRows
        /// "direct", "shared" or "prepared"
        let mode: String
        /// the shared prefix plan (k multiples of q, k * q tokens; k = 0 runs every row directly); nil for a direct run
        let shared: (k: Int, tokens: Int)?
        /// per row, the hidden rows [T * d] fp16 the head read (for shared: the prefix's rows, then the question's)
        let hidden: [[Float16]]
        /// per row, z before the temperature (float64)
        let logits: [[Double]]
        let probabilities: [[Float]]
        let answers: KevJSON
        /// json.dumps(answers) with Python's defaults: the text `output_tokens` counts
        let answersJSON: String
        let response: KevJSON
        /// every graph call's seconds, in order
        let callSeconds: [Double]
        /// every graph call's length (its pad included), in order
        let callLengths: [Int]
        /// state zeroing / restoring
        let resetSeconds: Double
        /// rows (request checks, record, tokens), graph (states + calls), head, answers (answers + output tokens),
        /// latency (graph + head = latency_ms), wall
        let seconds: [String: Double]
    }

    let bundle: URL
    let metadata: Metadata
    let tokenizer: KevTokenizer
    let head: KevHead
    let decoder: KevDecoder
    /// the response's `model` (the author's server echoes the request's; this host writes the bundle name)
    var modelName: String
    let loadSeconds: [String: Double]

    /// `<bundle>/../../bundles_aotc/<name>.h16c.aimodelc`, the AOT asset the Mac gates load.
    static func defaultAOT(bundle: URL, name: String) -> URL {
        bundle.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("bundles_aotc/\(name).h16c.aimodelc")
    }

    /// The `.aimodel`'s specialization here: GPU preferred with frequent reshapes, the exporter's AOT flags
    /// (`coreai-build compile --preferred-compute gpu --expect-frequent-reshapes`).
    static var jitOptions: SpecializationOptions {
        var o = SpecializationOptions(preferredComputeUnitKind: .gpu)
        o.expectFrequentReshapes = true
        return o
    }

    /// `asset`: the decoder to load, `.aimodelc` (AOT, loaded with SpecializationOptions.default) or `.aimodel`
    /// (specialized here with `jitOptions`); nil = the bundle's `.aimodel`. `options` overrides either choice.
    /// `callMax` / `multiple`: the longest call and the multiple every call length is (nil = metadata.json's).
    init(bundle: URL, asset: URL? = nil, options: SpecializationOptions? = nil, callMax: Int? = nil,
                multiple: Int? = nil) async throws {
        let t0 = ContinuousClock.now
        let meta = try Metadata(bundle: bundle, callMax: callMax, multiple: multiple)
        let tTok = ContinuousClock.now
        tokenizer = try await KevTokenizer.load(folder: bundle.appendingPathComponent("tokenizer"), expected: meta.delimiters)
        let tokS = kevSecondsSince(tTok)
        let tHead = ContinuousClock.now
        guard let hf = meta.headFiles.first else { throw KevError.bundle("metadata.json: no head files") }
        let headDir = bundle.appendingPathComponent(hf).deletingLastPathComponent()
        let names = Set(meta.headFiles.map { ($0 as NSString).lastPathComponent })
        guard let jsonName = names.first(where: { $0.hasSuffix(".json") }),
              let weightsName = names.first(where: { $0.hasSuffix(".safetensors") })
        else { throw KevError.bundle("metadata.json: head files \(meta.headFiles) (a .json and a .safetensors)") }
        head = try KevHead(directory: headDir, json: jsonName, weights: weightsName)
        guard head.scale == meta.headScale else {
            throw KevError.contract("head scale \(head.scale) != metadata \(meta.headScale)")
        }
        let headS = kevSecondsSince(tHead)
        let url = asset ?? bundle.appendingPathComponent(meta.asset)
        let opts = options ?? (url.pathExtension == "aimodelc" ? .default : Self.jitOptions)
        decoder = try await KevDecoder(contentsOf: url, shape: meta.shape, hidden: head.hiddenSize, maxContext: meta.maxContext,
                                       padID: meta.delimiters.pad, options: opts)
        self.bundle = bundle
        metadata = meta
        modelName = meta.name
        loadSeconds = ["tokenizer": tokS, "head": headS, "decoder_model": decoder.loadSeconds.model,
                       "decoder_function": decoder.loadSeconds.function, "wall": kevSecondsSince(t0)]
    }

    static func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
        let d = b - a
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    /// Every call length of the plan once from zero states (KevDecoder.warmUp) -> each length's seconds.
    func warmUp() async throws -> [(length: Int, seconds: Double)] { try await decoder.warmUp() }

    /// The response for one request (the request's JSON text).
    func decide(requestJSON: Data, shared: Bool = false) async throws -> KevJSON {
        try await trace(request: try KevRequest(data: requestJSON), shared: shared).response
    }

    /// The whole decision with its intermediate values and times.
    func trace(request: KevRequest, shared: Bool = false) async throws -> Trace {
        let t0 = ContinuousClock.now
        let rows = try tokenizer.rows(request)
        try KevTokenizer.graphCheck(rows.rows, maxContext: metadata.maxContext, shape: metadata.shape)
        let t1 = ContinuousClock.now
        var hidden: [[Float16]] = []
        var calls: [Double] = []
        var lengths: [Int] = []
        var reset = 0.0
        var plan: (k: Int, tokens: Int)? = nil
        let p = metadata.shape.sharedPrefix(stateLength: rows.stateLength)
        if shared { plan = p }
        if shared && p.k > 0 {
            let prefix = Array(rows.rows[0].ids[0..<p.tokens])
            let (pre, tails) = try await decoder.runShared(prefix: prefix, tails: rows.rows.map { Array($0.ids[p.tokens...]) })
            calls += pre.callSeconds
            lengths += pre.callLengths
            reset += pre.resetSeconds
            for t in tails {
                hidden.append(pre.hidden + t.hidden)
                calls += t.callSeconds
                lengths += t.callLengths
                reset += t.resetSeconds
            }
        } else {
            for r in rows.rows {
                let pass = try await decoder.run(ids: r.ids)
                hidden.append(pass.hidden)
                calls += pass.callSeconds
                lengths += pass.callLengths
                reset += pass.resetSeconds
            }
        }
        let t2 = ContinuousClock.now
        return answer(rows: rows, hidden: hidden, calls: calls, lengths: lengths, reset: reset, mode: shared ? "shared" : "direct",
                      plan: plan, times: (t0, t1, t2))
    }

    /// Round 15: the state run once (its first k = floor(Ls / q) * q tokens) and kept, for `decide(prepared:)` later.
    func prepare(state: KevJSON) async throws -> Prepared {
        let t0 = ContinuousClock.now
        let ids = try tokenizer.stateIDs(state)
        let p = metadata.shape.sharedPrefix(stateLength: ids.count)
        let (pass, states) = try await decoder.prepare(prefix: Array(ids[0..<p.tokens]))
        return Prepared(state: state, stateIDs: ids, plan: p, hidden: pass.hidden, callSeconds: pass.callSeconds,
                        callLengths: pass.callLengths, seconds: Self.seconds(t0, ContinuousClock.now), states: states)
    }

    /// The response to {state: the prepared state, questions} (the questions object's JSON text), on the kept states.
    func decide(prepared: Prepared, questionsJSON: Data) async throws -> KevJSON {
        try await trace(prepared: prepared, questions: try KevJSONParser.parse(questionsJSON)).response
    }

    /// The whole decision on a prepared state: per question its row's tokens from k on, from a copy of the kept states.
    /// The rows must start with the prepared state's ids; the graph limit is checked per question. `latency_ms` = these
    /// graph calls + the head (the prepare is not in it).
    func trace(prepared: Prepared, questions: KevJSON, model: String? = nil) async throws -> Trace {
        let t0 = ContinuousClock.now
        var members = [KevJSONMember("state", prepared.state), KevJSONMember("questions", questions)]
        if let model { members.insert(KevJSONMember("model", .string(model)), at: 0) }
        let rows = try tokenizer.rows(try KevRequest(json: .object(members)))
        guard rows.stateLength == prepared.stateIDs.count,
              rows.rows.allSatisfy({ Array($0.ids[0..<rows.stateLength]) == prepared.stateIDs })
        else { throw KevError.contract("the questions' rows do not start with the prepared state's ids") }
        try KevTokenizer.graphCheck(rows.rows, maxContext: metadata.maxContext, shape: metadata.shape)
        let t1 = ContinuousClock.now
        let k = prepared.plan.tokens
        let passes = try await decoder.runPrepared(states: prepared.states, from: k, tails: rows.rows.map { Array($0.ids[k...]) })
        let t2 = ContinuousClock.now
        return answer(rows: rows, hidden: passes.map { prepared.hidden + $0.hidden }, calls: passes.flatMap(\.callSeconds),
                      lengths: passes.flatMap(\.callLengths), reset: passes.reduce(0) { $0 + $1.resetSeconds }, mode: "prepared",
                      plan: prepared.plan, times: (t0, t1, t2))
    }

    /// The head, the answers and the response from a decision's hidden rows (times: start, after the rows, after the graph).
    private func answer(rows: KevRows, hidden: [[Float16]], calls: [Double], lengths: [Int], reset: Double, mode: String,
                        plan: (k: Int, tokens: Int)?, times: (ContinuousClock.Instant, ContinuousClock.Instant,
                                                              ContinuousClock.Instant)) -> Trace {
        let (t0, t1, t2) = times
        let d = head.hiddenSize
        var logits: [[Double]] = []
        var probs: [[Float]] = []
        for (r, h) in zip(rows.rows, hidden) {
            let z = head.logits(decide: h[(r.decide * d)..<((r.decide + 1) * d)],
                                options: r.opts.map { h[($0 * d)..<(($0 + 1) * d)] })
            logits.append(z)
            probs.append(head.probabilities(z))
        }
        let t3 = ContinuousClock.now
        let latency = Self.seconds(t1, t3)
        let answers = KevAnswers.answers(probs: probs, meta: rows.meta)
        let answersJSON = KevPythonFormat.dumps(answers)
        let response = KevAnswers.response(model: modelName, answers: answers, inputTokens: rows.inputTokens,
                                           outputTokens: tokenizer.plainTokens(answersJSON).count, latencyMs: latency * 1e3)
        let t4 = ContinuousClock.now
        let seconds = ["rows": Self.seconds(t0, t1), "graph": Self.seconds(t1, t2), "head": Self.seconds(t2, t3),
                       "answers": Self.seconds(t3, t4), "latency": latency, "wall": Self.seconds(t0, t4)]
        return Trace(rows: rows, mode: mode, shared: plan, hidden: hidden, logits: logits,
                     probabilities: probs, answers: answers, answersJSON: answersJSON, response: response,
                     callSeconds: calls, callLengths: lengths, resetSeconds: reset, seconds: seconds)
    }
}

#endif
