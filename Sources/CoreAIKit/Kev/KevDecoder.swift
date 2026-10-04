// From the model zoo's apps/Kev/Sources/Kev/Decoder.swift (9e06b5a, sha256 013d4261faea), identifiers prefixed Kev for the kit.
// Decoder — the Kev backbone on the low-level runtime (AIModel + loadFunction + MutableViews states): one function
// `main` that returns the final-norm hidden state at every position of a call (no vocabulary head in the graph). The
// loop is apps/ClefFlash's ClefDecoder (zoo main 5ef2247) without the image inputs, plus the shared prefix and the
// dynamic query length of round 14's graph.
//
//   inputs   input_ids [1, S] i32 (static S) or [1, -1] (dynamic: any length of the plan), position_ids [1, -1] i32
//            (0 ..< p + c for a call of c ids after p)
//   states   keyCache / valueCache [n_full, 1, n_kv, -1, head] f16 (the dynamic length resolved to
//            max_context_length), convState [n_lin, 1, conv, 3] f16, recState [n_lin, 1, n_v, 128, 128] f16 — zero
//            at the start of every row
//   output   hidden [1, S, d] f16 (static) or [1, -1, d] (dynamic: [1, c, d] for a call of c ids)
//
// Read order (the bundle's `decision.readout`, conversion/kev/host.py `plan`): a row of T ids runs from zero states in
// the calls of `KevGraphShape.plan(T)`: pieces of cap ids, the last one padded with <|endoftext|> up to a multiple of q
// (a static-S bundle is cap = q = S: ceil(T / S) calls of S); the padded positions' rows are dropped (causal: they cannot
// reach a real position). The shape, d, max_context_length and the pad id come from the bundle (metadata.json,
// kev_head.json); the descriptor of `main` is checked against them at load.
//
// The shared prefix: the first k = floor(Ls / q) * q tokens are state tokens for every question of a request. They run
// once from zero states; the four states are then copied whole into a snapshot, and every question runs the rest of
// its row from a copy of it, positions continuing at k. On a static-S bundle the calls and their inputs are a direct
// run's, so are the hidden rows; on a dynamic one the calls are cut at other places (other last bits).
//
// Every call length is specialized by the runtime on its first call in a process; `warmUp` runs each length of the plan
// once from zero states so that no request pays it.

import CoreAI
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitKevDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 27, iOS 27, *)
final class KevDecoder: @unchecked Sendable {
    /// One row's pass (or a question's part after the shared prefix): the hidden rows and each call's time.
    struct Pass: Sendable {
        /// [tokens * d] fp16, row-major: the final-norm hidden state of every real position.
        let hidden: [Float16]
        let tokens: Int
        let calls: Int
        let callSeconds: [Double]
        /// each call's length, its pad included
        let callLengths: [Int]
        /// zeroing (or restoring) the four states before the pass
        let resetSeconds: Double
        let seconds: Double
    }

    let url: URL
    let shape: KevGraphShape
    /// the longest call (S of a static-S bundle)
    var chunk: Int { shape.cap }
    let hidden: Int
    let maxContext: Int
    let padID: Int32
    let functionNames: [String]
    let loadSeconds: (model: Double, function: Double)
    let descriptor: KevJSON
    let options: SpecializationOptions

    private let main: InferenceFunction
    private let idsDescriptor: NDArrayDescriptor
    private let positions: NDArrayDescriptor
    private let hiddenDescriptor: NDArrayDescriptor
    /// The four states' descriptors, the dynamic length resolved to max_context_length.
    private let stateDescriptors: [String: NDArrayDescriptor]
    // The four states and the hidden buffers, allocated once. A pass moves them into locals for its whole run:
    // MutableViews borrows what it holds up to `run`, which a class property's access scope does not cover.
    private var buffers: Buffers?
    /// The states after a shared prefix (allocated on first use).
    private var snapshot: [String: NDArray]?

    struct Buffers {
        var keyCache: NDArray
        var valueCache: NDArray
        var convState: NDArray
        var recState: NDArray
        /// one hidden buffer per call length, allocated on its first call
        var hidden: [Int: NDArray]
    }

    static let stateNames = ["keyCache", "valueCache", "convState", "recState"]

    /// Loads the decoder asset (`.aimodelc` AOT or `.aimodel` JIT) and checks `main` against the contract.
    init(contentsOf url: URL, shape: KevGraphShape, hidden: Int, maxContext: Int, padID: Int,
                options: SpecializationOptions) async throws
    {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = kevSecondsSince(t0)
        functionNames = model.functionNames
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let mainFn = try model.loadFunction(named: "main") else {
            throw KevError.contract("decoder \(url.lastPathComponent): no \"main\" (functions \(model.functionNames))")
        }
        let tMain = kevSecondsSince(t1)
        try Self.check(md, url: url, shape: shape, hidden: hidden)
        var resolved: [String: NDArrayDescriptor] = [:]
        for n in Self.stateNames {
            let d = KevND.descriptor(md.stateDescriptor(of: n))!
            resolved[n] = d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? maxContext : $0 })
        }
        func state(_ n: String) -> NDArray {
            var a = NDArray(descriptor: resolved[n]!)
            KevND.zero(&a)
            return a
        }
        self.stateDescriptors = resolved
        self.url = url
        self.shape = shape
        self.hidden = hidden
        self.maxContext = maxContext
        self.padID = Int32(padID)
        self.options = options
        self.loadSeconds = (tModel, tMain)
        self.descriptor = kevDescribe(md)
        self.main = mainFn
        self.idsDescriptor = KevND.descriptor(md.inputDescriptor(of: "input_ids"))!
        self.positions = KevND.descriptor(md.inputDescriptor(of: "position_ids"))!
        self.hiddenDescriptor = KevND.descriptor(md.outputDescriptor(of: "hidden"))!
        buffers = Buffers(keyCache: state("keyCache"), valueCache: state("valueCache"), convState: state("convState"),
                          recState: state("recState"), hidden: [:])
    }

    /// The graph's contract: inputs input_ids [1, S] i32 (dynamic: [1, -1]) and position_ids [1, -1] i32, output
    /// hidden [1, S, d] f16 (dynamic: [1, -1, d]), the four states fp16 (KV rank 5 with one dynamic axis, conv rank 4,
    /// recurrent rank 5).
    static func check(_ d: InferenceFunctionDescriptor, url: URL, shape: KevGraphShape, hidden: Int) throws {
        var bad: [String] = []
        let n = shape.dynamic ? -1 : shape.cap
        if Set(d.inputNames) != ["input_ids", "position_ids"] { bad.append("inputs \(d.inputNames.sorted())") }
        if KevTensorSpec.of(d.inputDescriptor(of: "input_ids")) != KevTensorSpec(shape: [1, n], type: .int32) {
            bad.append("input_ids \(KevTensorSpec.of(d.inputDescriptor(of: "input_ids"))?.description ?? "missing") != [1, \(n)] int32")
        }
        if KevTensorSpec.of(d.inputDescriptor(of: "position_ids")) != KevTensorSpec(shape: [1, -1], type: .int32) {
            bad.append("position_ids \(KevTensorSpec.of(d.inputDescriptor(of: "position_ids"))?.description ?? "missing") != [1, -1] int32")
        }
        if d.outputNames != ["hidden"] || KevTensorSpec.of(d.outputDescriptor(of: "hidden")) != KevTensorSpec(shape: [1, n, hidden], type: .float16) {
            bad.append("outputs \(d.outputNames) \(KevTensorSpec.of(d.outputDescriptor(of: "hidden"))?.description ?? "") != hidden [1, \(n), \(hidden)] float16")
        }
        if Set(d.stateNames) != Set(stateNames) { bad.append("states \(d.stateNames.sorted())") }
        let k = KevTensorSpec.of(d.stateDescriptor(of: "keyCache"))
        let v = KevTensorSpec.of(d.stateDescriptor(of: "valueCache"))
        let conv = KevTensorSpec.of(d.stateDescriptor(of: "convState"))
        let rec = KevTensorSpec.of(d.stateDescriptor(of: "recState"))
        if k == nil || k != v || k!.shape.count != 5 || k!.shape.filter({ $0 < 0 }).count != 1 || k!.shape[3] >= 0 {
            bad.append("keyCache / valueCache \(k?.description ?? "missing") / \(v?.description ?? "missing")")
        }
        if conv?.shape.count != 4 || conv?.shape.contains(where: { $0 < 0 }) != false { bad.append("convState \(conv?.description ?? "missing")") }
        if rec?.shape.count != 5 || rec?.shape.contains(where: { $0 < 0 }) != false { bad.append("recState \(rec?.description ?? "missing")") }
        for s in [k, v, conv, rec] where s?.type != .float16 { bad.append("a state is not float16: \(s?.description ?? "missing")") }
        if !bad.isEmpty { throw KevError.contract("decoder \(url.lastPathComponent) main: \(bad.joined(separator: "; "))") }
    }

    /// Zeroes the states and runs one row in the plan's calls.
    func run(ids: [Int]) async throws -> Pass {
        guard var b = buffers else { throw KevError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        KevND.zero(&b.keyCache)
        KevND.zero(&b.valueCache)
        KevND.zero(&b.convState)
        KevND.zero(&b.recState)
        let reset = kevSecondsSince(t0)
        let (h, secs, lens) = try await pieces(ids, from: 0, &b)
        return Pass(hidden: h, tokens: ids.count, calls: secs.count, callSeconds: secs, callLengths: lens, resetSeconds: reset,
                    seconds: kevSecondsSince(t0))
    }

    /// The shared prefix: `prefix` (a multiple of q, no pad) once from zero states, its states kept; then each of
    /// `tails` (a question's row after the prefix) from a copy of them, positions continuing at prefix.count. -> the
    /// prefix's pass and one pass per tail.
    func runShared(prefix: [Int], tails: [[Int]]) async throws -> (prefix: Pass, tails: [Pass]) {
        guard prefix.count % shape.q == 0 else {
            throw KevError.contract("a shared prefix of \(prefix.count) tokens is not a multiple of \(shape.q)")
        }
        guard var b = buffers else { throw KevError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        KevND.zero(&b.keyCache)
        KevND.zero(&b.valueCache)
        KevND.zero(&b.convState)
        KevND.zero(&b.recState)
        let reset = kevSecondsSince(t0)
        let (hp, sp, lp) = try await pieces(prefix, from: 0, &b)
        let tSnap = ContinuousClock.now
        var snap = snapshot ?? Dictionary(uniqueKeysWithValues: Self.stateNames.map { ($0, NDArray(descriptor: stateDescriptors[$0]!)) })
        KevND.copy(b.keyCache, into: &snap["keyCache"]!)
        KevND.copy(b.valueCache, into: &snap["valueCache"]!)
        KevND.copy(b.convState, into: &snap["convState"]!)
        KevND.copy(b.recState, into: &snap["recState"]!)
        snapshot = snap
        let snapSeconds = kevSecondsSince(tSnap)
        let pre = Pass(hidden: hp, tokens: prefix.count, calls: sp.count, callSeconds: sp, callLengths: lp,
                       resetSeconds: reset + snapSeconds, seconds: kevSecondsSince(t0))
        var out: [Pass] = []
        for tail in tails {
            let t1 = ContinuousClock.now
            KevND.copy(snap["keyCache"]!, into: &b.keyCache)
            KevND.copy(snap["valueCache"]!, into: &b.valueCache)
            KevND.copy(snap["convState"]!, into: &b.convState)
            KevND.copy(snap["recState"]!, into: &b.recState)
            let restore = kevSecondsSince(t1)
            let (h, secs, lens) = try await pieces(tail, from: prefix.count, &b)
            out.append(Pass(hidden: h, tokens: tail.count, calls: secs.count, callSeconds: secs, callLengths: lens,
                            resetSeconds: restore, seconds: kevSecondsSince(t1)))
        }
        return (pre, out)
    }

    /// The prepared state (round 15): `prefix` (a multiple of q, no pad; empty = nothing to keep) once from zero states
    /// -> its pass and a copy of the four states after it, newly allocated (the caller keeps it past this call).
    func prepare(prefix: [Int]) async throws -> (pass: Pass, states: [String: NDArray]) {
        guard prefix.count % shape.q == 0 else {
            throw KevError.contract("a prepared prefix of \(prefix.count) tokens is not a multiple of \(shape.q)")
        }
        guard var b = buffers else { throw KevError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        KevND.zero(&b.keyCache)
        KevND.zero(&b.valueCache)
        KevND.zero(&b.convState)
        KevND.zero(&b.recState)
        let reset = kevSecondsSince(t0)
        let (hp, sp, lp) = prefix.isEmpty ? ([Float16](), [Double](), [Int]()) : try await pieces(prefix, from: 0, &b)
        var states = Dictionary(uniqueKeysWithValues: Self.stateNames.map { ($0, NDArray(descriptor: stateDescriptors[$0]!)) })
        KevND.copy(b.keyCache, into: &states["keyCache"]!)
        KevND.copy(b.valueCache, into: &states["valueCache"]!)
        KevND.copy(b.convState, into: &states["convState"]!)
        KevND.copy(b.recState, into: &states["recState"]!)
        return (Pass(hidden: hp, tokens: prefix.count, calls: sp.count, callSeconds: sp, callLengths: lp, resetSeconds: reset,
                     seconds: kevSecondsSince(t0)), states)
    }

    /// Each of `tails` (a question's row after a prepared prefix of k tokens) from a copy of the prepared `states`,
    /// positions continuing at k: the shared prefix's second half, on states kept from an earlier `prepare`.
    func runPrepared(states: [String: NDArray], from k: Int, tails: [[Int]]) async throws -> [Pass] {
        guard Set(states.keys) == Set(Self.stateNames) else { throw KevError.contract("prepared states \(states.keys.sorted())") }
        guard var b = buffers else { throw KevError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        var out: [Pass] = []
        for tail in tails {
            let t1 = ContinuousClock.now
            KevND.copy(states["keyCache"]!, into: &b.keyCache)
            KevND.copy(states["valueCache"]!, into: &b.valueCache)
            KevND.copy(states["convState"]!, into: &b.convState)
            KevND.copy(states["recState"]!, into: &b.recState)
            let restore = kevSecondsSince(t1)
            let (h, secs, lens) = try await pieces(tail, from: k, &b)
            out.append(Pass(hidden: h, tokens: tail.count, calls: secs.count, callSeconds: secs, callLengths: lens,
                            resetSeconds: restore, seconds: kevSecondsSince(t1)))
        }
        return out
    }

    /// Every call length of the plan (`shape.callLengths`), one call each of <|endoftext|> ids from zero states, the
    /// hidden read back and dropped: the runtime's one-time specialization of each length, done before any request.
    /// -> each length's seconds.
    func warmUp() async throws -> [(length: Int, seconds: Double)] {
        guard var b = buffers else { throw KevError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        var out: [(length: Int, seconds: Double)] = []
        for c in shape.callLengths {
            KevND.zero(&b.keyCache)
            KevND.zero(&b.valueCache)
            KevND.zero(&b.convState)
            KevND.zero(&b.recState)
            let t = ContinuousClock.now
            _ = try await call([Int32](repeating: padID, count: c), from: 0, &b)
            out.append((c, kevSecondsSince(t)))
        }
        return out
    }

    /// ids from position p0 on, in the plan's calls (the last padded) -> hidden fp16 rows, each call's seconds and length.
    private func pieces(_ ids: [Int], from p0: Int, _ b: inout Buffers) async throws -> ([Float16], [Double], [Int]) {
        let plan = try shape.plan(ids.count)
        let end = p0 + plan.reduce(0) { $0 + $1.length }
        guard end <= maxContext - 1 else {
            throw KevError.graphLimit("\(p0 + ids.count) tokens (padded \(end)) > the graph's \(maxContext - 1) positions")
        }
        var out = [Float16]()
        out.reserveCapacity(ids.count * hidden)
        var secs: [Double] = []
        var lens: [Int] = []
        var q0 = 0
        for (c, r) in plan {
            let t = ContinuousClock.now
            var x = [Int32](repeating: padID, count: c)
            for i in 0..<r { x[i] = Int32(ids[q0 + i]) }
            let h = try await call(x, from: p0 + q0, &b)
            out += h[0..<(r * hidden)]
            secs.append(kevSecondsSince(t))
            lens.append(c)
            q0 += r
        }
        return (out, secs, lens)
    }

    /// One call: ids x after p earlier ids (position_ids 0 ..< p + c), the states in place -> hidden [c * d] fp16.
    private func call(_ x: [Int32], from p: Int, _ b: inout Buffers) async throws -> [Float16] {
        let c = x.count
        let end = p + c
        let inputs: [String: NDArray] = [
            "input_ids": KevND.make(x, shape.dynamic ? idsDescriptor.resolvingDynamicDimensions([1, c]) : idsDescriptor),
            "position_ids": KevND.make((0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
        ]
        var h = b.hidden.removeValue(forKey: c)
            ?? NDArray(descriptor: hiddenDescriptor.resolvingDynamicDimensions(shape.dynamic ? [1, c, hidden] : hiddenDescriptor.shape))
        var states = InferenceFunction.MutableViews()
        states.insert(&b.keyCache, for: "keyCache")
        states.insert(&b.valueCache, for: "valueCache")
        states.insert(&b.convState, for: "convState")
        states.insert(&b.recState, for: "recState")
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&h, for: "hidden")
        _ = try await main.run(inputs: inputs, states: consume states, outputViews: consume outputs)
        let rows = KevND.read(h, as: Float16.self)
        b.hidden[c] = h
        guard rows.count == c * hidden else {
            throw KevError.contract("hidden of \(rows.count) values for a call of \(c) ids")
        }
        return rows
    }
}

#endif
