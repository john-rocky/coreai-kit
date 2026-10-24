// Decoder — the WeVisOne-VL-2B text decoder with its second output, on the low-level runtime (AIModel + loadFunction +
// MutableViews states). The kit's pipelined engine samples on the GPU and hands back token ids only; the mask decoder
// needs the decoder's last hidden state at each <OBJ> the model writes, so the app runs the graph itself. The loop is
// DeciderVision's DecisionDecoder (zoo apps/DeciderVision, 2307ecf, sha256 b5cdf022598b) with WeVisOne's inputs, the
// second output and a greedy free run.
//
//   main      input_ids [1, 1]            prefill  input_ids [1, 64]
//   both      position_ids [1, -1] (0 ..< t + S for a call after t ids), image_embeds fp16 [196, 2048],
//             deepstack_embeds fp16 [588, 2048], rope_shift_start [1] i32, rope_shift_amount [1] i32
//             -> logits fp16 [1, S, 152680], last_hidden fp16 [1, S, 2048] (the last decoder layer's output, before
//             the final norm: the package's LastHiddenCapture)
//   states    keyCache / valueCache fp16 [28, 1, 8, -1, 128], the dynamic axis bound at `kvLength` (1,024): the
//             iOS cap of the pipelined engine (CoreAIPipelinedEngine.swift, apple/coreai-models#124: the device
//             compiler miscompiles this graph class once the bound KV axis reaches 2,048), on the Mac too
//
// The prompt goes in one of three orders (`Schedule`); generation is "main" one token at a time, greedy (the first
// largest fp16 logit), no repetition penalty, until <|im_end|> / <|endoftext|> or `maxNew`. The hidden state of the
// call that feeds an <OBJ> back is that <OBJ>'s: no second forward pass (its cosine to the package's teacher-forced
// state is ≥ 0.999994 on Core AI).

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public final class WeVisOneDecoder: @unchecked Sendable {
    public static let vocab = 152_680
    public static let hidden = 2048
    public static let chunkSize = 64

    public enum Schedule: String, Sendable, Codable, CaseIterable {
        /// Every prompt token through "main" S = 1 (the order of the zoo's mask_e2e.py gate).
        case s1
        /// 64-token "prefill" calls while 64 remain; the rest padded with <|endoftext|> to one more "prefill" call whose
        /// row r − 1 holds the next token's logits (the padded rows' KV slots are rewritten before any call reads them).
        case pad
        /// The pipelined engine's split: "prefill" while remaining − 1 ≥ 64, then "main" S = 1 for the rest.
        case engine
    }

    public struct Call: Sendable {
        /// "main", "prefill" or "pad"
        public let kind: String
        public let tokens: Int
        public let seconds: Double
    }

    public struct Generation: Sendable {
        /// The generated ids; the stop id is the last one when it came.
        public let ids: [Int]
        /// last_hidden of each call that fed an <OBJ>, fp16 [2048], in order.
        public let objHidden: [[Float16]]
        public let objPositions: [Int]
        /// The softmax top-2 probability gap of the logits each id was taken from (when asked; else empty).
        public let gaps: [Double]
        public let promptCalls: [Call]
        public let promptSeconds: Double
        /// From the first generated id to the end of the last call.
        public let decodeSeconds: Double
        public let decodeCalls: Int
        /// "stop" (a stop id), "max" (maxNew), "cancelled"
        public let stop: String
        public let resetSeconds: Double
    }

    public struct StaticInputs: @unchecked Sendable {
        let embeds: NDArray
        let deepstack: NDArray
        let start: NDArray
        let amount: NDArray
    }

    public let url: URL
    public let kvLength: Int
    /// 64 when the asset has a "prefill" function, nil for an S = 1 asset.
    public let chunk: Int?
    public let functionNames: [String]
    public let loadSeconds: Double
    public let descriptors: [String: Any]
    public let options: String

    private let main: InferenceFunction
    private let prefill: InferenceFunction?
    private let idsMain: NDArrayDescriptor
    private let idsPrefill: NDArrayDescriptor?
    private let positions: NDArrayDescriptor
    private let startDescriptor: NDArrayDescriptor
    private let amountDescriptor: NDArrayDescriptor
    // The two states and the output buffers, allocated once. A pass moves them into a local for its whole run:
    // MutableViews borrows what it holds up to `run`, which a class property's access scope does not cover.
    private var buffers: Buffers?

    struct Buffers {
        var keyCache: NDArray
        var valueCache: NDArray
        var logits1: NDArray
        var hidden1: NDArray
        var logits64: NDArray
        var hidden64: NDArray
    }

    static let stateNames = ["keyCache", "valueCache"]
    static let inputNames: Set<String> = ["input_ids", "position_ids", "image_embeds", "deepstack_embeds",
                                          "rope_shift_start", "rope_shift_amount"]

    /// Loads the decoder asset (`.aimodel` or `.aimodelc`) and checks its functions against the contract.
    public init(contentsOf url: URL, kvLength: Int = 1024, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let md = model.functionDescriptor(for: "main"), let mainFn = try model.loadFunction(named: "main") else {
            throw CutoutError.contract("decoder \(url.lastPathComponent): no \"main\" (functions \(model.functionNames))")
        }
        var pd: InferenceFunctionDescriptor? = nil
        var prefillFn: InferenceFunction? = nil
        if model.functionNames.contains("prefill") {
            guard let d = model.functionDescriptor(for: "prefill"), let f = try model.loadFunction(named: "prefill") else {
                throw CutoutError.contract("decoder \(url.lastPathComponent): \"prefill\" does not load")
            }
            pd = d
            prefillFn = f
        }
        loadSeconds = secondsSince(t0)
        try Self.check(md, name: "main", queryLength: 1)
        if let pd {
            try Self.check(pd, name: "prefill", queryLength: Self.chunkSize)
            for s in Self.stateNames where TensorSpec.of(pd.stateDescriptor(of: s)) != TensorSpec.of(md.stateDescriptor(of: s)) {
                throw CutoutError.contract("prefill state \(s) differs from main's")
            }
        }
        func nd(_ d: InferenceFunctionDescriptor, input n: String) -> NDArrayDescriptor { ND.descriptor(d.inputDescriptor(of: n))! }
        func out(_ d: InferenceFunctionDescriptor, _ n: String) -> NDArray {
            let o = ND.descriptor(d.outputDescriptor(of: n))!
            return NDArray(descriptor: o.resolvingDynamicDimensions(o.shape))
        }
        func state(_ n: String) -> NDArray {
            let d = ND.descriptor(md.stateDescriptor(of: n))!
            var a = NDArray(descriptor: d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? kvLength : $0 }))
            ND.zero(&a)
            return a
        }
        self.url = url
        self.kvLength = kvLength
        self.chunk = prefillFn == nil ? nil : Self.chunkSize
        self.functionNames = model.functionNames
        var desc: [String: Any] = ["main": describe(md)]
        if let pd { desc["prefill"] = describe(pd) }
        self.descriptors = desc
        self.options = describe(options)
        self.main = mainFn
        self.prefill = prefillFn
        self.idsMain = nd(md, input: "input_ids")
        self.idsPrefill = pd.map { nd($0, input: "input_ids") }
        self.positions = nd(md, input: "position_ids")
        self.startDescriptor = nd(md, input: "rope_shift_start")
        self.amountDescriptor = nd(md, input: "rope_shift_amount")
        // an S = 1 asset gets 1-row stand-ins for the prefill buffers (never bound)
        let p = pd ?? md
        buffers = Buffers(keyCache: state("keyCache"), valueCache: state("valueCache"),
                          logits1: out(md, "logits"), hidden1: out(md, "last_hidden"),
                          logits64: out(p, "logits"), hidden64: out(p, "last_hidden"))
    }

    private static func check(_ d: InferenceFunctionDescriptor, name: String, queryLength: Int) throws {
        let want: [String: TensorSpec] = [
            "input_ids": TensorSpec(shape: [1, queryLength], type: .int32),
            "position_ids": TensorSpec(shape: [1, -1], type: .int32),
            "image_embeds": TensorSpec(shape: [196, hidden], type: .float16),
            "deepstack_embeds": TensorSpec(shape: [588, hidden], type: .float16),
            "rope_shift_start": TensorSpec(shape: [1], type: .int32),
            "rope_shift_amount": TensorSpec(shape: [1], type: .int32),
        ]
        var bad: [String] = []
        if Set(d.inputNames) != inputNames { bad.append("inputs \(d.inputNames)") }
        for (n, w) in want where TensorSpec.of(d.inputDescriptor(of: n)) != w {
            bad.append("\(n) \(TensorSpec.of(d.inputDescriptor(of: n))?.description ?? "missing") != \(w)")
        }
        if d.outputNames != ["logits", "last_hidden"]
            || TensorSpec.of(d.outputDescriptor(of: "logits")) != TensorSpec(shape: [1, queryLength, vocab], type: .float16)
            || TensorSpec.of(d.outputDescriptor(of: "last_hidden")) != TensorSpec(shape: [1, queryLength, hidden], type: .float16)
        {
            bad.append("outputs \(d.outputNames) (the app needs logits [1, \(queryLength), \(vocab)] and last_hidden "
                + "[1, \(queryLength), \(hidden)], fp16: the two-output export)")
        }
        if Set(d.stateNames) != Set(stateNames) { bad.append("states \(d.stateNames)") }
        for s in stateNames where TensorSpec.of(d.stateDescriptor(of: s))?.type != .float16 { bad.append("state \(s) not f16") }
        if !bad.isEmpty { throw CutoutError.contract("decoder function \(name): \(bad.joined(separator: "; "))") }
    }

    /// The four static inputs of one request: the tower's two outputs as they came, and the rope shift of the row.
    public func staticInputs(_ tower: VisionTaps.Output, row: PromptRow) -> StaticInputs {
        StaticInputs(embeds: tower.imageEmbeds, deepstack: tower.deepstackEmbeds,
                     start: ND.make([row.ropeShiftStart], startDescriptor), amount: ND.make([row.ropeShiftAmount], amountDescriptor))
    }

    /// Zeroes the states, feeds the prompt in `schedule`'s order and generates greedily. One request at a time per decoder:
    /// the states are this object's.
    public func generate(prompt ids: [Int], inputs s: StaticInputs, schedule: Schedule, maxNew: Int, recordGaps: Bool = false,
                         onToken: (@Sendable (Int) -> Void)? = nil) async throws -> Generation
    {
        let n = ids.count
        let cap = min(maxNew, kvLength - n)
        guard cap >= 1 else { throw CutoutError.prompt("\(n) prompt ids leave no room in the \(kvLength)-slot KV cache") }
        guard schedule == .s1 || chunk != nil || schedule == .engine else {
            throw CutoutError.contract("schedule \(schedule) needs the \"prefill\" function; this asset has \(functionNames)")
        }
        guard var b = buffers else { throw CutoutError.contract("decoder busy: one request at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        ND.zero(&b.keyCache)
        ND.zero(&b.valueCache)
        let reset = secondsSince(t0)

        // the prompt's calls: (start, count, kind); a pad call carries count real ids
        var plan: [(start: Int, count: Int, kind: String)] = []
        var t = 0
        if let c = chunk, schedule != .s1 {
            if schedule == .pad {
                while n - t >= c { plan.append((t, c, "prefill")); t += c }
                if t < n { plan.append((t, n - t, "pad")); t = n }
            } else {
                while n - t - 1 >= c { plan.append((t, c, "prefill")); t += c }
            }
        }
        while t < n { plan.append((t, 1, "main")); t += 1 }

        var calls: [Call] = []
        var last: [Float16] = []
        let tPrompt = ContinuousClock.now
        for (i, step) in plan.enumerated() {
            let tc = ContinuousClock.now
            let isPrefill = step.kind != "main"
            let end = step.start + (isPrefill ? Self.chunkSize : 1)
            var callIDs = ids[step.start..<(step.start + step.count)].map { Int32($0) }
            if step.kind == "pad" {
                callIDs += [Int32](repeating: Int32(PromptBuilder.endOfText), count: Self.chunkSize - step.count)
            }
            let inputs: [String: NDArray] = [
                "input_ids": ND.make(callIDs, isPrefill ? idsPrefill! : idsMain),
                "position_ids": ND.make((0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
                "image_embeds": s.embeds, "deepstack_embeds": s.deepstack,
                "rope_shift_start": s.start, "rope_shift_amount": s.amount,
            ]
            var states = InferenceFunction.MutableViews()
            states.insert(&b.keyCache, for: "keyCache")
            states.insert(&b.valueCache, for: "valueCache")
            var outputs = InferenceFunction.MutableViews()
            if isPrefill {
                outputs.insert(&b.logits64, for: "logits")
                outputs.insert(&b.hidden64, for: "last_hidden")
                _ = try await prefill!.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            } else {
                outputs.insert(&b.logits1, for: "logits")
                outputs.insert(&b.hidden1, for: "last_hidden")
                _ = try await main.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            }
            if i == plan.count - 1 {
                last = isPrefill ? ND.readRow(b.logits64, row: step.count - 1, as: Float16.self)
                    : ND.read(b.logits1, as: Float16.self)
            }
            calls.append(Call(kind: step.kind, tokens: step.count, seconds: secondsSince(tc)))
        }
        let promptSeconds = secondsSince(tPrompt)

        var gen: [Int] = []
        var hidden: [[Float16]] = []
        var objPos: [Int] = []
        var gaps: [Double] = []
        var stop = "max"
        var decodeCalls = 0
        let tDecode = ContinuousClock.now
        var next = argmaxF16(last)
        if recordGaps { gaps.append(top2Gap(last)) }
        while true {
            gen.append(next)
            onToken?(next)
            if PromptBuilder.stopIDs.contains(next) { stop = "stop"; break }
            if gen.count >= cap { break }
            if Task.isCancelled { stop = "cancelled"; break }
            let pos = n + gen.count - 1
            let inputs: [String: NDArray] = [
                "input_ids": ND.make([Int32(next)], idsMain),
                "position_ids": ND.make((0...pos).map { Int32($0) }, positions.resolvingDynamicDimensions([1, pos + 1])),
                "image_embeds": s.embeds, "deepstack_embeds": s.deepstack,
                "rope_shift_start": s.start, "rope_shift_amount": s.amount,
            ]
            var states = InferenceFunction.MutableViews()
            states.insert(&b.keyCache, for: "keyCache")
            states.insert(&b.valueCache, for: "valueCache")
            var outputs = InferenceFunction.MutableViews()
            outputs.insert(&b.logits1, for: "logits")
            outputs.insert(&b.hidden1, for: "last_hidden")
            _ = try await main.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            decodeCalls += 1
            if next == PromptBuilder.obj {
                hidden.append(ND.read(b.hidden1, as: Float16.self))
                objPos.append(pos)
            }
            let logits = ND.read(b.logits1, as: Float16.self)
            next = argmaxF16(logits)
            if recordGaps { gaps.append(top2Gap(logits)) }
        }
        return Generation(ids: gen, objHidden: hidden, objPositions: objPos, gaps: gaps, promptCalls: calls,
                          promptSeconds: promptSeconds, decodeSeconds: secondsSince(tDecode), decodeCalls: decodeCalls,
                          stop: stop, resetSeconds: reset)
    }

    // MARK: - The prompt's tail, measured

    /// One measurement of the two ways to feed the prompt's last r ids (r < 64) after `prefix` ids in 64-chunks.
    public struct TailRun: Sendable {
        public let r: Int
        public let prefix: Int
        /// "s1" (r "main" calls) or "pad" (one "prefill" call with the rest padded by <|endoftext|>)
        public let form: String
        /// the tail alone: the calls, their inputs and the read of the next token's logits
        public let seconds: Double
        public let argmax: Int
        /// the greedy ids after the tail (`follow` of them), to check that the padded rows are never read
        public let follow: [Int]
        public let logits: [Float16]
    }

    /// The tail in both forms on the same prompt, alternated A B B A per repetition: for each r, the prompt's first
    /// `prefix + r` ids with prefix the largest multiple of 64 below; the prefix runs in "prefill" chunks from zero
    /// states, untimed; then the tail is timed.
    public func tailBench(prompt ids: [Int], inputs s: StaticInputs, tails: [Int], reps: Int, follow: Int = 16) async throws -> [TailRun] {
        guard chunk != nil else { throw CutoutError.contract("the tail measurement needs the \"prefill\" function") }
        guard var b = buffers else { throw CutoutError.contract("decoder busy: one request at a time") }
        buffers = nil
        defer { buffers = b }
        var runs: [TailRun] = []
        func call(_ fn: InferenceFunction, _ callIDs: [Int32], prefill: Bool, end: Int) async throws {
            let inputs: [String: NDArray] = [
                "input_ids": ND.make(callIDs, prefill ? idsPrefill! : idsMain),
                "position_ids": ND.make((0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
                "image_embeds": s.embeds, "deepstack_embeds": s.deepstack,
                "rope_shift_start": s.start, "rope_shift_amount": s.amount,
            ]
            var states = InferenceFunction.MutableViews()
            states.insert(&b.keyCache, for: "keyCache")
            states.insert(&b.valueCache, for: "valueCache")
            var outputs = InferenceFunction.MutableViews()
            if prefill {
                outputs.insert(&b.logits64, for: "logits")
                outputs.insert(&b.hidden64, for: "last_hidden")
                _ = try await fn.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            } else {
                outputs.insert(&b.logits1, for: "logits")
                outputs.insert(&b.hidden1, for: "last_hidden")
                _ = try await fn.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            }
        }
        for r in tails {
            let c = Self.chunkSize
            let prefix = ((ids.count - r) / c) * c
            guard r > 0, r < c, prefix + r <= ids.count, prefix > 0 else {
                throw CutoutError.prompt("tail \(r) does not fit the \(ids.count)-id prompt")
            }
            for rep in 0..<reps {
                let order = rep % 2 == 0 ? ["s1", "pad", "pad", "s1"] : ["pad", "s1", "s1", "pad"]
                for form in order {
                    ND.zero(&b.keyCache)
                    ND.zero(&b.valueCache)
                    var t = 0
                    while t < prefix {
                        try await call(prefill!, ids[t..<(t + c)].map { Int32($0) }, prefill: true, end: t + c)
                        t += c
                    }
                    let t0 = ContinuousClock.now
                    var last: [Float16]
                    if form == "s1" {
                        for q in prefix..<(prefix + r) { try await call(main, [Int32(ids[q])], prefill: false, end: q + 1) }
                        last = ND.read(b.logits1, as: Float16.self)
                    } else {
                        let tail = ids[prefix..<(prefix + r)].map { Int32($0) }
                            + [Int32](repeating: Int32(PromptBuilder.endOfText), count: c - r)
                        try await call(prefill!, tail, prefill: true, end: prefix + c)
                        last = ND.readRow(b.logits64, row: r - 1, as: Float16.self)
                    }
                    let seconds = secondsSince(t0)
                    var next = argmaxF16(last)
                    var gen: [Int] = []
                    if rep == 0 {
                        for k in 0..<follow {
                            gen.append(next)
                            let pos = prefix + r + k
                            try await call(main, [Int32(next)], prefill: false, end: pos + 1)
                            next = argmaxF16(ND.read(b.logits1, as: Float16.self))
                        }
                    }
                    runs.append(TailRun(r: r, prefix: prefix, form: form, seconds: seconds, argmax: argmaxF16(last),
                                        follow: gen, logits: rep == 0 ? last : []))
                }
            }
        }
        return runs
    }
}
