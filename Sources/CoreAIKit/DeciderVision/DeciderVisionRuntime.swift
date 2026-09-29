// DeciderVisionRuntime.swift — the two Core AI graphs behind decider-2b-vision, on the low-level runtime
// (`AIModel` + `loadFunction` + `MutableViews` states): the pipelined engine exposes no logits, and none of the
// kit's engines takes a static input and returns logits at every slot of one pass. Ported from the model zoo's
// `apps/DeciderVision/Sources/DeciderVision/{VisionTower,DecisionDecoder,Support}.swift` (2307ecf).
//
// The tower: patches float32 [4 G², 1536] → image_embeds float32 [G², 2048], G = 8 (g256) or 14 (g448). Stateless.
//
// The decoder: two functions share four states —
//   main     input_ids [1, 1]   prefill  input_ids [1, C] (C = 16, the bundle's language.prefill_chunk)
//   both     position_ids [1, t + S] (0 … t + S − 1), image_embeds [256, 2048] f16, image_rc [256, 2] i32,
//            rope_shift_start [1] i32, rope_shift_amount [1] i32 → logits [1, 1, V] f16 (the last position)
//   states   keyCache / valueCache [6, 1, 2, −1, 256] f16 (the dynamic length resolved to max_context_length),
//            convState [18, 1, 6144, 3] f16, recState [18, 1, 16, 128, 128] f16 — zeroed at the start of every row.
//
// Read order (the bundle's `decision.readout`): cursor = 0; for each slot s ascending, while s − cursor + 1 ≥ C run
// "prefill" on ids[cursor ..< cursor + C] (its logits are the slot's when cursor + C − 1 == s) and advance C; then
// "main" one token at a time up to and including s (the logits at cursor == s are the slot's). The last slot is the
// row's last token. Both graphs are checked against this contract at load (names, shapes and types of the inputs,
// states and outputs), so an asset that differs fails there, not in a probability.
//
// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles for: every fp16 access
// sits behind the same architecture guard as the rest of the kit's fp16 code.

import CoreAI
import Foundation

/// A decider-2b-vision asset, image or prompt that does not meet the model's contract.
public enum DeciderVisionError: Error, LocalizedError, Equatable {
    /// A graph whose inputs, states or outputs are not the ones the read-out drives.
    case contract(String)
    /// An image the host cannot turn into patches.
    case image(String)
    /// A row the author's slot rule reads differently from the questions asked.
    case prompt(String)
    /// A bundle whose metadata or tokenizer is not decider-2b-vision's.
    case bundle(String)
    /// A decision at a grid whose tower is not loaded and cannot be downloaded (a decider loaded from local files).
    case towerNotLoaded(grid: String)

    public var errorDescription: String? {
        switch self {
        case .contract(let s): return "decider-2b-vision contract: \(s)"
        case .image(let s): return "decider-2b-vision image: \(s)"
        case .prompt(let s): return "decider-2b-vision prompt: \(s)"
        case .bundle(let s): return "decider-2b-vision bundle: \(s)"
        case .towerNotLoaded(let grid):
            return "No \(grid) vision tower is loaded, and this decider has no download source for one; "
                + "pass its graph in towersAt."
        }
    }
}

// MARK: - NDArray helpers

@available(macOS 27, iOS 27, *)
enum DeciderVisionND {
    static func descriptor(_ value: InferenceValue.Descriptor?) -> NDArrayDescriptor? {
        guard case .ndArray(let d) = value else { return nil }
        return d
    }

    /// A new array for `descriptor`, filled row-major from `values`.
    static func make<T: BitwiseCopyable>(_ values: [T], _ descriptor: NDArrayDescriptor) -> NDArray {
        var array = NDArray(descriptor: descriptor)
        var view = array.mutableView(as: T.self)
        view.copyElements(fromContentsOf: values)
        return array
    }

    /// A new fp16 array for `descriptor`, each value rounded to the nearest half.
    static func makeHalf(_ values: [Float], _ descriptor: NDArrayDescriptor) -> NDArray {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        return make(values.map { Float16($0) }, descriptor)
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }

    /// Row-major copy of an array's scalars, honouring its strides (in elements).
    static func read<T: BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [T] {
        let shape = array.shape
        let count = shape.reduce(1, *)
        return array.view(as: T.self).withUnsafePointer { ptr, _, strides in
            var expected = 1
            var contiguous = true
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                if shape[d] > 1 && strides[d] != expected {
                    contiguous = false
                    break
                }
                expected *= shape[d]
            }
            if contiguous { return Array(UnsafeBufferPointer(start: ptr, count: count)) }
            var out = [T]()
            out.reserveCapacity(count)
            var index = [Int](repeating: 0, count: shape.count)
            for _ in 0..<count {
                var offset = 0
                for d in 0..<shape.count { offset += index[d] * strides[d] }
                out.append(ptr[offset])
                var d = shape.count - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
            return out
        }
    }

    /// An fp16 array's scalars widened to Float (exactly: every half is a float).
    static func readHalf(_ array: NDArray) -> [Float] {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        return read(array, as: Float16.self).map(Float.init)
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }

    /// Sets every scalar of the array (and any padding between its rows) to zero.
    static func zero(_ array: inout NDArray) {
        let shape = array.shape
        switch array.scalarType {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16: zeroScalars(&array, Float16(0), shape)
        #endif
        case .float32: zeroScalars(&array, Float(0), shape)
        case .int32: zeroScalars(&array, Int32(0), shape)
        default: fatalError("DeciderVisionND.zero: unsupported scalar type \(array.scalarType)")
        }
    }

    private static func zeroScalars<T: BitwiseCopyable>(_ array: inout NDArray, _ zero: T, _ shape: [Int]) {
        array.mutableView(as: T.self).withUnsafeMutablePointer { ptr, _, strides in
            var extent = 1
            for d in 0..<shape.count where shape[d] > 0 { extent += (shape[d] - 1) * strides[d] }
            ptr.update(repeating: zero, count: extent)
        }
    }
}

/// What a loaded function must declare: a shape (−1 = dynamic) and a scalar type.
@available(macOS 27, iOS 27, *)
struct DeciderVisionTensorSpec: Equatable, CustomStringConvertible {
    let shape: [Int]
    let type: NDArray.ScalarType

    var description: String { "\(shape) \(type)" }

    static func of(_ value: InferenceValue.Descriptor?) -> DeciderVisionTensorSpec? {
        DeciderVisionND.descriptor(value).map { DeciderVisionTensorSpec(shape: $0.shape, type: $0.scalarType) }
    }
}

func deciderVisionSeconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

// MARK: - Tower

/// One fixed-grid vision tower (`.aimodel` for the runtime to specialize, or a compiled `.aimodelc`).
@available(macOS 27, iOS 27, *)
final class DeciderVisionTower: @unchecked Sendable {
    static let hidden = 2048

    /// Merged grid side: 8 or 14.
    let grid: Int
    let url: URL
    /// `AIModel(contentsOf:)` + `loadFunction`, in seconds.
    let loadSeconds: Double
    private let function: InferenceFunction
    private let patchesDescriptor: NDArrayDescriptor
    private static let outputName = "image_embeds"

    init(contentsOf url: URL, grid: Int, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let name = model.functionNames.first, let fd = model.functionDescriptor(for: name),
            let fn = try model.loadFunction(named: name)
        else { throw DeciderVisionError.contract("tower \(url.lastPathComponent): no function") }
        loadSeconds = deciderVisionSeconds(since: t0)
        let n = 4 * grid * grid
        guard fd.stateNames.isEmpty, fd.inputNames == ["patches"], fd.outputNames == [Self.outputName],
            let pin = DeciderVisionND.descriptor(fd.inputDescriptor(of: "patches")),
            pin.shape == [n, DeciderVisionPreprocessor.patchVector], pin.scalarType == .float32,
            DeciderVisionTensorSpec.of(fd.outputDescriptor(of: Self.outputName))
                == DeciderVisionTensorSpec(shape: [grid * grid, Self.hidden], type: .float32)
        else {
            throw DeciderVisionError.contract(
                "tower \(url.lastPathComponent) is not patches f32 [\(n), 1536] -> image_embeds f32 "
                    + "[\(grid * grid), 2048] (inputs \(fd.inputNames), outputs \(fd.outputNames))")
        }
        self.grid = grid
        self.url = url
        self.function = fn
        self.patchesDescriptor = pin
    }

    /// patches [4 G² · 1536] → image_embeds [G² · 2048], row-major.
    func encode(patches: [Float]) async throws -> [Float] {
        let input = DeciderVisionND.make(patches, patchesDescriptor)
        var outputs = try await function.run(inputs: ["patches": input])
        guard let array = outputs.remove(Self.outputName)?.ndArray else {
            throw DeciderVisionError.contract("tower: no \(Self.outputName) in the outputs")
        }
        let embeds = DeciderVisionND.read(array, as: Float.self)
        guard embeds.count == grid * grid * Self.hidden else {
            throw DeciderVisionError.contract(
                "tower: \(embeds.count) values for \(grid * grid) x \(Self.hidden)")
        }
        return embeds
    }
}

// MARK: - Decoder

/// The ids-input decoder with its two functions and four states. One row at a time: the states are this object's.
@available(macOS 27, iOS 27, *)
final class DeciderVisionDecoder: @unchecked Sendable {
    static let imageRows = 256
    static let hidden = 2048

    /// The logits of one slot and the call that produced them.
    struct Slot: Sendable {
        let position: Int
        /// Full-vocabulary logits at the slot, widened from fp16.
        let logits: [Float]
        /// "prefill" or "main".
        let readFrom: String
    }

    /// One row's pass: the slots, each call's kind and wall time.
    struct Pass: Sendable {
        let slots: [Slot]
        /// true = prefill, false = main, per call in order.
        let callIsPrefill: [Bool]
        let callSeconds: [Double]
        /// Zeroing the four states before the row.
        let resetSeconds: Double
        let seconds: Double
    }

    /// The static inputs of one row: tower rows 0 ..< N as fp16 (the rest zero), image_rc[k] = (k / G, k % G).
    struct StaticInputs: @unchecked Sendable {
        let embeds: NDArray
        let rc: NDArray
        let start: NDArray
        let amount: NDArray
    }

    /// One call of a row's schedule: prefill on ids[start ..< start + count] (count = C), or main on ids[start].
    struct Step: Equatable {
        let prefill: Bool
        let start: Int
        let count: Int
        /// The slot this call's logits answer, if any.
        let reads: Int?
    }

    let url: URL
    let vocab: Int
    let maxContext: Int
    /// nil when the asset has no "prefill" function (S = 1 order only).
    let chunk: Int?
    let loadSeconds: Double

    private let main: InferenceFunction
    private let prefill: InferenceFunction?
    private let idsMain: NDArrayDescriptor
    private let idsPrefill: NDArrayDescriptor?
    private let positions: NDArrayDescriptor
    private let embedsDescriptor: NDArrayDescriptor
    private let rcDescriptor: NDArrayDescriptor
    private let shiftStartDescriptor: NDArrayDescriptor
    private let shiftAmountDescriptor: NDArrayDescriptor
    // The four states and the logits buffer, allocated once. A row moves them into locals for its whole pass:
    // MutableViews borrows what it holds up to `run`, which a class property's access scope does not cover.
    private var buffers: Buffers?

    struct Buffers {
        var keyCache: NDArray
        var valueCache: NDArray
        var convState: NDArray
        var recState: NDArray
        var logits: NDArray
    }

    static let stateNames = ["keyCache", "valueCache", "convState", "recState"]
    static let inputNames: Set<String> = [
        "input_ids", "position_ids", "image_embeds", "image_rc", "rope_shift_start", "rope_shift_amount",
    ]

    /// Loads the decoder asset (`.aimodel` or `.aimodelc`) and checks both functions against the contract.
    init(
        contentsOf url: URL, vocab: Int, maxContext: Int, prefillChunk: Int?, options: SpecializationOptions
    ) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let md = model.functionDescriptor(for: "main"), let mainFn = try model.loadFunction(named: "main")
        else {
            throw DeciderVisionError.contract(
                "decoder \(url.lastPathComponent): no \"main\" (functions \(model.functionNames))")
        }
        var pd: InferenceFunctionDescriptor? = nil
        var prefillFn: InferenceFunction? = nil
        if let c = prefillChunk, c > 1 {
            guard let d = model.functionDescriptor(for: "prefill"), let f = try model.loadFunction(named: "prefill")
            else {
                throw DeciderVisionError.contract(
                    "decoder \(url.lastPathComponent): prefill_chunk \(c) but no \"prefill\" "
                        + "(functions \(model.functionNames))")
            }
            pd = d
            prefillFn = f
        }
        try Self.check(md, name: "main", queryLength: 1, vocab: vocab)
        if let pd, let c = prefillChunk {
            try Self.check(pd, name: "prefill", queryLength: c, vocab: vocab)
            for s in Self.stateNames
            where DeciderVisionTensorSpec.of(pd.stateDescriptor(of: s))
                != DeciderVisionTensorSpec.of(md.stateDescriptor(of: s))
            {
                throw DeciderVisionError.contract("prefill state \(s) differs from main's")
            }
        }
        func nd(_ d: InferenceFunctionDescriptor, input n: String) -> NDArrayDescriptor {
            DeciderVisionND.descriptor(d.inputDescriptor(of: n))!
        }
        func state(_ n: String) -> NDArray {
            let d = DeciderVisionND.descriptor(md.stateDescriptor(of: n))!
            var a = NDArray(descriptor: d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? maxContext : $0 }))
            DeciderVisionND.zero(&a)
            return a
        }
        self.url = url
        self.vocab = vocab
        self.maxContext = maxContext
        self.chunk = prefillFn == nil ? nil : prefillChunk
        self.main = mainFn
        self.prefill = prefillFn
        self.idsMain = nd(md, input: "input_ids")
        self.idsPrefill = pd.map { nd($0, input: "input_ids") }
        self.positions = nd(md, input: "position_ids")
        self.embedsDescriptor = nd(md, input: "image_embeds")
        self.rcDescriptor = nd(md, input: "image_rc")
        self.shiftStartDescriptor = nd(md, input: "rope_shift_start")
        self.shiftAmountDescriptor = nd(md, input: "rope_shift_amount")
        let ld = DeciderVisionND.descriptor(md.outputDescriptor(of: "logits"))!
        buffers = Buffers(
            keyCache: state("keyCache"), valueCache: state("valueCache"), convState: state("convState"),
            recState: state("recState"), logits: NDArray(descriptor: ld.resolvingDynamicDimensions(ld.shape)))
        loadSeconds = deciderVisionSeconds(since: t0)
    }

    private static func check(
        _ d: InferenceFunctionDescriptor, name: String, queryLength: Int, vocab: Int
    ) throws {
        let want: [String: DeciderVisionTensorSpec] = [
            "input_ids": .init(shape: [1, queryLength], type: .int32),
            "position_ids": .init(shape: [1, -1], type: .int32),
            "image_embeds": .init(shape: [imageRows, hidden], type: .float16),
            "image_rc": .init(shape: [imageRows, 2], type: .int32),
            "rope_shift_start": .init(shape: [1], type: .int32),
            "rope_shift_amount": .init(shape: [1], type: .int32),
        ]
        var bad: [String] = []
        if Set(d.inputNames) != inputNames { bad.append("inputs \(d.inputNames)") }
        for (n, w) in want.sorted(by: { $0.key < $1.key })
        where DeciderVisionTensorSpec.of(d.inputDescriptor(of: n)) != w {
            bad.append("\(n) \(DeciderVisionTensorSpec.of(d.inputDescriptor(of: n))?.description ?? "missing") != \(w)")
        }
        if d.outputNames != ["logits"]
            || DeciderVisionTensorSpec.of(d.outputDescriptor(of: "logits"))
                != DeciderVisionTensorSpec(shape: [1, 1, vocab], type: .float16)
        {
            bad.append(
                "outputs \(d.outputNames) \(DeciderVisionTensorSpec.of(d.outputDescriptor(of: "logits"))?.description ?? "")")
        }
        if Set(d.stateNames) != Set(stateNames) { bad.append("states \(d.stateNames)") }
        for s in stateNames where DeciderVisionTensorSpec.of(d.stateDescriptor(of: s))?.type != .float16 {
            bad.append("state \(s) not f16")
        }
        if !bad.isEmpty {
            throw DeciderVisionError.contract("decoder function \(name): \(bad.joined(separator: "; "))")
        }
    }

    /// `towerEmbeds` = the tower's output [G² · 2048] float32 (nil for a text-only row).
    func staticInputs(towerEmbeds: [Float]?, grid: Int?, start: Int32, amount: Int32) throws -> StaticInputs {
        var embeds = [Float](repeating: 0, count: Self.imageRows * Self.hidden)
        var rc = [Int32](repeating: 0, count: Self.imageRows * 2)
        if let towerEmbeds, let g = grid {
            let n = g * g
            guard n <= Self.imageRows, towerEmbeds.count == n * Self.hidden else {
                throw DeciderVisionError.contract("image_embeds: \(towerEmbeds.count) values for \(n) rows")
            }
            embeds.replaceSubrange(0..<(n * Self.hidden), with: towerEmbeds)
            for k in 0..<n {
                rc[2 * k] = Int32(k / g)
                rc[2 * k + 1] = Int32(k % g)
            }
        }
        return StaticInputs(
            embeds: DeciderVisionND.makeHalf(embeds, embedsDescriptor), rc: DeciderVisionND.make(rc, rcDescriptor),
            start: DeciderVisionND.make([start], shiftStartDescriptor),
            amount: DeciderVisionND.make([amount], shiftAmountDescriptor))
    }

    /// The chunk order (the bundle's `decision.readout`); S = 1 throughout when `chunk` is nil.
    static func schedule(slots: [Int], chunk: Int?) -> [Step] {
        var steps: [Step] = []
        var cursor = 0
        for slot in slots {
            if let c = chunk {
                while slot - cursor + 1 >= c {
                    steps.append(Step(prefill: true, start: cursor, count: c, reads: cursor + c - 1 == slot ? slot : nil))
                    cursor += c
                }
            }
            while cursor <= slot {
                steps.append(Step(prefill: false, start: cursor, count: 1, reads: cursor == slot ? slot : nil))
                cursor += 1
            }
        }
        return steps
    }

    /// Zeroes the states and reads one row's slots in the chunk order.
    func run(ids: [Int], slots: [Int], inputs s: StaticInputs) async throws -> Pass {
        guard !slots.isEmpty, slots == slots.sorted(), let last = slots.last, last == ids.count - 1 else {
            throw DeciderVisionError.prompt("slots \(slots) do not end the row (\(ids.count) ids)")
        }
        guard ids.count <= maxContext else {
            throw DecisionError.promptTooLong(tokens: ids.count, max: maxContext)
        }
        guard var b = buffers else { throw DeciderVisionError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        DeciderVisionND.zero(&b.keyCache)
        DeciderVisionND.zero(&b.valueCache)
        DeciderVisionND.zero(&b.convState)
        DeciderVisionND.zero(&b.recState)
        let reset = deciderVisionSeconds(since: t0)
        let steps = Self.schedule(slots: slots, chunk: prefill != nil ? chunk : nil)
        var got: [Slot] = []
        var kinds: [Bool] = [], secs: [Double] = []
        for step in steps {
            let t = ContinuousClock.now
            let fn = step.prefill ? prefill! : main
            let end = step.start + step.count
            let inputs: [String: NDArray] = [
                "input_ids": DeciderVisionND.make(
                    ids[step.start..<end].map { Int32($0) }, step.prefill ? idsPrefill! : idsMain),
                "position_ids": DeciderVisionND.make(
                    (0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
                "image_embeds": s.embeds, "image_rc": s.rc, "rope_shift_start": s.start, "rope_shift_amount": s.amount,
            ]
            var states = InferenceFunction.MutableViews()
            states.insert(&b.keyCache, for: "keyCache")
            states.insert(&b.valueCache, for: "valueCache")
            states.insert(&b.convState, for: "convState")
            states.insert(&b.recState, for: "recState")
            var outputs = InferenceFunction.MutableViews()
            outputs.insert(&b.logits, for: "logits")
            _ = try await fn.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            if let slot = step.reads {
                got.append(
                    Slot(
                        position: slot, logits: DeciderVisionND.readHalf(b.logits),
                        readFrom: step.prefill ? "prefill" : "main"))
            }
            kinds.append(step.prefill)
            secs.append(deciderVisionSeconds(since: t))
        }
        guard steps.last.map({ $0.start + $0.count }) == ids.count, got.count == slots.count else {
            throw DeciderVisionError.prompt(
                "read \(got.count) slots; the schedule does not end at the row's last id")
        }
        return Pass(
            slots: got, callIsPrefill: kinds, callSeconds: secs, resetSeconds: reset,
            seconds: deciderVisionSeconds(since: t0))
    }
}
