// From the model zoo's apps/ClefFlash/Sources/ClefFlash/Decoder.swift (5ef2247, sha256 2ee502b96977, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// Decoder — the clef-flash backbone on the low-level runtime (AIModel + loadFunction + MutableViews states): one
// static-S function `main` (S = 64, the bundle's language.prefill_chunk) that returns the final-norm hidden state at
// every position; no vocabulary head in the graph.
//
//   inputs   input_ids [1, 64] i32, position_ids [1, -1] i32 (0 ... 64k + 63), image_embeds [1024, 4096] f16,
//            image_rc [1024, 2] i32, rope_shift_start [1] i32, rope_shift_amount [1] i32
//   states   keyCache / valueCache [8, 1, 4, -1, 256] f16 (the dynamic length resolved to max_context_length 4096),
//            convState [24, 1, 8192, 3] f16, recState [24, 1, 32, 128, 128] f16 — zero at the start of every row
//   output   hidden [1, 64, 4096] f16
//
// Read order (the bundle's `decision.readout`, decide.py): a row of T ids runs as ceil(T / 64) calls; call k gets
// ids[64k ..< 64k + 64] with position_ids 0 ..< 64k + 64; the last call is padded with <|endoftext|> (248044) and
// the hidden rows of the padded positions are dropped (causal: they cannot reach a real position).

import CoreAI
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitClefDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 27, iOS 27, *)
final class ClefDecoder: @unchecked Sendable {
    static let hidden = 4096
    static let imageRows = 1024

    /// One row's pass: the hidden rows and each call's time.
    struct Pass: Sendable {
        /// [T * 4096] fp16, row-major: the final-norm hidden state of every real position.
        let hidden: [Float16]
        let tokens: Int
        let calls: Int
        let callSeconds: [Double]
        /// zeroing the four states before the row
        let resetSeconds: Double
        let seconds: Double
    }

    /// The static inputs of one row.
    struct StaticInputs: @unchecked Sendable {
        let embeds: NDArray
        let rc: NDArray
        let start: NDArray
        let amount: NDArray
    }

    let url: URL
    let chunk: Int
    let maxContext: Int
    let functionNames: [String]
    let loadSeconds: (model: Double, function: Double)
    let descriptor: ClefJSON

    private let main: InferenceFunction
    private let idsDescriptor: NDArrayDescriptor
    private let positions: NDArrayDescriptor
    private let embedsDescriptor: NDArrayDescriptor
    private let rcDescriptor: NDArrayDescriptor
    private let shiftStartDescriptor: NDArrayDescriptor
    private let shiftAmountDescriptor: NDArrayDescriptor
    // The four states and the hidden buffer, allocated once. A row moves them into locals for its whole pass:
    // MutableViews borrows what it holds up to `run`, which a class property's access scope does not cover.
    private var buffers: Buffers?

    struct Buffers {
        var keyCache: NDArray
        var valueCache: NDArray
        var convState: NDArray
        var recState: NDArray
        var hidden: NDArray
    }

    static let stateNames = ["keyCache", "valueCache", "convState", "recState"]

    /// Loads the decoder asset (`.aimodel` or `.aimodelc`) and checks `main` against the contract.
    init(contentsOf url: URL, chunk: Int, maxContext: Int, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = clefSecondsSince(t0)
        functionNames = model.functionNames
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let mainFn = try model.loadFunction(named: "main") else {
            throw ClefFlashError.contract("decoder \(url.lastPathComponent): no \"main\" (functions \(model.functionNames))")
        }
        let tMain = clefSecondsSince(t1)
        let h = Self.hidden
        try clefCheckFunction(md, what: "decoder \(url.lastPathComponent) main", inputs: [
            "input_ids": ClefTensorSpec(shape: [1, chunk], type: .int32),
            "position_ids": ClefTensorSpec(shape: [1, -1], type: .int32),
            "image_embeds": ClefTensorSpec(shape: [Self.imageRows, h], type: .float16),
            "image_rc": ClefTensorSpec(shape: [Self.imageRows, 2], type: .int32),
            "rope_shift_start": ClefTensorSpec(shape: [1], type: .int32),
            "rope_shift_amount": ClefTensorSpec(shape: [1], type: .int32),
        ], outputs: [
            "hidden": ClefTensorSpec(shape: [1, chunk, h], type: .float16),
        ], states: [
            "keyCache": ClefTensorSpec(shape: [8, 1, 4, -1, 256], type: .float16),
            "valueCache": ClefTensorSpec(shape: [8, 1, 4, -1, 256], type: .float16),
            "convState": ClefTensorSpec(shape: [24, 1, 8192, 3], type: .float16),
            "recState": ClefTensorSpec(shape: [24, 1, 32, 128, 128], type: .float16),
        ])
        func nd(input n: String) -> NDArrayDescriptor { ClefND.descriptor(md.inputDescriptor(of: n))! }
        func state(_ n: String) -> NDArray {
            let d = ClefND.descriptor(md.stateDescriptor(of: n))!
            var a = NDArray(descriptor: d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? maxContext : $0 }))
            ClefND.zero(&a)
            return a
        }
        self.url = url
        self.chunk = chunk
        self.maxContext = maxContext
        self.loadSeconds = (tModel, tMain)
        self.descriptor = clefDescribe(md)
        self.main = mainFn
        self.idsDescriptor = nd(input: "input_ids")
        self.positions = nd(input: "position_ids")
        self.embedsDescriptor = nd(input: "image_embeds")
        self.rcDescriptor = nd(input: "image_rc")
        self.shiftStartDescriptor = nd(input: "rope_shift_start")
        self.shiftAmountDescriptor = nd(input: "rope_shift_amount")
        let hd = ClefND.descriptor(md.outputDescriptor(of: "hidden"))!
        buffers = Buffers(keyCache: state("keyCache"), valueCache: state("valueCache"), convState: state("convState"),
                          recState: state("recState"), hidden: NDArray(descriptor: hd.resolvingDynamicDimensions(hd.shape)))
    }

    /// `embeds` = the image rows [N * 4096] (float32 tower output, cast to fp16 here as decide.py does; nil for a
    /// text-only row), the rest of the 1024-row buffer zero.
    func staticInputs(embeds: [Float]?, row: ClefPromptRow) throws -> StaticInputs {
        var emb = [Float16](repeating: 0, count: Self.imageRows * Self.hidden)
        if let embeds {
            guard let g = row.grid, embeds.count == g.count * Self.hidden else {
                throw ClefFlashError.contract("image_embeds: \(embeds.count) values for the row's grid \(row.grid?.description ?? "none")")
            }
            for i in 0..<embeds.count { emb[i] = Float16(embeds[i]) }
        } else if let g = row.grid {
            throw ClefFlashError.contract("an image row (\(g)) without image rows")
        }
        guard row.imageRC.count == Self.imageRows * 2 else {
            throw ClefFlashError.contract("image_rc has \(row.imageRC.count / 2) rows, the graph \(Self.imageRows)")
        }
        return StaticInputs(embeds: ClefND.make(emb, embedsDescriptor), rc: ClefND.make(row.imageRC, rcDescriptor),
                            start: ClefND.make([row.ropeShiftStart], shiftStartDescriptor),
                            amount: ClefND.make([row.ropeShiftAmount], shiftAmountDescriptor))
    }

    /// Zeroes the states and runs one row in the chunk order. One row at a time per decoder: the states are this
    /// object's.
    func run(ids: [Int32], inputs s: StaticInputs) async throws -> Pass {
        let T = ids.count
        let S = chunk
        let nCalls = (T + S - 1) / S
        guard T > 0, nCalls * S <= maxContext else {
            throw ClefFlashError.prompt("\(T) tokens (padded \(nCalls * S)) > the decoder's context \(maxContext)")
        }
        guard var b = buffers else { throw ClefFlashError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        ClefND.zero(&b.keyCache)
        ClefND.zero(&b.valueCache)
        ClefND.zero(&b.convState)
        ClefND.zero(&b.recState)
        let reset = clefSecondsSince(t0)
        var padded = [Int32](repeating: Int32(ClefPromptBuilder.padID), count: nCalls * S)
        padded.replaceSubrange(0..<T, with: ids)
        var hidden = [Float16]()
        hidden.reserveCapacity(nCalls * S * Self.hidden)
        var secs: [Double] = []
        for c in 0..<nCalls {
            let t = ContinuousClock.now
            let end = (c + 1) * S
            let inputs: [String: NDArray] = [
                "input_ids": ClefND.make(Array(padded[(c * S)..<end]), idsDescriptor),
                "position_ids": ClefND.make((0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
                "image_embeds": s.embeds, "image_rc": s.rc, "rope_shift_start": s.start, "rope_shift_amount": s.amount,
            ]
            var states = InferenceFunction.MutableViews()
            states.insert(&b.keyCache, for: "keyCache")
            states.insert(&b.valueCache, for: "valueCache")
            states.insert(&b.convState, for: "convState")
            states.insert(&b.recState, for: "recState")
            var outputs = InferenceFunction.MutableViews()
            outputs.insert(&b.hidden, for: "hidden")
            _ = try await main.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            hidden += ClefND.read(b.hidden, as: Float16.self)
            secs.append(clefSecondsSince(t))
        }
        hidden.removeLast((nCalls * S - T) * Self.hidden)
        return Pass(hidden: hidden, tokens: T, calls: nCalls, callSeconds: secs, resetSeconds: reset,
                    seconds: clefSecondsSince(t0))
    }
}

#endif
