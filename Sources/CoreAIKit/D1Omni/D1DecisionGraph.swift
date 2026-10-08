// From the model zoo's apps/D1Omni/Sources/D1Omni/DecisionGraph.swift (f9e0e09, sha256 57b0a9e403e4), identifiers prefixed D1 for the kit.
// D1DecisionGraph — one bucket's decision graph on the system CoreAI runtime (AIModel + loadFunction + run), and the
// graph's six inputs for a row (conversion/d1_omni/host.py §2 `graph_inputs`, `bucket_for`):
//
//   input_ids     [1,L] int32    0 on [0, P), the row's ids on [P, P+n), 0 (<|pad|>) after
//   prefix_embeds [1,L,1024] f32 the media prefix on [0, P), 0.0 elsewhere
//   pad_mask      [1,L] f32      1.0 on [0, P+n)
//   prefix_mask   [1,L] f32      1.0 on [0, P)
//   keep_right    [1,L] f32      0.0 at P-1 when P > 0, 1.0 elsewhere
//   qtype_onehot  [1,3] f32      choice / score / noul
//   -> scores     [1,L] f32      read at P + markers
//
// The asset is the bucket's AOT `.aimodelc` or its `.aimodel`, which the runtime specializes here (JIT). Either way
// the options are explicit: GPU preferred, no expectFrequentReshapes (the graphs are static; the AOT compile used
// `--preferred-compute gpu` without `--expect-frequent-reshapes`). `main`'s descriptor is checked against the contract
// at load. A text row's prefix_embeds is one zero array per graph, made at load and reused (never written).

import CoreAI
import Foundation

/// The graph's inputs for one row at bucket length L (host.py `graph_inputs`), as flat row-major arrays.
@available(macOS 27, iOS 27, *)
struct D1GraphInputs: Sendable {
    let length: Int
    let inputIDs: [Int32]
    let padMask: [Float]
    let prefixMask: [Float]
    let keepRight: [Float]
    let qtype: [Float]
    /// P + markers: the positions of the scores the host reads
    let markers: [Int]
    let prefixLength: Int

    static let inputNames = ["input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot"]

    init(row: D1Row, length L: Int) throws {
        let p = row.prefixLength, n = row.ids.count
        guard p + n <= L else { throw D1OmniError.graphLimit("a row of \(p) + \(n) positions does not fit length \(L)") }
        var ids = [Int32](repeating: 0, count: L)
        for i in 0..<n { ids[p + i] = Int32(row.ids[i]) }
        var pad = [Float](repeating: 0, count: L)
        for i in 0..<(p + n) { pad[i] = 1 }
        var prefix = [Float](repeating: 0, count: L)
        for i in 0..<p { prefix[i] = 1 }
        var keep = [Float](repeating: 1, count: L)
        if p > 0 { keep[p - 1] = 0 }
        var q = [Float](repeating: 0, count: 3)
        q[row.question.type.index] = 1
        length = L
        inputIDs = ids
        padMask = pad
        prefixMask = prefix
        keepRight = keep
        qtype = q
        markers = row.markers.map { p + $0 }
        prefixLength = p
    }

    /// host.py `bucket_for`: the smallest bucket that holds the positions, nil if none does.
    static func bucket(positions: Int, buckets: [Int]) -> Int? { buckets.sorted().first { positions <= $0 } }
}

@available(macOS 27, iOS 27, *)
final class D1DecisionGraph: @unchecked Sendable {
    let length: Int
    let url: URL
    /// "aot" (.aimodelc) or "jit" (.aimodel specialized here)
    let kind: String
    let options: SpecializationOptions
    let functionNames: [String]
    let descriptor: D1JSONValue
    /// AIModel(contentsOf:) and loadFunction(named: "main"), seconds
    let loadSeconds: (model: Double, function: Double)
    static let hidden = 1024

    private let model: AIModel
    private let main: InferenceFunction
    private let descriptors: [String: NDArrayDescriptor]
    private let zeroPrefix: NDArray

    /// GPU preferred, no frequent reshapes: the AOT compile's flags, for either asset (never `.default`, which can fall
    /// back to the CPU and hide the accelerator's numbers).
    static var gpuOptions: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .gpu) }

    init(contentsOf url: URL, length L: Int, options: SpecializationOptions = D1DecisionGraph.gpuOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = d1SecondsSince(t0)
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let fn = try model.loadFunction(named: "main") else {
            throw D1OmniError.contract("\(url.lastPathComponent): no function \"main\" (functions \(model.functionNames))")
        }
        let tMain = d1SecondsSince(t1)
        let want: [String: D1TensorSpec] = [
            "input_ids": D1TensorSpec(shape: [1, L], type: .int32),
            "prefix_embeds": D1TensorSpec(shape: [1, L, Self.hidden], type: .float32),
            "pad_mask": D1TensorSpec(shape: [1, L], type: .float32), "prefix_mask": D1TensorSpec(shape: [1, L], type: .float32),
            "keep_right": D1TensorSpec(shape: [1, L], type: .float32), "qtype_onehot": D1TensorSpec(shape: [1, 3], type: .float32),
        ]
        var bad: [String] = []
        if Set(md.inputNames) != Set(want.keys) { bad.append("inputs \(md.inputNames.sorted())") }
        for (name, spec) in want where D1TensorSpec.of(md.inputDescriptor(of: name)) != spec {
            bad.append("\(name) \(D1TensorSpec.of(md.inputDescriptor(of: name))?.description ?? "missing") != \(spec)")
        }
        if md.outputNames != ["scores"] || D1TensorSpec.of(md.outputDescriptor(of: "scores")) != D1TensorSpec(shape: [1, L], type: .float32) {
            bad.append("outputs \(md.outputNames) \(D1TensorSpec.of(md.outputDescriptor(of: "scores"))?.description ?? "") != scores [1, \(L)] float32")
        }
        if !md.stateNames.isEmpty { bad.append("states \(md.stateNames)") }
        if !bad.isEmpty { throw D1OmniError.contract("\(url.lastPathComponent) main: \(bad.joined(separator: "; "))") }
        var d: [String: NDArrayDescriptor] = [:]
        for name in D1GraphInputs.inputNames { d[name] = D1ND.descriptor(md.inputDescriptor(of: name))! }
        self.length = L
        self.url = url
        self.kind = url.pathExtension == "aimodelc" ? "aot" : "jit"
        self.options = options
        self.functionNames = model.functionNames
        self.descriptor = d1Describe(md)
        self.loadSeconds = (tModel, tMain)
        self.model = model
        self.main = fn
        self.descriptors = d
        self.zeroPrefix = D1ND.zeros(d["prefix_embeds"]!)
    }

    /// One call: the inputs (and the media prefix [P * 1024] row-major when P > 0) -> scores [L].
    func scores(_ x: D1GraphInputs, prefix: [Float]? = nil) async throws -> [Float] {
        guard x.length == length else { throw D1OmniError.contract("inputs for L = \(x.length) on the L = \(length) graph") }
        let prefixArray: NDArray
        if x.prefixLength > 0 {
            guard let prefix, prefix.count == x.prefixLength * Self.hidden else {
                throw D1OmniError.request("a row with a \(x.prefixLength)-position prefix needs \(x.prefixLength) x \(Self.hidden) values")
            }
            var full = [Float](repeating: 0, count: length * Self.hidden)
            full.replaceSubrange(0..<prefix.count, with: prefix)
            prefixArray = D1ND.make(full, descriptors["prefix_embeds"]!)
        } else {
            prefixArray = zeroPrefix
        }
        let inputs: [String: NDArray] = [
            "input_ids": D1ND.make(x.inputIDs, descriptors["input_ids"]!),
            "prefix_embeds": prefixArray,
            "pad_mask": D1ND.make(x.padMask, descriptors["pad_mask"]!),
            "prefix_mask": D1ND.make(x.prefixMask, descriptors["prefix_mask"]!),
            "keep_right": D1ND.make(x.keepRight, descriptors["keep_right"]!),
            "qtype_onehot": D1ND.make(x.qtype, descriptors["qtype_onehot"]!),
        ]
        var outputs = try await main.run(inputs: inputs)
        guard let array = outputs.remove("scores")?.ndArray else { throw D1OmniError.contract("no scores in the outputs") }
        let s = D1ND.read(array, as: Float.self)
        guard s.count == length else { throw D1OmniError.contract("\(s.count) scores for L = \(length)") }
        return s
    }
}
