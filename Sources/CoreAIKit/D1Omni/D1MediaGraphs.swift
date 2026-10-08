// From the model zoo's apps/D1Omni/Sources/D1Omni/MediaGraphs.swift (f9e0e09, sha256 07fe8e68db3c), identifiers prefixed D1 for the kit.
// MediaGraphs — the vision and audio graphs on the system CoreAI runtime (AIModel + loadFunction("main") + run), each
// with the contract of its bundle's metadata.json checked at load:
//
//   vision  pixel_values [1,1024,768] f32, pos_embed [1,1024,768] f32, patch_mask [1,1024] f32,
//           unshuffle_index [256,4] int32                                   -> prefix [1,256,1024] f32
//   audio   mel [1,128,F] f32, mask_f [1,F], mask_f2 [1,F2], mask_f4 [1,F4], mask_t [1,T] f32
//                                                                           -> prefix [1,T,1024] f32
//           one bundle per clip bucket: sec 5 / 10 / 20 / 30 -> F 501 / 1001 / 2001 / 3001, T 63 / 126 / 251 / 376
//
// As with the decision graph, the asset is the bundle's `.aimodel` (the runtime specializes it at load: JIT) or the
// Mac's AOT `.aimodelc` of the same bundle, always with SpecializationOptions(preferredComputeUnitKind: .gpu).

import CoreAI
import Foundation

/// One loaded media graph: `main`'s inputs checked against `contract`, one call at a time.
@available(macOS 27, iOS 27, *)
final class D1MediaGraph: @unchecked Sendable {
    enum Input: Sendable {
        case float([Float])
        case int32([Int32])
    }

    let url: URL
    /// "aot" (.aimodelc) or "jit" (.aimodel specialized here)
    let kind: String
    let options: SpecializationOptions
    let descriptor: D1JSONValue
    /// AIModel(contentsOf:) and loadFunction(named: "main"), seconds
    let loadSeconds: (model: Double, function: Double)
    let output: String
    let outputShape: [Int]

    private let model: AIModel
    private let main: InferenceFunction
    private let descriptors: [String: NDArrayDescriptor]
    private let specs: [String: D1TensorSpec]

    init(contentsOf url: URL, inputs want: [String: D1TensorSpec], output: (String, D1TensorSpec),
         options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = d1SecondsSince(t0)
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let fn = try model.loadFunction(named: "main") else {
            throw D1OmniError.contract("\(url.lastPathComponent): no function \"main\" (functions \(model.functionNames))")
        }
        let tMain = d1SecondsSince(t1)
        var bad: [String] = []
        if Set(md.inputNames) != Set(want.keys) { bad.append("inputs \(md.inputNames.sorted()) != \(want.keys.sorted())") }
        for (name, spec) in want where D1TensorSpec.of(md.inputDescriptor(of: name)) != spec {
            bad.append("\(name) \(D1TensorSpec.of(md.inputDescriptor(of: name))?.description ?? "missing") != \(spec)")
        }
        if md.outputNames != [output.0] || D1TensorSpec.of(md.outputDescriptor(of: output.0)) != output.1 {
            bad.append("outputs \(md.outputNames) \(D1TensorSpec.of(md.outputDescriptor(of: output.0))?.description ?? "") != \(output.0) \(output.1)")
        }
        if !md.stateNames.isEmpty { bad.append("states \(md.stateNames)") }
        if !bad.isEmpty { throw D1OmniError.contract("\(url.lastPathComponent) main: \(bad.joined(separator: "; "))") }
        var d: [String: NDArrayDescriptor] = [:]
        for name in want.keys { d[name] = D1ND.descriptor(md.inputDescriptor(of: name))! }
        self.url = url
        self.kind = url.pathExtension == "aimodelc" ? "aot" : "jit"
        self.options = options
        self.descriptor = d1Describe(md)
        self.loadSeconds = (tModel, tMain)
        self.output = output.0
        self.outputShape = output.1.shape
        self.model = model
        self.main = fn
        self.descriptors = d
        self.specs = want
    }

    /// One call -> the output, flat row-major float32.
    func run(_ inputs: [String: Input]) async throws -> [Float] {
        var arrays: [String: NDArray] = [:]
        for (name, spec) in specs {
            guard let x = inputs[name] else { throw D1OmniError.request("missing input \(name)") }
            let count = spec.shape.reduce(1, *)
            switch x {
            case .float(let v):
                guard spec.type == .float32, v.count == count else { throw D1OmniError.request("\(name): \(v.count) float32 for \(spec)") }
                arrays[name] = D1ND.make(v, descriptors[name]!)
            case .int32(let v):
                guard spec.type == .int32, v.count == count else { throw D1OmniError.request("\(name): \(v.count) int32 for \(spec)") }
                arrays[name] = D1ND.make(v, descriptors[name]!)
            }
        }
        var outputs = try await main.run(inputs: arrays)
        guard let array = outputs.remove(output)?.ndArray else { throw D1OmniError.contract("no \(output) in the outputs") }
        let out = D1ND.read(array, as: Float.self)
        guard out.count == outputShape.reduce(1, *) else { throw D1OmniError.contract("\(out.count) values of \(output) for \(outputShape)") }
        return out
    }
}

/// The vision graph: one crop -> its prefix rows.
@available(macOS 27, iOS 27, *)
final class D1VisionGraph: @unchecked Sendable {
    static let names = ["pixel_values", "pos_embed", "patch_mask", "unshuffle_index"]
    let graph: D1MediaGraph

    init(contentsOf url: URL, options: SpecializationOptions = D1DecisionGraph.gpuOptions) async throws {
        let p = D1ImagePreprocess.maxPatches, d = D1ImagePreprocess.patchDim
        graph = try await D1MediaGraph(
            contentsOf: url,
            inputs: ["pixel_values": D1TensorSpec(shape: [1, p, d], type: .float32),
                     "pos_embed": D1TensorSpec(shape: [1, p, D1ImagePreprocess.hidden], type: .float32),
                     "patch_mask": D1TensorSpec(shape: [1, p], type: .float32),
                     "unshuffle_index": D1TensorSpec(shape: [D1ImagePreprocess.maxTokens, 4], type: .int32)],
            output: ("prefix", D1TensorSpec(shape: [1, D1ImagePreprocess.maxTokens, D1DecisionGraph.hidden], type: .float32)),
            options: options)
    }

    /// The graph's whole output [256 * 1024] for one crop's inputs.
    func output(_ x: D1CropInputs) async throws -> [Float] {
        try await graph.run(["pixel_values": .float(x.pixelValues), "pos_embed": .float(x.posEmbed),
                             "patch_mask": .float(x.patchMask), "unshuffle_index": .int32(x.unshuffleIndex)])
    }

    /// The crop's prefix: the first (ph / 2)(pw / 2) rows of the output, [tokens * 1024].
    func prefix(_ x: D1CropInputs) async throws -> [Float] {
        Array(try await output(x).prefix(x.tokens * D1DecisionGraph.hidden))
    }
}

/// The audio graph of one clip bucket: one clip's mel and masks -> its prefix rows.
@available(macOS 27, iOS 27, *)
final class D1AudioGraph: @unchecked Sendable {
    static let names = ["mel", "mask_f", "mask_f2", "mask_f4", "mask_t"]
    let graph: D1MediaGraph
    let bucket: D1AudioBucket

    init(contentsOf url: URL, bucket: D1AudioBucket, options: SpecializationOptions = D1DecisionGraph.gpuOptions) async throws {
        self.bucket = bucket
        graph = try await D1MediaGraph(
            contentsOf: url,
            inputs: ["mel": D1TensorSpec(shape: [1, D1AudioPreprocess.features, bucket.F], type: .float32),
                     "mask_f": D1TensorSpec(shape: [1, bucket.F], type: .float32),
                     "mask_f2": D1TensorSpec(shape: [1, bucket.F2], type: .float32),
                     "mask_f4": D1TensorSpec(shape: [1, bucket.F4], type: .float32),
                     "mask_t": D1TensorSpec(shape: [1, bucket.T], type: .float32)],
            output: ("prefix", D1TensorSpec(shape: [1, bucket.T, D1DecisionGraph.hidden], type: .float32)),
            options: options)
    }

    /// The graph's whole output [T * 1024] for one clip's inputs (at this bucket).
    func output(_ x: D1AudioInputs) async throws -> [Float] {
        guard x.bucket == bucket else { throw D1OmniError.request("inputs of the \(x.bucket.seconds) s bucket on the \(bucket.seconds) s graph") }
        return try await graph.run(["mel": .float(x.mel), "mask_f": .float(x.maskF), "mask_f2": .float(x.maskF2),
                                    "mask_f4": .float(x.maskF4), "mask_t": .float(x.maskT)])
    }

    /// The clip's prefix: the first P rows of the output, [P * 1024].
    func prefix(_ x: D1AudioInputs) async throws -> [Float] {
        Array(try await output(x).prefix(x.prefixRows * D1DecisionGraph.hidden))
    }
}
