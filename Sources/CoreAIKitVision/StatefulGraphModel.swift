// StatefulGraphModel.swift — run a STATEFUL .aimodel graph with a 2-buffer KV cache (the exact shape
// the language-model decode bundles use: two state buffers + one output). State persists and mutates
// in place across `step` calls. Companion to GraphModel (stateless). Built on the system CoreAI
// framework, mirroring the state-threading pattern of apple/coreai-models' engines (BSD-3-Clause):
// the state/output buffers are stored properties (stable storage) because InferenceFunction.MutableViews
// is non-escapable and borrows them up to `function.run`.
//
//   let dec = try await StatefulGraphModel(contentsOf: url, computeUnits: .gpu)
//   dec.resetState()
//   let out = try await dec.step(["inputs_embeds": .float16(e, shape: [1,1,1024]),
//                                 "pos": .int32([Int32(t)], shape: [1])])   // KV grows in place

import CoreAI
import Foundation

public final class StatefulGraphModel: @unchecked Sendable {
    private let function: InferenceFunction
    private let descriptor: InferenceFunctionDescriptor
    public let inputNames: [String]
    public let outputName: String

    private let keyName: String
    private let valueName: String
    private let keyDescriptor: NDArrayDescriptor
    private let valueDescriptor: NDArrayDescriptor

    private var keyCache: NDArray
    private var valueCache: NDArray
    private var outputArray: NDArray

    /// Loads a stateful `.aimodel` that declares exactly two state buffers (key/value cache) and one
    /// output. `kvCapacity` resolves any dynamic state dimension (the VoxCPM bundles bake a fixed one).
    public init(
        contentsOf url: URL, function name: String = "main",
        computeUnits: GraphModel.ComputeUnits = .gpu, kvCapacity: Int = 512
    ) async throws {
        let options: SpecializationOptions
        switch computeUnits {
        case .neuralEngine: options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .gpu: options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .cpu: options = SpecializationOptions(preferredComputeUnitKind: .cpu)
        }
        let model = try await AIModel(contentsOf: url, options: options)
        guard let descriptor = model.functionDescriptor(for: name) else {
            throw VisionError.functionNotFound(name)
        }
        guard descriptor.stateNames.count == 2 else {
            throw VisionError.statefulGraphUnsupported(descriptor.stateNames)
        }
        guard descriptor.outputNames.count == 1 else {
            throw VisionError.missingOutput("expected exactly 1 output, got \(descriptor.outputNames)")
        }
        guard let function = try model.loadFunction(named: name) else {
            throw VisionError.functionNotFound(name)
        }
        self.descriptor = descriptor
        self.function = function
        self.inputNames = descriptor.inputNames
        self.outputName = descriptor.outputNames[0]
        self.keyName = descriptor.stateNames[0]
        self.valueName = descriptor.stateNames[1]

        guard case .ndArray(let kd) = descriptor.stateDescriptor(of: keyName),
              case .ndArray(let vd) = descriptor.stateDescriptor(of: valueName),
              case .ndArray(let od) = descriptor.outputDescriptor(of: outputName)
        else { throw VisionError.unsupportedScalarType("state/output descriptor") }
        self.keyDescriptor = kd.resolvingDynamicDimensions(kd.shape.map { $0 < 0 ? kvCapacity : $0 })
        self.valueDescriptor = vd.resolvingDynamicDimensions(vd.shape.map { $0 < 0 ? kvCapacity : $0 })

        self.keyCache = NDArray(descriptor: keyDescriptor)
        self.valueCache = NDArray(descriptor: valueDescriptor)
        self.outputArray = NDArray(descriptor: od.resolvingDynamicDimensions(od.shape))
        resetState()
    }

    /// Re-allocates the KV buffers as zeros — call before generating a fresh sequence.
    public func resetState() {
        keyCache = NDArray(descriptor: keyDescriptor)
        valueCache = NDArray(descriptor: valueDescriptor)
        zero(&keyCache)
        zero(&valueCache)
    }

    /// Seed this model's KV buffers from another model's (same shape) — e.g. continue a q=1 decode
    /// from a q=T prefill bundle that shares the cache layout. One-time per utterance (cheap copy).
    public func adoptState(from other: StatefulGraphModel) {
        copyF16(other.keyCache, &keyCache)
        copyF16(other.valueCache, &valueCache)
    }

    /// Runs one step; the KV state carries over to the next call.
    public func step(_ inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        var nd: [String: NDArray] = [:]
        for (name, value) in inputs {
            guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else {
                throw VisionError.unknownInput(name)
            }
            nd[name] = try value.makeNDArray(
                descriptor: d.resolvingDynamicDimensions(value.shape), inputName: name)
        }
        var states = InferenceFunction.MutableViews()
        states.insert(&keyCache, for: keyName)
        states.insert(&valueCache, for: valueName)
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&outputArray, for: outputName)
        _ = try await function.run(
            inputs: nd, states: consume states, outputViews: consume outputViews)
        return [outputName: try TensorValue(reading: outputArray)]
    }
}

// Direct fp16->fp16 copy (no Float32 round-trip): read src into a buffer, write into dst's view.
private func copyF16(_ src: NDArray, _ dst: inout NDArray) {
    let n = dst.shape.reduce(1, *)
    var buf = [Float16](repeating: 0, count: n)
    src.view(as: Float16.self).withUnsafePointer { ptr, _, _ in
        buf.withUnsafeMutableBufferPointer { b in b.baseAddress!.update(from: ptr, count: n) }
    }
    var v = dst.mutableView(as: Float16.self)
    v.copyElements(fromContentsOf: buf)
}

private func zero(_ array: inout NDArray) {
    let n = array.shape.reduce(1, *)
    switch array.scalarType {
    case .float16:
        var v = array.mutableView(as: Float16.self); v.copyElements(fromContentsOf: [Float16](repeating: 0, count: n))
    case .float32:
        var v = array.mutableView(as: Float.self); v.copyElements(fromContentsOf: [Float](repeating: 0, count: n))
    case .int32:
        var v = array.mutableView(as: Int32.self); v.copyElements(fromContentsOf: [Int32](repeating: 0, count: n))
    default:
        break
    }
}
