// GraphModel.swift — run any stateless .aimodel graph (CV encoders, detectors, embedders)
// with dictionary-in / dictionary-out tensors. Built directly on the system CoreAI
// framework, following the public-API usage of apple/coreai-models (BSD-3-Clause).

import CoreAI
import Foundation

/// A loaded, specialized graph function.
///
/// ```swift
/// let model = try await GraphModel(contentsOf: aimodelURL, computeUnits: .neuralEngine)
/// let out = try await model.run(["pixel_values": .float32(pixels, shape: [1, 3, 224, 224])])
/// ```
public final class GraphModel: @unchecked Sendable {
    public enum ComputeUnits: Sendable {
        /// `neuralEngine`/`gpu`/`cpu` express a *preference* over the full allowed set
        /// (`[cpu, gpu, neuralEngine]`); `cpuOnly` collapses the allowed set to the CPU alone.
        /// The distinction is load-bearing: a preference over a heterogeneous allowed set is the
        /// trigger for a Core AI placement defect that returns silently wrong numerics on some
        /// graphs (filed with Apple; see `knowledge/pocket-tts-port.md`). Parity and gate runs
        /// should use `cpuOnly`, never `cpu`.
        case neuralEngine, gpu, cpu, cpuOnly

        /// The Core AI options this choice lowers to. Exposed so callers that hold a raw
        /// `AIModel` — the graphs whose shape `GraphModel`/`StatefulGraphModel` cannot
        /// accept — specialize identically instead of re-deriving the mapping.
        public var specializationOptions: SpecializationOptions {
            switch self {
            case .neuralEngine: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
            case .cpu: return SpecializationOptions(preferredComputeUnitKind: .cpu)
            case .cpuOnly: return SpecializationOptions.cpuOnly
            }
        }
    }

    private let function: InferenceFunction
    private let descriptor: InferenceFunctionDescriptor
    public let inputNames: [String]
    public let outputNames: [String]

    /// Loads and specializes a `.aimodel` (first load compiles on-device; cached afterwards).
    public init(
        contentsOf url: URL, function name: String = "main",
        computeUnits: ComputeUnits = .gpu
    ) async throws {
        let model = try await AIModel(contentsOf: url, options: computeUnits.specializationOptions)
        guard let descriptor = model.functionDescriptor(for: name) else {
            throw VisionError.functionNotFound(name)
        }
        guard descriptor.stateNames.isEmpty else {
            throw VisionError.statefulGraphUnsupported(descriptor.stateNames)
        }
        guard let function = try model.loadFunction(named: name) else {
            throw VisionError.functionNotFound(name)
        }
        self.descriptor = descriptor
        self.function = function
        self.inputNames = descriptor.inputNames
        self.outputNames = descriptor.outputNames
    }

    /// Declared input shape (dynamic dimensions are negative), or nil for unknown names.
    public func shape(ofInput name: String) -> [Int]? {
        guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else { return nil }
        return d.shape
    }

    /// Declared output shape (dynamic dimensions are negative), or nil for unknown names.
    public func shape(ofOutput name: String) -> [Int]? {
        guard case .ndArray(let d) = descriptor.outputDescriptor(of: name) else { return nil }
        return d.shape
    }

    /// Runs one inference. Every declared input must be provided; scalar types convert
    /// automatically between float16/float32 where needed.
    public func run(_ inputs: [String: TensorValue]) async throws -> [String: TensorValue] {
        var ndInputs: [String: NDArray] = [:]
        for (name, value) in inputs {
            guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else {
                throw VisionError.unknownInput(name)
            }
            guard d.shape.count == value.shape.count,
                zip(d.shape, value.shape).allSatisfy({ $0 < 0 || $0 == $1 })
            else {
                throw VisionError.shapeMismatch(
                    input: name, expected: d.shape, got: value.shape)
            }
            let resolved = d.resolvingDynamicDimensions(value.shape)
            ndInputs[name] = try value.makeNDArray(descriptor: resolved, inputName: name)
        }

        var rawOutputs = try await function.run(inputs: ndInputs)
        var outputs: [String: TensorValue] = [:]
        for name in outputNames {
            guard let array = rawOutputs.remove(name)?.ndArray else {
                throw VisionError.missingOutput(name)
            }
            outputs[name] = try TensorValue(reading: array)
        }
        return outputs
    }
}
