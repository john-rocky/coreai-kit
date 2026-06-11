// TensorValue.swift — a minimal host-side tensor for graph inputs/outputs. Copies in/out of
// the runtime's NDArray; fine at CV sizes. The NDArray bridge follows the public-API usage
// patterns of apple/coreai-models (BSD-3-Clause), see NOTICE.txt.

import CoreAI
import Foundation

public struct TensorValue: Sendable, Equatable {
    enum Storage: Equatable {
        case float16([Float16])
        case float32([Float])
        case int32([Int32])
    }

    let storage: Storage
    public let shape: [Int]

    public var count: Int { shape.reduce(1, *) }

    public static func float16(_ scalars: [Float16], shape: [Int]) -> TensorValue {
        precondition(scalars.count == shape.reduce(1, *), "scalar count must match shape")
        return TensorValue(storage: .float16(scalars), shape: shape)
    }

    public static func float32(_ scalars: [Float], shape: [Int]) -> TensorValue {
        precondition(scalars.count == shape.reduce(1, *), "scalar count must match shape")
        return TensorValue(storage: .float32(scalars), shape: shape)
    }

    public static func int32(_ scalars: [Int32], shape: [Int]) -> TensorValue {
        precondition(scalars.count == shape.reduce(1, *), "scalar count must match shape")
        return TensorValue(storage: .int32(scalars), shape: shape)
    }

    /// Converting accessor: the scalars as Float, row-major.
    public func floats() -> [Float] {
        switch storage {
        case .float16(let v): return v.map(Float.init)
        case .float32(let v): return v
        case .int32(let v): return v.map(Float.init)
        }
    }
}

// MARK: - NDArray bridge (internal)

extension TensorValue {
    /// Fills a fresh NDArray for the (resolved) descriptor, converting scalar type as needed.
    func makeNDArray(descriptor: NDArrayDescriptor, inputName: String) throws -> NDArray {
        var array = NDArray(descriptor: descriptor)
        switch (storage, descriptor.scalarType) {
        case (.float16(let v), .float16): fill(&array, with: v)
        case (.float32(let v), .float16): fill(&array, with: v.map(Float16.init))
        case (.float16(let v), .float32): fill(&array, with: v.map(Float.init))
        case (.float32(let v), .float32): fill(&array, with: v)
        case (.int32(let v), .int32): fill(&array, with: v)
        default:
            throw VisionError.dtypeMismatch(
                input: inputName, expected: "\(descriptor.scalarType)")
        }
        return array
    }

    /// Reads an output NDArray into host storage, branching on its own scalar type.
    init(reading array: NDArray) throws {
        let shape = array.shape
        let count = shape.reduce(1, *)
        switch array.scalarType {
        case .float16:
            self.init(storage: .float16(read(array, Float16.self, count)), shape: shape)
        case .float32:
            self.init(storage: .float32(read(array, Float.self, count)), shape: shape)
        case .int32:
            self.init(storage: .int32(read(array, Int32.self, count)), shape: shape)
        default:
            throw VisionError.unsupportedScalarType("\(array.scalarType)")
        }
    }
}

private func fill<T: BitwiseCopyable>(_ array: inout NDArray, with values: [T]) {
    var view = array.mutableView(as: T.self)
    view.copyElements(fromContentsOf: values)
}

private func read<T: BitwiseCopyable>(_ array: NDArray, _ type: T.Type, _ count: Int) -> [T] {
    array.view(as: T.self).withUnsafePointer { ptr, _, _ in
        Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
