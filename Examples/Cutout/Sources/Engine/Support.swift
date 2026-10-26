// Support — NDArray fill / read / zero on the system CoreAI framework (strides honored), the contract checks the
// loaders share, and the clock. From the model zoo's apps/DeciderVision/Sources/DeciderVision/Support.swift (zoo
// 2307ecf, sha256 483697ecc4ad): the error type is Cutout's, the float64 reductions are left out, and `readRow` and
// the Float16 argmax / top-2 gap are added for the decoder's logits.

import CoreAI
import Foundation

public enum CutoutError: Error, CustomStringConvertible, LocalizedError, Sendable {
    /// An asset that is not the graph this code was written for (names, shapes, types).
    case contract(String)
    case image(String)
    case prompt(String)
    /// A model file that is not where the app looks for it.
    case files(String)

    public var description: String {
        switch self {
        case .contract(let s): return "contract: \(s)"
        case .image(let s): return "image: \(s)"
        case .prompt(let s): return "prompt: \(s)"
        case .files(let s): return s
        }
    }

    public var errorDescription: String? { description }
}

@available(macOS 27, iOS 27, *)
enum ND {
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

    /// A new array of the descriptor's scalar type (float16 or float32) from float32 values: float16 rounds to nearest
    /// even, as NumPy's `astype(float16)` does.
    static func make(floats values: [Float], _ descriptor: NDArrayDescriptor) throws -> NDArray {
        switch descriptor.scalarType {
        case .float32: return make(values, descriptor)
        case .float16: return make(values.map { Float16($0) }, descriptor)
        default: throw CutoutError.contract("an input of type \(descriptor.scalarType), expected float16 or float32")
        }
    }

    /// Row-major copy of an array's scalars, honoring its strides (in elements).
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

    /// One row of a rank-3 array [1, R, C] (or rank-2 [R, C]): the C scalars at index `row` of the second-to-last axis.
    static func readRow<T: BitwiseCopyable>(_ array: NDArray, row: Int, as type: T.Type) -> [T] {
        let shape = array.shape
        let cols = shape[shape.count - 1]
        return array.view(as: T.self).withUnsafePointer { ptr, _, strides in
            let rowStride = strides[shape.count - 2]
            let colStride = strides[shape.count - 1]
            let base = ptr + row * rowStride
            if colStride == 1 { return Array(UnsafeBufferPointer(start: base, count: cols)) }
            return (0..<cols).map { base[$0 * colStride] }
        }
    }

    /// Reads float16 or float32 scalars as float32.
    static func readFloats(_ array: NDArray) throws -> [Float] {
        switch array.scalarType {
        case .float32: return read(array, as: Float.self)
        case .float16: return read(array, as: Float16.self).map { Float($0) }
        default: throw CutoutError.contract("an output of type \(array.scalarType), expected float16 or float32")
        }
    }

    /// Sets every scalar of the array (and any padding between its rows) to zero.
    static func zero(_ array: inout NDArray) {
        let shape = array.shape
        switch array.scalarType {
        case .float16: zeroScalars(&array, Float16(0), shape)
        case .float32: zeroScalars(&array, Float(0), shape)
        case .int32: zeroScalars(&array, Int32(0), shape)
        default:
            fatalError("ND.zero: unsupported scalar type \(array.scalarType)")
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

/// The contract a loaded function must meet: input / output / state names with shapes (-1 = dynamic) and types.
@available(macOS 27, iOS 27, *)
struct TensorSpec: Equatable, CustomStringConvertible {
    let shape: [Int]
    let type: NDArray.ScalarType

    var description: String { "\(shape) \(type)" }

    static func of(_ value: InferenceValue.Descriptor?) -> TensorSpec? {
        ND.descriptor(value).map { TensorSpec(shape: $0.shape, type: $0.scalarType) }
    }
}

@available(macOS 27, iOS 27, *)
func describe(_ d: InferenceFunctionDescriptor) -> [String: Any] {
    func table(_ names: [String], _ get: (String) -> InferenceValue.Descriptor?) -> [String: Any] {
        var out: [String: Any] = [:]
        for n in names {
            if let s = TensorSpec.of(get(n)) { out[n] = ["shape": s.shape, "type": "\(s.type)"] } else { out[n] = "non-ndarray" }
        }
        return out
    }
    return ["inputs": table(d.inputNames, d.inputDescriptor(of:)),
            "outputs": table(d.outputNames, d.outputDescriptor(of:)),
            "states": table(d.stateNames, d.stateDescriptor(of:))]
}

/// `SpecializationOptions` in words, for the records.
@available(macOS 27, iOS 27, *)
func describe(_ o: SpecializationOptions) -> String {
    if o == .default { return "SpecializationOptions.default" }
    let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
    return "preferred \(pref), expectFrequentReshapes \(o.expectFrequentReshapes)"
}

/// The first index of the largest value (numpy.argmax over the logits cast to float32: the same order).
func argmaxF16(_ xs: [Float16]) -> Int {
    var best = 0
    var bestValue = xs[0]
    for i in 1..<xs.count where xs[i] > bestValue {
        best = i
        bestValue = xs[i]
    }
    return best
}

/// The softmax top-2 probability gap at T = 1 in float64 (the knife-edge measure of coreai_gate.py: < 0.1 is a tie).
func top2Gap(_ xs: [Float16]) -> Double {
    var m1 = -Double.infinity, m2 = -Double.infinity
    for x in xs {
        let v = Double(x)
        if v > m1 {
            m2 = m1
            m1 = v
        } else if v > m2 {
            m2 = v
        }
    }
    var sum = 0.0
    for x in xs { sum += Foundation.exp(Double(x) - m1) }
    return (1.0 - Foundation.exp(m2 - m1)) / sum
}

func secondsSince(_ t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

/// This process's memory footprint (`phys_footprint`, what jetsam counts) and its lifetime maximum, in bytes.
func memoryFootprint() -> (now: UInt64, peak: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return (0, 0) }
    return (info.phys_footprint, UInt64(max(0, info.ledger_phys_footprint_peak)))
}
