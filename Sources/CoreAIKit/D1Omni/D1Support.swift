// From the model zoo's apps/D1Omni/Sources/D1Omni/Support.swift (f9e0e09, sha256 6b7746215ea3), identifiers prefixed D1 for the kit.
// Support — the error type, NDArray fill / read on the system CoreAI framework (strides honored), the contract check a
// loaded function must pass, and small helpers. `D1ND`, `D1TensorSpec`, `d1Describe` and `d1SecondsSince` (the zoo's `ND`,
// `TensorSpec`, `describe`, `secondsSince`) are copied from
// apps/Kev/Sources/Kev/Support.swift (zoo main 9e06b5a, itself from apps/ClefFlash).

import CoreAI
import Darwin
import Foundation

public enum D1OmniError: Error, CustomStringConvertible, Sendable {
    /// the asset or the bundle does not match the contract the host was written for
    case contract(String)
    /// the request is not one the publisher's code accepts (its ValueError, with its message)
    case request(String)
    /// a row does not fit the largest decision graph
    case graphLimit(String)
    case bundle(String)
    case json(String)

    public var description: String {
        switch self {
        case .contract(let s): return "contract: \(s)"
        case .request(let s): return "request: \(s)"
        case .graphLimit(let s): return "graph limit: \(s)"
        case .bundle(let s): return "bundle: \(s)"
        case .json(let s): return "json: \(s)"
        }
    }
}

@available(macOS 27, iOS 27, *)
enum D1ND {
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

    /// A float32 array of `descriptor` with every scalar (and any padding between rows) set to zero.
    static func zeros(_ descriptor: NDArrayDescriptor) -> NDArray {
        var array = NDArray(descriptor: descriptor)
        let shape = array.shape
        array.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, strides in
            ptr.update(repeating: 0, count: extent(shape, strides))
        }
        return array
    }

    /// The elements an array spans from its first scalar to its last, padding included.
    static func extent(_ shape: [Int], _ strides: Span<Int>) -> Int {
        var e = 1
        for d in 0..<shape.count where shape[d] > 0 { e += (shape[d] - 1) * strides[d] }
        return e
    }
}

/// The contract a loaded function must meet: input / output names with shapes and types.
@available(macOS 27, iOS 27, *)
struct D1TensorSpec: Equatable, CustomStringConvertible {
    let shape: [Int]
    let type: NDArray.ScalarType

    var description: String { "\(shape) \(type)" }

    static func of(_ value: InferenceValue.Descriptor?) -> D1TensorSpec? {
        D1ND.descriptor(value).map { D1TensorSpec(shape: $0.shape, type: $0.scalarType) }
    }
}

@available(macOS 27, iOS 27, *)
func d1Describe(_ d: InferenceFunctionDescriptor) -> D1JSONValue {
    func table(_ names: [String], _ get: (String) -> InferenceValue.Descriptor?) -> D1JSONValue {
        .object(names.map { n in
            if let s = D1TensorSpec.of(get(n)) {
                return D1JSONMember(n, .object([D1JSONMember("shape", .array(s.shape.map { .int($0) })),
                                              D1JSONMember("type", .string("\(s.type)"))]))
            }
            return D1JSONMember(n, .string("non-ndarray"))
        })
    }
    return .object([D1JSONMember("inputs", table(d.inputNames, d.inputDescriptor(of:))),
                    D1JSONMember("outputs", table(d.outputNames, d.outputDescriptor(of:))),
                    D1JSONMember("states", table(d.stateNames, d.stateDescriptor(of:)))])
}

@available(macOS 27, iOS 27, *)
func d1Describe(_ o: SpecializationOptions) -> String {
    if o == .default { return "SpecializationOptions.default" }
    if o == .cpuOnly { return "SpecializationOptions.cpuOnly" }
    let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
    return "preferred \(pref), allowed \(o.allowedComputeUnitKinds.map { "\($0)" }.sorted()), "
        + "expectFrequentReshapes \(o.expectFrequentReshapes)"
}

@available(macOS 27, iOS 27, *)
func d1SecondsSince(_ t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

/// Python's `a // b` for ints (floor division).
@available(macOS 27, iOS 27, *)
func d1FloorDiv(_ a: Int, _ b: Int) -> Int {
    let q = a / b
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q
}
