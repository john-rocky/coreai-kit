// From the model zoo's apps/Kev/Sources/Kev/Support.swift (9e06b5a, sha256 1e6fe7148f16), identifiers prefixed Kev for the kit.
// Support — the error type, NDArray fill / read / zero / copy on the system CoreAI framework (strides honored), and
// the contract check the loader runs on a function's descriptor.
// `KevND`, `KevTensorSpec`, `kevDescribe` and `kevSecondsSince` are copied from apps/ClefFlash/Sources/ClefFlash/Support.swift
// (zoo main 5ef2247); `KevND.copy` and `functionContract` are new.

import CoreAI
import Darwin
import Foundation

public enum KevError: Error, CustomStringConvertible, Sendable {
    /// the asset or the bundle does not match the contract the host was written for
    case contract(String)
    /// the request is not a valid SystemOne request (the author's server answers 422)
    case request(String)
    /// the request is over a serving limit (kev.model.ContextOverflow; the author's server answers 422)
    case contextOverflow(String)
    /// a row does not fit the exported graph (ceil(T / S) * S > max_context_length - 1)
    case graphLimit(String)
    case bundle(String)
    case json(String)

    public var description: String {
        switch self {
        case .contract(let s): return "contract: \(s)"
        case .request(let s): return "request: \(s)"
        case .contextOverflow(let s): return "context overflow: \(s)"
        case .graphLimit(let s): return "graph limit: \(s)"
        case .bundle(let s): return "bundle: \(s)"
        case .json(let s): return "json: \(s)"
        }
    }
}

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: the code below
// is Apple silicon only, and KitKevDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
@available(macOS 27, iOS 27, *)
enum KevND {
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

    /// Sets every scalar of the array (and any padding between its rows) to zero.
    static func zero(_ array: inout NDArray) {
        let shape = array.shape
        switch array.scalarType {
        case .float16: zeroScalars(&array, Float16(0), shape)
        case .float32: zeroScalars(&array, Float(0), shape)
        case .int32: zeroScalars(&array, Int32(0), shape)
        default:
            fatalError("KevND.zero: unsupported scalar type \(array.scalarType)")
        }
    }

    private static func zeroScalars<T: BitwiseCopyable>(_ array: inout NDArray, _ zero: T, _ shape: [Int]) {
        array.mutableView(as: T.self).withUnsafeMutablePointer { ptr, _, strides in
            ptr.update(repeating: zero, count: extent(shape, strides))
        }
    }

    /// The elements an array spans from its first scalar to its last, padding included.
    static func extent(_ shape: [Int], _ strides: Span<Int>) -> Int {
        var e = 1
        for d in 0..<shape.count where shape[d] > 0 { e += (shape[d] - 1) * strides[d] }
        return e
    }

    /// Copies `source` into `dest`, scalar for scalar and padding included: two arrays of one descriptor (the state
    /// snapshot of the shared prefix). Both are fp16 here; the element type only sets the copy's unit.
    static func copy(_ source: NDArray, into dest: inout NDArray) {
        precondition(source.shape == dest.shape && source.scalarType == dest.scalarType && source.strides == dest.strides,
                     "KevND.copy: arrays of different descriptors")
        let shape = source.shape
        source.view(as: Float16.self).withUnsafePointer { src, _, strides in
            let n = extent(shape, strides)
            dest.mutableView(as: Float16.self).withUnsafeMutablePointer { dst, _, _ in
                dst.update(from: src, count: n)
            }
        }
    }
}

/// The contract a loaded function must meet: input / output / state names with shapes (-1 = dynamic) and types.
@available(macOS 27, iOS 27, *)
struct KevTensorSpec: Equatable, CustomStringConvertible {
    let shape: [Int]
    let type: NDArray.ScalarType

    var description: String { "\(shape) \(type)" }

    static func of(_ value: InferenceValue.Descriptor?) -> KevTensorSpec? {
        KevND.descriptor(value).map { KevTensorSpec(shape: $0.shape, type: $0.scalarType) }
    }
}

@available(macOS 27, iOS 27, *)
func kevDescribe(_ d: InferenceFunctionDescriptor) -> KevJSON {
    func table(_ names: [String], _ get: (String) -> InferenceValue.Descriptor?) -> KevJSON {
        .object(names.map { n in
            if let s = KevTensorSpec.of(get(n)) {
                return KevJSONMember(n, .object([KevJSONMember("shape", .array(s.shape.map { .int($0) })),
                                              KevJSONMember("type", .string("\(s.type)"))]))
            }
            return KevJSONMember(n, .string("non-ndarray"))
        })
    }
    return .object([KevJSONMember("inputs", table(d.inputNames, d.inputDescriptor(of:))),
                    KevJSONMember("outputs", table(d.outputNames, d.outputDescriptor(of:))),
                    KevJSONMember("states", table(d.stateNames, d.stateDescriptor(of:)))])
}

@available(macOS 27, iOS 27, *)
func kevDescribe(_ o: SpecializationOptions) -> String {
    if o == .default { return "SpecializationOptions.default" }
    let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
    return "preferred \(pref), allowed \(o.allowedComputeUnitKinds.map { "\($0)" }.sorted()), "
        + "expectFrequentReshapes \(o.expectFrequentReshapes)"
}

func kevSecondsSince(_ t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

#endif
