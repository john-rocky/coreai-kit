// From the model zoo's apps/ClefFlash/Sources/ClefFlash/Support.swift (5ef2247, sha256 a96745ace5cd, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// Support — NDArray fill/read/zero on the system CoreAI framework (strides honored), the contract checks the loaders
// share, and the NumPy float32 reductions the reference numbers go through (pairwise sum, the per-question softmax),
// so a probability here is the same float32 the Python reference (`clef_head.question_probs`) computes from the same
// logits.

import CoreAI
import Darwin
import Foundation

public enum ClefFlashError: Error, CustomStringConvertible, LocalizedError, Sendable {
    case contract(String)
    case request(String)
    case image(String)
    case prompt(String)
    case bundle(String)
    case json(String)

    public var description: String {
        switch self {
        case .contract(let s): return "contract: \(s)"
        case .request(let s): return "request: \(s)"
        case .image(let s): return "image: \(s)"
        case .prompt(let s): return "prompt: \(s)"
        case .bundle(let s): return "bundle: \(s)"
        case .json(let s): return "json: \(s)"
        }
    }

    public var errorDescription: String? { "clef-flash \(description)" }
}

@available(macOS 27, iOS 27, *)
enum ClefND {
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
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case .float16: zeroScalars(&array, Float16(0), shape)
        #endif
        case .float32: zeroScalars(&array, Float(0), shape)
        case .int32: zeroScalars(&array, Int32(0), shape)
        default:
            fatalError("ClefND.zero: unsupported scalar type \(array.scalarType)")
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
struct ClefTensorSpec: Equatable, CustomStringConvertible {
    let shape: [Int]
    let type: NDArray.ScalarType

    var description: String { "\(shape) \(type)" }

    static func of(_ value: InferenceValue.Descriptor?) -> ClefTensorSpec? {
        ClefND.descriptor(value).map { ClefTensorSpec(shape: $0.shape, type: $0.scalarType) }
    }
}

/// Checks a function's inputs, outputs and states against `want` (names exactly, shapes and types per name).
@available(macOS 27, iOS 27, *)
func clefCheckFunction(_ d: InferenceFunctionDescriptor, what: String, inputs: [String: ClefTensorSpec],
                   outputs: [String: ClefTensorSpec], states: [String: ClefTensorSpec]) throws
{
    var bad: [String] = []
    for (part, names, want, get) in [
        ("inputs", d.inputNames, inputs, d.inputDescriptor(of:)),
        ("outputs", d.outputNames, outputs, d.outputDescriptor(of:)),
        ("states", d.stateNames, states, d.stateDescriptor(of:)),
    ] as [(String, [String], [String: ClefTensorSpec], (String) -> InferenceValue.Descriptor?)] {
        if Set(names) != Set(want.keys) {
            bad.append("\(part) \(names.sorted()) != \(want.keys.sorted())")
            continue
        }
        for (n, w) in want.sorted(by: { $0.key < $1.key }) where ClefTensorSpec.of(get(n)) != w {
            bad.append("\(part) \(n): \(ClefTensorSpec.of(get(n))?.description ?? "not an ndarray") != \(w)")
        }
    }
    if !bad.isEmpty { throw ClefFlashError.contract("\(what): \(bad.joined(separator: "; "))") }
}

@available(macOS 27, iOS 27, *)
func clefDescribe(_ d: InferenceFunctionDescriptor) -> ClefJSON {
    func table(_ names: [String], _ get: (String) -> InferenceValue.Descriptor?) -> ClefJSON {
        .object(names.map { n in
            if let s = ClefTensorSpec.of(get(n)) {
                return ClefJSONMember(n, .object([ClefJSONMember("shape", .array(s.shape.map { .int($0) })),
                                              ClefJSONMember("type", .string("\(s.type)"))]))
            }
            return ClefJSONMember(n, .string("non-ndarray"))
        })
    }
    return .object([ClefJSONMember("inputs", table(d.inputNames, d.inputDescriptor(of:))),
                    ClefJSONMember("outputs", table(d.outputNames, d.outputDescriptor(of:))),
                    ClefJSONMember("states", table(d.stateNames, d.stateDescriptor(of:)))])
}

@available(macOS 27, iOS 27, *)
func clefDescribe(_ o: SpecializationOptions) -> String {
    if o == .default { return "SpecializationOptions.default" }
    let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
    return "preferred \(pref), allowed \(o.allowedComputeUnitKinds.map { "\($0)" }.sorted()), "
        + "expectFrequentReshapes \(o.expectFrequentReshapes)"
}

// MARK: - NumPy float32 reductions

/// NumPy's float32 `add.reduce` over a contiguous 1-D array (`FLOAT_pairwise_sum`: a plain loop below 8 elements, 8
/// accumulators up to 128, halves above), accumulated in float32 like NumPy does for a float32 input.
func clefNumpyPairwiseSum(_ a: [Float]) -> Float {
    a.withUnsafeBufferPointer { pairwise($0.baseAddress!, a.count) }
}

private func pairwise(_ a: UnsafePointer<Float>, _ n: Int) -> Float {
    if n < 8 {
        var res: Float = -0.0
        for i in 0..<n { res += a[i] }
        return res
    } else if n <= 128 {
        var r = (0..<8).map { a[$0] }
        var i = 8
        while i < n - (n % 8) {
            for j in 0..<8 { r[j] += a[i + j] }
            i += 8
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n {
            res += a[i]
            i += 1
        }
        return res
    } else {
        var n2 = n / 2
        n2 -= n2 % 8
        return pairwise(a, n2) + pairwise(a + n2, n - n2)
    }
}

/// `clef_head.question_probs` for one question: float32 `exp(z - max(z)) / sum` (the author's
/// `question_logits.float().softmax(-1)`), NumPy's operation order: the max, the float32 differences, `expf`, the
/// pairwise float32 sum (added to NumPy's 0.0 identity), one float32 division per option.
func clefSoftmaxFloat32(_ z: [Float]) -> [Float] {
    guard var m = z.first else { return [] }
    for v in z.dropFirst() where v > m || v.isNaN { m = v }
    let e = z.map { expf($0 - m) }
    let s: Float = 0.0 + clefNumpyPairwiseSum(e)
    return e.map { $0 / s }
}

/// The first index of the largest value (numpy.clefArgmax / Python's max over a sequence).
func clefArgmax<T: Comparable>(_ xs: [T]) -> Int {
    var best = 0
    for i in 1..<xs.count where xs[i] > xs[best] { best = i }
    return best
}

func clefSecondsSince(_ t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}
