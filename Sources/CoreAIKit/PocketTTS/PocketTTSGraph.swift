// PocketTTSGraph.swift — the raw-CoreAI layer this port needs instead of `GraphModel`.
//
// `GraphModel` rejects any graph carrying state, and `StatefulGraphModel` accepts exactly two
// state buffers, exactly one output, and seeds/copies them through fp16-only helpers. Every
// graph here misses on at least one count: the flow-LM `step` emits two outputs (`cond` and
// `eos_logit`), the Mimi decoder threads twelve state tensors of which one is int32 and the
// rest fp32, and the flow-LM bundle is multifunction. Same situation, and the same answer, as
// `DocDecoder` in `OCR/KitDocReader.swift`: hold the `AIModel` and its functions locally, and
// keep the kit-shaped surface at the `PocketTTS` type above this.

import CoreAI
import CoreAIKitVision
import Foundation

/// Short local name for the marshalling namespace defined at the bottom of this file.
private typealias ND = PocketTTSND

enum PocketTTSError: Error, CustomStringConvertible {
    case message(String)
    var description: String { switch self { case .message(let m): return m } }
}

/// One loaded `.aimodel` bundle, holding the `AIModel` so `function(_:)` can vend several
/// entry points from it. The flow-LM asset is **multifunction** — `prefill` and `step` over
/// one weight set — and loading the bundle once is the point: the two functions share their
/// KV cache through host-owned `NDArray`s handed over as `MutableViews`, so it is the buffers
/// that carry the state, not the `AIModel` identity. What a second load would duplicate is
/// the weights.
final class PocketTTSAsset {
    let url: URL
    let unit: GraphModel.ComputeUnits
    let model: AIModel
    let loadSeconds: Double

    init(url: URL, unit: GraphModel.ComputeUnits) async throws {
        self.url = url
        self.unit = unit
        let t0 = ND.nowNanos()
        self.model = try await AIModel(contentsOf: url, options: unit.specializationOptions)
        self.loadSeconds = Double(ND.nowNanos() - t0) / 1e9
    }

    var functionNames: [String] { model.functionNames }

    func function(_ name: String) throws -> InferenceFunction {
        guard let fn = try model.loadFunction(named: name) else {
            throw PocketTTSError.message("\(url.lastPathComponent): no function '\(name)'")
        }
        return fn
    }
}

// MARK: - NDArray helpers

/// Marshalling helpers for this port, namespaced rather than left at module scope.
///
/// `nd`, `flat` and `take` are the names the *next* port will want too, and everything in
/// `CoreAIKit` shares one module namespace, so at file scope the first port to land simply
/// takes them. A caseless `enum` is Swift's idiom for a namespace: it has no cases, so it
/// cannot be instantiated, and it exists only to qualify the members inside it. Consumers
/// shorten it back with a file-private `typealias ND = PocketTTSND`, which keeps call sites
/// as readable as they were while leaving the bare names free for everyone else.
enum PocketTTSND {
    @inline(__always) static func nowNanos() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    }

    /// Build a float32 NDArray from a Swift array.
    static func nd(_ values: [Float], _ shape: [Int]) -> NDArray {
        NDArray(scalars: values, shape: shape)
    }

    static func nd(_ values: [Int32], _ shape: [Int]) -> NDArray {
        NDArray(scalars: values, shape: shape)
    }

    /// Build a float16 NDArray by narrowing float32 values. Used when the flow-LM / flow
    /// decoder assets are the fp16 export — the graph's declared input dtype must be matched
    /// exactly or the runtime rejects the call.
    static func ndHalf(_ values: [Float], _ shape: [Int]) -> NDArray {
        NDArray(scalars: values.map { Float16($0) }, shape: shape)
    }

    /// Allocate an NDArray of the given scalar type and fill it from float32 source data.
    ///
    /// This is how the KV state gets created: a *state* is a buffer the runtime mutates in
    /// place across calls; the host owns it, hands the runtime a mutable view each call, and
    /// never reads it back. Zero-initialised, not NaN — a masked SDPA still multiplies V by a
    /// zero weight and `0 * NaN` is `NaN` (NOTES.md §6, blocker iii-b).
    static func makeState(_ values: [Float], shape: [Int], half: Bool) -> NDArray {
        var a = NDArray(shape: shape, scalarType: half ? .float16 : .float32)
        if half {
            var mv = a.mutableView(as: Float16.self)
            mv.withUnsafeMutablePointer { p, _, _ in
                for i in 0..<values.count { p[i] = Float16(values[i]) }
            }
        } else {
            var mv = a.mutableView(as: Float.self)
            mv.withUnsafeMutablePointer { p, _, _ in
                for i in 0..<values.count { p[i] = values[i] }
            }
        }
        return a
    }

    /// Flatten any float output to `[Float]`, row-major, widening fp16.
    static func flat(_ array: NDArray) -> [Float] {
        switch array.scalarType {
        case .float16: return flatten(array, as: Float16.self)
        case .float32: return flatten(array, as: Float.self)
        default: preconditionFailure("unsupported output scalar type \(array.scalarType)")
        }
    }

    private static func flatten<T: BinaryFloatingPoint & BitwiseCopyable>(_ a: NDArray, as _: T.Type) -> [Float] {
        let total = a.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: total)
        a.view(as: T.self).withUnsafePointer { p, shp, strides in
            var expected = 1
            var contiguous = true
            for d in stride(from: shp.count - 1, through: 0, by: -1) {
                if strides[d] != expected { contiguous = false; break }
                expected *= shp[d]
            }
            if contiguous {
                for i in 0..<total { out[i] = Float(p[i]) }
                return
            }
            var idx = [Int](repeating: 0, count: shp.count)
            for i in 0..<total {
                var off = 0
                for d in 0..<shp.count { off += idx[d] * strides[d] }
                out[i] = Float(p[off])
                var d = shp.count - 1
                while d >= 0 {
                    idx[d] += 1
                    if idx[d] < shp[d] { break }
                    idx[d] = 0
                    d -= 1
                }
            }
        }
        return out
    }

    /// Pull one named output as `[Float]`, consuming it out of the `Outputs` bag.
    static func take(_ outputs: inout InferenceFunction.Outputs, _ name: String) throws -> [Float] {
        guard let v = outputs.remove(name)?.ndArray else {
            throw PocketTTSError.message("missing output '\(name)'")
        }
        return flat(v)
    }
}
