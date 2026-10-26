// MaskDecoder — the 9.5M-parameter matting decoder as two static graphs (the zoo's conversion/wevisone/mask/mask_graph.py):
//
//   graph A  once per 16 instances: image [1, 3, 1024, 1024], tap5 / tap11 / tap17 / tap23 [1, 1024, 28, 28],
//            t_seg [16, 2048] (the <OBJ> hidden states, real rows first, padding rows zero), boxes01 [16, 4] (padding
//            rows [0.25, 0.25, 0.75, 0.75]), box_pe [16, 256] (BoxFourier, padding rows from the padding box), valid [16]
//            (1 / 0) → queries [16, 192], coarse_logits [16, 512, 512], semantic_s2 [1, 48, 512, 512], memory
//   graph B  once per instance: image, semantic_s2, query [1, 192], coarse_logits [1, 512, 512] → mask_logits fp32
//            [1, 1024, 1024] (+ mask_logits_aux); a graph B built for K instances per call (query [K, 192],
//            coarse_logits [K, 512, 512] → mask_logits [K, 1024, 1024]; measured, not shipped) takes them K at a time, a
//            short last chunk filled with copies of its last instance whose outputs are dropped (`MaskGraphB`)
//
// Instances attend to each other inside graph A (the package's decoder runs 16 at a time, `decoder_batch_size`), so the
// rows of one answer go in together; more than 16 go in chunks of 16, as the package does. Graph B has no coupling
// between instances (its normalisations are GroupNorm, per instance). Inputs take the graph's type (fp16 for the
// shipped form, fp32 for the reference form). The alpha is the package's `_masks_from_logits`: sigmoid, then bilinear
// (align_corners false) to the picture's size (`Alpha`).

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public final class MaskGraphs: @unchecked Sendable {
    public static let rows = 16
    public static let side = 1024
    static let padBox: [Float] = [0.25, 0.25, 0.75, 0.75]

    public struct Timing: Sendable {
        public var boxPE: [Double] = []
        public var graphA: [Double] = []
        /// one entry per graph B call (per instance for the shipped graph B; per K instances for a K-row graph B)
        public var graphB: [Double] = []
        /// the real instances in each graph B call
        public var graphBInstances: [Int] = []
        public var inputs: Double = 0
    }

    /// Graph A's outputs for one chunk of up to 16 instances, as graph B takes them.
    public struct StageA: @unchecked Sendable {
        public let n: Int
        public let image: NDArray
        public let semantic: NDArray
        public let queries: [Float]
        public let coarse: [Float]
    }

    public let urls: (a: URL, b: URL)
    public let loadSeconds: (a: Double, b: Double)
    /// "fp16" or "fp32", from graph A's and graph B's t_seg / query types.
    public let precision: (a: String, b: String)
    public let descriptors: [String: Any]
    public let graphB: MaskGraphB
    private let a: InferenceFunction
    private let aIn: [String: NDArrayDescriptor]

    /// `graphAOptions` (default: `options`) lets graph A take another compute unit than graph B.
    public init(graphA: URL, graphB urlB: URL, options: SpecializationOptions,
                graphAOptions: SpecializationOptions? = nil) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: graphA, options: graphAOptions ?? options)
        guard let fname = model.functionNames.first, let da = model.functionDescriptor(for: fname),
              let fa = try model.loadFunction(named: fname)
        else { throw CutoutError.contract("\(graphA.lastPathComponent): no function") }
        let sa = secondsSince(t0)
        let gb = try await MaskGraphB(contentsOf: urlB, options: options)
        let ia = MaskGraphB.inputs(da)
        let r = Self.rows
        let shapesA: [String: [Int]] = ["image": [1, 3, 1024, 1024], "tap5": [1, 1024, 28, 28], "tap11": [1, 1024, 28, 28],
                                        "tap17": [1, 1024, 28, 28], "tap23": [1, 1024, 28, 28], "t_seg": [r, 2048],
                                        "boxes01": [r, 4], "box_pe": [r, BoxFourier.width], "valid": [r]]
        guard Set(ia.keys) == Set(shapesA.keys), shapesA.allSatisfy({ ia[$0.key]?.shape == $0.value }),
              ia.values.allSatisfy({ $0.scalarType == .float16 || $0.scalarType == .float32 })
        else {
            throw CutoutError.contract("graph A inputs \(ia.mapValues { "\($0.shape) \($0.scalarType)" }) are not the "
                + "host-Fourier mask graph's (\(shapesA))")
        }
        let outA: [String: [Int]] = ["queries": [r, 192], "coarse_logits": [r, 512, 512], "semantic_s2": [1, 48, 512, 512]]
        for (n, s) in outA where ND.descriptor(da.outputDescriptor(of: n))?.shape != s {
            throw CutoutError.contract("graph A output \(n): \(describe(da)["outputs"] ?? "?")")
        }
        urls = (graphA, urlB)
        loadSeconds = (sa, gb.loadSeconds)
        precision = (ia["t_seg"]!.scalarType == .float16 ? "fp16" : "fp32", gb.precision)
        descriptors = ["graph_a": describe(da), "graph_b": gb.descriptor]
        a = fa
        aIn = ia
        graphB = gb
    }

    /// One answer's instances → mask_logits [1024 · 1024] fp32 each. `canvas` = ImageHost.canvas (3 · 1024²),
    /// `taps` = the tower's taps [4, 1024, 28, 28] fp16, `hidden` = the <OBJ> states, `boxes` = fp32 boxes in [0, 1].
    public func logits(canvas: [Float], taps: NDArray, hidden: [[Float16]], boxes: [[Float]], libmBoxPE: Bool = false,
                       boxPE keep: ((Int, [Float]) -> Void)? = nil) async throws -> (logits: [[Float]], timing: Timing)
    {
        var timing = Timing()
        var out: [[Float]] = []
        for stage in try await stagesA(canvas: canvas, taps: taps, hidden: hidden, boxes: boxes, libmBoxPE: libmBoxPE,
                                       boxPE: keep, timing: &timing) {
            out += try await graphB.run(stage, timing: &timing)
        }
        return (out, timing)
    }

    /// Graph A for every chunk of 16 instances (step 6 of the host procedure), its outputs ready for graph B.
    public func stagesA(canvas: [Float], taps: NDArray, hidden: [[Float16]], boxes: [[Float]], libmBoxPE: Bool = false,
                        boxPE keep: ((Int, [Float]) -> Void)? = nil, timing: inout Timing) async throws -> [StageA]
    {
        precondition(hidden.count == boxes.count && !boxes.isEmpty)
        let t0 = ContinuousClock.now
        let imageA = try ND.make(floats: canvas, aIn["image"]!)
        let imageB = aIn["image"]!.scalarType == graphB.inputType("image") ? imageA : try ND.make(floats: canvas, graphB.inputDescriptor("image"))
        let tapValues = ND.read(taps, as: Float16.self)
        let per = 1024 * 28 * 28
        var tapArrays: [String: NDArray] = [:]
        for (k, name) in ["tap5", "tap11", "tap17", "tap23"].enumerated() {
            let slice = Array(tapValues[(k * per)..<((k + 1) * per)])
            let d = aIn[name]!
            tapArrays[name] = d.scalarType == .float16 ? ND.make(slice, d) : ND.make(slice.map { Float($0) }, d)
        }
        timing.inputs = secondsSince(t0)
        var stages: [StageA] = []
        let r = Self.rows
        for start in stride(from: 0, to: boxes.count, by: r) {
            let end = min(boxes.count, start + r)
            let n = end - start
            var seg = [Float](repeating: 0, count: r * 2048)
            var box = [Float](repeating: 0, count: r * 4)
            var valid = [Float](repeating: 0, count: r)
            var padded: [[Float]] = []
            for i in 0..<r {
                let bx = i < n ? boxes[start + i] : Self.padBox
                padded.append(bx)
                for j in 0..<4 { box[i * 4 + j] = bx[j] }
                if i < n {
                    for j in 0..<2048 { seg[i * 2048 + j] = Float(hidden[start + i][j]) }
                    valid[i] = 1
                }
            }
            let tp = ContinuousClock.now
            let pe = BoxFourier.rows(padded, libm: libmBoxPE)
            timing.boxPE.append(secondsSince(tp))
            keep?(start, pe)
            var inputs = tapArrays
            inputs["image"] = imageA
            inputs["t_seg"] = try ND.make(floats: seg, aIn["t_seg"]!)
            inputs["boxes01"] = try ND.make(floats: box, aIn["boxes01"]!)
            inputs["box_pe"] = try ND.make(floats: pe, aIn["box_pe"]!)
            inputs["valid"] = try ND.make(floats: valid, aIn["valid"]!)
            let ta = ContinuousClock.now
            var outA = try await a.run(inputs: inputs)
            timing.graphA.append(secondsSince(ta))
            guard let queries = outA.remove("queries")?.ndArray, let coarse = outA.remove("coarse_logits")?.ndArray,
                  let semantic = outA.remove("semantic_s2")?.ndArray
            else { throw CutoutError.contract("graph A: an output is missing") }
            let q = try ND.readFloats(queries)
            let semanticB = semantic.scalarType == graphB.inputType("semantic_s2") ? semantic
                : try ND.make(floats: try ND.readFloats(semantic), graphB.inputDescriptor("semantic_s2"))
            let coarseValues = try ND.readFloats(coarse)
            stages.append(StageA(n: n, image: imageB, semantic: semanticB, queries: Array(q[0..<(n * 192)]),
                                 coarse: Array(coarseValues[0..<(n * 512 * 512)])))
        }
        return stages
    }
}

/// Graph B, K instances per call (K = the graph's `query` rows: 1 for the shipped graph, 2 / 4 / 8 for batched forms,
/// not shipped): image, semantic_s2 (shared), query [K, 192], coarse_logits [K, 512, 512] → mask_logits fp32 [K, 1024, 1024].
@available(macOS 27, iOS 27, *)
public final class MaskGraphB: @unchecked Sendable {
    public let url: URL
    /// K, the instances per call
    public let rows: Int
    public let loadSeconds: Double
    /// "fp16" or "fp32", from the query's type
    public let precision: String
    public let descriptor: [String: Any]
    private let function: InferenceFunction
    private let ins: [String: NDArrayDescriptor]

    static func inputs(_ d: InferenceFunctionDescriptor) -> [String: NDArrayDescriptor] {
        Dictionary(uniqueKeysWithValues: d.inputNames.compactMap { n in ND.descriptor(d.inputDescriptor(of: n)).map { (n, $0) } })
    }

    public init(contentsOf url: URL, options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        guard let name = model.functionNames.first, let fd = model.functionDescriptor(for: name),
              let fn = try model.loadFunction(named: name)
        else { throw CutoutError.contract("\(url.lastPathComponent): no function") }
        let ib = Self.inputs(fd)
        let k = ib["query"]?.shape.first ?? 0
        let want: [String: [Int]] = ["image": [1, 3, 1024, 1024], "semantic_s2": [1, 48, 512, 512], "query": [k, 192],
                                     "coarse_logits": [k, 512, 512]]
        guard k >= 1, Set(ib.keys) == Set(want.keys), want.allSatisfy({ ib[$0.key]?.shape == $0.value }),
              ib.values.allSatisfy({ $0.scalarType == .float16 || $0.scalarType == .float32 })
        else {
            throw CutoutError.contract("graph B inputs \(ib.mapValues { "\($0.shape) \($0.scalarType)" }) are not the mask "
                + "graph's (\(want))")
        }
        guard TensorSpec.of(fd.outputDescriptor(of: "mask_logits")) == TensorSpec(shape: [k, 1024, 1024], type: .float32) else {
            throw CutoutError.contract("graph B output mask_logits: \(describe(fd)["outputs"] ?? "?")")
        }
        self.url = url
        rows = k
        loadSeconds = secondsSince(t0)
        precision = ib["query"]!.scalarType == .float16 ? "fp16" : "fp32"
        descriptor = describe(fd)
        function = fn
        ins = ib
    }

    func inputType(_ name: String) -> NDArray.ScalarType { ins[name]!.scalarType }

    func inputDescriptor(_ name: String) -> NDArrayDescriptor { ins[name]! }

    /// mask_logits [1024 · 1024] per instance of one graph A chunk, K at a time.
    public func run(_ stage: MaskGraphs.StageA, timing: inout MaskGraphs.Timing) async throws -> [[Float]] {
        let k = rows, plane = 512 * 512, side = 1024 * 1024
        var out: [[Float]] = []
        for start in stride(from: 0, to: stage.n, by: k) {
            let n = min(k, stage.n - start)
            var q = [Float](), c = [Float]()
            q.reserveCapacity(k * 192)
            c.reserveCapacity(k * plane)
            for slot in 0..<k {
                let i = start + min(slot, n - 1)   // padding slots repeat the chunk's last instance
                q += stage.queries[(i * 192)..<((i + 1) * 192)]
                c += stage.coarse[(i * plane)..<((i + 1) * plane)]
            }
            let tb = ContinuousClock.now
            let inB: [String: NDArray] = ["image": stage.image, "semantic_s2": stage.semantic,
                                          "query": try ND.make(floats: q, ins["query"]!),
                                          "coarse_logits": try ND.make(floats: c, ins["coarse_logits"]!)]
            var outB = try await function.run(inputs: inB)
            guard let m = outB.remove("mask_logits")?.ndArray else { throw CutoutError.contract("graph B: no mask_logits") }
            let values = ND.read(m, as: Float.self)
            if k == 1 { out.append(values) } else { for slot in 0..<n { out.append(Array(values[(slot * side)..<((slot + 1) * side)])) } }
            timing.graphB.append(secondsSince(tb))
            timing.graphBInstances.append(n)
        }
        return out
    }
}

/// The package's `_masks_from_logits`: sigmoid, then `F.interpolate(…, size=(height, width), mode="bilinear",
/// align_corners=False)` (torch's source index: scale · (dst + 0.5) − 0.5, clamped at 0; scale = in / out in fp32).
public enum Alpha {
    struct Taps {
        let i0: [Int]
        let i1: [Int]
        let l1: [Float]
    }

    static func taps(input: Int, output: Int) -> Taps {
        let scale = Float(input) / Float(output)
        var i0 = [Int](repeating: 0, count: output), i1 = i0
        var l1 = [Float](repeating: 0, count: output)
        for o in 0..<output {
            let src = max(scale * (Float(o) + 0.5) - 0.5, 0)
            let lo = Int(src)
            i0[o] = lo
            i1[o] = lo + (lo < input - 1 ? 1 : 0)
            l1[o] = src - Float(lo)
        }
        return Taps(i0: i0, i1: i1, l1: l1)
    }

    /// Sigmoid of mask_logits [side²] resampled to width × height, as probabilities in [0, 1].
    public static func probabilities(_ logits: [Float], side: Int = 1024, width: Int, height: Int) -> [Float] {
        var p = [Float](repeating: 0, count: logits.count)
        for i in 0..<logits.count { p[i] = 1 / (1 + Foundation.exp(-logits[i])) }
        let ty = taps(input: side, output: height), tx = taps(input: side, output: width)
        var out = [Float](repeating: 0, count: width * height)
        p.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<height {
                    let r0 = ty.i0[y] * side, r1 = ty.i1[y] * side
                    let h1 = ty.l1[y], h0 = 1 - h1
                    for x in 0..<width {
                        let w1 = tx.l1[x], w0 = 1 - w1
                        let c0 = tx.i0[x], c1 = tx.i1[x]
                        d[y * width + x] = h0 * (w0 * s[r0 + c0] + w1 * s[r0 + c1]) + h1 * (w0 * s[r1 + c0] + w1 * s[r1 + c1])
                    }
                }
            }
        }
        return out
    }

    /// The same, as 8-bit alpha (round(p · 255)).
    public static func alpha8(_ logits: [Float], side: Int = 1024, width: Int, height: Int) -> [UInt8] {
        probabilities(logits, side: side, width: width, height: height).map { UInt8(min(255, max(0, ($0 * 255).rounded()))) }
    }

    /// Several instances as one 8-bit matte at width × height: per pixel min(1, Σ p_i), each p_i resampled as
    /// `probabilities` does, without a full-size float plane per instance (a 12-megapixel photo with eight instances
    /// would need 400 MB of them).
    public static func combined8(_ logits: [[Float]], side: Int = 1024, width: Int, height: Int) -> [UInt8] {
        let planes = logits.map { l in l.map { 1 / (1 + Foundation.exp(-$0)) } }
        let ty = taps(input: side, output: height), tx = taps(input: side, output: width)
        var out = [UInt8](repeating: 0, count: width * height)
        out.withUnsafeMutableBufferPointer { d in
            for y in 0..<height {
                let r0 = ty.i0[y] * side, r1 = ty.i1[y] * side
                let h1 = ty.l1[y], h0 = 1 - h1
                for x in 0..<width {
                    let w1 = tx.l1[x], w0 = 1 - w1
                    let c0 = tx.i0[x], c1 = tx.i1[x]
                    var s: Float = 0
                    for p in planes {
                        s += h0 * (w0 * p[r0 + c0] + w1 * p[r0 + c1]) + h1 * (w0 * p[r1 + c0] + w1 * p[r1 + c1])
                    }
                    d[y * width + x] = UInt8(min(255, max(0, (min(s, 1) * 255).rounded())))
                }
            }
        }
        return out
    }
}
