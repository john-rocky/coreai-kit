// From the model zoo's apps/Kev/Sources/Kev/Head.swift (9e06b5a, sha256 fd69e2c65db5), identifiers prefixed Kev for the kit.
// Head — the author's pointer head (kev.model.PointerHead) on the host, in float64 (conversion/kev/host.py §5):
//
//   q   = Wq h_decide + bq;   k_j = Wk h_opt_j + bk          Wq, Wk [head_dim, d] and the biases [head_dim], fp32
//   z_j = (k_j . q) * scale                                  scale = 1 / sqrt(head_dim) (kev_head.json, = metadata)
//   p   = softmax_j(z / T)                                   T = kev_head.json temperature; max subtracted first
//
// The fp16 hidden rows and the fp32 weights are converted to Double exactly; the whole head and the softmax run in
// Double and p is rounded to Float once at the end, as host.py does: two hosts then agree on p bit for bit whatever
// their summation order (a float64 difference near 1e-16 lands on the same Float). The softmax's sum is NumPy's
// pairwise sum added to the identity 0.0 (float64 `e.sum()`), the exponential is the C library's `exp`.
//
// head/head.safetensors is read here: an 8-byte little-endian header length, the JSON header, then the raw tensors
// (offsets relative to the end of the header); every tensor F32.

import Accelerate
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitKevDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

struct KevHead: Sendable {
    let hiddenSize: Int
    let headDim: Int
    let scale: Double
    let temperature: Double
    /// the file's sha256 is not checked here; the gate records it
    let url: URL
    let qWeight: [Double]
    let qBias: [Double]
    let kWeight: [Double]
    let kBias: [Double]

    /// `host.load_head`: head.safetensors (q / k weight [head_dim, d] and bias [head_dim], fp32) + kev_head.json.
    init(directory: URL, json: String = "kev_head.json", weights: String = "head.safetensors") throws {
        let info = try KevJSONParser.parse(Data(contentsOf: directory.appendingPathComponent(json)))
        guard let scale = info["scale"]?.double, let temperature = info["temperature"]?.double,
              let headDim = info["head_dim"]?.intValue, let hidden = info["hidden_size"]?.intValue
        else { throw KevError.bundle("\(json): scale / temperature / head_dim / hidden_size missing") }
        guard abs(scale - 1.0 / Double(headDim).squareRoot()) <= 1e-12 else {
            throw KevError.contract("\(json): scale \(scale) != 1 / sqrt(head_dim \(headDim))")
        }
        let url = directory.appendingPathComponent(weights)
        let t = try Self.readSafetensors(url)
        func tensor(_ name: String, _ shape: [Int]) throws -> [Double] {
            guard let (s, v) = t[name] else { throw KevError.bundle("\(weights): no tensor \(name)") }
            guard s == shape else { throw KevError.contract("\(weights): \(name) has shape \(s), the head \(shape)") }
            return v.map(Double.init)
        }
        qWeight = try tensor("q.weight", [headDim, hidden])
        qBias = try tensor("q.bias", [headDim])
        kWeight = try tensor("k.weight", [headDim, hidden])
        kBias = try tensor("k.bias", [headDim])
        self.hiddenSize = hidden
        self.headDim = headDim
        self.scale = scale
        self.temperature = temperature
        self.url = url
    }

    /// A .safetensors file's F32 tensors: name -> (shape, values).
    static func readSafetensors(_ url: URL) throws -> [String: ([Int], [Float])] {
        let raw = try Data(contentsOf: url)
        guard raw.count >= 8 else { throw KevError.bundle("\(url.lastPathComponent): shorter than its header length") }
        let n = raw.prefix(8).enumerated().reduce(0) { $0 | (Int($1.element) << (8 * $1.offset)) }
        guard 8 + n <= raw.count else { throw KevError.bundle("\(url.lastPathComponent): header length \(n)") }
        let header = try KevJSONParser.parse(raw.subdata(in: 8..<(8 + n)))
        let base = 8 + n
        var out: [String: ([Int], [Float])] = [:]
        for m in header.members ?? [] where m.key != "__metadata__" {
            guard m.value["dtype"]?.string == "F32", let shape = m.value["shape"]?.array?.compactMap(\.intValue),
                  let off = m.value["data_offsets"]?.array?.compactMap(\.intValue), off.count == 2
            else { throw KevError.bundle("\(url.lastPathComponent): \(m.key) is not an F32 tensor") }
            let count = shape.reduce(1, *)
            guard off[1] - off[0] == count * 4, base + off[1] <= raw.count else {
                throw KevError.bundle("\(url.lastPathComponent): \(m.key) offsets \(off) for \(count) floats")
            }
            var values = [Float](repeating: 0, count: count)
            values.withUnsafeMutableBytes { dst in
                raw.withUnsafeBytes { src in
                    dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[(base + off[0])..<(base + off[1])]))
                }
            }
            out[m.key] = (shape, values.map { Float(bitPattern: UInt32(littleEndian: $0.bitPattern)) })
        }
        return out
    }

    private func linear(_ w: [Double], _ b: [Double], _ h: UnsafePointer<Double>) -> [Double] {
        var out = [Double](repeating: 0, count: headDim)
        w.withUnsafeBufferPointer { wp in
            for i in 0..<headDim {
                var s = 0.0
                vDSP_dotprD(wp.baseAddress! + i * hiddenSize, 1, h, 1, &s, vDSP_Length(hiddenSize))
                out[i] = s + b[i]
            }
        }
        return out
    }

    /// `host.head_logits`: z [K] before the temperature, from the fp16 hidden rows at <decide> and at every </opt>.
    func logits(decide: ArraySlice<Float16>, options: [ArraySlice<Float16>]) -> [Double] {
        let hd = decide.map(Double.init)
        let q = hd.withUnsafeBufferPointer { linear(qWeight, qBias, $0.baseAddress!) }
        return options.map { h in
            let ho = h.map(Double.init)
            let k = ho.withUnsafeBufferPointer { linear(kWeight, kBias, $0.baseAddress!) }
            var z = 0.0
            vDSP_dotprD(k, 1, q, 1, &z, vDSP_Length(headDim))
            return z * scale
        }
    }

    /// `host.head_probs`: softmax(z / T) over one question's options in float64, rounded to Float once.
    func probabilities(_ z: [Double]) -> [Float] {
        let zt = z.map { $0 / temperature }
        guard var m = zt.first else { return [] }
        for v in zt.dropFirst() where v > m || v.isNaN { m = v }
        let e = zt.map { Foundation.exp($0 - m) }
        let s = 0.0 + Self.pairwiseSum(e)
        return e.map { Float($0 / s) }
    }

    /// NumPy's `add.reduce` pairwise sum over a contiguous 1-D float64 array (a plain loop from -0.0 below 8
    /// elements, 8 accumulators up to 128, halves above).
    static func pairwiseSum(_ a: [Double]) -> Double {
        a.withUnsafeBufferPointer { pairwise($0.baseAddress!, a.count) }
    }

    private static func pairwise(_ a: UnsafePointer<Double>, _ n: Int) -> Double {
        if n < 8 {
            var res = -0.0
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
}

#endif
