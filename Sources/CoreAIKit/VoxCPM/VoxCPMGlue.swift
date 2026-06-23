// VoxCPMGlue.swift — the small host-side math VoxCPM runs OUTSIDE the engine bundles: the token
// embedding lookup (prefill) and the per-frame projections that sit between the AR backbone and the
// diffusion feat_decoder. These are tiny matrix-vector products over a single [1024] vector, so they
// run on the CPU via Accelerate (cblas_sgemv) — cheaper than a round-trip through a separate bundle,
// and it keeps the engine to the 5 model bundles. Weights come from `export_host_glue.py`
// (artifacts/voxcpm_host_glue/{manifest.json,<name>.bin}); embed is fp16, the matmuls are fp32.

import Accelerate
import Foundation

/// Loads + runs VoxCPM's host-side glue weights.
final class VoxCPMGlue {
    private struct Spec: Decodable { let dtype: String; let shape: [Int]; let bytes: Int }

    private let embed: [Float16]          // [V, 1024] row-major token embedding table
    private let vocab: Int
    private let hidden: Int               // 1024
    private let fsqLatent: Int            // 256

    private let lmToDitW: [Float], lmToDitB: [Float]     // [1024,1024],[1024]
    private let resToDitW: [Float], resToDitB: [Float]
    private let fsqInW: [Float], fsqInB: [Float]         // [256,1024],[256]
    private let fsqOutW: [Float], fsqOutB: [Float]       // [1024,256],[1024]
    private let stopProjW: [Float], stopProjB: [Float]   // [1024,1024],[1024]
    private let stopHeadW: [Float]                       // [2,1024]

    init(directory dir: URL) throws {
        let manifest = try JSONDecoder().decode(
            [String: Spec].self,
            from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))

        func f32(_ name: String) throws -> [Float] {
            guard manifest[name] != nil else { throw VoxCPMError.glueMissing(name) }
            let d = try Data(contentsOf: dir.appendingPathComponent("\(name).bin"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        func f16(_ name: String) throws -> ([Float16], [Int]) {
            guard let s = manifest[name] else { throw VoxCPMError.glueMissing(name) }
            let d = try Data(contentsOf: dir.appendingPathComponent("\(name).bin"))
            return (d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }, s.shape)
        }

        let (e, eShape) = try f16("embed_tokens")
        self.embed = e
        self.vocab = eShape[0]
        self.hidden = eShape[1]
        self.lmToDitW = try f32("lm_to_dit_w"); self.lmToDitB = try f32("lm_to_dit_b")
        self.resToDitW = try f32("res_to_dit_w"); self.resToDitB = try f32("res_to_dit_b")
        self.fsqInW = try f32("fsq_in_w"); self.fsqInB = try f32("fsq_in_b")
        self.fsqOutW = try f32("fsq_out_w"); self.fsqOutB = try f32("fsq_out_b")
        self.stopProjW = try f32("stop_proj_w"); self.stopProjB = try f32("stop_proj_b")
        self.stopHeadW = try f32("stop_head_w")
        self.fsqLatent = manifest["fsq_in_w"]!.shape[0]   // 256
    }

    var hiddenSize: Int { hidden }

    /// Token-embedding row (fp16, length 1024) for prefill.
    func embedRow(_ id: Int) -> [Float] {
        precondition(id >= 0 && id < vocab, "token id \(id) out of range \(vocab)")
        let base = id * hidden
        return (0..<hidden).map { Float(embed[base + $0]) }
    }

    /// dit_hidden = lm_to_dit(lm_h) + res_to_dit(res_h)   (the feat_decoder `mu`)
    func dit(lm: [Float], res: [Float]) -> [Float] {
        var out = matvec(lmToDitW, lm, rows: hidden, cols: hidden, bias: lmToDitB)
        let r = matvec(resToDitW, res, rows: hidden, cols: hidden, bias: resToDitB)
        vDSP_vadd(out, 1, r, 1, &out, 1, vDSP_Length(hidden))
        return out
    }

    /// FSQ: out_proj(round(tanh(in_proj(h)) * 9) / 9)   — the scalar-quantization bottleneck.
    func fsq(_ h: [Float]) -> [Float] {
        var z = matvec(fsqInW, h, rows: fsqLatent, cols: hidden, bias: fsqInB)
        z.withUnsafeMutableBufferPointer { p in
            var n = Int32(p.count)
            vvtanhf(p.baseAddress!, p.baseAddress!, &n)   // in-place tanh (one access region)
        }
        for i in 0..<fsqLatent { z[i] = (z[i] * 9).rounded() / 9 }
        return matvec(fsqOutW, z, rows: hidden, cols: fsqLatent, bias: fsqOutB)
    }

    /// stop-head argmax: 1 => end of utterance.  argmax(stop_head(silu(stop_proj(lm)))).
    func stop(_ lm: [Float]) -> Int {
        var p = matvec(stopProjW, lm, rows: hidden, cols: hidden, bias: stopProjB)
        for i in 0..<hidden { p[i] = p[i] / (1 + expf(-p[i])) }       // silu = x * sigmoid(x)
        let logits = matvec(stopHeadW, p, rows: 2, cols: hidden, bias: nil)
        return logits[1] > logits[0] ? 1 : 0
    }

    // MARK: - helpers

    /// y[rows] = W[rows,cols] · x[cols] (+ bias). W row-major; C = A[rows×cols]·B[cols×1].
    private func matvec(_ w: [Float], _ x: [Float], rows: Int, cols: Int, bias: [Float]?) -> [Float] {
        var y = [Float](repeating: 0, count: rows)
        vDSP_mmul(w, 1, x, 1, &y, 1, vDSP_Length(rows), 1, vDSP_Length(cols))
        if let b = bias { vDSP_vadd(y, 1, b, 1, &y, 1, vDSP_Length(rows)) }
        return y
    }
}
