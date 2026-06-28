// VoxCPM2Glue.swift — host-side glue for VoxCPM2 (2B), the scaled successor to VoxCPMGlue (0.5B).
// Same idea (token embedding lookup + per-frame projections run on the CPU via Accelerate, outside the
// engine bundles), but the v2 dataflow differs:
//   * hidden 2048 (v1 1024), FSQ latent 512 (v1 256)
//   * dit `mu` = CONCAT(lm_to_dit(lm_h), res_to_dit(res_h)) -> 2048   (v1 ADDED them -> 1024)
//   * residual input = fusion_concat_proj(cat(A, B)) where A=fsq(lm_h)|enc_out, B=curr_embed|0  (NEW;
//     v1 just added lm_h + curr)
// Weights from `export_host_glue_v2.py` (artifacts/voxcpm2_host_glue/{manifest.json,<name>.bin});
// embed is fp16, the matmuls are fp32. enc_to_lm_proj is folded into the feat_encoder bundle.

import Accelerate
import Foundation

/// Loads + runs VoxCPM2's host-side glue weights.
final class VoxCPM2Glue {
    private struct Spec: Decodable { let dtype: String; let shape: [Int]; let bytes: Int }

    private let embed: [Float16]          // [V, 2048] row-major token embedding table
    private let vocab: Int
    private let hidden: Int               // 2048
    private let ditDim: Int               // 1024 (each of lm_to_dit / res_to_dit outputs)
    private let fsqLatent: Int            // 512

    private let lmToDitW: [Float], lmToDitB: [Float]     // [1024,2048],[1024]
    private let resToDitW: [Float], resToDitB: [Float]
    private let fusionW: [Float], fusionB: [Float]       // [2048,4096],[2048]
    private let fsqInW: [Float], fsqInB: [Float]         // [512,2048],[512]
    private let fsqOutW: [Float], fsqOutB: [Float]       // [2048,512],[2048]
    private let stopProjW: [Float], stopProjB: [Float]   // [2048,2048],[2048]
    private let stopHeadW: [Float]                       // [2,2048]

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
        self.hidden = eShape[1]                          // 2048
        self.lmToDitW = try f32("lm_to_dit_w"); self.lmToDitB = try f32("lm_to_dit_b")
        self.resToDitW = try f32("res_to_dit_w"); self.resToDitB = try f32("res_to_dit_b")
        self.fusionW = try f32("fusion_w"); self.fusionB = try f32("fusion_b")
        self.fsqInW = try f32("fsq_in_w"); self.fsqInB = try f32("fsq_in_b")
        self.fsqOutW = try f32("fsq_out_w"); self.fsqOutB = try f32("fsq_out_b")
        self.stopProjW = try f32("stop_proj_w"); self.stopProjB = try f32("stop_proj_b")
        self.stopHeadW = try f32("stop_head_w")
        self.ditDim = manifest["lm_to_dit_w"]!.shape[0]  // 1024
        self.fsqLatent = manifest["fsq_in_w"]!.shape[0]  // 512
    }

    var hiddenSize: Int { hidden }

    /// Token-embedding row (fp16, length 2048) for prefill.
    func embedRow(_ id: Int) -> [Float] {
        precondition(id >= 0 && id < vocab, "token id \(id) out of range \(vocab)")
        let base = id * hidden
        return (0..<hidden).map { Float(embed[base + $0]) }
    }

    /// dit_hidden = cat(lm_to_dit(lm_h), res_to_dit(res_h))  -> [2048] (the feat_decoder `mu`, two tokens).
    func dit(lm: [Float], res: [Float]) -> [Float] {
        let a = matvec(lmToDitW, lm, rows: ditDim, cols: hidden, bias: lmToDitB)
        let b = matvec(resToDitW, res, rows: ditDim, cols: hidden, bias: resToDitB)
        return a + b   // array concatenation -> [2*ditDim] = [2048]
    }

    /// residual input = fusion_concat_proj(cat(a, b))   (a,b each [2048]) -> [2048].
    func fusion(_ a: [Float], _ b: [Float]) -> [Float] {
        let x = a + b   // concat -> [4096]
        return matvec(fusionW, x, rows: hidden, cols: 2 * hidden, bias: fusionB)
    }

    /// FSQ: out_proj(round(tanh(in_proj(h)) * 9) / 9)  (latent 512).
    func fsq(_ h: [Float]) -> [Float] {
        var z = matvec(fsqInW, h, rows: fsqLatent, cols: hidden, bias: fsqInB)
        z.withUnsafeMutableBufferPointer { p in
            var n = Int32(p.count)
            vvtanhf(p.baseAddress!, p.baseAddress!, &n)
        }
        for i in 0..<fsqLatent { z[i] = (z[i] * 9).rounded() / 9 }
        return matvec(fsqOutW, z, rows: hidden, cols: fsqLatent, bias: fsqOutB)
    }

    /// stop-head argmax: 1 => end of utterance.  argmax(stop_head(silu(stop_proj(lm)))).
    func stop(_ lm: [Float]) -> Int {
        var p = matvec(stopProjW, lm, rows: hidden, cols: hidden, bias: stopProjB)
        for i in 0..<hidden { p[i] = p[i] / (1 + expf(-p[i])) }
        let logits = matvec(stopHeadW, p, rows: 2, cols: hidden, bias: nil)
        return logits[1] > logits[0] ? 1 : 0
    }

    // MARK: - helpers

    /// y[rows] = W[rows,cols] · x[cols] (+ bias). W row-major.
    private func matvec(_ w: [Float], _ x: [Float], rows: Int, cols: Int, bias: [Float]?) -> [Float] {
        var y = [Float](repeating: 0, count: rows)
        vDSP_mmul(w, 1, x, 1, &y, 1, vDSP_Length(rows), 1, vDSP_Length(cols))
        if let b = bias { vDSP_vadd(y, 1, b, 1, &y, 1, vDSP_Length(rows)) }
        return y
    }
}
