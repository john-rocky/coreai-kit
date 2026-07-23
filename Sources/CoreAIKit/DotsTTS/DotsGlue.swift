// DotsGlue.swift — host-side glue for dots.tts (rednote-hilab, 2B multilingual TTS). Community port,
// NOT an Apple model. Mirrors VoxCPM2Glue: token-embedding lookup + per-patch projections run on the
// CPU (Accelerate), outside the four engine bundles. Weights from conversion/dots_tts/export_host_glue.py
// (artifacts/dots_host_glue/{manifest.json,<name>.bin}); embed is fp16, matmuls are fp32.
//
// dots.tts glue (see conversion/dots_tts/e2e_full.py, the validated Python blueprint):
//   embedRow(id)         -> [1536]   token embedding (prefill text)
//   hiddenProj(h1536)    -> [1024]   fm hidden       (append the LLM hidden to fm_sequence)
//   latentProj(l128x4)   -> [1024]x4 fm history      (append the NORMALIZED solver patch)
//   coordinateProj(z128) -> [1024]   noise -> DiT space (solver scatter, per euler step)
//   eosProb(h1536)       -> Float    softmax(eos_proj2(silu(eos_proj0(h))))[...,1]  (stop when > 0.8)
//   denormalize(p128x4)  -> p*std+mean  (latent_stats; the vocoder + patch_encoder input)

import Accelerate
import Foundation

final class DotsGlue {
    private struct Spec: Decodable { let dtype: String; let shape: [Int]; let bytes: Int }

    private let embed: [Float16]                 // [V,1536] row-major
    let vocab: Int
    let hidden: Int                              // 1536
    let ditDim = 1024
    let latentDim = 128
    let patch = 4

    private let hiddenPW: [Float], hiddenPB: [Float]      // [1024,1536],[1024]
    private let latentPW: [Float], latentPB: [Float]      // [1024,128],[1024]
    private let coordPW: [Float], coordPB: [Float]        // [1024,128],[1024]
    private let eos0W: [Float], eos0B: [Float]            // [1536,1536],[1536]
    private let eos2W: [Float], eos2B: [Float]            // [2,1536],[2]
    private let mean: [Float], std: [Float]               // [128]

    init(directory dir: URL) throws {
        let manifest = try JSONDecoder().decode(
            [String: Spec].self,
            from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        func f32(_ n: String) throws -> [Float] {
            guard manifest[n] != nil else { throw DotsError.glueMissing(n) }
            return try Data(contentsOf: dir.appendingPathComponent("\(n).bin"))
                .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        func f16(_ n: String) throws -> ([Float16], [Int]) {
            guard let s = manifest[n] else { throw DotsError.glueMissing(n) }
            let d = try Data(contentsOf: dir.appendingPathComponent("\(n).bin"))
            return (d.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }, s.shape)
        }
        let (e, eShape) = try f16("embed_tokens")
        self.embed = e; self.vocab = eShape[0]; self.hidden = eShape[1]   // 1536
        self.hiddenPW = try f32("hidden_proj_w"); self.hiddenPB = try f32("hidden_proj_b")
        self.latentPW = try f32("latent_proj_w"); self.latentPB = try f32("latent_proj_b")
        self.coordPW = try f32("coordinate_proj_w"); self.coordPB = try f32("coordinate_proj_b")
        self.eos0W = try f32("eos_proj0_w"); self.eos0B = try f32("eos_proj0_b")
        self.eos2W = try f32("eos_proj2_w"); self.eos2B = try f32("eos_proj2_b")
        self.mean = try f32("latent_mean"); self.std = try f32("latent_std")
    }

    /// Token embedding row (fp16 -> fp32, length 1536).
    func embedRow(_ id: Int) -> [Float] {
        precondition(id >= 0 && id < vocab, "token id \(id) out of range \(vocab)")
        let base = id * hidden
        return (0..<hidden).map { Float(embed[base + $0]) }
    }

    func hiddenProj(_ h: [Float]) -> [Float] { matvec(hiddenPW, h, rows: ditDim, cols: hidden, bias: hiddenPB) }

    /// latent_proj over 4 latent columns [128*4] (layout [p*128+c]) -> [1024*4] (layout [p*1024+d]).
    func latentProj4(_ l: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: patch * ditDim)
        for p in 0..<patch {
            let row = Array(l[p * latentDim ..< (p + 1) * latentDim])
            let y = matvec(latentPW, row, rows: ditDim, cols: latentDim, bias: latentPB)
            for d in 0..<ditDim { out[p * ditDim + d] = y[d] }
        }
        return out
    }

    /// coordinate_proj over 4 columns [128*4] -> [1024*4].
    func coordProj4(_ z: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: patch * ditDim)
        for p in 0..<patch {
            let row = Array(z[p * latentDim ..< (p + 1) * latentDim])
            let y = matvec(coordPW, row, rows: ditDim, cols: latentDim, bias: coordPB)
            for d in 0..<ditDim { out[p * ditDim + d] = y[d] }
        }
        return out
    }

    /// P(eos) = softmax(eos_proj2(silu(eos_proj0(h))))[1].
    func eosProb(_ h: [Float]) -> Float {
        var x = matvec(eos0W, h, rows: hidden, cols: hidden, bias: eos0B)
        for i in 0..<hidden { x[i] = x[i] / (1 + expf(-x[i])) }   // SiLU
        let l = matvec(eos2W, x, rows: 2, cols: hidden, bias: eos2B)
        let m = max(l[0], l[1])
        let e0 = expf(l[0] - m), e1 = expf(l[1] - m)
        return e1 / (e0 + e1)
    }

    /// denormalize a [128*4] patch (layout [p*128+c]) in place: x*std + mean (per channel c).
    func denormalize(_ p: [Float]) -> [Float] {
        var out = p
        for pi in 0..<patch { for c in 0..<latentDim {
            out[pi * latentDim + c] = p[pi * latentDim + c] * std[c] + mean[c]
        } }
        return out
    }

    /// y[rows] = W[rows,cols] · x[cols] (+ bias). W row-major.
    private func matvec(_ w: [Float], _ x: [Float], rows: Int, cols: Int, bias: [Float]?) -> [Float] {
        var y = [Float](repeating: 0, count: rows)
        vDSP_mmul(w, 1, x, 1, &y, 1, vDSP_Length(rows), 1, vDSP_Length(cols))
        if let b = bias { vDSP_vadd(y, 1, b, 1, &y, 1, vDSP_Length(rows)) }
        return y
    }
}

enum DotsError: Error { case glueMissing(String), missingOutput(String) }
