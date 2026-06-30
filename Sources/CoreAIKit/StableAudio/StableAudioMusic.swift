// StableAudioMusic — on-device text→music/audio (Stable Audio Open Small) on Core AI.
// Three stateless .aimodel bundles (run via CoreAIKitVision's GraphModel) + a host 8-step
// rectified-flow euler sampler:
//   cond(input_ids[1,64], attn[1,64], seconds_norm[1]) -> cross_attn_cond[1,65,768], global_embed[1,768], cond_mask[1,65]
//   8× dit(x[1,64,256], t[1], cross_attn_cond, global_embed, cross_attn_cond_mask) -> v ; x += (t_next-t)·v
//   vae(latent[1,64,256]) -> audio[1,2,524288]  (~11.9s stereo 44.1kHz)
// CFG-free (cfg_scale 1.0 — the model is ARC-distilled). Returns interleaved-by-channel stereo [2*N].
import CoreAIKitVision
import Foundation
import Tokenizers

public struct StableAudioPaths: Sendable {
    public var cond: URL, dit: URL, vae: URL, tokenizerDir: URL
    public init(cond: URL, dit: URL, vae: URL, tokenizerDir: URL) {
        self.cond = cond; self.dit = dit; self.vae = vae; self.tokenizerDir = tokenizerDir
    }
    /// Resolve from a bundle root holding `sa_{cond_fp16b,dit_fp16,vae_fp16}.aimodel(c)` + `t5_tokenizer/`.
    public static func resolve(root: URL, aot: Bool) -> StableAudioPaths {
        let sfx = aot ? ".h18p.aimodelc" : ".aimodel"
        return .init(cond: root.appendingPathComponent("sa_cond_fp16b\(sfx)"),
                     dit: root.appendingPathComponent("sa_dit_fp16\(sfx)"),
                     vae: root.appendingPathComponent("sa_vae_fp16\(sfx)"),
                     tokenizerDir: root.appendingPathComponent("t5_tokenizer"))
    }
}

public final class StableAudioMusic: @unchecked Sendable {
    public static let sampleRate = 44100
    public static let audioSamples = 524288         // ~11.9 s
    private static let LAT_C = 64, LAT_T = 256, MAXTOK = 64, COND_LEN = 65, COND_DIM = 768
    private static let tSched: [Float] = [1.0, 0.9943756, 0.9844802, 0.9579123, 0.8909032, 0.7455466, 0.5124974, 0.27388501, 0.0]

    private let cond: GraphModel, dit: GraphModel, vae: GraphModel
    private let tokenizer: any Tokenizer

    public init(paths: StableAudioPaths, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        cond = try await GraphModel(contentsOf: paths.cond, computeUnits: computeUnits)
        dit = try await GraphModel(contentsOf: paths.dit, computeUnits: computeUnits)
        vae = try await GraphModel(contentsOf: paths.vae, computeUnits: computeUnits)
        tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
    }

    private func gaussianNoise() -> [Float] {
        var rng = SystemRandomNumberGenerator()
        var out = [Float](repeating: 0, count: Self.LAT_C * Self.LAT_T)
        var i = 0
        while i < out.count {
            let u1 = max(Double.random(in: 0..<1, using: &rng), 1e-12), u2 = Double.random(in: 0..<1, using: &rng)
            let r = (-2 * Foundation.log(u1)).squareRoot()
            out[i] = Float(r * Foundation.cos(2 * Double.pi * u2)); i += 1
            if i < out.count { out[i] = Float(r * Foundation.sin(2 * Double.pi * u2)); i += 1 }
        }
        return out
    }

    /// Generate ~11s of stereo audio for `prompt`. Returns interleaved-by-channel [2 * audioSamples].
    public func generate(prompt: String, seconds: Float = 11) async throws -> [Float] {
        var ids = tokenizer.encode(text: prompt)
        if ids.count > Self.MAXTOK { ids = Array(ids.prefix(Self.MAXTOK)) }
        let realLen = ids.count
        while ids.count < Self.MAXTOK { ids.append(0) }
        let idsI32 = ids.map { Int32($0) }
        let attn = (0..<Self.MAXTOK).map { Float($0 < realLen ? 1 : 0) }

        let cOut = try await cond.run([
            "input_ids": .int32(idsI32, shape: [1, Self.MAXTOK]),
            "attention_mask": .float32(attn, shape: [1, Self.MAXTOK]),
            "seconds_norm": .float32([seconds / 256.0], shape: [1]),
        ])
        let cross = cOut["cross_attn_cond"]!.floats(), glob = cOut["global_embed"]!.floats(), cmask = cOut["cond_mask"]!.floats()

        var x = gaussianNoise()
        for i in 0..<8 {
            let out = try await dit.run([
                "x": .float32(x, shape: [1, Self.LAT_C, Self.LAT_T]),
                "t": .float32([Self.tSched[i]], shape: [1]),
                "cross_attn_cond": .float32(cross, shape: [1, Self.COND_LEN, Self.COND_DIM]),
                "global_embed": .float32(glob, shape: [1, Self.COND_DIM]),
                "cross_attn_cond_mask": .float32(cmask, shape: [1, Self.COND_LEN]),
            ])
            let v = out["v"]!.floats()
            let dt = Self.tSched[i + 1] - Self.tSched[i]
            for k in 0..<x.count { x[k] += dt * v[k] }
        }
        let aOut = try await vae.run(["latent": .float32(x, shape: [1, Self.LAT_C, Self.LAT_T])])
        return aOut["audio"]!.floats()
    }
}
