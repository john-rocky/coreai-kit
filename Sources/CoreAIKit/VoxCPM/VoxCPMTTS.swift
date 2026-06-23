// VoxCPMTTS.swift — on-device VoxCPM-0.5B text-to-speech host (the zoo's first voice-capable TTS,
// MiniCPM4 TSLM/RALM backbone + LocDiT diffusion + AudioVAE). Drives the five exported Core AI
// bundles through the autoregressive loop; the small projections run host-side (VoxCPMGlue).
//
// Pipeline (per the validated torch reference generate.py / engine_generate.py):
//   tokenize(text, addSpecialTokens:false) + [audioStart]
//   prefill  : run the q=1 base/res DECODE bundles over the text embeds (pos 0..T-1) — bit-exact
//              with a batched prefill, so no separate prefill bundle is shipped (any text length).
//   per frame: dit = lm_to_dit(lm_h)+res_to_dit(res_h)              [host]
//              pred = feat_decoder(mu=dit, cond, z~N(0,1))           [engine, diffusion]
//              curr = feat_encoder(pred)                             [engine]
//              stop = argmax(stop_head(silu(stop_proj(lm_h))))       [host]
//              lm_h = FSQ(base_decode(curr, pos))                    [engine + host FSQ]
//              res_h = res_decode(lm_h + curr, pos)                  [engine]
//   vocoder  : AudioVAE decode of the [64, 2T] latents in 12-column chunks -> 16 kHz mono.
//
// Plain TTS (fixed speaker). Voice-clone prompt branch (vae_encode + prompt prefill) is a follow-on.

import Accelerate
import CoreAIKitVision
import Foundation
import Tokenizers

public enum VoxCPMError: Error, LocalizedError {
    case bundleNotFound(String)
    case glueMissing(String)
    case missingOutput(String)

    public var errorDescription: String? {
        switch self {
        case .bundleNotFound(let n): return "VoxCPM bundle not found: \(n)"
        case .glueMissing(let n): return "VoxCPM host-glue weight missing: \(n)"
        case .missingOutput(let n): return "VoxCPM bundle produced no '\(n)' output"
        }
    }
}

/// File locations for the five bundles + host glue + tokenizer.
public struct VoxCPMPaths: Sendable {
    public var baseDecode: URL, resDecode: URL, featDecoder: URL, featEncoder: URL, vocoder: URL
    public var glueDir: URL, tokenizerDir: URL

    public init(baseDecode: URL, resDecode: URL, featDecoder: URL, featEncoder: URL,
                vocoder: URL, glueDir: URL, tokenizerDir: URL) {
        self.baseDecode = baseDecode; self.resDecode = resDecode; self.featDecoder = featDecoder
        self.featEncoder = featEncoder; self.vocoder = vocoder
        self.glueDir = glueDir; self.tokenizerDir = tokenizerDir
    }

    // base/res LM are int8 weight-only (the size driver; mlx-community VoxCPM2 quantizes exactly these
    // two and leaves the diffusion/VAE full-precision). feat_decoder/encoder/vocoder stay fp16 — the
    // continuous-feedback diffusion path is quant-sensitive.
    private static let bundleNames = [
        "voxcpm_base_int8_decode_cl512", "voxcpm_res_int8_decode_cl512",
        "voxcpm_feat_decoder_fp16", "voxcpm_feat_encoder_fp16", "voxcpm_vocoder_fp16_t12"]

    private static func make(_ resolve: (String) -> URL, root: URL, tokenizerDir: URL?) -> VoxCPMPaths {
        VoxCPMPaths(
            baseDecode: resolve(bundleNames[0]), resDecode: resolve(bundleNames[1]),
            featDecoder: resolve(bundleNames[2]), featEncoder: resolve(bundleNames[3]),
            vocoder: resolve(bundleNames[4]),
            glueDir: root.appendingPathComponent("voxcpm_host_glue"),
            tokenizerDir: tokenizerDir ?? root.appendingPathComponent("tokenizer"))
    }

    /// macOS dev layout from `export_*.py`: each bundle is `<name>/<name>.aimodel` (JIT-specialized).
    public static func standard(artifactsRoot root: URL, tokenizerDir: URL? = nil) -> VoxCPMPaths {
        make({ root.appendingPathComponent($0).appendingPathComponent("\($0).aimodel") },
             root: root, tokenizerDir: tokenizerDir)
    }

    /// iOS AOT layout: flat `<root>/<name>.<arch>.aimodelc` (precompiled, no on-device JIT spike),
    /// produced by `coreai-build compile --platform iOS --architecture <arch>`.
    public static func aot(root: URL, arch: String = "h18p", tokenizerDir: URL? = nil) -> VoxCPMPaths {
        make({ root.appendingPathComponent("\($0).\(arch).aimodelc") },
             root: root, tokenizerDir: tokenizerDir)
    }
}

public final class VoxCPMTTS: @unchecked Sendable {
    public static let sampleRate = 16_000
    private static let audioStart = 101          // <|audio_start|>
    private static let hidden = 1024
    private static let feat = 64
    private static let patch = 2                  // columns per decode frame
    private static let vaeFrames = 12            // columns baked into the vocoder bundle

    private let baseDecode: StatefulGraphModel
    private let resDecode: StatefulGraphModel
    private let featDecoder: GraphModel
    private let featEncoder: GraphModel
    private let vocoder: GraphModel
    private let glue: VoxCPMGlue
    private let tokenizer: any Tokenizer

    public init(paths: VoxCPMPaths, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        // Load order base -> res -> feat_decoder -> feat_encoder -> vocoder (the order that keeps the
        // heavy diffusion graph's specialization happy under co-residency; see knowledge doc).
        self.baseDecode = try await StatefulGraphModel(contentsOf: paths.baseDecode, computeUnits: computeUnits)
        self.resDecode = try await StatefulGraphModel(contentsOf: paths.resDecode, computeUnits: computeUnits)
        self.featDecoder = try await GraphModel(contentsOf: paths.featDecoder, computeUnits: computeUnits)
        self.featEncoder = try await GraphModel(contentsOf: paths.featEncoder, computeUnits: computeUnits)
        self.vocoder = try await GraphModel(contentsOf: paths.vocoder, computeUnits: computeUnits)
        self.glue = try VoxCPMGlue(directory: paths.glueDir)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
        try await warm()
    }

    /// Synthesize speech for `text`. Returns mono Float PCM at 16 kHz.
    public func synthesize(
        _ text: String, seed: UInt64 = 0, maxFrames: Int = 300, minFrames: Int = 2
    ) async throws -> [Float] {
        let H = Self.hidden
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        ids.append(Self.audioStart)
        let T = ids.count

        // ---- prefill via the decode bundle (causal, so step-by-step == batched) ----
        baseDecode.resetState()
        resDecode.resetState()
        var baseHidden: [[Float]] = []
        baseHidden.reserveCapacity(T)
        for t in 0..<T {
            let o = try await baseDecode.step([
                "inputs_embeds": .float32(glue.embedRow(ids[t]), shape: [1, 1, H]),
                "pos": .int32([Int32(t)], shape: [1])])
            baseHidden.append(try out(o, "hidden"))
        }
        var resHidden: [Float] = []
        for t in 0..<T {
            let o = try await resDecode.step([
                "inputs_embeds": .float32(baseHidden[t], shape: [1, 1, H]),
                "pos": .int32([Int32(t)], shape: [1])])
            resHidden = try out(o, "hidden")
        }
        var lmH = baseHidden[T - 1]
        var resH = resHidden

        // ---- autoregressive diffusion loop ----
        var rng = GaussianRNG(seed: seed)
        var cond = [Float](repeating: 0, count: Self.feat * Self.patch)   // [1,64,2], layout [c*2+p]
        var frames: [[Float]] = []
        frames.reserveCapacity(maxFrames)
        for i in 0..<maxFrames {
            let dit = glue.dit(lm: lmH, res: resH)
            let z = rng.normals(Self.feat * Self.patch)
            let predTV = try await featDecoder.run([
                "mu": .float32(dit, shape: [1, H]),
                "cond": .float32(cond, shape: [1, Self.feat, Self.patch]),
                "z": .float32(z, shape: [1, Self.feat, Self.patch])])
            let pred = try out(predTV, "pred_feat")                       // [64,2], layout [c*2+p]
            frames.append(pred)

            // feat_encoder wants pred_feat [1,1,2,64] (layout [p*64+c]); cond feeds back as pred [1,64,2]
            var predFeat = [Float](repeating: 0, count: Self.patch * Self.feat)
            for c in 0..<Self.feat { for p in 0..<Self.patch { predFeat[p * Self.feat + c] = pred[c * Self.patch + p] } }
            let currTV = try await featEncoder.run([
                "pred_feat": .float32(predFeat, shape: [1, 1, Self.patch, Self.feat])])
            let curr = try out(currTV, "curr_embed")                     // [1024]
            cond = pred

            if i > minFrames && glue.stop(lmH) == 1 { break }

            let pos = Int32(T + i)
            let bo = try await baseDecode.step([
                "inputs_embeds": .float32(curr, shape: [1, 1, H]), "pos": .int32([pos], shape: [1])])
            lmH = glue.fsq(try out(bo, "hidden"))
            var resIn = [Float](repeating: 0, count: H)
            vDSP_vadd(lmH, 1, curr, 1, &resIn, 1, vDSP_Length(H))
            let ro = try await resDecode.step([
                "inputs_embeds": .float32(resIn, shape: [1, 1, H]), "pos": .int32([pos], shape: [1])])
            resH = try out(ro, "hidden")
        }

        return try await vocode(frames)
    }

    // MARK: - vocoder

    /// [64, 2T] latents -> 16 kHz wav, in 12-column chunks (the baked vocoder width).
    private func vocode(_ frames: [[Float]]) async throws -> [Float] {
        let cols = frames.count * Self.patch
        guard cols > 0 else { return [] }
        // flat latents [c * cols + (t*patch + p)] = frames[t][c*patch + p]
        var lat = [Float](repeating: 0, count: Self.feat * cols)
        for t in frames.indices {
            for c in 0..<Self.feat {
                for p in 0..<Self.patch {
                    lat[c * cols + t * Self.patch + p] = frames[t][c * Self.patch + p]
                }
            }
        }
        var wav: [Float] = []
        wav.reserveCapacity(cols * 640)
        var s = 0
        while s < cols {
            let w = min(Self.vaeFrames, cols - s)
            var chunk = [Float](repeating: 0, count: Self.feat * Self.vaeFrames)
            for c in 0..<Self.feat {
                for j in 0..<w { chunk[c * Self.vaeFrames + j] = lat[c * cols + s + j] }
            }
            let o = try await vocoder.run(["latents": .float32(chunk, shape: [1, Self.feat, Self.vaeFrames])])
            wav.append(contentsOf: try out(o, "wav").prefix(w * 640))
            s += Self.vaeFrames
        }
        return wav
    }

    // MARK: - helpers

    private func out(_ d: [String: TensorValue], _ name: String) throws -> [Float] {
        guard let v = d[name] else { throw VoxCPMError.missingOutput(name) }
        return v.floats()
    }

    /// One dummy call per model (load order) to specialize the graphs before the real loop.
    private func warm() async throws {
        let H = Self.hidden
        _ = try await baseDecode.step(["inputs_embeds": .float32([Float](repeating: 0, count: H), shape: [1, 1, H]),
                                       "pos": .int32([0], shape: [1])])
        _ = try await resDecode.step(["inputs_embeds": .float32([Float](repeating: 0, count: H), shape: [1, 1, H]),
                                      "pos": .int32([0], shape: [1])])
        _ = try await featDecoder.run(["mu": .float32([Float](repeating: 0, count: H), shape: [1, H]),
                                       "cond": .float32([Float](repeating: 0, count: Self.feat * Self.patch), shape: [1, Self.feat, Self.patch]),
                                       "z": .float32([Float](repeating: 0, count: Self.feat * Self.patch), shape: [1, Self.feat, Self.patch])])
        _ = try await featEncoder.run(["pred_feat": .float32([Float](repeating: 0, count: Self.patch * Self.feat), shape: [1, 1, Self.patch, Self.feat])])
        _ = try await vocoder.run(["latents": .float32([Float](repeating: 0, count: Self.feat * Self.vaeFrames), shape: [1, Self.feat, Self.vaeFrames])])
        baseDecode.resetState()
        resDecode.resetState()
    }
}

/// Deterministic standard-normal generator (SplitMix64 + Box–Muller) — VoxCPM's only stochastic input.
private struct GaussianRNG {
    private var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    private mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    private mutating func uniform() -> Float {   // (0,1]
        Float(next() >> 40) * (1.0 / Float(1 << 24)) + Float.leastNonzeroMagnitude
    }

    mutating func normals(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            let u1 = uniform(), u2 = uniform()
            let r = (-2 * logf(u1)).squareRoot()
            out[i] = r * cosf(2 * .pi * u2)
            if i + 1 < n { out[i + 1] = r * sinf(2 * .pi * u2) }
            i += 2
        }
        return out
    }
}
