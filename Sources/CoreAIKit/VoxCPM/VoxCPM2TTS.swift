// VoxCPM2TTS.swift — on-device VoxCPM2 (2B) text-to-speech host, the scaled 48 kHz successor to
// VoxCPMTTS (0.5B / 16 kHz). Same five-bundle + host-glue structure, but the v2 dataflow (verified
// bit-exact vs the official model in conversion/voxcpm/gate_v2_*_torch.py and engine-gated in
// export_v2.py) differs:
//   hidden 2048, patch 4, AudioVAE 48 kHz (1920 samples/latent-column), vocoder bundle baked Tf=8.
//   per frame: mu  = cat(lm_to_dit(lm_h), res_to_dit(res_h))            [host, 2048 -> two DiT tokens]
//              pred = feat_decoder(mu, cond, z~N(0,1))                   [engine, diffusion, patch 4]
//              curr = feat_encoder(pred)  (+ enc_to_lm folded in)        [engine, -> 2048]
//              stop = argmax(stop_head(silu(stop_proj(lm_h))))           [host]
//              lm_h = FSQ(base_decode(curr, pos))                        [engine + host FSQ-512]
//              res_h = res_decode(fusion(cat(lm_h, curr)), pos)          [host fusion + engine]
//   prefill (zero-shot): res input is fusion(cat(base_out, 0)) per position (feat_mask 0).
//
// Kept separate from the shipped VoxCPMTTS (0.5B) so v1 is untouched. Plain TTS (fixed speaker).

import Accelerate
import CoreAIKitVision
import Foundation
import Tokenizers

/// File locations for the five VoxCPM2 bundles + host glue + tokenizer.
public struct VoxCPM2Paths: Sendable {
    public var baseDecode: URL, resDecode: URL, featDecoder: URL, featEncoder: URL, vocoder: URL
    public var glueDir: URL, tokenizerDir: URL
    public var prefillBase: URL?, prefillRes: URL?
    public var prefillLen: Int = 32

    public init(baseDecode: URL, resDecode: URL, featDecoder: URL, featEncoder: URL,
                vocoder: URL, glueDir: URL, tokenizerDir: URL,
                prefillBase: URL? = nil, prefillRes: URL? = nil, prefillLen: Int = 32) {
        self.baseDecode = baseDecode; self.resDecode = resDecode; self.featDecoder = featDecoder
        self.featEncoder = featEncoder; self.vocoder = vocoder
        self.glueDir = glueDir; self.tokenizerDir = tokenizerDir
        self.prefillBase = prefillBase; self.prefillRes = prefillRes; self.prefillLen = prefillLen
    }

    /// base/res LM precision. int8 = the size driver (mlx-community VoxCPM2 quantizes exactly these two,
    /// leaves diffusion/VAE full precision); fp16 = higher quality, ~2x size. feat_decoder/encoder/vocoder
    /// are always fp16 (continuous-feedback diffusion is quant-sensitive). Both decode bundles exist in
    /// artifacts, so this is a pure path swap.
    public enum LMPrecision: Sendable { case int8, fp16 }

    private static func bundleNames(_ p: LMPrecision) -> [String] {
        let q = (p == .int8) ? "int8" : "fp16"
        return ["voxcpm2_base_\(q)_decode_cl512", "voxcpm2_res_\(q)_decode_cl512",
                "voxcpm2_feat_decoder_fp16", "voxcpm2_feat_encoder_fp16", "voxcpm2_vocoder_fp16_t8"]
    }

    private static func make(_ resolve: (String) -> URL, root: URL, lm: LMPrecision,
                             tokenizerDir: URL?) -> VoxCPM2Paths {
        let n = bundleNames(lm)
        let q = (lm == .int8) ? "int8" : "fp16"
        return VoxCPM2Paths(
            baseDecode: resolve(n[0]), resDecode: resolve(n[1]),
            featDecoder: resolve(n[2]), featEncoder: resolve(n[3]), vocoder: resolve(n[4]),
            glueDir: root.appendingPathComponent("voxcpm2_host_glue"),
            tokenizerDir: tokenizerDir ?? root.appendingPathComponent("tokenizer"),
            prefillBase: resolve("voxcpm2_base_\(q)_prefill_t32"),
            prefillRes: resolve("voxcpm2_res_\(q)_prefill_t32"))
    }

    /// macOS dev layout: each bundle is `<name>/<name>.aimodel` (JIT-specialized).
    public static func standard(artifactsRoot root: URL, lm: LMPrecision = .int8,
                                tokenizerDir: URL? = nil) -> VoxCPM2Paths {
        make({ root.appendingPathComponent($0).appendingPathComponent("\($0).aimodel") },
             root: root, lm: lm, tokenizerDir: tokenizerDir)
    }

    /// iOS AOT layout: flat `<root>/<name>.<arch>.aimodelc`.
    public static func aot(root: URL, arch: String = "h18p", lm: LMPrecision = .int8,
                           tokenizerDir: URL? = nil) -> VoxCPM2Paths {
        make({ root.appendingPathComponent("\($0).\(arch).aimodelc") },
             root: root, lm: lm, tokenizerDir: tokenizerDir)
    }
}

public final class VoxCPM2TTS: @unchecked Sendable {
    public static let sampleRate = 48_000
    private static let audioStart = 101
    private static let hidden = 2048
    private static let feat = 64
    private static let patch = 4                  // latent columns per decode frame
    private static let vaeFrames = 8              // columns baked into the vocoder bundle
    private static let colSamples = 1920          // 48 kHz samples per latent column (1920x upsample)

    private let baseDecode: StatefulGraphModel
    private let resDecode: StatefulGraphModel
    private let featDecoder: GraphModel
    private let featEncoder: GraphModel
    private let vocoder: GraphModel
    private let glue: VoxCPM2Glue
    private let tokenizer: any Tokenizer
    private let prefillBase: StatefulGraphModel?
    private let prefillRes: StatefulGraphModel?
    private let prefillLen: Int

    public private(set) var profile: [String: Double] = [:]
    private func tick(_ key: String, _ since: Date) { profile[key, default: 0] += Date().timeIntervalSince(since) }

    public init(paths: VoxCPM2Paths, compute: ComputeConfig = ComputeConfig()) async throws {
        self.baseDecode = try await StatefulGraphModel(contentsOf: paths.baseDecode, computeUnits: compute.base)
        self.resDecode = try await StatefulGraphModel(contentsOf: paths.resDecode, computeUnits: compute.res)
        self.featDecoder = try await GraphModel(contentsOf: paths.featDecoder, computeUnits: compute.featDecoder)
        self.featEncoder = try await GraphModel(contentsOf: paths.featEncoder, computeUnits: compute.featEncoder)
        self.vocoder = try await GraphModel(contentsOf: paths.vocoder, computeUnits: compute.vocoder)
        self.glue = try VoxCPM2Glue(directory: paths.glueDir)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
        self.prefillLen = paths.prefillLen
        if let pb = paths.prefillBase, let pr = paths.prefillRes,
           FileManager.default.fileExists(atPath: pb.path), FileManager.default.fileExists(atPath: pr.path) {
            self.prefillBase = try await StatefulGraphModel(contentsOf: pb, computeUnits: .gpu)
            self.prefillRes = try await StatefulGraphModel(contentsOf: pr, computeUnits: .gpu)
        } else {
            self.prefillBase = nil; self.prefillRes = nil
        }
        try await warm()
    }

    /// Synthesize the whole clip as mono Float PCM at 48 kHz (collects the streaming chunks).
    public func synthesize(
        _ text: String, seed: UInt64 = 0, maxFrames: Int = 300, minFrames: Int = 2
    ) async throws -> [Float] {
        var out: [Float] = []
        _ = try await generate(text, seed: seed, maxFrames: maxFrames, minFrames: minFrames) { chunk in
            out.append(contentsOf: chunk)
        }
        return out
    }

    /// Streaming synthesis: invokes `onChunk` with each ~0.16 s audio window as soon as it is decoded.
    @discardableResult
    public func synthesizeStreaming(
        _ text: String, seed: UInt64 = 0, maxFrames: Int = 300, minFrames: Int = 2,
        onChunk: @Sendable ([Float]) async -> Void
    ) async throws -> StreamStats {
        try await generate(text, seed: seed, maxFrames: maxFrames, minFrames: minFrames) { chunk in
            await onChunk(chunk)
        }
    }

    private func generate(
        _ text: String, seed: UInt64, maxFrames: Int, minFrames: Int,
        emit: ([Float]) async throws -> Void
    ) async throws -> StreamStats {
        let H = Self.hidden
        let framesPerChunk = Self.vaeFrames / Self.patch     // 8 / 4 = 2 frames per vocoder window
        let start = Date()
        var firstChunk = -1.0
        var totalSamples = 0
        profile = [:]
        let zeros = [Float](repeating: 0, count: H)

        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        ids.append(Self.audioStart)
        let T = ids.count

        // ---- prefill. lm_h = base_out[T-1] (NOT fsq'd, zero-shot text mask); residual input per
        // position = fusion(cat(base_out_t, 0)) (feat_mask 0). ----
        let pf = Date()
        baseDecode.resetState(); resDecode.resetState()
        var lmH: [Float]
        var resH: [Float]
        if let pBase = prefillBase, let pRes = prefillRes, T <= prefillLen {
            pBase.resetState(); pRes.resetState()
            let L = prefillLen
            var emb = [Float](repeating: 0, count: L * H)
            for t in 0..<T {
                glue.embedRow(ids[t]).withUnsafeBufferPointer { src in
                    emb.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.advanced(by: t * H).update(from: src.baseAddress!, count: H) } }
            }
            let bo = try await pBase.step(["inputs_embeds": .float32(emb, shape: [1, L, H])])
            let baseAll = try out(bo, "hidden")                       // [L*H]
            var resInSeq = [Float](repeating: 0, count: L * H)
            for t in 0..<L {
                let row = Array(baseAll[t * H ..< (t + 1) * H])
                let fz = glue.fusion(row, zeros)
                fz.withUnsafeBufferPointer { src in
                    resInSeq.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.advanced(by: t * H).update(from: src.baseAddress!, count: H) } }
            }
            let ro = try await pRes.step(["inputs_embeds": .float32(resInSeq, shape: [1, L, H])])
            let resAll = try out(ro, "hidden")
            lmH = Array(baseAll[(T - 1) * H ..< T * H])
            resH = Array(resAll[(T - 1) * H ..< T * H])
            baseDecode.adoptState(from: pBase)
            resDecode.adoptState(from: pRes)
        } else {
            var baseHidden: [[Float]] = []; baseHidden.reserveCapacity(T)
            for t in 0..<T {
                let o = try await baseDecode.step([
                    "inputs_embeds": .float32(glue.embedRow(ids[t]), shape: [1, 1, H]),
                    "pos": .int32([Int32(t)], shape: [1])])
                baseHidden.append(try out(o, "hidden"))
            }
            var resHidden: [Float] = []
            for t in 0..<T {
                let o = try await resDecode.step([
                    "inputs_embeds": .float32(glue.fusion(baseHidden[t], zeros), shape: [1, 1, H]),
                    "pos": .int32([Int32(t)], shape: [1])])
                resHidden = try out(o, "hidden")
            }
            lmH = baseHidden[T - 1]; resH = resHidden
        }
        tick("prefill", pf)

        // ---- autoregressive diffusion loop ----
        var rng = GaussianRNG(seed: seed)
        var cond = [Float](repeating: 0, count: Self.feat * Self.patch)   // [1,64,4], layout [c*4+p]
        var pending: [[Float]] = []; pending.reserveCapacity(framesPerChunk)

        func flush() async throws {
            guard !pending.isEmpty else { return }
            let vt = Date()
            let wav = try await vocodeChunk(pending)
            tick("vocoder", vt)
            pending.removeAll(keepingCapacity: true)
            if firstChunk < 0 { firstChunk = Date().timeIntervalSince(start) }
            totalSamples += wav.count
            try await emit(wav)
        }

        for i in 0..<maxFrames {
            let gt = Date()
            let mu = glue.dit(lm: lmH, res: resH)                         // [2048]
            let z = rng.normals(Self.feat * Self.patch)
            tick("hostGlue", gt)
            let ft = Date()
            let predTV = try await featDecoder.run([
                "mu": .float32(mu, shape: [1, H]),
                "cond": .float32(cond, shape: [1, Self.feat, Self.patch]),
                "z": .float32(z, shape: [1, Self.feat, Self.patch])])
            let pred = try out(predTV, "pred_feat")                       // [64*4], layout [c*4+p]
            tick("featDecoder", ft)
            pending.append(pred)

            // feat_encoder wants pred_feat [1,1,4,64] (layout [p*64+c]); cond feeds back as pred [1,64,4]
            var predFeat = [Float](repeating: 0, count: Self.patch * Self.feat)
            for c in 0..<Self.feat { for p in 0..<Self.patch { predFeat[p * Self.feat + c] = pred[c * Self.patch + p] } }
            let et = Date()
            let currTV = try await featEncoder.run([
                "pred_feat": .float32(predFeat, shape: [1, 1, Self.patch, Self.feat])])
            let curr = try out(currTV, "curr_embed")                     // [2048]
            tick("featEncoder", et)
            cond = pred

            if pending.count == framesPerChunk { try await flush() }

            if i > minFrames && glue.stop(lmH) == 1 { break }

            let pos = Int32(T + i)
            let bt = Date()
            let bo = try await baseDecode.step([
                "inputs_embeds": .float32(curr, shape: [1, 1, H]), "pos": .int32([pos], shape: [1])])
            lmH = glue.fsq(try out(bo, "hidden"))
            tick("baseDecode", bt)
            let resIn = glue.fusion(lmH, curr)                           // fusion(cat(fsq(lm_h), curr))
            let rt = Date()
            let ro = try await resDecode.step([
                "inputs_embeds": .float32(resIn, shape: [1, 1, H]), "pos": .int32([pos], shape: [1])])
            resH = try out(ro, "hidden")
            tick("resDecode", rt)
        }
        try await flush()

        let total = Date().timeIntervalSince(start)
        return StreamStats(samples: totalSamples,
                           firstChunkSeconds: firstChunk < 0 ? total : firstChunk,
                           totalSeconds: total, sampleRate: Self.sampleRate)
    }

    // MARK: - vocoder

    /// Decode one window (≤ `vaeFrames` columns) of `frames` -> 48 kHz wav. The bundle bakes the 8-column
    /// width and is stateless, so windowing here matches decoding the full latents and slicing.
    private func vocodeChunk(_ frames: [[Float]]) async throws -> [Float] {
        let cols = frames.count * Self.patch
        guard cols > 0 else { return [] }
        var chunk = [Float](repeating: 0, count: Self.feat * Self.vaeFrames)
        for t in frames.indices {
            for c in 0..<Self.feat {
                for p in 0..<Self.patch {
                    chunk[c * Self.vaeFrames + t * Self.patch + p] = frames[t][c * Self.patch + p]
                }
            }
        }
        let o = try await vocoder.run(["latents": .float32(chunk, shape: [1, Self.feat, Self.vaeFrames])])
        return Array(try out(o, "wav").prefix(cols * Self.colSamples))
    }

    // MARK: - helpers

    private func out(_ d: [String: TensorValue], _ name: String) throws -> [Float] {
        guard let v = d[name] else { throw VoxCPMError.missingOutput(name) }
        return v.floats()
    }

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
        if let pBase = prefillBase, let pRes = prefillRes {
            let L = prefillLen
            _ = try await pBase.step(["inputs_embeds": .float32([Float](repeating: 0, count: L * H), shape: [1, L, H])])
            _ = try await pRes.step(["inputs_embeds": .float32([Float](repeating: 0, count: L * H), shape: [1, L, H])])
            pBase.resetState(); pRes.resetState()
        }
        baseDecode.resetState(); resDecode.resetState()
    }
}

/// Deterministic standard-normal generator (SplitMix64 + Box–Muller) — VoxCPM2's only stochastic input.
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

    private mutating func uniform() -> Float {
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
