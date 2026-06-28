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
    /// Optional q=T batched prefill bundles (base/res). When present + text fits, prefill runs as ONE
    /// call each instead of T sequential q=1 decodes (~25x faster prefill, lower TTFB). nil = decode-loop.
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

    // base/res LM precision. int8 weight-only is the size driver (mlx-community VoxCPM2 quantizes exactly
    // these two, leaves diffusion/VAE full-precision); fp16 is FASTER (no GPU dequant per frame) and
    // higher quality, at ~2x the bundle size. feat_decoder/encoder/vocoder always fp16 (the
    // continuous-feedback diffusion path is quant-sensitive). The fp16 decode bundles also exist in
    // artifacts, so this is a pure path swap.
    // fp16metal = fp16 weights with the bandwidth-dominant Linears on the fused "simd" Metal kernel
    // (lossless: bit-exact vs fp16, just removes the q=1 GEMV launch/occupancy overhead).
    public enum LMPrecision: Sendable { case int8, fp16, fp16metal }

    private static func bundleNames(_ p: LMPrecision) -> [String] {
        let q: String
        switch p { case .int8: q = "int8"; case .fp16: q = "fp16"; case .fp16metal: q = "fp16metal" }
        return ["voxcpm_base_\(q)_decode_cl512", "voxcpm_res_\(q)_decode_cl512",
                "voxcpm_feat_decoder_fp16", "voxcpm_feat_encoder_fp16", "voxcpm_vocoder_fp16_t12"]
    }

    private static func make(_ resolve: (String) -> URL, root: URL, lm: LMPrecision,
                             tokenizerDir: URL?) -> VoxCPMPaths {
        let n = bundleNames(lm)
        // Precision-matched q=32 batched prefill bundle (int8 decode -> int8 prefill, fp16 -> fp16) so
        // the prefill KV matches what decode expects. fp16metal decode reads fp16 KV (k/v not metalized).
        let pq = lm == .int8 ? "int8" : "fp16"
        let pBase: URL? = resolve("voxcpm_base_\(pq)_prefill_t32")
        let pRes: URL? = resolve("voxcpm_res_\(pq)_prefill_t32")
        return VoxCPMPaths(
            baseDecode: resolve(n[0]), resDecode: resolve(n[1]),
            featDecoder: resolve(n[2]), featEncoder: resolve(n[3]), vocoder: resolve(n[4]),
            glueDir: root.appendingPathComponent("voxcpm_host_glue"),
            tokenizerDir: tokenizerDir ?? root.appendingPathComponent("tokenizer"),
            prefillBase: pBase, prefillRes: pRes)
    }

    /// macOS dev layout from `export_*.py`: each bundle is `<name>/<name>.aimodel` (JIT-specialized).
    public static func standard(artifactsRoot root: URL, lm: LMPrecision = .int8,
                                tokenizerDir: URL? = nil) -> VoxCPMPaths {
        make({ root.appendingPathComponent($0).appendingPathComponent("\($0).aimodel") },
             root: root, lm: lm, tokenizerDir: tokenizerDir)
    }

    /// iOS AOT layout: flat `<root>/<name>.<arch>.aimodelc` (precompiled, no on-device JIT spike),
    /// produced by `coreai-build compile --platform iOS --architecture <arch>`.
    public static func aot(root: URL, arch: String = "h18p", lm: LMPrecision = .int8,
                           tokenizerDir: URL? = nil) -> VoxCPMPaths {
        make({ root.appendingPathComponent("\($0).\(arch).aimodelc") },
             root: root, lm: lm, tokenizerDir: tokenizerDir)
    }
}

/// Per-bundle compute-unit placement. The backbone (base/res int8) and the diffusion/VAE (fp16) have
/// different op mixes, so the fastest unit can differ per bundle — expose it instead of forcing one.
/// Default = all `.gpu` (the shipped, device-validated placement). Note: on iOS the AOT `.aimodelc`
/// bakes its compute unit at `coreai-build compile` time, so to *change* a bundle's unit on device you
/// must re-AOT it with that `--preferred-compute`; on macOS the JIT `.aimodel` honours this at load.
public struct ComputeConfig: Sendable {
    public var base: GraphModel.ComputeUnits
    public var res: GraphModel.ComputeUnits
    public var featDecoder: GraphModel.ComputeUnits
    public var featEncoder: GraphModel.ComputeUnits
    public var vocoder: GraphModel.ComputeUnits

    public init(base: GraphModel.ComputeUnits = .gpu, res: GraphModel.ComputeUnits = .gpu,
                featDecoder: GraphModel.ComputeUnits = .gpu, featEncoder: GraphModel.ComputeUnits = .gpu,
                vocoder: GraphModel.ComputeUnits = .gpu) {
        self.base = base; self.res = res; self.featDecoder = featDecoder
        self.featEncoder = featEncoder; self.vocoder = vocoder
    }

    /// Same unit for every bundle.
    public static func uniform(_ u: GraphModel.ComputeUnits) -> ComputeConfig {
        ComputeConfig(base: u, res: u, featDecoder: u, featEncoder: u, vocoder: u)
    }
}

/// Timing for a streaming synthesis: how soon the first audio chunk was ready (the metric that
/// governs perceived latency) and the total generation time vs. the audio length (RTF).
public struct StreamStats: Sendable {
    public var samples: Int
    public var firstChunkSeconds: Double
    public var totalSeconds: Double
    /// Output sample rate of the model that produced these samples (VoxCPM 16 kHz / VoxCPM2 48 kHz).
    /// Defaults to 16 kHz so the shipped 0.5B call sites are unchanged.
    public var sampleRate: Int = VoxCPMTTS.sampleRate
    public var audioSeconds: Double { Double(samples) / Double(sampleRate) }
    public var realTimeFactor: Double { audioSeconds > 0 ? totalSeconds / audioSeconds : 0 }
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
    // Optional q=T batched prefill bundles (one call replaces T sequential decodes). Loaded on GPU
    // (q=32 is compute-bound — GPU is fast and reliable for it). nil -> decode-loop prefill.
    private let prefillBase: StatefulGraphModel?
    private let prefillRes: StatefulGraphModel?
    private let prefillLen: Int

    /// Wall-clock seconds spent per pipeline stage during the last `generate` (for profiling where the
    /// time goes: `prefill`, `featDecoder` (diffusion), `featEncoder`, `baseDecode`, `resDecode`,
    /// `vocoder`, `hostGlue`). Reset at the start of each call.
    public private(set) var profile: [String: Double] = [:]
    private func tick(_ key: String, _ since: Date) { profile[key, default: 0] += Date().timeIntervalSince(since) }

    public init(paths: VoxCPMPaths, compute: ComputeConfig = ComputeConfig()) async throws {
        // Load order base -> res -> feat_decoder -> feat_encoder -> vocoder (the order that keeps the
        // heavy diffusion graph's specialization happy under co-residency; see knowledge doc).
        self.baseDecode = try await StatefulGraphModel(contentsOf: paths.baseDecode, computeUnits: compute.base)
        self.resDecode = try await StatefulGraphModel(contentsOf: paths.resDecode, computeUnits: compute.res)
        self.featDecoder = try await GraphModel(contentsOf: paths.featDecoder, computeUnits: compute.featDecoder)
        self.featEncoder = try await GraphModel(contentsOf: paths.featEncoder, computeUnits: compute.featEncoder)
        self.vocoder = try await GraphModel(contentsOf: paths.vocoder, computeUnits: compute.vocoder)
        self.glue = try VoxCPMGlue(directory: paths.glueDir)
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

    /// Synthesize speech for `text`. Returns the whole clip as mono Float PCM at 16 kHz
    /// (collects the streaming chunks). Prefer `synthesizeStreaming` for low perceived latency.
    public func synthesize(
        _ text: String, seed: UInt64 = 0, maxFrames: Int = 300, minFrames: Int = 2
    ) async throws -> [Float] {
        var out: [Float] = []
        _ = try await generate(text, seed: seed, maxFrames: maxFrames, minFrames: minFrames) { chunk in
            out.append(contentsOf: chunk)
        }
        return out
    }

    /// Streaming synthesis: invokes `onChunk` with each ~0.48 s audio chunk (one baked vocoder window)
    /// as soon as it is decoded, so playback can start after ~6 frames instead of waiting for the whole
    /// clip. The concatenation of all chunks is byte-identical to `synthesize` (same vocoder windows).
    @discardableResult
    public func synthesizeStreaming(
        _ text: String, seed: UInt64 = 0, maxFrames: Int = 300, minFrames: Int = 2,
        onChunk: @Sendable ([Float]) async -> Void
    ) async throws -> StreamStats {
        try await generate(text, seed: seed, maxFrames: maxFrames, minFrames: minFrames) { chunk in
            await onChunk(chunk)
        }
    }

    /// Core AR loop. Runs prefill, then per-frame [diffusion -> encode -> backbone], vocoding and
    /// emitting every `framesPerChunk` frames so audio streams out during generation.
    private func generate(
        _ text: String, seed: UInt64, maxFrames: Int, minFrames: Int,
        emit: ([Float]) async throws -> Void
    ) async throws -> StreamStats {
        let H = Self.hidden
        let framesPerChunk = Self.vaeFrames / Self.patch     // 12 cols / 2 = 6 frames per vocoder window
        let start = Date()
        var firstChunk = -1.0
        var totalSamples = 0
        profile = [:]

        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        ids.append(Self.audioStart)
        let T = ids.count

        // ---- prefill: batched bundle (one q=T call each) when available + text fits, else decode-loop.
        // Both are causal so they're bit-identical; the bundle reads weights once instead of T times. ----
        let pf = Date()
        baseDecode.resetState()
        resDecode.resetState()
        var lmH: [Float]
        var resH: [Float]
        if let pBase = prefillBase, let pRes = prefillRes, T <= prefillLen {
            pBase.resetState(); pRes.resetState()
            let L = prefillLen
            var emb = [Float](repeating: 0, count: L * H)                 // [1, L, H], real tokens 0..<T, rest 0
            for t in 0..<T { glue.embedRow(ids[t]).withUnsafeBufferPointer { src in
                emb.withUnsafeMutableBufferPointer { dst in dst.baseAddress!.advanced(by: t * H).update(from: src.baseAddress!, count: H) } } }
            let bo = try await pBase.step(["inputs_embeds": .float32(emb, shape: [1, L, H])])
            let baseAll = try out(bo, "hidden")                           // [L*H]; positions T..L-1 are padding (masked out of 0..T-1)
            let ro = try await pRes.step(["inputs_embeds": .float32(baseAll, shape: [1, L, H])])
            let resAll = try out(ro, "hidden")
            lmH = Array(baseAll[(T - 1) * H ..< T * H])
            resH = Array(resAll[(T - 1) * H ..< T * H])
            baseDecode.adoptState(from: pBase)                            // seed decode KV from prefill KV
            resDecode.adoptState(from: pRes)
        } else {
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
            lmH = baseHidden[T - 1]
            resH = resHidden
        }
        tick("prefill", pf)

        // ---- autoregressive diffusion loop ----
        var rng = GaussianRNG(seed: seed)
        var cond = [Float](repeating: 0, count: Self.feat * Self.patch)   // [1,64,2], layout [c*2+p]
        var pending: [[Float]] = []                                       // frames awaiting a vocoder window
        pending.reserveCapacity(framesPerChunk)

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
            let dit = glue.dit(lm: lmH, res: resH)
            let z = rng.normals(Self.feat * Self.patch)
            tick("hostGlue", gt)
            let ft = Date()
            let predTV = try await featDecoder.run([
                "mu": .float32(dit, shape: [1, H]),
                "cond": .float32(cond, shape: [1, Self.feat, Self.patch]),
                "z": .float32(z, shape: [1, Self.feat, Self.patch])])
            let pred = try out(predTV, "pred_feat")                       // [64,2], layout [c*2+p]
            tick("featDecoder", ft)
            pending.append(pred)

            // feat_encoder wants pred_feat [1,1,2,64] (layout [p*64+c]); cond feeds back as pred [1,64,2]
            var predFeat = [Float](repeating: 0, count: Self.patch * Self.feat)
            for c in 0..<Self.feat { for p in 0..<Self.patch { predFeat[p * Self.feat + c] = pred[c * Self.patch + p] } }
            let et = Date()
            let currTV = try await featEncoder.run([
                "pred_feat": .float32(predFeat, shape: [1, 1, Self.patch, Self.feat])])
            let curr = try out(currTV, "curr_embed")                     // [1024]
            tick("featEncoder", et)
            cond = pred

            if pending.count == framesPerChunk { try await flush() }     // stream the window out

            if i > minFrames && glue.stop(lmH) == 1 { break }            // stopping frame is kept (above)

            let pos = Int32(T + i)
            let bt = Date()
            let bo = try await baseDecode.step([
                "inputs_embeds": .float32(curr, shape: [1, 1, H]), "pos": .int32([pos], shape: [1])])
            lmH = glue.fsq(try out(bo, "hidden"))
            tick("baseDecode", bt)
            var resIn = [Float](repeating: 0, count: H)
            vDSP_vadd(lmH, 1, curr, 1, &resIn, 1, vDSP_Length(H))
            let rt = Date()
            let ro = try await resDecode.step([
                "inputs_embeds": .float32(resIn, shape: [1, 1, H]), "pos": .int32([pos], shape: [1])])
            resH = try out(ro, "hidden")
            tick("resDecode", rt)
        }
        try await flush()                                                // final partial window (< 6 frames)

        let total = Date().timeIntervalSince(start)
        return StreamStats(samples: totalSamples,
                           firstChunkSeconds: firstChunk < 0 ? total : firstChunk,
                           totalSeconds: total)
    }

    // MARK: - vocoder

    /// Decode one vocoder window (≤ `vaeFrames` columns) of `frames` -> 16 kHz wav. Independent per
    /// window (the bundle bakes the 12-column width and is stateless), so windowing here is identical
    /// to decoding the full `[64, 2T]` latents and slicing — just produced incrementally.
    private func vocodeChunk(_ frames: [[Float]]) async throws -> [Float] {
        let cols = frames.count * Self.patch                              // ≤ vaeFrames
        guard cols > 0 else { return [] }
        var chunk = [Float](repeating: 0, count: Self.feat * Self.vaeFrames)   // zero-padded to the baked width
        for t in frames.indices {
            for c in 0..<Self.feat {
                for p in 0..<Self.patch {
                    chunk[c * Self.vaeFrames + t * Self.patch + p] = frames[t][c * Self.patch + p]
                }
            }
        }
        let o = try await vocoder.run(["latents": .float32(chunk, shape: [1, Self.feat, Self.vaeFrames])])
        return Array(try out(o, "wav").prefix(cols * 640))
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
        if let pBase = prefillBase, let pRes = prefillRes {
            let L = prefillLen
            _ = try await pBase.step(["inputs_embeds": .float32([Float](repeating: 0, count: L * H), shape: [1, L, H])])
            _ = try await pRes.step(["inputs_embeds": .float32([Float](repeating: 0, count: L * H), shape: [1, L, H])])
            pBase.resetState(); pRes.resetState()
        }
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
