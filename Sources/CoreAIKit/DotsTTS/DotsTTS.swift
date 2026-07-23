// DotsTTS.swift — on-device dots.tts (rednote-hilab, 2B, 24-language) text-to-speech host.
// Community port, NOT an Apple model. Four Core AI bundles + host glue, mirroring the validated
// Python blueprint conversion/dots_tts/e2e_full.py (engine wav cos 0.9959 vs the fp32 oracle):
//
//   backbone     Qwen2.5-1.5B decode (static KV)     inputs_embeds[1,1,1536] -> hidden      [Stateful]
//   patchEncoder streaming causal re-encoder (24L KV) latent[1,4,128]+conv_tail+pos -> emb   [Stateful]
//   dit          flow head, bucket S=164 (CFG b2)    x[2,164,1024]+t+mask+pos+g -> vel[2,164,128]
//   vocoder      AudioVAE/BigVGAN 48 kHz             latents[1,128,60] -> wav[1,1,115200]
//
// Per patch (soar, 10-step euler + CFG): solver drives the dit bundle over the growing fm_sequence
// (padded to 164, host-built mask/pos_ids) -> denormalize -> append latent_proj(NORMALIZED patch) to
// fm_seq -> patch_encoder(DENORMALIZED patch) -> backbone step -> append hidden_proj(hidden). Stop when
// eos_proj(hidden) softmax > 0.8. Every 15 patches (= 60 latent columns) -> vocoder -> 48 kHz chunk.

import CoreAIKitVision
import Foundation
import Tokenizers

public struct DotsTTSPaths: Sendable {
    public var backbone: URL, patchEncoder: URL, dit: URL, vocoder: URL
    public var glueDir: URL, tokenizerDir: URL
    public var ditSmall: URL?         // optional smaller DiT bucket (S=84) — early/short patches skip S=164
    public var prefill: URL?          // optional q=32 backbone prefill bundle (batched, ~5× cheaper than 15 decode steps)
    public var prefillLen: Int = 32
    public init(backbone: URL, patchEncoder: URL, dit: URL, vocoder: URL, glueDir: URL, tokenizerDir: URL,
                ditSmall: URL? = nil, prefill: URL? = nil, prefillLen: Int = 32) {
        self.backbone = backbone; self.patchEncoder = patchEncoder; self.dit = dit
        self.vocoder = vocoder; self.glueDir = glueDir; self.tokenizerDir = tokenizerDir
        self.ditSmall = ditSmall; self.prefill = prefill; self.prefillLen = prefillLen
    }

    /// macOS dev layout: `<root>/<name>/<name>.aimodel`. `lm` picks the backbone precision
    /// (fp16 = the reference-faithful gate; int8/int4 = smaller/faster, device is the truth).
    public enum LMPrecision: String, Sendable { case fp16, int8, int4 }
    /// Decoder: `mf` (MeanFlow, NFE 4, no CFG batch-1 — ~5× faster, iPhone default) or `soar`
    /// (flow-matching, 10-step CFG batch-2 — reference quality). Same backbone/patchenc/vocoder/glue;
    /// only the DiT bundle differs.
    public enum Decoder: String, Sendable { case mf, soar }

    private static func make(_ resolve: (String) -> URL, root: URL, lm: LMPrecision,
                             decoder: Decoder, tokenizerDir: URL) -> DotsTTSPaths {
        DotsTTSPaths(
            backbone: resolve("dots_backbone_\(lm.rawValue)_decode_cl512"),
            patchEncoder: resolve("dots_patchenc_fp16_buf256"),
            dit: resolve("dots_dit_\(decoder.rawValue)_fp16_s164"),
            vocoder: resolve("dots_vocoder_fp16_t60"),
            glueDir: root.appendingPathComponent("dots_host_glue"),
            tokenizerDir: tokenizerDir,
            ditSmall: resolve("dots_dit_\(decoder.rawValue)_fp16_s84"),
            prefill: resolve("dots_backbone_\(lm.rawValue)_prefill_t32"))
    }

    /// macOS dev layout: `<root>/<name>/<name>.aimodel` (JIT-specialized).
    public static func standard(artifactsRoot root: URL, lm: LMPrecision = .fp16,
                                decoder: Decoder = .soar, tokenizerDir: URL) -> DotsTTSPaths {
        make({ root.appendingPathComponent($0).appendingPathComponent("\($0).aimodel") },
             root: root, lm: lm, decoder: decoder, tokenizerDir: tokenizerDir)
    }

    /// iOS AOT layout: flat `<root>/<name>.<arch>.aimodelc`.
    public static func aot(root: URL, arch: String = "h18p", lm: LMPrecision = .int4,
                           decoder: Decoder = .mf, tokenizerDir: URL) -> DotsTTSPaths {
        make({ root.appendingPathComponent("\($0).\(arch).aimodelc") },
             root: root, lm: lm, decoder: decoder, tokenizerDir: tokenizerDir)
    }
}

public final class DotsTTS: @unchecked Sendable {
    public static let sampleRate = 48_000
    private static let hidden = 1536
    private static let ditDim = 1024
    private static let latentDim = 128
    private static let patch = 4
    private static let cap = 164              // DiT bucket-32 total_len
    private static let latentStart = 160      // cap - patch
    // Vocode: a single Tf=60 call over up to 15 patches (the reusable `vocodeWindow` degenerates to
    // non-streaming with ctx=0, emit=15). Streaming was tried but the overlap re-vocode raised TOTAL
    // time and at RTF>1 the playback underruns — a single vocode is faster + gap-free. (Re-enable
    // streaming with ctxPast/ctxFuture>0 once RTF<1 makes continuous playback smooth.)
    private static let vaeFrames = 60         // vocoder Tf window (= 15 patches)
    private static let ctxPast = 0            // past-context patches
    private static let emitPatches = 15       // patches emitted per chunk (= non-streaming)
    private static let ctxFuture = 0          // future-context patches (holdback)
    private static let colSamples = 1920      // 48 kHz samples per latent column
    private static let numSteps = 10          // soar euler steps
    private static let nfe = 2                 // mf MeanFlow steps
    private static let guidance: Float = 1.2  // model.py generate default (soar)
    // template: "[文本]" + text + "[文本对应语音]<|audio_gen_start|>", then <|audio_gen_span|> until EOS.
    private static let prefixIds: [Int] = [58, 108704, 60]
    private static let suffixIds: [Int] = [58, 108704, 103124, 105761, 60, 151668]

    private let backbone: StatefulGraphModel
    private let patchEncoder: StatefulGraphModel
    private let dit: GraphModel           // S=164 bucket (fm_len up to 160)
    private let ditSmall: GraphModel?     // S=84 bucket (fm_len up to 80) — early/short patches
    private static let capSmall = 84
    private let vocoder: GraphModel
    private let glue: DotsGlue
    private let tokenizer: any Tokenizer
    private let decoder: DotsTTSPaths.Decoder
    private let prefillLM: StatefulGraphModel?     // optional batched prefill (q=32)
    private let prefillLen: Int

    public private(set) var profile: [String: Double] = [:]
    private func tick(_ k: String, _ s: Date) { profile[k, default: 0] += Date().timeIntervalSince(s) }

    public init(paths: DotsTTSPaths, decoder: DotsTTSPaths.Decoder = .mf,
                computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        self.decoder = decoder
        self.prefillLen = paths.prefillLen
        self.backbone = try await StatefulGraphModel(contentsOf: paths.backbone, computeUnits: computeUnits, kvCapacity: 512)
        self.patchEncoder = try await StatefulGraphModel(contentsOf: paths.patchEncoder, computeUnits: computeUnits, kvCapacity: 256)
        self.dit = try await GraphModel(contentsOf: paths.dit, computeUnits: computeUnits)
        self.vocoder = try await GraphModel(contentsOf: paths.vocoder, computeUnits: computeUnits)
        self.glue = try DotsGlue(directory: paths.glueDir)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
        if let pf = paths.prefill, FileManager.default.fileExists(atPath: pf.path) {
            self.prefillLM = try await StatefulGraphModel(contentsOf: pf, computeUnits: computeUnits, kvCapacity: 512)
        } else {
            self.prefillLM = nil
        }
        if let ds = paths.ditSmall, FileManager.default.fileExists(atPath: ds.path) {
            self.ditSmall = try await GraphModel(contentsOf: ds, computeUnits: computeUnits)
        } else {
            self.ditSmall = nil
        }
        try await warm()
    }

    /// Pre-run each bundle once so the first real generate() call is warm (moves GPU specialization
    /// out of the timed loop; otherwise the first decode/solver step pays a cold-start).
    private func warm() async throws {
        let H = Self.hidden
        backbone.resetState(); patchEncoder.resetState()
        _ = try? await backbone.step(["inputs_embeds": .float32([Float](repeating: 0, count: H), shape: [1, 1, H]),
                                      "pos": .int32([0], shape: [1])])
        _ = try? await patchEncoder.step([
            "latent_patch": .float32([Float](repeating: 0, count: Self.patch * Self.latentDim), shape: [1, Self.patch, Self.latentDim]),
            "conv_tail": .float32([Float](repeating: 0, count: Self.latentDim), shape: [1, Self.latentDim, 1]),
            "pos": .int32([0], shape: [1])])
        for (m, cap) in [(dit, Self.cap), (ditSmall, Self.capSmall)] {
            guard let m else { continue }
            _ = try? await warmDit(m, cap: cap)
        }
        _ = try? await vocoder.run(["latents": .float32([Float](repeating: 0, count: Self.latentDim * Self.vaeFrames),
                                                        shape: [1, Self.latentDim, Self.vaeFrames])])
        if let pLM = prefillLM {
            pLM.resetState()
            _ = try? await pLM.step(["inputs_embeds": .float32([Float](repeating: 0, count: prefillLen * H), shape: [1, prefillLen, H])])
            pLM.resetState()
        }
        backbone.resetState(); patchEncoder.resetState()
    }

    private func warmDit(_ m: GraphModel, cap: Int) async throws {
        let b = decoder == .mf ? 1 : 2
        var inp: [String: TensorValue] = [
            "x": .float32([Float](repeating: 0, count: b * cap * Self.ditDim), shape: [b, cap, Self.ditDim]),
            "timesteps": .float32([Float](repeating: 0, count: b), shape: [b]),
            "attn_mask": .float32(buildFmAttnMask(1, cap) as [Float], shape: [1, cap, cap]),
            "pos_ids": .float32(buildFmPosIds(1, cap) as [Float], shape: [1, cap]),
            "g_cond": .float32([Float](repeating: 0, count: b * Self.ditDim), shape: [b, Self.ditDim])]
        if decoder == .mf { inp["duration"] = .float32([0], shape: [1]) }
        _ = try await m.run(inp)
    }

    /// Synthesize the whole clip as mono Float PCM at 48 kHz.
    public func synthesize(_ text: String, seed: UInt64 = 0, maxFrames: Int = 500, minFrames: Int = 2) async throws -> [Float] {
        var out: [Float] = []
        _ = try await generate(text, seed: seed, maxFrames: maxFrames, minFrames: minFrames) { out.append(contentsOf: $0) }
        return out
    }

    @discardableResult
    public func synthesizeStreaming(_ text: String, seed: UInt64 = 0, maxFrames: Int = 500, minFrames: Int = 2,
                                    onChunk: @Sendable ([Float]) async -> Void) async throws -> StreamStats {
        try await generate(text, seed: seed, maxFrames: maxFrames, minFrames: minFrames) { await onChunk($0) }
    }

    private func generate(_ text: String, seed: UInt64, maxFrames: Int, minFrames: Int,
                          emit: ([Float]) async throws -> Void) async throws -> StreamStats {
        let start = Date(); profile = [:]
        var firstChunk = -1.0; var totalSamples = 0

        // ---- schedule (prefill tokens) ----
        var ids = Self.prefixIds
        ids.append(contentsOf: tokenizer.encode(text: text, addSpecialTokens: false))
        ids.append(contentsOf: Self.suffixIds)
        let P = ids.count

        // ---- prefill: one batched q=32 call (adoptState into decode) if it fits, else decode-via-decode ----
        let pf = Date()
        backbone.resetState(); patchEncoder.resetState()
        var lmH = [Float](repeating: 0, count: Self.hidden)
        if let pLM = prefillLM, P <= prefillLen {
            pLM.resetState()
            let L = prefillLen
            var emb = [Float](repeating: 0, count: L * Self.hidden)
            for t in 0..<P {
                glue.embedRow(ids[t]).withUnsafeBufferPointer { src in
                    emb.withUnsafeMutableBufferPointer { dst in
                        dst.baseAddress!.advanced(by: t * Self.hidden).update(from: src.baseAddress!, count: Self.hidden) } }
            }
            let o = try await pLM.step(["inputs_embeds": .float32(emb, shape: [1, L, Self.hidden])])
            let all = try out(o, pLM.outputName)                        // [L*hidden]
            lmH = Array(all[(P - 1) * Self.hidden ..< P * Self.hidden])  // hidden at the last real token
            backbone.adoptState(from: pLM)                              // continue decode from the prefill KV
        } else {
            for t in 0..<P {
                let o = try await backbone.step(["inputs_embeds": .float32(glue.embedRow(ids[t]), shape: [1, 1, Self.hidden]),
                                                 "pos": .int32([Int32(t)], shape: [1])])
                lmH = try out(o, backbone.outputName)
            }
        }
        tick("prefill", pf)

        // ---- fm_sequence buffers (cap x ditDim, flat) + append the prefill hidden ----
        var fm = [Float](repeating: 0, count: Self.cap * Self.ditDim)
        var fmCfg = [Float](repeating: 0, count: Self.cap * Self.ditDim)
        var fmLen = 0
        let zeroHidden = [Float](repeating: 0, count: Self.hidden)
        let nullHiddenProj = glue.hiddenProj(zeroHidden)
        func appendHidden(_ h: [Float]) {
            let p = glue.hiddenProj(h)
            for d in 0..<Self.ditDim { fm[fmLen * Self.ditDim + d] = p[d]; fmCfg[fmLen * Self.ditDim + d] = nullHiddenProj[d] }
            fmLen += 1
        }
        func appendHistory(_ normPatch: [Float]) {   // latent_proj of the NORMALIZED patch
            let p = glue.latentProj4(normPatch)       // [patch*ditDim]
            for pi in 0..<Self.patch { for d in 0..<Self.ditDim {
                let v = p[pi * Self.ditDim + d]
                fm[(fmLen + pi) * Self.ditDim + d] = v; fmCfg[(fmLen + pi) * Self.ditDim + d] = v
            } }
            fmLen += Self.patch
        }
        appendHidden(lmH)

        var rng = GaussianRNG(seed: seed)
        var convTail = [Float](repeating: 0, count: Self.latentDim)   // [128]
        var peSeq = 0
        var allDenorm: [[Float]] = []          // every denorm patch (for the vocode past-context window)
        var emitted = 0                        // patches whose audio has been emitted

        // emit ready patches: vocode [ctxPast | emit | ctxFuture] and emit only the middle `emit` part.
        // Non-final needs `ctxFuture` patches beyond the emit region available (holdback); final flushes
        // the tail with zero future-pad (= utterance end).
        func drain(final: Bool) async throws {
            let step = Self.emitPatches
            while true {
                let avail = allDenorm.count - emitted
                let ready = final ? (avail > 0) : (avail >= step + Self.ctxFuture)
                if !ready { break }
                let n = min(step, avail)
                let vt = Date()
                let wav = try await vocodeWindow(allDenorm, emit0: emitted, emitN: n)
                tick("vocoder", vt)
                emitted += n
                if firstChunk < 0 { firstChunk = Date().timeIntervalSince(start) }
                totalSamples += wav.count
                try await emit(wav)
            }
        }

        for i in 0..<maxFrames {
            if i > minFrames && glue.eosProb(lmH) > 0.8 { break }
            // ---- solver (soar): denoise one patch over the padded fm_sequence ----
            let st = Date()
            // pick the smallest DiT bucket that fits fm_len+4 (early/short patches skip the big S=164)
            let useSmall = ditSmall != nil && fmLen + Self.patch <= Self.capSmall
            let cap = useSmall ? Self.capSmall : Self.cap
            let ditM = useSmall ? ditSmall! : dit
            let mask = buildFmAttnMask(fmLen, cap)
            let pos = buildFmPosIds(fmLen, cap)
            let noise = rng.normals(Self.patch * Self.latentDim)
            let patch = decoder == .mf
                ? try await solveMf(dit: ditM, cap: cap, fm: fm, mask: mask, pos: pos, noise: noise)
                : try await solveSoar(dit: ditM, cap: cap, fm: fm, fmCfg: fmCfg, mask: mask, pos: pos, noise: noise)
            tick("solver", st)
            let denorm = glue.denormalize(patch)
            allDenorm.append(denorm)

            // ---- feedback ----
            appendHistory(patch)                                       // NORMALIZED patch
            let pt = Date()
            let po = try await patchEncoder.step([
                "latent_patch": .float32(denorm, shape: [1, Self.patch, Self.latentDim]),
                "conv_tail": .float32(convTail, shape: [1, Self.latentDim, 1]),
                "pos": .int32([Int32(peSeq)], shape: [1])])
            let emb = try out(po, patchEncoder.outputName)             // [1536]
            tick("patchEncoder", pt)
            // new_conv_tail = last latent frame of the DENORMALIZED patch (layout [p*128+c] -> [c])
            for c in 0..<Self.latentDim { convTail[c] = denorm[(Self.patch - 1) * Self.latentDim + c] }
            peSeq += 2

            let bt = Date()
            let bo = try await backbone.step(["inputs_embeds": .float32(emb, shape: [1, 1, Self.hidden]),
                                              "pos": .int32([Int32(P + i)], shape: [1])])
            lmH = try out(bo, backbone.outputName)
            tick("backbone", bt)
            appendHidden(lmH)

            try await drain(final: false)
        }
        try await drain(final: true)

        let total = Date().timeIntervalSince(start)
        return StreamStats(samples: totalSamples, firstChunkSeconds: firstChunk < 0 ? total : firstChunk,
                           totalSeconds: total, sampleRate: Self.sampleRate)
    }

    // MARK: - solver

    /// One denoised patch [patch*latentDim] (layout [p*128+c], normalized) via 10-step euler + CFG.
    private func solveSoar(dit m: GraphModel, cap: Int, fm: [Float], fmCfg: [Float], mask: [Float], pos: [Float], noise: [Float]) async throws -> [Float] {
        let latentStart = cap - Self.patch
        var z = noise                                                  // [patch*latentDim]
        let g2 = [Float](repeating: 0, count: 2 * Self.ditDim)         // no-prompt g_cond = 0 (cond+uncond)
        for step in 0..<Self.numSteps {
            let t = Float(step) / Float(Self.numSteps)
            let zp = glue.coordProj4(z)                                // [patch*ditDim]
            var x = [Float](repeating: 0, count: 2 * cap * Self.ditDim)
            let condBase = 0, cfgBase = cap * Self.ditDim
            for k in 0..<(latentStart * Self.ditDim) { x[condBase + k] = fm[k]; x[cfgBase + k] = fmCfg[k] }
            for pi in 0..<Self.patch { for d in 0..<Self.ditDim {
                let v = zp[pi * Self.ditDim + d]
                x[condBase + (latentStart + pi) * Self.ditDim + d] = v
                x[cfgBase + (latentStart + pi) * Self.ditDim + d] = v
            } }
            let o = try await m.run([
                "x": .float32(x, shape: [2, cap, Self.ditDim]),
                "timesteps": .float32([t, t], shape: [2]),
                "attn_mask": .float32(mask, shape: [1, cap, cap]),
                "pos_ids": .float32(pos, shape: [1, cap]),
                "g_cond": .float32(g2, shape: [2, Self.ditDim])])
            let vel = try out(o, "velocity")                           // [2*cap*latentDim]
            for pi in 0..<Self.patch { for c in 0..<Self.latentDim {
                let cIdx = (latentStart + pi) * Self.latentDim + c
                let uIdx = cap * Self.latentDim + (latentStart + pi) * Self.latentDim + c
                let v = vel[cIdx] + Self.guidance * (vel[cIdx] - vel[uIdx])
                z[pi * Self.latentDim + c] += (1.0 / Float(Self.numSteps)) * v
            } }
        }
        return z
    }

    /// MeanFlow: NFE steps, batch-1, NO CFG, +duration. z += velocity*dt each step. ~5× cheaper than soar.
    private func solveMf(dit m: GraphModel, cap: Int, fm: [Float], mask: [Float], pos: [Float], noise: [Float]) async throws -> [Float] {
        let latentStart = cap - Self.patch
        var z = noise
        let g1 = [Float](repeating: 0, count: Self.ditDim)   // no-prompt g_cond = 0
        for step in 0..<Self.nfe {
            let t = Float(step) / Float(Self.nfe)
            let dt = 1.0 / Float(Self.nfe)
            let zp = glue.coordProj4(z)                       // [patch*ditDim]
            var x = [Float](repeating: 0, count: cap * Self.ditDim)  // batch-1: [cap*ditDim]
            for k in 0..<(latentStart * Self.ditDim) { x[k] = fm[k] }
            for pi in 0..<Self.patch { for d in 0..<Self.ditDim {
                x[(latentStart + pi) * Self.ditDim + d] = zp[pi * Self.ditDim + d]
            } }
            let o = try await m.run([
                "x": .float32(x, shape: [1, cap, Self.ditDim]),
                "timesteps": .float32([t], shape: [1]),
                "attn_mask": .float32(mask, shape: [1, cap, cap]),
                "pos_ids": .float32(pos, shape: [1, cap]),
                "g_cond": .float32(g1, shape: [1, Self.ditDim]),
                "duration": .float32([dt], shape: [1])])
            let vel = try out(o, "velocity")                  // [cap*latentDim]
            for pi in 0..<Self.patch { for c in 0..<Self.latentDim {
                z[pi * Self.latentDim + c] += dt * vel[(latentStart + pi) * Self.latentDim + c]
            } }
        }
        return z
    }

    // MARK: - fm mask / pos_ids (verbatim conversion/dots_tts/export_dit.py)

    private func buildFmAttnMask(_ fmLen: Int, _ cap: Int) -> [Float] {
        var m = [Float](repeating: 0, count: cap * cap)
        let ls = cap - Self.patch, blk = fmLen - 1
        func set(_ r: Int, _ c: Int) { m[r * cap + c] = 1 }
        if blk > 0 { for r in 0..<blk { for c in 0...r { set(r, c) } } }            // causal
        for r in max(blk, 0)..<fmLen { for c in 0..<fmLen { set(r, c) }; for c in ls..<cap { set(r, c) } }
        for r in ls..<cap { for c in 0..<fmLen { set(r, c) }; for c in ls..<cap { set(r, c) } }
        if ls > fmLen { for d in fmLen..<ls { set(d, d) } }                          // padding diagonal
        return m
    }

    private func buildFmPosIds(_ fmLen: Int, _ cap: Int) -> [Float] {
        var p = [Float](repeating: 0, count: cap)
        for i in 0..<fmLen { p[i] = Float(i) }
        for i in 0..<Self.patch { p[(cap - Self.patch) + i] = Float(fmLen + i) }
        return p
    }

    // MARK: - vocoder (streaming, overlap window)

    /// Vocode a Tf=16 window laid out as [ctxPatches past-context | emitN new patches | zero pad], and
    /// return ONLY the emitN new patches' samples. The causal vocoder's ~2-patch field is covered by the
    /// past context, so the emitted samples equal a full-clip vocode (clean boundary, cos 0.999998). The
    /// leading context is zeros for the utterance start (= correct silence).
    private func vocodeWindow(_ all: [[Float]], emit0: Int, emitN: Int) async throws -> [Float] {
        let ctx = Self.ctxPast, F = Self.vaeFrames, D = Self.latentDim, P = Self.patch
        let slots = F / P                                // 8 window patch-slots
        var win = [Float](repeating: 0, count: D * F)    // [128,32], zero-padded (past/future edges)
        // slot s -> source patch (emit0 - ctx + s); emit patches occupy slots [ctx .. ctx+emitN)
        for s in 0..<slots {
            let src = emit0 - ctx + s
            guard src >= 0 && src < all.count else { continue }
            let pat = all[src]                           // [p*128+c]
            for c in 0..<D { for pp in 0..<P {
                win[c * F + s * P + pp] = pat[pp * D + c]
            } }
        }
        let o = try await vocoder.run(["latents": .float32(win, shape: [1, D, F])])
        let wav = try out(o, "wav")
        let lo = ctx * P * Self.colSamples, hi = (ctx + emitN) * P * Self.colSamples
        return Array(wav[lo..<min(hi, wav.count)])
    }

    // MARK: - helpers

    private func out(_ d: [String: TensorValue], _ name: String) throws -> [Float] {
        guard let v = d[name] else { throw DotsError.missingOutput(name) }
        return v.floats()
    }
}

/// Deterministic standard-normal generator (SplitMix64 + Box–Muller) — the solver's only stochastic input.
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
    private mutating func uniform() -> Float { Float(next() >> 40) * (1.0 / Float(1 << 24)) + Float.leastNonzeroMagnitude }
    mutating func normals(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n); var i = 0
        while i < n {
            let u1 = uniform(), u2 = uniform(); let r = (-2 * logf(u1)).squareRoot()
            out[i] = r * cosf(2 * .pi * u2); if i + 1 < n { out[i + 1] = r * sinf(2 * .pi * u2) }
            i += 2
        }
        return out
    }
}
