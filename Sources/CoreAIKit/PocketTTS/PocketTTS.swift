// PocketTTS.swift — on-device Kyutai pocket-tts text-to-speech host.
// Community port, NOT an Apple model. Three Core AI assets over four graphs:
//
//   flowlm_<dtype>_s512          multifunction: `prefill` + `step`, one weight set, one
//                                shared in-graph KV (`k_cache`/`v_cache`, packed
//                                [6,1,16,512,64])          text_emb|latent -> cond, eos_logit
//   flow_decoder_<dtype>_lsd1    stateless                 cond + noise -> latent[1,32]
//   mimi_decoder_float32_...     12 in-graph state tensors  latent[1,32] -> pcm[1,1,1920]
//
// Per AR step: flow-LM step -> flow decoder -> Mimi frame, 12.5 Hz, 1920 samples a frame.
// EOS is a host-side threshold compare on `eos_logit`; the latent produced on the breaking
// step is discarded, matching upstream's `_autoregressive_generation`.
//
// This type talks to `CoreAI` directly rather than through `GraphModel`/`StatefulGraphModel`
// — see the header of `PocketTTSGraph.swift` for why, and `OCR/KitDocReader.swift` for the
// same pattern applied to the OCR decoder.

import CoreAI
import CoreAIKitVision
import Foundation

private typealias ND = PocketTTSND

public struct PocketTTSPaths: Sendable {
    /// Graph precision of the flow-LM and flow decoder. The Mimi decoder is fp32 at either
    /// setting: twelve state tensors feeding back at 12.5 Hz is exactly where fp16 error
    /// compounds audibly, so there is no fp16 Mimi asset to point at.
    public enum Precision: String, Sendable { case float16, float32 }

    public var flowLM: URL, flowDecoder: URL, mimi: URL
    /// `model.safetensors` — the text-embedding LUT the host looks up outside any graph.
    public var weights: URL
    /// `tokenizer.model`, sentencepiece.
    public var tokenizer: URL
    /// `embeddings/`, one `<voice>.safetensors` per voice holding its pre-baked flow-LM KV.
    public var voicesDir: URL

    public init(flowLM: URL, flowDecoder: URL, mimi: URL,
                weights: URL, tokenizer: URL, voicesDir: URL) {
        self.flowLM = flowLM; self.flowDecoder = flowDecoder; self.mimi = mimi
        self.weights = weights; self.tokenizer = tokenizer; self.voicesDir = voicesDir
    }

    private static func make(_ resolve: (String) -> URL, precision: Precision,
                             weights: URL, tokenizer: URL, voicesDir: URL) -> PocketTTSPaths {
        PocketTTSPaths(
            flowLM: resolve("flowlm_\(precision.rawValue)_s\(PocketTTSModel.sMax)"),
            flowDecoder: resolve("flow_decoder_\(precision.rawValue)_lsd1"),
            mimi: resolve("mimi_decoder_float32_ring272_outer_q_gs"),
            weights: weights, tokenizer: tokenizer, voicesDir: voicesDir)
    }

    /// macOS dev layout: flat `<root>/<name>.aimodel` (JIT-specialized).
    public static func standard(artifactsRoot root: URL, precision: Precision = .float16,
                                weights: URL, tokenizer: URL, voicesDir: URL) -> PocketTTSPaths {
        make({ root.appendingPathComponent("\($0).aimodel") },
             precision: precision, weights: weights, tokenizer: tokenizer, voicesDir: voicesDir)
    }

    /// iOS AOT layout: flat `<root>/<name>.<arch>.aimodelc`.
    public static func aot(root: URL, arch: String = "h18p", precision: Precision = .float16,
                           weights: URL, tokenizer: URL, voicesDir: URL) -> PocketTTSPaths {
        make({ root.appendingPathComponent("\($0).\(arch).aimodelc") },
             precision: precision, weights: weights, tokenizer: tokenizer, voicesDir: voicesDir)
    }

    /// Resolve the three weight pieces out of a HuggingFace snapshot tree, where the
    /// checkpoint revision and the voice-embedding revision are two different snapshots, so
    /// the directory is globbed rather than pinned.
    public static func discoverWeights(root: URL) throws -> (weights: URL, tokenizer: URL, voicesDir: URL) {
        let l = try PocketTTSWeightsLayout.discover(root: root)
        return (l.modelURL, l.tokenizerURL, l.embeddingsDir)
    }
}

/// Per-chunk telemetry from one synthesis run, kept because the chunker's budget is the
/// invariant that makes the whole pipeline safe: `posStart + steps` must never reach S_MAX.
public struct PocketTTSChunkStat: Sendable {
    public var text = ""
    public var tokens = 0
    public var windows = 0
    public var posStart = 0
    public var headroom = 0
    public var steps = 0
    public var frames = 0
    public var eosStep: Int?
    public var hitMaxGenLen = false
    public var durationSeconds = 0.0
}

public final class PocketTTS: @unchecked Sendable {
    public static let sampleRate = PocketTTSModel.sampleRate

    private let lm: PocketTTSAsset
    private let flow: PocketTTSAsset
    private let mimiAsset: PocketTTSAsset
    private let fPrefill: InferenceFunction
    private let fStep: InferenceFunction
    private let fFlow: InferenceFunction
    private let fMimi: InferenceFunction
    private let half: Bool

    private let weights: PocketTTSHostWeights
    private let sp: PocketTTSSentencePiece
    private let chunker: PocketTTSChunker
    private let voicesDir: URL

    /// The voices shipped beside the bundles, sorted. Each is a pre-baked flow-LM KV state.
    public let voices: [String]
    public private(set) var loadSeconds: Double = 0
    /// Wall-clock attribution for the last run, in milliseconds: `prefill`, `step`, `flow`,
    /// `mimi` are engine time; `lut`, `marshal`, `flatten` are host glue; `engineCalls` is a
    /// count. Whatever is left of `wall` is loop and async overhead.
    public private(set) var profile: [String: Double] = [:]
    /// Per-chunk detail for the last run.
    public private(set) var chunks: [PocketTTSChunkStat] = []

    public init(paths: PocketTTSPaths, precision: PocketTTSPaths.Precision = .float16,
                computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        self.half = (precision == .float16)
        for u in [paths.flowLM, paths.flowDecoder, paths.mimi]
        where !FileManager.default.fileExists(atPath: u.path) {
            throw PocketTTSError.message("missing asset \(u.lastPathComponent)")
        }
        let t0 = ND.nowNanos()
        lm = try await PocketTTSAsset(url: paths.flowLM, unit: computeUnits)
        flow = try await PocketTTSAsset(url: paths.flowDecoder, unit: computeUnits)
        mimiAsset = try await PocketTTSAsset(url: paths.mimi, unit: computeUnits)
        fPrefill = try lm.function("prefill")
        fStep = try lm.function("step")
        fFlow = try flow.function("main")
        fMimi = try mimiAsset.function("main")
        weights = try PocketTTSHostWeights(modelURL: paths.weights)
        sp = try PocketTTSSentencePiece(url: paths.tokenizer)
        chunker = PocketTTSChunker(sp: sp)
        voicesDir = paths.voicesDir
        voices = ((try? FileManager.default.contentsOfDirectory(atPath: paths.voicesDir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }
            .map { String($0.dropLast(".safetensors".count)) }
            .sorted()
        loadSeconds = Double(ND.nowNanos() - t0) / 1e9
    }

    /// Graph-dtype NDArray from float32 host values.
    @inline(__always) private func gd(_ v: [Float], _ shape: [Int]) -> NDArray {
        half ? ND.ndHalf(v, shape) : ND.nd(v, shape)
    }

    /// The 12 Mimi streaming-state buffers, host-owned for the lifetime of one run. The
    /// runtime mutates them in place through `MutableViews` — nothing round-trips. Allocated
    /// once per *run* and never per chunk: upstream never resets Mimi state, and resetting it
    /// clicks at every chunk boundary.
    private struct MimiState {
        var upP = zeros([1, 512, 16])
        var kv0 = zeros([2, 1, PocketTTSModel.mimiRing, 8, 64])
        var kv1 = zeros([2, 1, PocketTTSModel.mimiRing, 8, 64])
        var offset: NDArray = {
            var a = NDArray(shape: [1], scalarType: .int32)
            var mv = a.mutableView(as: Int32.self)
            mv.withUnsafeMutablePointer { p, _, _ in p[0] = 0 }
            return a
        }()
        var c0 = zeros([1, 512, 6]); var c2 = zeros([1, 256, 6]); var c3 = zeros([1, 256, 2])
        var c5 = zeros([1, 128, 5]); var c6 = zeros([1, 128, 2])
        var c8 = zeros([1, 64, 4]); var c9 = zeros([1, 64, 2]); var c11 = zeros([1, 64, 2])

        /// Zero-initialised, never NaN: a NaN-filled fixed-capacity cache poisons the first
        /// frames through the causal field even though those slots are logically unreachable.
        static func zeros(_ shape: [Int]) -> NDArray {
            var a = NDArray(shape: shape, scalarType: .float32)
            var mv = a.mutableView(as: Float.self)
            mv.withUnsafeMutablePointer { p, shp, _ in
                let n = shp.indices.reduce(1) { $0 * shp[$1] }
                for i in 0..<n { p[i] = 0 }
            }
            return a
        }
    }

    /// One Mimi frame: raw flow-decoder latent `[1,32]` in, 1920 PCM samples out. State
    /// advances in place inside the graph.
    private func mimiFrame(latent: NDArray, state: inout MimiState,
                           engineNanos: inout UInt64, flattenNanos: inout UInt64) async throws -> [Float] {
        var states = InferenceFunction.MutableViews()
        states.insert(&state.upP, for: "up_p")
        states.insert(&state.kv0, for: "kv0")
        states.insert(&state.kv1, for: "kv1")
        states.insert(&state.offset, for: "offset")
        states.insert(&state.c0, for: "c0"); states.insert(&state.c2, for: "c2")
        states.insert(&state.c3, for: "c3"); states.insert(&state.c5, for: "c5")
        states.insert(&state.c6, for: "c6"); states.insert(&state.c8, for: "c8")
        states.insert(&state.c9, for: "c9"); states.insert(&state.c11, for: "c11")
        let te = ND.nowNanos()
        var out = try await fMimi.run(inputs: ["latent": latent], states: states)
        engineNanos &+= ND.nowNanos() &- te
        let tf = ND.nowNanos()
        let pcm = try ND.take(&out, "pcm")
        flattenNanos &+= ND.nowNanos() &- tf
        return pcm
    }

    // MARK: - synthesis

    /// Synthesize the whole clip as mono Float PCM at 24 kHz.
    public func synthesize(_ text: String, voice: String, seed: UInt64 = 0,
                           applyGain: Bool = true) async throws -> [Float] {
        var out: [Float] = []
        _ = try await generate(text, voice: voice, seed: seed, applyGain: applyGain) {
            out.append(contentsOf: $0)
        }
        return out
    }

    /// Same generation, emitting each chunk's audio as it completes.
    ///
    /// `applyGain` is off here by default: the per-voice gain needs the whole clip's peak to
    /// clamp against, so applying it chunk-wise would change the normalisation mid-stream.
    @discardableResult
    public func synthesizeStreaming(_ text: String, voice: String, seed: UInt64 = 0,
                                    applyGain: Bool = false,
                                    onChunk: @Sendable ([Float]) async -> Void) async throws -> StreamStats {
        try await generate(text, voice: voice, seed: seed, applyGain: applyGain) { await onChunk($0) }
    }

    /// Lifetime note: `InferenceFunction.MutableViews` is `~Escapable` and borrows its arrays
    /// up to `function.run`, so each state buffer needs storage that outlives every call it is
    /// inserted into. Here the KV cache is per-chunk and the Mimi state per-run, so both are
    /// locals of this function rather than stored properties — which is why prefill and the AR
    /// loop share one scope. (`DocDecoder` makes the other choice, holding its cache as stored
    /// properties, because its KV lives as long as the object.)
    @discardableResult
    private func generate(_ text: String, voice: String, seed: UInt64, applyGain: Bool,
                          emit: ([Float]) async throws -> Void) async throws -> StreamStats {
        let voiceState = try PocketTTSVoiceState(name: voice, embeddingsDir: voicesDir)
        var noise = PocketTTSNoiseSource(seed: seed, temp: PocketTTSModel.temp)
        var mimiState = MimiState()        // NEVER reset across chunks
        var stepN: UInt64 = 0, flowN: UInt64 = 0, mimiN: UInt64 = 0, preN: UInt64 = 0
        var lutN: UInt64 = 0, marshN: UInt64 = 0, flatN: UInt64 = 0
        var engineCalls = 0
        var totalSamples = 0
        var firstChunk = -1.0
        chunks = []

        let chunkTexts = try chunker.chunk(text, voicePos: voiceState.positions)
        let t0 = ND.nowNanos()

        for chunkText in chunkTexts {
            var cs = PocketTTSChunkStat()
            cs.text = chunkText

            // frames_after_eos: model_recommended is nil for this config, so it is the
            // prepare_text_prompt guess (3 for <=4 words else 1) + 2.
            let framesAfterEOS = try PocketTTSChunker.prepareTextPrompt(chunkText).framesGuess + 2

            // Upstream tokenizes the CHUNK, not the prepared text; prepare_text_prompt is
            // consulted only for the frames guess.
            let tokens = sp.encode(chunkText)
            cs.tokens = tokens.count
            let maxGenLen = PocketTTSModel.maxGenLen(tokens: tokens.count)
            precondition(voiceState.positions + tokens.count + maxGenLen <= PocketTTSModel.sMax,
                         "chunker invariant violated: \(voiceState.positions)+\(tokens.count)+\(maxGenLen) > \(PocketTTSModel.sMax)")
            let tl = ND.nowNanos()
            let textEmb = weights.embed(tokens)
            lutN &+= ND.nowNanos() &- tl

            // Fresh KV per chunk, re-seeded from the voice — upstream deep-copies it per
            // chunk, so a chunk never sees the previous chunk's text.
            var kCache = ND.makeState(voiceState.kSeed, shape: [6, 1, 16, PocketTTSModel.sMax, 64], half: half)
            var vCache = ND.makeState(voiceState.vSeed, shape: [6, 1, 16, PocketTTSModel.sMax, 64], half: half)
            var pos = Int32(voiceState.positions)

            // Windowed prefill over the static T_PRE=16 graph, carrying `pos` across windows.
            // Pad rows write junk into slots strictly after every real position; the causal
            // mask makes them unreachable and the first AR steps overwrite them unread.
            var w = 0
            while w < tokens.count {
                let width = min(PocketTTSModel.tPre, tokens.count - w)
                let tm = ND.nowNanos()
                var win = [Float](repeating: 0, count: PocketTTSModel.tPre * PocketTTSModel.dModel)
                for j in 0..<(width * PocketTTSModel.dModel) { win[j] = textEmb[w * PocketTTSModel.dModel + j] }
                let winND = gd(win, [1, PocketTTSModel.tPre, PocketTTSModel.dModel])
                let posND = ND.nd([pos], [1])
                marshN &+= ND.nowNanos() &- tm
                var states = InferenceFunction.MutableViews()
                states.insert(&kCache, for: "k_cache")
                states.insert(&vCache, for: "v_cache")
                let ta = ND.nowNanos()
                var out = try await fPrefill.run(inputs: ["text_emb": winND, "pos": posND], states: states)
                preN &+= ND.nowNanos() &- ta
                engineCalls += 1
                _ = out.remove("cond")   // upstream discards the prefill latent too
                pos += Int32(width)
                w += width
                cs.windows += 1
            }
            cs.posStart = Int(pos)
            cs.headroom = PocketTTSModel.sMax - Int(pos)

            let chunkStart = totalSamples
            var chunkAudio: [Float] = []
            var latent = [Float](repeating: 0, count: PocketTTSModel.ldim)
            var isBOS: Float = 1.0
            var eosStep: Int?
            var i = 0
            while i < maxGenLen {
                // Past S_MAX the write corrupts context silently and EOS fires late. The
                // chunker's budget makes this unreachable; it stays as a hard stop.
                precondition(Int(pos) + i < PocketTTSModel.sMax,
                             "AR write position \(Int(pos) + i) reached S_MAX=\(PocketTTSModel.sMax)")

                let tm = ND.nowNanos()
                let latND = gd(latent, [1, 1, PocketTTSModel.ldim])
                let bosND = gd([isBOS], [1])
                let posND = ND.nd([pos + Int32(i)], [1])
                marshN &+= ND.nowNanos() &- tm
                var states = InferenceFunction.MutableViews()
                states.insert(&kCache, for: "k_cache")
                states.insert(&vCache, for: "v_cache")
                let ta = ND.nowNanos()
                var out = try await fStep.run(
                    inputs: ["latent_in": latND, "is_bos": bosND, "pos": posND], states: states)
                stepN &+= ND.nowNanos() &- ta
                engineCalls += 1
                isBOS = 0
                guard let condND = out.remove("cond")?.ndArray else {
                    throw PocketTTSError.message("step: missing 'cond'")
                }
                var tf = ND.nowNanos()
                let eos = try ND.take(&out, "eos_logit")[0]
                flatN &+= ND.nowNanos() &- tf
                if eos > PocketTTSModel.eosThreshold, eosStep == nil { eosStep = i }

                // `cond` goes straight from the step graph's output into the flow decoder's
                // input — same dtype, no host round-trip.
                let tn = ND.nowNanos()
                let nsND = gd(noise.nextNoise(count: PocketTTSModel.ldim), [1, PocketTTSModel.ldim])
                marshN &+= ND.nowNanos() &- tn
                let tb = ND.nowNanos()
                var fout = try await fFlow.run(inputs: ["cond": condND, "noise": nsND])
                flowN &+= ND.nowNanos() &- tb
                engineCalls += 1
                tf = ND.nowNanos()
                latent = try ND.take(&fout, "latent")
                flatN &+= ND.nowNanos() &- tf

                // The latent produced on the breaking step is DISCARDED.
                if let e = eosStep, i >= e + framesAfterEOS { i += 1; break }

                let tw = ND.nowNanos()
                let latMimiND = ND.nd(latent, [1, PocketTTSModel.ldim])
                marshN &+= ND.nowNanos() &- tw
                let pcm = try await mimiFrame(latent: latMimiND, state: &mimiState,
                                              engineNanos: &mimiN, flattenNanos: &flatN)
                engineCalls += 1
                chunkAudio.append(contentsOf: pcm)
                totalSamples += pcm.count
                i += 1
            }
            cs.steps = i
            cs.eosStep = eosStep
            cs.hitMaxGenLen = (eosStep == nil)
            cs.frames = (totalSamples - chunkStart) / PocketTTSModel.frameSamples
            cs.durationSeconds = Double(totalSamples - chunkStart) / Double(PocketTTSModel.sampleRate)
            chunks.append(cs)

            if applyGain { _ = PocketTTSVoiceGain.apply(&chunkAudio, voice: voice) }
            if firstChunk < 0 { firstChunk = Double(ND.nowNanos() - t0) / 1e9 }
            try await emit(chunkAudio)
        }

        let total = Double(ND.nowNanos() - t0) / 1e9
        profile = ["wall": total * 1000, "prefill": Double(preN) / 1e6, "step": Double(stepN) / 1e6,
                   "flow": Double(flowN) / 1e6, "mimi": Double(mimiN) / 1e6,
                   "lut": Double(lutN) / 1e6, "marshal": Double(marshN) / 1e6,
                   "flatten": Double(flatN) / 1e6, "engineCalls": Double(engineCalls)]
        return StreamStats(samples: totalSamples,
                           firstChunkSeconds: firstChunk < 0 ? total : firstChunk,
                           totalSeconds: total, sampleRate: Self.sampleRate)
    }
}
