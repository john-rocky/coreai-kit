// Audio8TTS.swift — on-device Audio8-TTS-Preview-0.6b (Edge0, Apache-2.0): multilingual text-to-speech with
// zero-shot voice cloning, a DualAR model (Fish Audio S2 Pro design) on two Core AI assets.
//
//   dualar   prefill(codes [1,11,32], pos)                          -> logits [32,4097] f16, hidden [32,896] f16
//            frame(codes [1,11,1], pos, noise_slow [2,4097], window [10], noise_fast [9,4096], forced [11], use_forced [1])
//                                                                    -> semantic [1], codes [10], logits, hidden, fast_logits, …
//            first_frame(logits [4097], hidden [896], noise…)        -> the same, for the frame right after the prefill
//            one KV state pair [24, 1, 2, 2048, 64] f16 shared by prefill and frame
//   codec    main(codes [1,10,160]) -> wav [1, 160*2048] f16 at 44.1 kHz    stateless, causal
//
// One graph call per frame: the slow AR step, the semantic draw (top-k / top-p / temperature, Gumbel-max, the RAS
// repetition rule), and the fast AR's ten rows with their nine codebook draws all run inside the `frame` function; the
// host supplies the uniform draws (`Audio8NoiseSource`) and keeps the 10-token RAS window. `forced` / `use_forced`
// let the gate teacher-force a frame. Audio streams out every `chunkFrames` frames: the codec decodes the last 160
// frames (128 of context — the codec transformer's attention window — plus the new chunk, right-padded) and only the
// new chunk's samples are emitted; every op in the codec is causal, so this equals a whole-utterance decode.
//
// Raw-CoreAI layer (`PocketTTSAsset` + `PocketTTSND`, shared with the pocket-tts host): the assets are multifunction
// with two state tensors and several outputs, which `GraphModel` / `StatefulGraphModel` do not model.

import CoreAI
import CoreAIKitVision
import Foundation
import Tokenizers

@available(macOS 27, iOS 27, *)
private typealias ND = PocketTTSND

/// File locations for the two assets + tokenizer.
public struct Audio8Paths: Sendable {
    public var dualar: URL
    public var codec: URL
    public var tokenizerDir: URL

    public init(dualar: URL, codec: URL, tokenizerDir: URL) {
        self.dualar = dualar; self.codec = codec; self.tokenizerDir = tokenizerDir
    }

    public static let dualarName = "audio8_dualar_int8_cl2048_w32"
    public static let codecName = "audio8_codec_decoder_fp16_t160"

    /// The zoo export / Hugging Face layout: `<root>/<name>.aimodel` (JIT) with `<root>/tokenizer/`.
    public static func standard(root: URL, tokenizerDir: URL? = nil, dualar: String = dualarName,
                                codec: String = codecName) -> Audio8Paths {
        Audio8Paths(dualar: root.appendingPathComponent("\(dualar).aimodel"),
                    codec: root.appendingPathComponent("\(codec).aimodel"),
                    tokenizerDir: tokenizerDir ?? root.appendingPathComponent("tokenizer"))
    }

    /// AOT layout: `<root>/<name>.<arch>.aimodelc`.
    @available(macOS 27, iOS 27, *)
    public static func aot(root: URL, arch: String = AIModel.deviceArchitectureName, tokenizerDir: URL? = nil,
                           dualar: String = dualarName, codec: String = codecName) -> Audio8Paths {
        Audio8Paths(dualar: root.appendingPathComponent("\(dualar).\(arch).aimodelc"),
                    codec: root.appendingPathComponent("\(codec).\(arch).aimodelc"),
                    tokenizerDir: tokenizerDir ?? root.appendingPathComponent("tokenizer"))
    }
}

/// Per-run statistics (engine time attribution and the frame count).
@available(macOS 27, iOS 27, *)
public struct Audio8RunStats: Sendable {
    public var promptTokens = 0
    public var frames = 0
    public var samples = 0
    public var endedWithEOS = false
    public var wallSeconds = 0.0
    public var prefillSeconds = 0.0
    /// Engine time of the `frame` / `first_frame` calls (slow step + sampling + fast AR).
    public var frameSeconds = 0.0
    public var codecSeconds = 0.0
    public var firstAudioSeconds = 0.0
    public var engineCalls = 0
    public var audioSeconds: Double { Double(samples) / Double(Audio8TTS.sampleRate) }
    /// Real-time factor of the whole run (compute time / audio time).
    public var rtf: Double { audioSeconds > 0 ? wallSeconds / audioSeconds : .infinity }
    /// The generated codes `[10][frames]`, for tests and voice registration.
    public var codes: [[Int32]] = []
}

@available(macOS 27, iOS 27, *)
public final class Audio8TTS: @unchecked Sendable {
    public static let sampleRate = 44100
    public static let frameSamples = 2048
    public static let prefillWidth = 32
    public static let cacheLength = 2048
    public static let codecFrames = 160
    public static let codecContext = 128
    static let dim = 896
    static let rows = 11

    private let dualarAsset: PocketTTSAsset
    private let codecAsset: PocketTTSAsset
    private let fPrefill: InferenceFunction
    private let fFrame: InferenceFunction
    private let fFirstFrame: InferenceFunction
    private let fCodec: InferenceFunction
    private let prompt: Audio8Prompt
    public private(set) var loadSeconds = 0.0
    /// Frames per emitted chunk in `synthesizeStreaming` (≤ 160 − 128 = 32 keeps every chunk exact).
    public var chunkFrames = 32
    /// Generation length cap (the checkpoint's generation_config: 512 frames ≈ 23.8 s).
    public var maxNewTokens = 512

    public init(paths: Audio8Paths, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        for u in [paths.dualar, paths.codec] where !FileManager.default.fileExists(atPath: u.path) {
            throw Audio8Error.message("missing asset \(u.lastPathComponent)")
        }
        let t0 = ND.nowNanos()
        dualarAsset = try await PocketTTSAsset(url: paths.dualar, unit: computeUnits)
        codecAsset = try await PocketTTSAsset(url: paths.codec, unit: computeUnits)
        fPrefill = try dualarAsset.function("prefill")
        fFrame = try dualarAsset.function("frame")
        fFirstFrame = try dualarAsset.function("first_frame")
        fCodec = try codecAsset.function("main")
        prompt = Audio8Prompt(tokenizer: try await AutoTokenizer.from(modelFolder: paths.tokenizerDir))
        loadSeconds = Double(ND.nowNanos() - t0) / 1e9
    }

    /// The packed prompt for `text` (+ voice), for tests: row 0 = ids, rows 1..10 = codebooks.
    public func promptRows(text: String, voice: Audio8Voice? = nil) throws -> (rows: [Int32], length: Int) {
        try prompt.build(text: text, voice: voice)
    }

    /// One utterance -> 44.1 kHz mono PCM in [-1, 1]. The codec runs once the frames are all there, in as few
    /// 160-frame windows as the utterance needs (one for anything up to 7.4 s) — the cheapest path; use
    /// `synthesizeStreaming` when audio should start before the utterance ends.
    public func synthesize(_ text: String, voice: Audio8Voice? = nil, seed: UInt64 = 0,
                           maxFrames: Int? = nil) async throws -> [Float] {
        var out: [Float] = []
        var noise = Audio8SeededNoise(seed: seed)
        _ = try await generate(text, voice: voice, noise: &noise, maxFrames: maxFrames, streaming: false) { out.append(contentsOf: $0) }
        return out
    }

    /// Streaming synthesis: `onChunk` gets each `chunkFrames`-frame chunk (~1.5 s at 32) as it decodes. Every chunk
    /// costs a 160-frame codec window (128 frames of context + the chunk), so the codec does about five times the
    /// work of `synthesize`; the concatenation of the chunks equals `synthesize` up to fp16 rounding.
    @discardableResult
    public func synthesizeStreaming(_ text: String, voice: Audio8Voice? = nil, seed: UInt64 = 0, maxFrames: Int? = nil,
                                    onChunk: @Sendable ([Float]) async -> Void) async throws -> Audio8RunStats {
        var noise = Audio8SeededNoise(seed: seed)
        return try await generate(text, voice: voice, noise: &noise, maxFrames: maxFrames, streaming: true) { await onChunk($0) }
    }

    /// The loop with an injectable noise source: an app passes `Audio8SeededNoise`; the smoke test and the gate app
    /// replay the oracle's recorded draws so the host can be compared with the Python engine run choice for choice.
    /// `forced` (11 ids per frame: semantic id, then the ten codebooks) teacher-forces the frames it covers.
    /// `streaming` decodes every `chunkFrames` frames (audio starts early); otherwise the codec runs at the end in
    /// 160-frame windows that share 128 frames of context (the same samples, a fraction of the codec work).
    public func generate<N: Audio8NoiseSource>(_ text: String, voice: Audio8Voice?, noise: inout N, maxFrames: Int?,
                                               forced: [[Int32]]? = nil, streaming: Bool = true,
                                               emit: ([Float]) async throws -> Void) async throws -> Audio8RunStats {
        var stats = Audio8RunStats()
        let t0 = ND.nowNanos()
        var nPre: UInt64 = 0, nFrame: UInt64 = 0, nCodec: UInt64 = 0

        let (rows, P) = try prompt.build(text: text, voice: voice)
        stats.promptTokens = P
        let budget = min(maxFrames ?? maxNewTokens, Self.cacheLength - P)
        guard budget > 0 else { throw Audio8Error.message("prompt of \(P) tokens leaves no room in the \(Self.cacheLength)-slot cache") }

        // ---- slow KV state, windowed prefill (pad rows write slots after every real position; the causal mask hides
        // them and the first frames overwrite them unread). Zero-initialised: a masked SDPA still multiplies V by 0
        // and 0 * NaN = NaN. ----
        let slowZeros = [Float](repeating: 0, count: 24 * 2 * Self.cacheLength * 64)
        var kCache = ND.makeState(slowZeros, shape: [24, 1, 2, Self.cacheLength, 64], half: true)
        var vCache = ND.makeState(slowZeros, shape: [24, 1, 2, Self.cacheLength, 64], half: true)
        var logits: [Float] = []
        var hidden: [Float] = []
        var start = 0
        while start < P {
            let real = min(Self.prefillWidth, P - start)
            var win = [Int32](repeating: 0, count: Self.rows * Self.prefillWidth)
            for c in 0..<Self.prefillWidth { win[c] = Audio8Sampling.pad }
            for r in 0..<Self.rows { for c in 0..<real { win[r * Self.prefillWidth + c] = rows[r * P + start + c] } }
            var states = InferenceFunction.MutableViews()
            states.insert(&kCache, for: "k_cache")
            states.insert(&vCache, for: "v_cache")
            let ta = ND.nowNanos()
            var out = try await fPrefill.run(inputs: ["codes": ND.nd(win, [1, Self.rows, Self.prefillWidth]),
                                                      "pos": ND.nd([Int32(start)], [1])], states: states)
            nPre &+= ND.nowNanos() &- ta
            stats.engineCalls += 1
            let l = try ND.take(&out, "logits")
            let h = try ND.take(&out, "hidden")
            let row = real - 1
            logits = Array(l[row * Audio8Sampling.allowedCount ..< (row + 1) * Audio8Sampling.allowedCount])
            hidden = Array(h[row * Self.dim ..< (row + 1) * Self.dim])
            start += real
        }
        stats.prefillSeconds = Double(nPre) / 1e9

        // ---- frames ----
        var window = Audio8Sampling.RASWindow()
        var frames: [[Int32]] = []          // frames[t] = the 10 codebooks
        var lastSemantic: Int32 = 0
        var emitted = 0

        /// Decode the frames [emitted, end) through one 160-frame window ending at `end` (the frames before
        /// `emitted` in the window are context and are dropped).
        func flush(upTo end: Int) async throws {
            guard end > emitted else { return }
            let winStart = max(0, end - Self.codecFrames)
            let real = end - winStart
            var codes = [Int32](repeating: 0, count: 10 * Self.codecFrames)
            for t in 0..<real { for cb in 0..<10 { codes[cb * Self.codecFrames + t] = frames[winStart + t][cb] } }
            let tc = ND.nowNanos()
            var out = try await fCodec.run(inputs: ["codes": ND.nd(codes, [1, 10, Self.codecFrames])])
            nCodec &+= ND.nowNanos() &- tc
            stats.engineCalls += 1
            let wav = try ND.take(&out, "wav")
            let from = (emitted - winStart) * Self.frameSamples
            let to = real * Self.frameSamples
            let chunk = Array(wav[from ..< to])
            emitted = end
            stats.samples += chunk.count
            if stats.firstAudioSeconds == 0 { stats.firstAudioSeconds = Double(ND.nowNanos() - t0) / 1e9 }
            try await emit(chunk)
        }

        for t in 0..<budget {
            let (uN, uH) = noise.slow()
            var uF = [Float]()
            uF.reserveCapacity(9 * Audio8Sampling.codebookSize)
            for _ in 0..<9 { uF.append(contentsOf: noise.fast()) }
            var forcedRow = [Int32](repeating: 0, count: Self.rows)
            var useForced: Float = 0
            if let forced, t < forced.count, forced[t].count == Self.rows { forcedRow = forced[t]; useForced = 1 }
            var inputs: [String: NDArray] = [
                "noise_slow": ND.nd(uN + uH, [2, Audio8Sampling.allowedCount]),
                "window": ND.nd(window.values ?? [Int32](repeating: -1, count: Audio8Sampling.rasWindow), [Audio8Sampling.rasWindow]),
                "noise_fast": ND.nd(uF, [9, Audio8Sampling.codebookSize]),
                "forced": ND.nd(forcedRow, [Self.rows]),
                "use_forced": ND.nd([useForced], [1]),
            ]
            let ta = ND.nowNanos()
            var out: InferenceFunction.Outputs
            if t == 0 {
                inputs["logits"] = ND.ndHalf(logits, [Audio8Sampling.allowedCount])
                inputs["hidden"] = ND.ndHalf(hidden, [Self.dim])
                out = try await fFirstFrame.run(inputs: inputs)
            } else {
                var column = [Int32](repeating: 0, count: Self.rows)
                column[0] = lastSemantic
                let prev = frames[frames.count - 1]
                for cb in 0..<10 { column[cb + 1] = prev[cb] }
                inputs["codes"] = ND.nd(column, [1, Self.rows, 1])
                inputs["pos"] = ND.nd([Int32(P + t - 1)], [1])
                var states = InferenceFunction.MutableViews()
                states.insert(&kCache, for: "k_cache")
                states.insert(&vCache, for: "v_cache")
                out = try await fFrame.run(inputs: inputs, states: states)
            }
            nFrame &+= ND.nowNanos() &- ta
            stats.engineCalls += 1
            let semantic = try ND.takeInt32(&out, "semantic")[0]
            if semantic == Audio8Sampling.eos { stats.endedWithEOS = true; break }
            let codes = try ND.takeInt32(&out, "codes")
            frames.append(codes)
            lastSemantic = semantic
            window.push(semantic)
            if streaming && frames.count - emitted >= chunkFrames { try await flush(upTo: frames.count) }
        }
        if streaming {
            try await flush(upTo: frames.count)
        } else {
            // whole utterance: the first window covers min(T, 160) frames; each later window adds 160 - 128 = 32
            let step = Self.codecFrames - Self.codecContext
            var end = min(frames.count, Self.codecFrames)
            while emitted < frames.count {
                try await flush(upTo: end)
                end = min(frames.count, end + step)
            }
        }
        stats.frames = frames.count
        stats.codes = (0..<10).map { cb in frames.map { $0[cb] } }
        stats.wallSeconds = Double(ND.nowNanos() - t0) / 1e9
        stats.frameSeconds = Double(nFrame) / 1e9
        stats.codecSeconds = Double(nCodec) / 1e9
        return stats
    }
}

@available(macOS 27, iOS 27, *)
extension PocketTTSND {
    /// Pull one named int32 output as `[Int32]`, consuming it out of the `Outputs` bag.
    static func takeInt32(_ outputs: inout InferenceFunction.Outputs, _ name: String) throws -> [Int32] {
        guard let v = outputs.remove(name)?.ndArray else { throw Audio8Error.message("missing output '\(name)'") }
        let total = v.shape.reduce(1, *)
        var out = [Int32](repeating: 0, count: total)
        v.view(as: Int32.self).withUnsafePointer { p, _, _ in
            for i in 0..<total { out[i] = p[i] }
        }
        return out
    }
}
