// VibeVoiceTTS.swift — VibeVoice-Realtime-0.5B: the zoo's multi-speaker / dialogue TTS.
//
// Five Core AI graphs + a host loop, mirroring conversion/vibevoice/host_e2e.py (Python E2E wav
// cos 0.9995 vs upstream) and the device-validated coreai-audio self-test:
//
//   main LM (4L, stateful KV)  — reads the script's text embeddings
//   tts  LM (20L, stateful KV) — the speech trunk; a SECOND instance carries the CFG-negative stream
//   diffusion head             — 5-step DPMSolver++ v-prediction over a 64-dim latent per frame
//   connector                  — latent -> next input embedding (the feedback path)
//   acoustic decoder           — latents[1,64,64] -> 24 kHz audio (3200 samples per frame)
//
// Everything the host needs beyond the graphs rides the same HF repo under `coreai_host/`:
// per-voice prefill KV (the upstream `.pt` presets, packed to flat fp16 by
// conversion/vibevoice/pack_voice_presets.py), the shared glue (type embeddings, EOS classifier,
// DPMSolver schedule), the Qwen2.5 tokenizer, and the fp16 `embed_tokens` table (mmapped, not read
// whole). fp16 throughout — int8 LMs diverge inside the speech feedback loop.

import CoreAIKitVision
import Foundation
import Tokenizers

/// One of the packaged voice presets (`coreai_host/voices/<name>`).
public struct VibeVoiceVoice: Sendable, Hashable, Codable {
    public let name: String
    public let language: String
    let mainPrefillLen: Int
    let ttsPrefillLen: Int
    let negPrefillLen: Int

    enum CodingKeys: String, CodingKey {
        case name, language
        case mainPrefillLen = "main_prefill_len"
        case ttsPrefillLen = "tts_prefill_len"
        case negPrefillLen = "neg_prefill_len"
    }
}

public enum VibeVoiceError: Error, Sendable {
    case assetMissing(String)
    case voiceNotFound(String)
    case emptyText
}

struct VibeVoiceGlue: Sendable, Codable {
    struct Step: Sendable, Codable { let t: Int; let alpha: Double; let sigma: Double; let lambda: Double }
    let hidden, vae_dim, hop, text_window, speech_window: Int
    let cfg: Double
    let ddpm_steps: Int
    let schedule: [Step]
    let scaling, bias: Double
    let main_layers, tts_layers, n_kv, head_dim: Int
    let sample_rate, decoder_frames: Int
}

/// File layout for the five bundles + the host assets.
public struct VibeVoicePaths: Sendable {
    public var mainLM: URL, ttsLM: URL, head: URL, connector: URL, decoder: URL
    public var glueDir: URL, voicesDir: URL, embedTokens: URL

    public init(mainLM: URL, ttsLM: URL, head: URL, connector: URL, decoder: URL,
                glueDir: URL, voicesDir: URL, embedTokens: URL) {
        self.mainLM = mainLM; self.ttsLM = ttsLM; self.head = head
        self.connector = connector; self.decoder = decoder
        self.glueDir = glueDir; self.voicesDir = voicesDir; self.embedTokens = embedTokens
    }

    /// Resolves the five bundles inside a platform directory (`macos/` or `ios/` of the HF repo).
    public static func inBundleDir(
        _ dir: URL, glueDir: URL, voicesDir: URL, embedTokens: URL
    ) -> VibeVoicePaths {
        #if os(iOS)
        let ext = "h18p.aimodelc"
        #else
        let ext = "aimodel"
        #endif
        func u(_ base: String) -> URL { dir.appendingPathComponent("\(base).\(ext)") }
        return VibeVoicePaths(
            mainLM: u("vibevoice_mainlm_fp16_decode_cl512"),
            ttsLM: u("vibevoice_ttslm_fp16_decode_cl512"),
            head: u("vibevoice_diffusion_head_fp16"),
            connector: u("vibevoice_connector_fp16"),
            decoder: u("vibevoice_decoder_fp16_t64"),
            glueDir: glueDir, voicesDir: voicesDir, embedTokens: embedTokens)
    }
}

// Flat-blob readers for the host assets (file scope: an actor's `init` cannot call local
// functions before every stored property is initialized).
private func readF32(_ url: URL) throws -> [Float] {
    guard let d = try? Data(contentsOf: url) else {
        throw VibeVoiceError.assetMissing(url.lastPathComponent)
    }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func readF16(_ url: URL) throws -> [Float] {
    guard let d = try? Data(contentsOf: url) else {
        throw VibeVoiceError.assetMissing(url.lastPathComponent)
    }
    return d.withUnsafeBytes { $0.bindMemory(to: Float16.self).map { Float($0) } }
}

/// The VibeVoice engine. Prefer `KitDialogue` (scripts) or `KitSpeaker` (one line of text);
/// this is the direct handle when you want per-voice control.
public actor VibeVoiceTTS {
    private let mainLM: StatefulGraphModel
    private let ttsLM: StatefulGraphModel
    private let negLM: StatefulGraphModel
    private let head: GraphModel
    private let connector: GraphModel
    private let decoder: GraphModel

    private let glue: VibeVoiceGlue
    private let tokenizer: any Tokenizer
    private let embedTokens: Data          // mmapped fp16 (vocab × hidden)
    private let typeEmb: [Float]           // [2, hidden]
    private let eosW1: [Float], eosB1: [Float], eosW2: [Float], eosB2: [Float]
    private let voicesDir: URL

    /// The voices packaged with this model, in catalog order.
    public let voices: [VibeVoiceVoice]

    // MARK: - Loading

    public init(paths: VibeVoicePaths, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        guard let glueData = try? Data(contentsOf: paths.glueDir.appendingPathComponent("glue.json"))
        else { throw VibeVoiceError.assetMissing("glue.json") }
        self.glue = try JSONDecoder().decode(VibeVoiceGlue.self, from: glueData)

        // The two LM instances share one bundle but need independent KV — the negative (CFG) stream
        // runs the same graph on its own state, exactly as the Python host does.
        self.mainLM = try await StatefulGraphModel(
            contentsOf: paths.mainLM, computeUnits: computeUnits, kvCapacity: 512)
        self.ttsLM = try await StatefulGraphModel(
            contentsOf: paths.ttsLM, computeUnits: computeUnits, kvCapacity: 512)
        self.negLM = try await StatefulGraphModel(
            contentsOf: paths.ttsLM, computeUnits: computeUnits, kvCapacity: 512)
        self.head = try await GraphModel(contentsOf: paths.head, computeUnits: computeUnits)
        self.connector = try await GraphModel(contentsOf: paths.connector, computeUnits: computeUnits)
        self.decoder = try await GraphModel(contentsOf: paths.decoder, computeUnits: computeUnits)

        self.tokenizer = try await AutoTokenizer.from(modelFolder: paths.glueDir)
        guard let emb = try? Data(contentsOf: paths.embedTokens, options: .mappedIfSafe) else {
            throw VibeVoiceError.assetMissing(paths.embedTokens.lastPathComponent)
        }
        self.embedTokens = emb
        self.typeEmb = try readF16(paths.glueDir.appendingPathComponent("type_emb.f16"))
        self.eosW1 = try readF32(paths.glueDir.appendingPathComponent("eos_fc1_w.f32"))
        self.eosB1 = try readF32(paths.glueDir.appendingPathComponent("eos_fc1_b.f32"))
        self.eosW2 = try readF32(paths.glueDir.appendingPathComponent("eos_fc2_w.f32"))
        self.eosB2 = try readF32(paths.glueDir.appendingPathComponent("eos_fc2_b.f32"))
        self.voicesDir = paths.voicesDir

        let indexURL = paths.voicesDir.appendingPathComponent("index.json")
        guard let idx = try? Data(contentsOf: indexURL) else {
            throw VibeVoiceError.assetMissing("voices/index.json")
        }
        struct Index: Codable { let voices: [VibeVoiceVoice] }
        self.voices = try JSONDecoder().decode(Index.self, from: idx).voices
    }

    /// Audio sample rate of everything this engine returns.
    public nonisolated var sampleRate: Int { 24_000 }

    /// Longest single utterance the fixed-T decoder renders in one pass (~8.5 s).
    public nonisolated var maxUtteranceSeconds: Double { 64.0 * 3200.0 / 24_000.0 }

    // MARK: - Synthesis

    /// Speaks one utterance in `voice`. Long text is split on sentence boundaries and concatenated,
    /// because the acoustic decoder renders a fixed 64-frame (~8.5 s) window per pass.
    ///
    /// `noiseFrames` overrides the per-frame DDPM init noise — pass it to reproduce a reference
    /// bit-for-bit (that is how the gates compare against the published golden); leave it nil for
    /// ordinary use, where `seed` drives a deterministic generator.
    public func synthesize(
        _ text: String, voice: String, seed: UInt64 = 0x5EED_1234,
        noiseFrames: [[Float]]? = nil
    ) async throws -> SpokenAudio {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw VibeVoiceError.emptyText }
        guard let v = voices.first(where: { $0.name == voice }) else {
            throw VibeVoiceError.voiceNotFound(voice)
        }
        var rng = SplitMix64(seed: seed)
        var out: [Float] = []
        // Upstream feeds the model the script line *including* its "Speaker N:" tag, so keep the tag
        // on every piece when a long utterance is split — otherwise the tail runs out of distribution.
        let (tag, body) = Self.splitSpeakerTag(trimmed)
        for sentence in Self.sentences(body) {
            let piece = try await generate(tag + sentence, voice: v, rng: &rng, noiseFrames: noiseFrames)
            out.append(contentsOf: piece)
        }
        return SpokenAudio(samples: out, sampleRate: sampleRate)
    }

    /// One pass of the upstream streaming `generate()` loop: text windows into the main LM, speech
    /// windows out of the tts LM, DDPM per frame, EOS to stop, then a single whole-sequence decode.
    private func generate(
        _ text: String, voice: VibeVoiceVoice, rng: inout SplitMix64, noiseFrames: [[Float]]?
    ) async throws -> [Float] {
        let H = glue.hidden, VD = glue.vae_dim, T = glue.decoder_frames
        let dir = voicesDir.appendingPathComponent(voice.name)
        func f16(_ name: String) throws -> [Float] {
            try readF16(dir.appendingPathComponent(name))
        }
        try mainLM.seedState(keys: try f16("main_k.f16"), values: try f16("main_v.f16"),
                             prefillLength: voice.mainPrefillLen)
        try ttsLM.seedState(keys: try f16("tts_k.f16"), values: try f16("tts_v.f16"),
                            prefillLength: voice.ttsPrefillLen)
        try negLM.seedState(keys: try f16("neg_k.f16"), values: try f16("neg_v.f16"),
                            prefillLength: voice.negPrefillLen)
        var mainPos = voice.mainPrefillLen, ttsPos = voice.ttsPrefillLen, negPos = voice.negPrefillLen

        // text: the processor's rule verbatim — encode(text.strip() + "\n") with no special tokens.
        let ids = tokenizer.encode(text: text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n",
                                   addSpecialTokens: false)
        guard !ids.isEmpty else { throw VibeVoiceError.emptyText }

        func typeVec(_ i: Int) -> [Float] { Array(typeEmb[i * H..<(i + 1) * H]) }
        func add(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map(+) }

        var latents: [[Float]] = []
        var negCond = try f16("negtts_last.f16")
        var ttsLast = [Float](repeating: 0, count: H)
        var winIdx = 0, noiseIndex = 0, finished = false

        while !finished {
            let lo = winIdx * glue.text_window
            let hi = min(lo + glue.text_window, ids.count)
            winIdx += 1
            if lo < hi {
                for k in lo..<hi {
                    let mh = try await step(mainLM, embedding(ids[k]), pos: mainPos)
                    mainPos += 1
                    ttsLast = try await step(ttsLM, add(mh, typeVec(1)), pos: ttsPos)
                    ttsPos += 1
                }
            }
            for _ in 0..<glue.speech_window {
                let noise = noiseFrames.map { $0[min(noiseIndex, $0.count - 1)] }
                    ?? (0..<(2 * VD)).map { _ in rng.nextGaussian() }
                noiseIndex += 1
                let lat = try await ddpm(Array(noise[0..<VD]), cond: ttsLast, negCond: negCond)
                latents.append(lat)
                let emb = try await connect(lat)
                ttsLast = try await step(ttsLM, add(emb, typeVec(0)), pos: ttsPos)
                ttsPos += 1
                negCond = try await step(negLM, add(emb, typeVec(0)), pos: negPos)
                negPos += 1
                if eos(ttsLast) > 0.5 || latents.count >= T { finished = true; break }
            }
            if lo >= hi && winIdx * glue.text_window > ids.count + glue.text_window { finished = true }
        }

        // whole-sequence decode of the fixed 64-frame window, trimmed to what was generated
        let n = latents.count
        guard n > 0 else { return [] }
        var lats = [Float](repeating: 0, count: VD * T)
        for f in 0..<n {
            for c in 0..<VD {
                lats[c * T + f] = latents[f][c] / Float(glue.scaling) - Float(glue.bias)
            }
        }
        let audio = try await decode(lats)
        return Array(audio[0..<min(n * glue.hop, audio.count)])
    }

    // MARK: - Graph steps

    private func step(_ lm: StatefulGraphModel, _ embedding: [Float], pos: Int) async throws -> [Float] {
        let out = try await lm.step([
            "inputs_embeds": .float16(embedding.map { Float16($0) }, shape: [1, 1, glue.hidden]),
            "pos": .int32([Int32(pos)], shape: [1]),
        ])
        guard let hidden = out["hidden"]?.floats() ?? out.values.first?.floats() else {
            throw VibeVoiceError.assetMissing("hidden")
        }
        return hidden
    }

    private func connect(_ latent: [Float]) async throws -> [Float] {
        let out = try await connector.run(["features": .float16(latent.map { Float16($0) }, shape: [1, 1, glue.vae_dim])])
        guard let e = out["embed"]?.floats() else { throw VibeVoiceError.assetMissing("embed") }
        return e
    }

    private func decode(_ latents: [Float]) async throws -> [Float] {
        let out = try await decoder.run([
            "latents": .float16(latents.map { Float16($0) }, shape: [1, glue.vae_dim, glue.decoder_frames])
        ])
        guard let a = out["audio"]?.floats() else { throw VibeVoiceError.assetMissing("audio") }
        return a
    }

    /// DPMSolver++ (v-prediction, multistep) over the 64-dim latent, with classifier-free guidance:
    /// the head runs cond and uncond as one batch of 2. Kept in Double — the schedule tables are.
    private func ddpm(_ noise: [Float], cond: [Float], negCond: [Float]) async throws -> [Float] {
        let VD = glue.vae_dim, S = glue.schedule
        var x = noise.map { Double($0) }
        let cond2 = cond + negCond
        var mPrev: [Double]? = nil
        for i in 0..<S.count {
            let xf = x.map { Float($0) }
            let out = try await head.run([
                "noisy_images": .float16((xf + xf).map { Float16($0) }, shape: [2, VD]),
                "timesteps": .float16([Float16(S[i].t), Float16(S[i].t)], shape: [2]),
                "condition": .float16(cond2.map { Float16($0) }, shape: [2, glue.hidden]),
            ])
            guard let e = out["eps"]?.floats() else { throw VibeVoiceError.assetMissing("eps") }
            var v = [Double](repeating: 0, count: VD)
            for j in 0..<VD {
                let c = Double(e[j]), u = Double(e[VD + j])
                v[j] = u + glue.cfg * (c - u)
            }
            let a = S[i].alpha, s = S[i].sigma, lam = S[i].lambda
            let aN = i + 1 < S.count ? S[i + 1].alpha : 1.0
            let sN = i + 1 < S.count ? S[i + 1].sigma : 0.0
            let lamN = i + 1 < S.count ? S[i + 1].lambda : 20.0
            let h = lamN - lam, em1 = exp(-h) - 1.0
            var m0 = [Double](repeating: 0, count: VD)
            for j in 0..<VD { m0[j] = a * x[j] - s * v[j] }
            if i > 0, i < S.count - 1, let mp = mPrev {
                let r0 = (lam - S[i - 1].lambda) / h
                for j in 0..<VD {
                    let d1 = (m0[j] - mp[j]) / r0
                    x[j] = (sN / s) * x[j] - aN * em1 * (m0[j] + 0.5 * d1)
                }
            } else {
                for j in 0..<VD { x[j] = (sN / s) * x[j] - aN * em1 * m0[j] }
            }
            mPrev = m0
        }
        return x.map { Float($0) }
    }

    /// The 2-layer EOS classifier (host-side): ReLU(W1·h + b1) → σ(W2·x + b2).
    private func eos(_ hidden: [Float]) -> Float {
        let H = glue.hidden
        var x = [Float](repeating: 0, count: H)
        for i in 0..<H {
            var a = eosB1[i]
            for j in 0..<H { a += eosW1[i * H + j] * hidden[j] }
            x[i] = max(0, a)
        }
        var a = eosB2[0]
        for j in 0..<H { a += eosW2[j] * x[j] }
        return 1 / (1 + exp(-a))
    }

    /// One row of the mmapped fp16 embedding table — no 272 MB read, no torch.
    private func embedding(_ tokenID: Int) -> [Float] {
        let H = glue.hidden
        let start = tokenID * H * MemoryLayout<Float16>.size
        return embedTokens.withUnsafeBytes { raw -> [Float] in
            let base = raw.baseAddress!.advanced(by: start).assumingMemoryBound(to: Float16.self)
            return (0..<H).map { Float(base[$0]) }
        }
    }

    /// `"Speaker 2: hi there"` → `("Speaker 2: ", "hi there")`; no tag → `("", text)`.
    static func splitSpeakerTag(_ text: String) -> (String, String) {
        guard text.lowercased().hasPrefix("speaker"), let colon = text.firstIndex(of: ":") else {
            return ("", text)
        }
        let head = text[text.index(text.startIndex, offsetBy: "speaker".count)..<colon]
        guard !head.isEmpty, head.allSatisfy({ $0.isNumber || $0.isWhitespace }) else {
            return ("", text)
        }
        let body = text[text.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        return (String(text[text.startIndex...colon]) + " ", body)
    }

    /// Split on sentence enders (ASCII + CJK) so every pass stays inside the decoder's window.
    static func sentences(_ text: String) -> [String] {
        var out: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ".!?。！？\n".contains(ch) {
                let s = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty { out.append(s) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out.isEmpty ? [text] : out
    }
}

/// Deterministic Gaussian source (SplitMix64 + Box–Muller) so a `seed` reproduces a take exactly.
struct SplitMix64 {
    private var state: UInt64
    private var spare: Float?

    init(seed: UInt64) { state = seed }

    private mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    private mutating func uniform() -> Double {
        Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    mutating func nextGaussian() -> Float {
        if let s = spare { spare = nil; return s }
        let u1 = max(uniform(), 1e-12), u2 = uniform()
        let r = (-2 * log(u1)).squareRoot(), theta = 2 * Double.pi * u2
        spare = Float(r * sin(theta))
        return Float(r * cos(theta))
    }
}
