// KokoroTTS.swift — Kokoro-82M (StyleTTS2 + iSTFTNet) text-to-speech on Core AI.
//
// Drives the three exported bundles (predictor / prosody / vocoder) through CoreAIKitVision's
// GraphModel on the CPU compute unit (the unrolled bi-LSTMs are ~8 ms on the Core AI CPU vs
// dispatch-bound on the GPU), with the host steps in Swift:
//
//   text -> (KokoroG2P) -> phoneme ids
//     predictor : ids, ref_s, attn_mask -> duration, d, t_en
//     host      : pred_dur = round(duration); one-hot alignment; frame mask
//     prosody   : d, t_en, aln, ref_s, frame_mask -> asr, F0, N
//     host      : har = STFT(SineGen(f0_upsamp(F0)))   — the hn-nsf source
//     vocoder   : asr, F0, N, har, ref_s, frame_mask -> audio
//     host      : trim to the real frames
//
// The host DSP (alignment + compute_har) mirrors conversion/export_kokoro.py exactly; the
// engine-vs-torch gate is spectral corr 0.999 (see the model card). The voice is data — a
// `ref_s` style row indexed by utterance length from a downloaded voice pack.

import CoreAIKitVision
import Foundation

/// Kokoro-82M text-to-speech: non-autoregressive, 24 kHz, English-first.
public final class KokoroTTS: Sendable {
    public static let sampleRate = 24000

    static let TB = 128          // token bucket
    static let LB = 512          // frame bucket
    static let UP = 300          // f0 upsample (prod(upsample_rates)*hop)
    static let NFFT = 20, HOP = 5, FREQ = 11, HARM = 9
    static let SR: Float = 24000

    private let predictor: GraphModel
    private let prosody: GraphModel
    private let vocoder: GraphModel
    private let vocab: [String: Int]
    private let llW: [Float]                    // l_linear weight[9] + bias[1]
    private let voices: [String: [Float]]       // name -> [N*256] style pack
    private let g2p: KokoroG2P

    // STFT DFT basis (cos/sin * periodic Hann), built once.
    private let basisR: [Float]                 // [FREQ*NFFT]
    private let basisI: [Float]

    /// Loads the three graph bundles plus the host glue (vocab, l_linear, lexicons, voices).
    /// `glueDir` is the repo's `kokoro_host_glue/` subtree.
    public init(
        predictorAt predictorURL: URL, prosodyAt prosodyURL: URL, vocoderAt vocoderURL: URL,
        glueDir: URL
    ) async throws {
        predictor = try await GraphModel(contentsOf: predictorURL, computeUnits: .cpu)
        prosody = try await GraphModel(contentsOf: prosodyURL, computeUnits: .cpu)
        vocoder = try await GraphModel(contentsOf: vocoderURL, computeUnits: .cpu)

        let vocabData = try Data(contentsOf: glueDir.appendingPathComponent("vocab.json"))
        vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        llW = Self.readFloats(glueDir.appendingPathComponent("l_linear.bin"))
        g2p = try KokoroG2P(
            goldURL: glueDir.appendingPathComponent("us_gold.json"),
            silverURL: glueDir.appendingPathComponent("us_silver.json"))

        let voicesDir = glueDir.appendingPathComponent("voices")
        var voices: [String: [Float]] = [:]
        for file in try FileManager.default.contentsOfDirectory(
            at: voicesDir, includingPropertiesForKeys: nil)
        where file.pathExtension == "bin" {
            voices[file.deletingPathExtension().lastPathComponent] = Self.readFloats(file)
        }
        self.voices = voices

        var basisR = [Float](), basisI = [Float]()
        basisR.reserveCapacity(Self.FREQ * Self.NFFT)
        basisI.reserveCapacity(Self.FREQ * Self.NFFT)
        for k in 0..<Self.FREQ {
            for n in 0..<Self.NFFT {
                let hann = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(Self.NFFT))
                let a = 2 * Float.pi * Float(k) * Float(n) / Float(Self.NFFT)
                basisR.append(cos(a) * hann)
                basisI.append(-sin(a) * hann)
            }
        }
        self.basisR = basisR
        self.basisI = basisI
    }

    /// The downloaded voice packs, e.g. `["af_heart", "af_bella", …]`.
    public var availableVoices: [String] { voices.keys.sorted() }

    /// Free text -> 24 kHz audio. Splits into sentences (each ≤ the token bucket),
    /// synthesizes each, and concatenates with a short gap. G2P is on-device.
    public func synthesize(_ text: String, voice: String = "af_heart") async throws -> [Float] {
        var audio: [Float] = []
        try await synthesizeBySentence(text, voice: voice) { audio += $0 }
        return audio
    }

    /// Streaming synthesis: `onChunk` receives one sentence's audio as it synthesizes. The
    /// concatenation equals `synthesize(_:voice:)`.
    @discardableResult
    public func synthesizeStreaming(
        _ text: String, voice: String = "af_heart", onChunk: @Sendable ([Float]) async -> Void
    ) async throws -> [Float] {
        var audio: [Float] = []
        try await synthesizeBySentence(text, voice: voice) { chunk in
            audio += chunk
            await onChunk(chunk)
        }
        return audio
    }

    private func synthesizeBySentence(
        _ text: String, voice: String, emit: ([Float]) async -> Void
    ) async throws {
        for sentence in Self.splitSentences(text) {
            let ids = ids(forText: sentence)
            guard ids.count > 2 else { continue }                        // skip empty
            guard ids.count <= Self.TB else { throw KokoroError.tooLong(ids.count) }
            var chunk = try await synthesize(ids: ids, voice: voice)
            chunk += [Float](repeating: 0, count: 3600)                  // ~0.15 s between sentences
            await emit(chunk)
        }
    }

    /// Phonemes for a chunk -> Kokoro token ids (incl. the [0, …, 0] bounds).
    func ids(forText text: String) -> [Int] {
        let phonemes = g2p.phonemize(text)
        return [0] + phonemes.compactMap { vocab[String($0)] } + [0]
    }

    static func splitSentences(_ text: String) -> [String] {
        var out: [String] = []
        var cur = ""
        for ch in text {
            cur.append(ch)
            if ch == "." || ch == "!" || ch == "?" {
                let s = cur.trimmingCharacters(in: .whitespacesAndNewlines)
                if !s.isEmpty { out.append(s) }
                cur = ""
            }
        }
        let tail = cur.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { out.append(tail) }
        return out.isEmpty ? [text] : out
    }

    /// Synthesize from already-tokenized phoneme ids (incl. the [0, …, 0] bounds).
    func synthesize(ids rawIds: [Int], voice: String) async throws -> [Float] {
        let T = rawIds.count
        guard T <= Self.TB else { throw KokoroError.tooLong(T) }
        guard let pack = voices[voice] else { throw KokoroError.noVoice(voice) }
        let refS = Array(pack[(T - 1) * 256 ..< T * 256])       // ref_s = pack[len-1]

        // pad ids to the token bucket
        let ids = rawIds.map { Int32($0) } + [Int32](repeating: 0, count: Self.TB - T)
        let attn = [Float](repeating: 1, count: T) + [Float](repeating: 0, count: Self.TB - T)

        let o1 = try await predictor.run([
            "input_ids": .int32(ids, shape: [1, Self.TB]),
            "ref_s": .float32(refS, shape: [1, 256]),
            "attn_mask": .float32(attn, shape: [1, Self.TB]),
        ])
        let duration = o1["duration"]!.floats()

        let (aln, frameMask, L) = buildAlignment(duration: Array(duration[0..<T]), T: T)

        let o2 = try await prosody.run([
            "d": o1["d"]!, "t_en": o1["t_en"]!,
            "aln": .float32(aln, shape: [1, Self.TB, Self.LB]),
            "ref_s": .float32(refS, shape: [1, 256]),
            "frame_mask": .float32(frameMask, shape: [1, Self.LB]),
        ])
        let F0 = o2["F0"]!.floats()                              // [2*LB]
        let (har, frames) = computeHar(F0: F0)

        let o3 = try await vocoder.run([
            "asr": o2["asr"]!, "F0": o2["F0"]!, "N": o2["N"]!,
            "har": .float32(har, shape: [1, 2 * Self.FREQ, frames]),
            "ref_s": .float32(refS, shape: [1, 256]),
            "frame_mask": .float32(frameMask, shape: [1, Self.LB]),
        ])
        let audio = o3["audio"]!.floats()
        return Array(audio[0 ..< min(L * 600, audio.count)])     // trim to real frames
    }

    // host step 1: duration -> one-hot alignment + frame mask (bucketed)
    private func buildAlignment(duration: [Float], T: Int) -> ([Float], [Float], Int) {
        var predDur = [Int](repeating: 1, count: T)
        var L = 0
        for i in 0..<T {
            predDur[i] = max(1, Int((duration[i]).rounded()))
            L += predDur[i]
        }
        var aln = [Float](repeating: 0, count: Self.TB * Self.LB)
        var frame = 0
        for i in 0..<T {
            for _ in 0..<predDur[i] where frame < Self.LB {
                aln[i * Self.LB + frame] = 1                     // aln[token i, frame]
                frame += 1
            }
        }
        var frameMask = [Float](repeating: 0, count: Self.LB)
        for f in 0..<min(L, Self.LB) { frameMask[f] = 1 }
        return (aln, frameMask, L)
    }

    // host step 2: hn-nsf source -> STFT (mag, phase). Mirrors compute_har.
    private func computeHar(F0: [Float]) -> ([Float], Int) {
        let twoL = F0.count                 // 2*LB
        let bigL = twoL * Self.UP
        // f0_upsamp: repeat each value UP times (nearest)
        // SineGen per harmonic: rad -> downsample(1/UP) -> cumsum -> upsample(UP) -> sin
        var harSource = [Float](repeating: 0, count: bigL)
        var radDS = [Float](repeating: 0, count: twoL)
        var phase = [Float](repeating: 0, count: twoL)
        for h in 1...Self.HARM {
            // rad on the bigL grid is f0_up*h/SR; its 1/UP downsample averages each
            // block of UP constant samples -> ~= F0[j]*h/SR (linear interp of a step).
            for j in 0..<twoL { radDS[j] = F0[j] * Float(h) / Self.SR }
            // cumsum * 2pi
            var acc: Float = 0
            for j in 0..<twoL { acc += radDS[j]; phase[j] = acc * 2 * Float.pi }
            // upsample (linear, align_corners=false) of phase*UP back to bigL, then sin
            let weight = llW[h - 1]
            for i in 0..<bigL {
                let s = max(0, min(Float(twoL - 1), (Float(i) + 0.5) * Float(twoL) / Float(bigL) - 0.5))
                let lo = Int(s.rounded(.down)); let hi = min(lo + 1, twoL - 1); let fr = s - Float(lo)
                let ph = (phase[lo] * (1 - fr) + phase[hi] * fr) * Float(Self.UP)
                let f0u = F0[min(twoL - 1, i / Self.UP)]                       // f0_upsamp value
                let uv: Float = f0u > 10 ? 1 : 0
                harSource[i] += sin(ph) * 0.1 * uv * weight                    // sine_amp * uv * l_linear[h-1]
            }
        }
        // l_linear bias + tanh
        let bias = llW[Self.HARM]
        for i in 0..<bigL { harSource[i] = tanh(harSource[i] + bias) }

        // STFT: replicate pad n_fft/2, stride HOP, DFT basis -> mag, phase
        let pad = Self.NFFT / 2
        let total = bigL + 2 * pad
        let frames = (total - Self.NFFT) / Self.HOP + 1
        var har = [Float](repeating: 0, count: 2 * Self.FREQ * frames)
        func sample(_ idx: Int) -> Float {                     // replicate (edge) pad
            if idx < pad { return harSource[0] }
            if idx >= pad + bigL { return harSource[bigL - 1] }
            return harSource[idx - pad]
        }
        for fi in 0..<frames {
            let base = fi * Self.HOP
            for k in 0..<Self.FREQ {
                var re: Float = 0, im: Float = 0
                let bk = k * Self.NFFT
                for n in 0..<Self.NFFT {
                    let v = sample(base + n)
                    re += basisR[bk + n] * v
                    im += basisI[bk + n] * v
                }
                let mag = (re * re + im * im + 1e-14).squareRoot()
                har[k * frames + fi] = mag
                har[(Self.FREQ + k) * frames + fi] = atan2(im, re)
            }
        }
        return (har, frames)
    }

    private static func readFloats(_ url: URL) -> [Float] {
        guard let d = try? Data(contentsOf: url) else { return [] }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Kokoro synthesis errors.
    public enum KokoroError: Error {
        /// One sentence phonemized past the 128-token bucket — split the text shorter.
        case tooLong(Int)
        /// The requested voice pack is not among the downloaded `voices/*.bin`.
        case noVoice(String)
    }
}
