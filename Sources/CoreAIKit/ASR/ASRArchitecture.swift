// ASRArchitecture.swift — fixed-shape geometry for a Qwen3-ASR speech-to-text decoder + its paired
// AuT audio encoder. The zoo's first ASR (transcription) model.
//
// Mirrors `AudioArchitecture` (Qwen-Omni audio understanding) but the AuT encoder geometry differs:
//   * 13 tokens/chunk (not 50): chunk = 100 mel frames, three stride-2 conv2d (8× down) -> 13.
//   * `attn_bias` is a FULL materialized `[1, S, S]` additive mask (not the per-chunk
//     `[chunks,1,1,headsFrames]`): valid tokens attend bidirectionally inside their 104-token
//     window; pad tokens (>= N) are masked as keys. (Replicates the HF `cu_seqlens` windows.)
//   * The clip's real token count N is a PER-CHUNK sum (full chunks ×13 + the partial last chunk),
//     NOT a single global downsample — the real tokens stay contiguous at `[0, N)`.
//
// The decoder is a plain Qwen3 LLM riding the audio on ONE static `audio_embeds [maxAudioTokens,
// hidden]` input (id-space recipe: audio tokens carry `vocab + slot`, the graph gathers row
// `id - vocab`). MRoPE is a no-op for ASR -> the engine's native 1-D positions are exact.

import Foundation

/// Geometry and token vocabulary for one Qwen3-ASR decoder + its paired AuT encoder.
public struct ASRArchitecture: Sendable, Hashable {
    /// Text vocabulary size (151936). Audio tokens are extension ids `vocab + slot`.
    public let vocab: Int32
    /// Decoder hidden width = encoder output width (Qwen3-ASR-1.7B = 2048).
    public let hidden: Int
    /// Fixed mel-chunk count baked into the encoder bundle (K=30 ≈ 30 s).
    public let chunks: Int
    /// Mel filterbank bins (Whisper = 128).
    public let melBins: Int
    /// Mel frames per AuT chunk before the conv stack (100 = n_window×2).
    public let chunkMel: Int
    /// Encoder tokens emitted per whole chunk (13).
    public let tokensPerChunk: Int
    /// Windowed-attention block width in tokens (104 = 8 chunks).
    public let window: Int

    // Special tokens (Qwen3-ASR template). `audioPad` must be a single token so the host can
    // splice exactly N copies and rewrite each to `vocab + slot`.
    public let imStart: String
    public let imEnd: String
    public let audioStart: String
    public let audioEnd: String
    public let audioPad: String
    public let asrText: String

    public init(
        vocab: Int32, hidden: Int, chunks: Int, melBins: Int = 128, chunkMel: Int = 100,
        tokensPerChunk: Int = 13, window: Int = 104,
        imStart: String = "<|im_start|>", imEnd: String = "<|im_end|>",
        audioStart: String = "<|audio_start|>", audioEnd: String = "<|audio_end|>",
        audioPad: String = "<|audio_pad|>", asrText: String = "<asr_text>"
    ) {
        self.vocab = vocab
        self.hidden = hidden
        self.chunks = chunks
        self.melBins = melBins
        self.chunkMel = chunkMel
        self.tokensPerChunk = tokensPerChunk
        self.window = window
        self.imStart = imStart
        self.imEnd = imEnd
        self.audioStart = audioStart
        self.audioEnd = audioEnd
        self.audioPad = audioPad
        self.asrText = asrText
    }

    /// Longest clip the bundle accepts in audio tokens (`chunks * tokensPerChunk` = 390 at K=30).
    public var maxAudioTokens: Int { chunks * tokensPerChunk }
    /// Total mel frames the encoder accepts (`chunkMel * chunks` = 3000 at K=30).
    public var melFrames: Int { chunkMel * chunks }
    /// `input_features` element count (`melBins * melFrames`).
    public var inputFeaturesCount: Int { melBins * melFrames }
    /// `attn_bias` element count (`S * S`, S = maxAudioTokens).
    public var attnBiasCount: Int { maxAudioTokens * maxAudioTokens }
    /// `audio_embeds` static-buffer element count (`maxAudioTokens * hidden`).
    public var audioEmbedsCount: Int { maxAudioTokens * hidden }

    /// AuT three stride-2 conv2d downsample of `m` mel frames -> output tokens (kernel 3, pad 1).
    private func downsample(_ m: Int) -> Int {
        guard m > 0 else { return 0 }
        let a = (m - 1) / 2 + 1
        let b = (a - 1) / 2 + 1
        return (b - 1) / 2 + 1
    }

    /// The clip's real audio-token count N = Σ over chunks of `downsample(mel-frames-in-chunk)`.
    /// Full chunks give `tokensPerChunk`; the single partial last chunk gives fewer; the rest 0.
    /// The real tokens stay contiguous at `[0, N)` (host left-aligns the mel), so `N` is the mask's
    /// `n_valid`.
    public func audioTokenCount(melValidFrames frames: Int) -> Int {
        var n = 0
        for c in 0..<chunks {
            let inChunk = max(0, min(chunkMel, frames - c * chunkMel))
            n += downsample(inChunk)
        }
        return n
    }

    /// Pack a Whisper log-mel `[melBins, frames]` (row-major) into the AuT encoder's fixed inputs:
    /// `input_features [1, melBins, melFrames]` (mel left-aligned, zero-padded to K chunks) +
    /// `attn_bias [1, S, S]` (0 in-window & both-valid, −65504 else) + the clip's token count N.
    /// Mirrors the python host contract (`build_attn_bias`).
    public func encoderInputs(fromMel mel: [Float], frames: Int)
        -> (inputFeatures: [Float16], attnBias: [Float16], audioTokenCount: Int)
    {
        var feats = [Float16](repeating: 0, count: inputFeaturesCount)
        let copyT = min(frames, melFrames)
        for c in 0..<melBins {
            let dst = c * melFrames, src = c * frames
            for t in 0..<copyT { feats[dst + t] = Float16(mel[src + t]) }
        }

        let n = audioTokenCount(melValidFrames: frames)
        let s = maxAudioTokens
        let neg: Float16 = -65504  // fp16 finfo.min, matches build_attn_bias
        // allow(i,j) = (i/window == j/window) && i < n && j < n  -> 0, else -65504.
        var bias = [Float16](repeating: neg, count: attnBiasCount)
        if n > 0 {
            for i in 0..<n {
                let wi = i / window
                let row = i * s
                for j in 0..<n where j / window == wi { bias[row + j] = 0 }
            }
        }
        return (feats, bias, n)
    }

    /// Qwen3-ASR-1.7B: 2048-wide, K=30 AuT encoder (≤30 s, 390 audio-embed rows). Apache-2.0.
    public static let qwen3ASR1_7B = ASRArchitecture(
        vocab: 151_936, hidden: 2048, chunks: 30)
}
