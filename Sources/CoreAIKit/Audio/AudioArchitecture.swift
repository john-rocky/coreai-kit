// AudioArchitecture.swift — fixed-shape geometry for a Qwen2.5-Omni-style audio-understanding
// decoder + its paired Whisper-style audio encoder.
//
// The decoder bundle (exported with the id-space recipe, see NOTICE.txt) keeps the text path a
// plain Qwen2.5 LLM and rides the audio state on ONE static graph input — `audio_embeds`
// `[maxAudioTokens, hidden]`. This is simpler than the VL rider: TMRoPE collapses to 1-D for
// audio+text (no image/video grid), so positions are the engine's native `arange` and there is
// NO rope-shift plumbing (the VL path's two i32 scalars are absent).
//
// The encoder is a fixed K-chunk graph: `input_features [1, melBins, chunkMel*chunks]` +
// `attn_bias [chunks,1,1,chunkMel/2]` -> `audio_embeds [chunks*50, hidden]`. The host zero-pads
// the mel to whole chunks and trims the output to the clip's real audio-token count `N`.

import Foundation

/// Geometry and token vocabulary for one audio decoder + its paired audio encoder.
public struct AudioArchitecture: Sendable, Hashable {
    /// Text vocabulary size. Audio tokens are encoded as extension ids `vocab + slot`, which the
    /// decoder graph gathers from `audio_embeds` by `id - vocab`.
    public let vocab: Int32
    /// Decoder hidden width and encoder output width (Qwen2.5-Omni-3B = 2048).
    public let hidden: Int
    /// Static `audio_embeds` row count = the longest clip the bundle accepts (`chunks * 50`).
    public let maxAudioTokens: Int
    /// Fixed mel-chunk count baked into the encoder bundle (K=15 ≈ 30 s).
    public let chunks: Int
    /// Mel frames per chunk before the encoder's stride-2 CNN (200).
    public let chunkMel: Int
    /// Mel filterbank bins (Whisper-large-v3 = 128).
    public let melBins: Int

    // Special tokens (Qwen ChatML + Qwen2.5-Omni audio framing). `audioPad` must be a single
    // token so the host can splice exactly N copies and rewrite them to `vocab + slot`.
    public let imStart: String
    public let imEnd: String
    public let audioBos: String
    public let audioEos: String
    public let audioPad: String

    public init(
        vocab: Int32, hidden: Int, maxAudioTokens: Int, chunks: Int, chunkMel: Int, melBins: Int,
        imStart: String = "<|im_start|>", imEnd: String = "<|im_end|>",
        audioBos: String = "<|audio_bos|>", audioEos: String = "<|audio_eos|>",
        audioPad: String = "<|AUDIO|>"
    ) {
        self.vocab = vocab
        self.hidden = hidden
        self.maxAudioTokens = maxAudioTokens
        self.chunks = chunks
        self.chunkMel = chunkMel
        self.melBins = melBins
        self.imStart = imStart
        self.imEnd = imEnd
        self.audioBos = audioBos
        self.audioEos = audioEos
        self.audioPad = audioPad
    }

    /// Post-CNN frames per chunk on the attention axis (`chunkMel / 2` = 100).
    public var headsFrames: Int { chunkMel / 2 }
    /// Total mel frames the encoder accepts (`chunkMel * chunks` = 3000 at K=15).
    public var melFrames: Int { chunkMel * chunks }
    /// `input_features` element count (`melBins * melFrames`).
    public var inputFeaturesCount: Int { melBins * melFrames }
    /// `attn_bias` element count (`chunks * headsFrames`).
    public var attnBiasCount: Int { chunks * headsFrames }
    /// `audio_embeds` element count (`maxAudioTokens * hidden`).
    public var audioEmbedsCount: Int { maxAudioTokens * hidden }

    /// Audio-token count for a clip of `melValidFrames` real (un-padded) mel frames, matching the
    /// encoder's conv1+conv2(stride2)+avg_pool(stride2) downsampling.
    public func audioTokenCount(melValidFrames: Int) -> Int {
        melValidFrames <= 0 ? 0 : ((melValidFrames - 1) / 2 + 1 - 2) / 2 + 1
    }

    /// Pack a Whisper log-mel `[melBins, frames]` (row-major) into the encoder's fixed inputs:
    /// `input_features [1, melBins, melFrames]` (mel left-aligned, zero-padded to K whole chunks) +
    /// `attn_bias [chunks,1,1,headsFrames]` (−30000 on each chunk's padded post-CNN frames) + the
    /// clip's audio-token count N. Mirrors the python host contract (`build_static_inputs`).
    public func encoderInputs(fromMel mel: [Float], frames: Int)
        -> (inputFeatures: [Float16], attnBias: [Float16], audioTokenCount: Int)
    {
        let neg: Float16 = -30000
        var feats = [Float16](repeating: 0, count: inputFeaturesCount)
        let copyT = min(frames, melFrames)
        for c in 0..<melBins {
            let dst = c * melFrames, src = c * frames
            for t in 0..<copyT { feats[dst + t] = Float16(mel[src + t]) }
        }
        var bias = [Float16](repeating: 0, count: attnBiasCount)
        for kc in 0..<chunks {
            let melInChunk = max(0, min(chunkMel, frames - kc * chunkMel))
            let valid = melInChunk == 0 ? 0 : (melInChunk - 1) / 2 + 1
            if valid < headsFrames {
                for j in valid..<headsFrames { bias[kc * headsFrames + j] = neg }
            }
        }
        return (feats, bias, audioTokenCount(melValidFrames: frames))
    }

    /// Qwen2.5-Omni-3B: 2048-wide, K=15 encoder (≈30 s, 750 audio-embed rows). Mac-class.
    public static let qwen2_5Omni3B = AudioArchitecture(
        vocab: 151_936, hidden: 2048, maxAudioTokens: 750, chunks: 15, chunkMel: 200, melBins: 128)
}
