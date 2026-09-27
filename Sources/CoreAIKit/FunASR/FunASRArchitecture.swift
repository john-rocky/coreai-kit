// FunASRArchitecture.swift — fixed-shape geometry for Fun-ASR-Nano-2512: the SAN-M audio encoder +
// adaptor graph and the Qwen3-0.6B text decoder that reads its output.
//
// Encoder graph: `feats [1, 500, 560]` (LFR frames, zero-padded to 30 s) + `mask [1, 500]` (1 valid /
// 0 pad) -> `audio_embeds [63, 1024]`, rows `[0, N)` valid, N = ceil(L / 8) for L LFR frames. The
// shipped encoder keeps fp16 weights but computes in fp32 (an fp16 activation run has only 1.4×
// residual headroom), so both inputs are Float32 and a Float16 input is refused; its Float32 output
// is narrowed to Float16 when it is written into the decoder's static `audio_embeds` buffer.
//
// Decoder: plain Qwen3-0.6B riding the audio on that ONE static input (id-space recipe: audio slot
// `slot` carries id `vocab + slot`, the graph gathers row `id - vocab`), 1-D positions from 0. The
// checkpoint's residual-stream scale (s = 1/4 with RMSNorm eps·s²) is folded into the graph.

import Foundation

/// Geometry, stop ids and window length for one Fun-ASR-Nano decoder + its paired audio encoder.
@available(macOS 27, iOS 27, *)
public struct FunASRArchitecture: Sendable, Hashable {
    /// Text vocabulary size (151936). Audio slots are extension ids `vocab + slot`.
    public let vocab: Int32
    /// Decoder hidden width = encoder output width (Qwen3-0.6B = 1024).
    public let hidden: Int
    /// LFR frames the encoder graph accepts (500 = 30 s).
    public let lfrMax: Int
    /// LFR feature width (80 mel × 7 stacked frames = 560).
    public let featureDim: Int
    /// Audio slots the decoder can gather (63 = ceil(500 / 8)).
    public let maxAudioTokens: Int
    /// Samples transcribed per encoder call (480,000 = 30 s at 16 kHz -> 2998 fbank -> 500 LFR frames).
    /// Longer audio is cut into windows of this length and transcribed in order.
    public let windowSamples: Int
    /// Generation stops at either id: `<|im_end|>` (151645) or `<|endoftext|>` (151643).
    public let eosTokenIDs: [Int32]

    public init(
        vocab: Int32, hidden: Int, lfrMax: Int = 500, featureDim: Int = 560, maxAudioTokens: Int = 63,
        windowSamples: Int = 480_000, eosTokenIDs: [Int32] = [151_645, 151_643]
    ) {
        precondition(
            FunASRFbankPreprocessor.lfrFrames(samples: windowSamples) <= lfrMax
                && FunASRFbankPreprocessor.audioTokenCount(lfrFrames: lfrMax) <= maxAudioTokens,
            "a window must fit the encoder graph")
        self.vocab = vocab
        self.hidden = hidden
        self.lfrMax = lfrMax
        self.featureDim = featureDim
        self.maxAudioTokens = maxAudioTokens
        self.windowSamples = windowSamples
        self.eosTokenIDs = eosTokenIDs
    }

    /// `audio_embeds` static-buffer element count (`maxAudioTokens * hidden`).
    public var audioEmbedsCount: Int { maxAudioTokens * hidden }

    /// Pack LFR features `[L, 560]` (row-major) into the encoder's fixed inputs: `feats [1, 500, 560]`
    /// zero-padded, `mask [1, 500]` (1 for the L valid rows), and the clip's audio-slot count N.
    /// Rows past `lfrMax` are dropped; callers cut the audio into windows first.
    public func encoderInputs(feats: [Float], lfrFrames: Int)
        -> (feats: [Float], mask: [Float], audioTokenCount: Int)
    {
        let l = max(0, min(lfrFrames, lfrMax, feats.count / featureDim))
        var padded = [Float](repeating: 0, count: lfrMax * featureDim)
        padded.replaceSubrange(0..<(l * featureDim), with: feats[0..<(l * featureDim)])
        var mask = [Float](repeating: 0, count: lfrMax)
        mask.replaceSubrange(0..<l, with: repeatElement(1, count: l))
        return (padded, mask, FunASRFbankPreprocessor.audioTokenCount(lfrFrames: l))
    }

    /// Fun-ASR-Nano-2512 (FunAudioLLM): 1024-wide Qwen3-0.6B decoder, 30 s windows.
    public static let funASRNano2512 = FunASRArchitecture(vocab: 151_936, hidden: 1024)
}
