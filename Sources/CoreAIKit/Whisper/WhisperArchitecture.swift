// WhisperArchitecture.swift — fixed-shape geometry + special-token vocabulary for an OpenAI
// Whisper encoder-decoder speech-to-text bundle exported to Core AI.
//
// Unlike Qwen3-ASR (an LLM decoder riding a separate AuT encoder on the pipelined engine), the
// Whisper bundle is ONE stateless graph: `input_features [1, melBins, melFrames]` +
// `decoder_input_ids [1, decoderWindow]` -> `logits [1, decoderWindow, vocab]`. The decoder buffer
// is a FIXED-LENGTH padded window: causal attention ignores the padding, so the constant shape
// compiles once and each greedy step re-runs the same graph reading logits at the real last
// position. See `conversion/export_whisper_fixed.py` in the zoo.

import Foundation

/// Geometry and special tokens for one fixed-window Whisper decode graph.
public struct WhisperArchitecture: Sendable, Hashable {
    /// Mel filterbank bins (Whisper-large-v3 = 128).
    public let melBins: Int
    /// Mel frames per 30 s window baked into the graph (3000).
    public let melFrames: Int
    /// Required input sample rate (16 kHz).
    public let sampleRate: Int
    /// Log-mel hop in samples (160 — `melFrames * hop` = 30 s).
    public let hop: Int
    /// Fixed decoder-buffer length the graph was exported with (128 tokens).
    public let decoderWindow: Int
    /// Output vocabulary size (large-v3 = 51866).
    public let vocab: Int

    // Special tokens (Whisper-large-v3 ids).
    /// `<|endoftext|>` — also the pad filler for unused decoder slots.
    public let eot: Int32
    /// `<|startoftranscript|>`.
    public let sot: Int32
    /// `<|transcribe|>` task token.
    public let transcribe: Int32
    /// `<|translate|>` task token.
    public let translate: Int32
    /// `<|notimestamps|>`.
    public let noTimestamps: Int32
    /// First language token (`<|en|>`); auto-detect picks the argmax right after `sot`.
    public let langFirst: Int32

    public init(
        melBins: Int = 128, melFrames: Int = 3000, sampleRate: Int = 16000, hop: Int = 160,
        decoderWindow: Int = 128, vocab: Int = 51866,
        eot: Int32 = 50257, sot: Int32 = 50258, transcribe: Int32 = 50360,
        translate: Int32 = 50359, noTimestamps: Int32 = 50364, langFirst: Int32 = 50259
    ) {
        self.melBins = melBins
        self.melFrames = melFrames
        self.sampleRate = sampleRate
        self.hop = hop
        self.decoderWindow = decoderWindow
        self.vocab = vocab
        self.eot = eot
        self.sot = sot
        self.transcribe = transcribe
        self.translate = translate
        self.noTimestamps = noTimestamps
        self.langFirst = langFirst
    }

    /// Samples in one 30 s window (`melFrames * hop` = 480000). Clips are padded/trimmed to this.
    public var nSamples: Int { melFrames * hop }
    /// `input_features` element count (`melBins * melFrames`).
    public var inputFeaturesCount: Int { melBins * melFrames }

    /// Whisper large-v3-turbo: 128 mel, 30 s window, 128-token decoder, 51866 vocab. MIT.
    public static let largeV3Turbo = WhisperArchitecture()
}
