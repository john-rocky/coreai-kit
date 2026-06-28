// WhisperRuntime.swift — the loaded pieces of a fixed-window Whisper transcription graph: the
// stateless `GraphModel`, the shared Whisper log-mel frontend, and the HF Whisper tokenizer.
//
// Pipeline: 16 kHz mono Float -> pad/trim to 30 s -> log-mel `[128, 3000]` -> greedy decode on the
// fixed 128-token decoder window (re-running the same graph each step, reading logits at the real
// last position) -> detokenize. Language is auto-detected (argmax right after `<|startoftranscript|>`)
// unless a Whisper code (e.g. "en", "ja") is forced.
//
// The mel frontend is CoreAIKit's bundled `AudioMelPreprocessor` — its filterbank is bit-exact with
// the HF `mel_filters_128.npy`, so no extra resource ships and the result is token-exact with the
// PyTorch reference.

import CoreAIKitVision
import Foundation
import Tokenizers

/// Failures specific to the Whisper transcription path.
public enum KitWhisperError: Error, LocalizedError {
    case unsupportedSampleRate(Int)
    case modelOutputMissing(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedSampleRate(let sr):
            return "Audio must be 16 kHz mono; got \(sr) Hz (resample before transcribing)."
        case .modelOutputMissing(let name):
            return "Whisper graph did not produce the expected output '\(name)'."
        }
    }
}

/// Owns a Whisper graph + mel frontend + tokenizer. Serial use (one transcription at a time).
public final class WhisperRuntime: @unchecked Sendable {
    public let arch: WhisperArchitecture
    private let graph: GraphModel
    private let mel: AudioMelPreprocessor
    public let tokenizer: any Tokenizer

    /// Loads a local Whisper `.aimodel` directory + a local `tokenizer/` directory.
    public init(
        modelAt modelURL: URL, tokenizerAt tokenizerURL: URL,
        arch: WhisperArchitecture = .largeV3Turbo,
        computeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        self.arch = arch
        self.graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.mel = try AudioMelPreprocessor.whisperLargeV3()
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
    }

    // MARK: - Transcribe

    /// Transcribe a raw 16 kHz mono waveform. The clip is padded/trimmed to the 30 s window.
    /// `language` is an optional forced Whisper code ("en", "ja", …); nil = auto-detect.
    public func transcribe(
        samples: [Float], sampleRate: Int = 16000, language: String? = nil,
        translate: Bool = false
    ) async throws -> Transcription {
        guard sampleRate == arch.sampleRate else {
            throw KitWhisperError.unsupportedSampleRate(sampleRate)
        }
        let feat = TensorValue.float16(melFeatures(samples), shape: [1, arch.melBins, arch.melFrames])
        let task = translate ? arch.translate : arch.transcribe

        // Seed the decoder. A forced language fixes the whole `<sot><lang><task><notimestamps>`
        // prefix; auto-detect starts at `<sot>` and reads the language token off the first step.
        var seq: [Int32] = [arch.sot]
        var detectedLang = language ?? ""
        if let language, let id = tokenizer.convertTokenToId("<|\(language)|>").map(Int32.init) {
            seq = [arch.sot, id, task, arch.noTimestamps]
        }
        let autoDetect = seq.count == 1

        var generated: [Int32] = []
        while seq.count < arch.decoderWindow {
            let next = try await step(feat: feat, seq: seq)

            if autoDetect && seq.count == 1 {  // first step emits the language token
                detectedLang = cleanLanguage(tokenizer.convertIdToToken(Int(next)) ?? "")
                seq.append(contentsOf: [next, task, arch.noTimestamps])
                continue
            }
            if next == arch.eot { break }
            seq.append(next)
            generated.append(next)
        }

        // Keep real text tokens only (drop the EOT and any timestamp/special token, all >= eot).
        let textTokens = generated.filter { $0 < arch.eot }.map(Int.init)
        let text = tokenizer.decode(tokens: textTokens)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return Transcription(language: detectedLang, text: text)
    }

    // MARK: - Internals

    /// Run the graph once with `seq` packed into the fixed decoder window and return the argmax of
    /// the logits at the real last position (`seq.count - 1`).
    private func step(feat: TensorValue, seq: [Int32]) async throws -> Int32 {
        var ids = [Int32](repeating: arch.eot, count: arch.decoderWindow)
        for i in 0..<seq.count { ids[i] = seq[i] }
        let dec = TensorValue.int32(ids, shape: [1, arch.decoderWindow])

        let out = try await graph.run(["input_features": feat, "decoder_input_ids": dec])
        guard let logits = out["logits"] else { throw KitWhisperError.modelOutputMissing("logits") }

        let flat = logits.floats()  // [1, decoderWindow, vocab]
        let base = (seq.count - 1) * arch.vocab
        var best = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for v in 0..<arch.vocab where flat[base + v] > bestVal {
            bestVal = flat[base + v]
            best = v
        }
        return Int32(best)
    }

    /// Pad/trim the waveform to the 30 s window, log-mel it, and pack into `input_features`
    /// `[melBins, melFrames]` (row-major, mel-major) as fp16.
    private func melFeatures(_ samples: [Float]) -> [Float16] {
        var x = samples
        if x.count < arch.nSamples {
            x.append(contentsOf: repeatElement(0, count: arch.nSamples - x.count))
        } else if x.count > arch.nSamples {
            x = Array(x[0..<arch.nSamples])
        }
        let (logmel, frames) = mel.logMel(x)  // [melBins, frames], frames == melFrames after padding

        var feats = [Float16](repeating: 0, count: arch.inputFeaturesCount)
        let copyT = min(frames, arch.melFrames)
        for c in 0..<arch.melBins {
            let dst = c * arch.melFrames
            let src = c * frames
            for t in 0..<copyT { feats[dst + t] = Float16(logmel[src + t]) }
        }
        return feats
    }

    /// "<|ja|>" -> "ja".
    private func cleanLanguage(_ token: String) -> String {
        token.replacingOccurrences(of: "<|", with: "").replacingOccurrences(of: "|>", with: "")
    }
}
