// ASRRuntime.swift — the loaded pieces of a Qwen3-ASR transcription model: a text decoder engine
// wired with ONE static `audio_embeds` input, the paired AuT audio encoder, and the owned host
// buffer the decoder gathers from. Mirrors `AudioRuntime` (audio understanding); the difference is
// the AuT encoder's `[1, S, S]` attn_bias and the ASR `{lang}<asr_text>{text}` output parse.
//
// Ship form (decided): the OMNI-STYLE static-[1,1]-query decoder driven by the high-level engine
// with `audio_embeds` bound as a static input buffer — the same path the shipped Qwen-Omni audio
// model uses on iPhone. The engine manages KV cache + positions, so there is no per-step wedge.

import CoreAILanguageModels
import CoreAIKitVision
import Foundation
import Metal
import Synchronization
import Tokenizers

/// Failures specific to the ASR / transcription path.
public enum KitASRError: Error, LocalizedError {
    case noMetalDevice
    case bufferAllocationFailed
    case encoderOutputMissing(String)
    case audioTooLong(tokens: Int, max: Int)
    case unsupportedSampleRate(Int)
    case noAudioAttached

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "No Metal device available for the ASR runtime."
        case .bufferAllocationFailed: return "Could not allocate the audio static-input buffer."
        case .encoderOutputMissing(let name):
            return "AuT encoder did not produce the expected output '\(name)'."
        case .audioTooLong(let tokens, let max):
            return "Clip has \(tokens) audio tokens but the bundle accepts at most \(max) (≈30 s)."
        case .unsupportedSampleRate(let sr):
            return "Audio must be 16 kHz mono; got \(sr) Hz (resample before transcribing)."
        case .noAudioAttached: return "No audio attached — call attach(samples:) before transcribing."
        }
    }
}

/// One transcription: the auto-detected language tag + the transcript text.
public struct Transcription: Sendable, Hashable {
    public let language: String
    public let text: String
}

/// Owns an ASR model's engine + AuT encoder + the static `audio_embeds` buffer. Serial use (one
/// transcription at a time) — the underlying engine traps on concurrent generate calls.
public final class ASRRuntime: @unchecked Sendable {
    public let arch: ASRArchitecture
    let engine: any InferenceEngine
    public let tokenizer: any Tokenizer
    let modelName: String
    let maxContextLength: Int
    let vocabSize: Int

    private let encoder: GraphModel
    private let mel: AudioMelPreprocessor
    private let audioBuffer: any MTLBuffer
    private let attachedTokenCount = Mutex<Int>(0)

    public init(
        decoderBundleAt decoderURL: URL,
        encoderModelAt encoderURL: URL,
        arch: ASRArchitecture,
        encoderComputeUnits: GraphModel.ComputeUnits = .gpu,
        engineVariant: EngineVariant = .pipelined
    ) async throws {
        // The shippable bundle is the static-[1,1]-query twin; chunked prefill stays off so prefill
        // runs as pipelined S=1 steps — the qwen3_vl / omni-proven config.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil { setenv("COREAI_CHUNK_THRESHOLD", "1", 1) }
        self.arch = arch

        guard let device = MTLCreateSystemDefaultDevice() else { throw KitASRError.noMetalDevice }
        let byteCount = arch.audioEmbedsCount * MemoryLayout<Float16>.size
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw KitASRError.bufferAllocationFailed
        }
        memset(buffer.contents(), 0, byteCount)
        self.audioBuffer = buffer

        let bundle = try LanguageBundle(at: decoderURL)
        self.modelName = bundle.name
        self.maxContextLength = bundle.maxContextLength
        self.vocabSize = bundle.vocabSize

        let config = ModelConfig(
            name: bundle.name, tokenizer: bundle.tokenizer, vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath], function: "main")
        let modelURL = try bundle.requireModelURL(for: "main")
        self.engine = try await EngineFactory.createEngine(
            config: try JSONEncoder().encode(config),
            modelURL: modelURL,
            options: EngineOptions(
                variant: engineVariant.factoryOverride,
                staticInputBuffers: ["audio_embeds": StaticInputBuffer(buffer)]))

        self.tokenizer = try await bundle.loadTokenizer()
        self.encoder = try await GraphModel(
            contentsOf: try GraphBundle.resolve(in: encoderURL),
            computeUnits: encoderComputeUnits)
        self.mel = try AudioMelPreprocessor.qwen2_5Omni()  // same Whisper frontend (128 mel, 400/160)
    }

    public var audioAttached: Bool { attachedTokenCount.withLock { $0 > 0 } }
    public var attachedAudioTokens: Int { attachedTokenCount.withLock { $0 } }

    // MARK: - Audio attach (AuT encoder -> static buffer)

    /// Encode host-prepared `inputFeatures`/`attnBias` through the AuT encoder and write its first
    /// `n` rows into the decoder's static buffer (the rest is zeroed; the decoder gathers `0..<N`).
    public func attach(inputFeatures: [Float16], attnBias: [Float16], audioTokenCount n: Int) async throws {
        guard n <= arch.maxAudioTokens else {
            throw KitASRError.audioTooLong(tokens: n, max: arch.maxAudioTokens)
        }
        let s = arch.maxAudioTokens
        let outputs = try await encoder.run([
            "input_features": .float16(inputFeatures, shape: [1, arch.melBins, arch.melFrames]),
            "attn_bias": .float16(attnBias, shape: [1, s, s]),
        ])
        guard let embeds = outputs["audio_embeds"] else {
            throw KitASRError.encoderOutputMissing("audio_embeds")
        }
        writeEmbeds(embeds.floats(), audioTokenCount: n)
    }

    /// Encode a raw 16 kHz mono waveform: Whisper log-mel -> AuT encoder -> static buffer. The clip
    /// is trimmed to the bundle's max duration (≈30 s).
    public func attach(samples: [Float], sampleRate: Int = 16000) async throws {
        guard sampleRate == 16000 else { throw KitASRError.unsupportedSampleRate(sampleRate) }
        let maxSamples = arch.melFrames * 160  // melFrames * hop(160) ≈ 30 s @ 16 kHz
        let clip = samples.count > maxSamples ? Array(samples[0..<maxSamples]) : samples
        let (logmel, frames) = mel.logMel(clip)
        let (feats, bias, n) = arch.encoderInputs(fromMel: logmel, frames: frames)
        try await attach(inputFeatures: feats, attnBias: bias, audioTokenCount: n)
    }

    public func detach() {
        memset(audioBuffer.contents(), 0, audioBuffer.length)
        attachedTokenCount.withLock { $0 = 0 }
    }

    private func writeEmbeds(_ values: [Float], audioTokenCount n: Int) {
        memset(audioBuffer.contents(), 0, audioBuffer.length)
        let pointer = audioBuffer.contents().assumingMemoryBound(to: Float16.self)
        let valid = min(n * arch.hidden, values.count, arch.audioEmbedsCount)
        for i in 0..<valid { pointer[i] = Float16(values[i]) }
        attachedTokenCount.withLock { $0 = n }
    }

    // MARK: - Transcribe (direct, low-level)

    /// Transcribe the already-attached clip. Builds the ASR prompt, greedy-decodes to EOS, and
    /// splits `{lang}<asr_text>{text}` at the `<asr_text>` token. `language` is the optional forced
    /// language (e.g. "Japanese"); nil = auto-detect.
    /// `onPartial` (optional) is called with the running transcript after each decoded text token, so
    /// a UI can show text as it streams. The `{lang}` prefix is dropped before the first emit.
    public func transcribe(
        language: String? = nil, maxTokens: Int = 448,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        let n = attachedAudioTokens
        guard n > 0 else { throw KitASRError.noAudioAttached }
        let prompt = ASRPromptRenderer.render(
            tokenizer: tokenizer, arch: arch, audioTokenCount: n, language: language)

        try await engine.reset()
        let asrTextID = tokenizer.convertTokenToId(arch.asrText)
        let eosTokenId = tokenizer.eosTokenId
        // With a forced language the prompt already primed `<asr_text>`, so the transcript starts now;
        // in auto mode the model first emits a `{lang}` prefix, then `<asr_text>`, then the text.
        var inText = (language != nil) || (asrTextID == nil)
        var markerSeen = false
        var langTokens: [Int] = []
        var transcriptTokens: [Int] = []
        var emitted = ""

        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        for try await output in stream {
            let token = Int(output.tokenId)
            if let eos = eosTokenId, token == eos { break }
            if !inText {
                if token == asrTextID { inText = true; markerSeen = true } else { langTokens.append(token) }
                continue  // still in the `{lang}` prefix
            }
            transcriptTokens.append(token)
            guard let onPartial else { continue }
            // Re-decode the running transcript; skip emits that land mid-multibyte (`\u{FFFD}`).
            let decoded = clean(tokenizer.decode(tokens: transcriptTokens))
            if decoded != emitted && !decoded.unicodeScalars.contains("\u{FFFD}") {
                emitted = decoded
                onPartial(decoded)
            }
        }

        // Auto mode where the model never emitted `<asr_text>`: the whole output is the transcript.
        if language == nil, asrTextID != nil, !markerSeen {
            return Transcription(language: "", text: clean(tokenizer.decode(tokens: langTokens)))
        }
        let lang = language ?? clean(tokenizer.decode(tokens: langTokens))
        return Transcription(language: lang, text: clean(tokenizer.decode(tokens: transcriptTokens)))
    }

    /// Greedy-generate from a `vocab + slot`-rewritten prompt; returns the generated ids
    /// (EOS-terminated, EOS excluded).
    public func generate(promptIDs: [Int32], maxTokens: Int) async throws -> [Int] {
        try await engine.reset()
        var ids: [Int] = []
        let stream = try await engine.generate(
            with: promptIDs, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        for try await output in stream {
            let token = Int(output.tokenId)
            if let eos = tokenizer.eosTokenId, token == eos { break }
            ids.append(token)
        }
        return ids
    }

    private func clean(_ s: String) -> String {
        s.replacingOccurrences(of: "language ", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func warmup() async throws {
        let seed = tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
        let stream = try await engine.generate(
            with: [seed], samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1))
        for try await _ in stream {}
        try await engine.reset()
    }
}
