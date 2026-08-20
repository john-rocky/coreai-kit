// AudioRuntime.swift — the loaded pieces of an audio-understanding model: a text decoder engine
// wired with ONE static `audio_embeds` input, the paired Whisper-style audio encoder, and the
// owned host buffer the decoder gathers from.
//
// Mirrors `VLRuntime` (the static-input-buffer wiring DUAL_PROFILE_STATE #8 added) but is the
// simpler rider: a single static buffer and NO rope-shift inputs (TMRoPE collapses to 1-D for
// audio+text, so the engine's native sequential positions are exact). The decoder graph gathers
// in-graph — audio tokens carry extension ids `vocab + slot`, the graph reads `audio_embeds`
// row `id - vocab`. The audio encoder runs once per clip through `CoreAIKitVision.GraphModel`
// (a stateless `.aimodel`), exactly as the VL path runs its ViT.

import CoreAILanguageModels
import CoreAIKitVision
import Foundation
import Metal
import Synchronization
import Tokenizers

/// Failures specific to the audio-understanding path.
public enum KitAudioError: Error, LocalizedError {
    case noMetalDevice
    case bufferAllocationFailed
    case encoderOutputMissing(String)
    case audioTooLong(tokens: Int, max: Int)
    case melFiltersMissing
    case unsupportedSampleRate(Int)
    case renderFailed(String)

    public var errorDescription: String? {
        switch self {
        case .renderFailed(let why): return "Audio prompt render failed: \(why)"
        case .noMetalDevice: return "No Metal device available for the audio runtime."
        case .bufferAllocationFailed: return "Could not allocate the audio static-input buffer."
        case .encoderOutputMissing(let name):
            return "Audio encoder did not produce the expected output '\(name)'."
        case .audioTooLong(let tokens, let max):
            return "Clip has \(tokens) audio tokens but the bundle accepts at most \(max)."
        case .melFiltersMissing:
            return "Bundled mel_filters.f32 resource was not found in CoreAIKit."
        case .unsupportedSampleRate(let sr):
            return "Audio must be 16 kHz mono; got \(sr) Hz (resample before attaching)."
        }
    }
}

/// Owns an audio model's engine + audio encoder + the static `audio_embeds` buffer, for the
/// engine's lifetime. One runtime assumes serial use (one `LanguageModelSession` at a time) —
/// the underlying engine traps on concurrent generate calls, same as the text path.
public final class AudioRuntime: @unchecked Sendable {
    public let arch: AudioArchitecture
    let engine: any InferenceEngine
    public let tokenizer: any Tokenizer
    let modelName: String
    let maxContextLength: Int
    let vocabSize: Int

    private let encoder: GraphModel
    private let mel: AudioMelPreprocessor
    // Owned static-input buffer ([maxAudioTokens, hidden] fp16), alive for the engine's lifetime.
    private let audioBuffer: any MTLBuffer
    /// How many leading rows of `audioBuffer` are valid audio embeds (0 = none attached).
    private let attachedTokenCount = Mutex<Int>(0)

    /// Loads the decoder bundle + audio encoder and wires the single static input.
    public init(
        decoderBundleAt decoderURL: URL,
        encoderModelAt encoderURL: URL,
        arch: AudioArchitecture,
        encoderComputeUnits: GraphModel.ComputeUnits = .gpu,
        engineVariant: EngineVariant = .pipelined
    ) async throws {
        // The shippable engine bundle is the static-[1,1]-query twin (the dynamic-query twin
        // crashes the macOS-beta pipelined engine); chunked prefill must stay off so prefill runs
        // as pipelined S=1 steps — the qwen3_vl-proven config.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        self.arch = arch

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw KitAudioError.noMetalDevice
        }
        let byteCount = arch.audioEmbedsCount * MemoryLayout<Float16>.size
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw KitAudioError.bufferAllocationFailed
        }
        memset(buffer.contents(), 0, byteCount)
        self.audioBuffer = buffer

        let bundle = try LanguageBundle(at: decoderURL)
        self.modelName = bundle.name
        self.maxContextLength = bundle.maxContextLength
        self.vocabSize = bundle.vocabSize

        // Build the engine through the factory so we can bind the static input (CoreAIRunner does
        // not expose EngineOptions). `"main"` avoids importing CoreAIShared's ComponentKey.
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: "main")
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
        self.mel = try AudioMelPreprocessor.qwen2_5Omni()
    }

    /// Whether any audio embeds are currently resident in the buffer.
    public var audioAttached: Bool { attachedTokenCount.withLock { $0 > 0 } }

    /// How many leading `audio_embeds` rows are valid (the renderer emits this many `<|AUDIO|>`).
    public var attachedAudioTokens: Int { attachedTokenCount.withLock { $0 } }

    // MARK: - Audio attach

    /// Encode `inputFeatures`/`attnBias` (host-prepared mel padded to whole chunks) through the
    /// audio encoder and write its first `audioTokenCount` rows into the decoder's static buffer.
    /// `audioTokenCount` is the clip's real audio-token count `N` (the rest of the buffer is
    /// zeroed; the decoder only gathers slots `0..<N`).
    public func attach(
        inputFeatures: [Float16], attnBias: [Float16], audioTokenCount n: Int
    ) async throws {
        guard n <= arch.maxAudioTokens else {
            throw KitAudioError.audioTooLong(tokens: n, max: arch.maxAudioTokens)
        }
        let outputs = try await encoder.run([
            "input_features": .float16(inputFeatures, shape: [1, arch.melBins, arch.melFrames]),
            "attn_bias": .float16(attnBias, shape: [arch.chunks, 1, 1, arch.headsFrames]),
        ])
        guard let embeds = outputs["audio_embeds"] else {
            throw KitAudioError.encoderOutputMissing("audio_embeds")
        }
        writeEmbeds(embeds.floats(), audioTokenCount: n)
    }

    /// Encode a raw 16 kHz mono waveform: Swift vDSP log-mel (bit-exact with the HF extractor) →
    /// encoder → static buffer. The clip is trimmed to the bundle's max duration (≈30 s). This is
    /// the production app path (the gate paths feed host-prepared tensors instead).
    public func attach(samples: [Float], sampleRate: Int = 16000) async throws {
        guard sampleRate == 16000 else { throw KitAudioError.unsupportedSampleRate(sampleRate) }
        let maxSamples = arch.melFrames * 160  // melFrames * hop(160) = ≈30 s @ 16 kHz
        let clip = samples.count > maxSamples ? Array(samples[0..<maxSamples]) : samples
        let (logmel, frames) = mel.logMel(clip)
        let (feats, bias, n) = arch.encoderInputs(fromMel: logmel, frames: frames)
        try await attach(inputFeatures: feats, attnBias: bias, audioTokenCount: n)
    }

    /// Write a precomputed `[maxAudioTokens, hidden]` embeds buffer directly (isolates the decoder
    /// + static-input wiring from the encoder; the values are taken as-is).
    public func attachEmbeds(_ embeds: [Float16], audioTokenCount n: Int) {
        let pointer = audioBuffer.contents().assumingMemoryBound(to: Float16.self)
        let count = min(embeds.count, arch.audioEmbedsCount)
        for i in 0..<count { pointer[i] = embeds[i] }
        attachedTokenCount.withLock { $0 = n }
    }

    /// Clears the resident audio embeds.
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

    // MARK: - Generate (low-level)

    /// Greedy-generate from a `vocab + slot`-rewritten prompt. Returns the generated token ids
    /// (EOS-terminated, EOS excluded). The static `audio_embeds` buffer must already be attached.
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

    /// Decode token ids to text (skips nothing; callers strip specials as needed).
    public func decode(_ ids: [Int]) -> String { tokenizer.decode(tokens: ids) }

    // MARK: - Warmup

    /// One real 1-token generate + reset: compiles the sampler graph and touches the weights.
    public func warmup() async throws {
        let seed = tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
        let stream = try await engine.generate(
            with: [seed], samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1))
        for try await _ in stream {}
        try await engine.reset()
    }
}
