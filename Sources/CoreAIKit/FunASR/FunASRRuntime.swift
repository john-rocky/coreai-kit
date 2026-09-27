// FunASRRuntime.swift — the loaded pieces of a Fun-ASR-Nano transcription model: the Qwen3-0.6B
// decoder engine wired with ONE static `audio_embeds` input, the audio encoder graph, the kaldi fbank
// + LFR front end, and the owned host buffer the decoder gathers from. Mirrors `ASRRuntime`
// (Qwen3-ASR); what differs is the front end, the encoder's two Float32 inputs, funasr's prompt
// options (hotwords / language / itn), its two stop ids and its output clean-up. Fun-ASR emits no
// language tag, so `Transcription.language` is empty.
//
// Ship form: the static-[1,1]-query decoder (`_s1`, dynamic KV) driven by the high-level pipelined
// engine with `audio_embeds` bound as a static input buffer — the Qwen3-ASR / Qwen-Omni path. The
// engine manages KV cache + positions; chunked prefill stays off so prefill runs as S=1 steps.

import CoreAILanguageModels
import CoreAIKitVision
import Foundation
import Metal
import Synchronization
import Tokenizers

/// Owns a Fun-ASR-Nano model's engine + audio encoder + the static `audio_embeds` buffer. Serial
/// use (one transcription at a time) — the underlying engine traps on concurrent generate calls.
@available(macOS 27, iOS 27, *)
public final class FunASRRuntime: @unchecked Sendable {
    /// One decoded window: funasr's text, the generated ids (the stop id included when the model
    /// stopped), and where the time went.
    struct Decoded: Sendable {
        var text: String
        var tokenIDs: [Int32]
        var hitCap: Bool
        var promptLength: Int
        var frontendMs: Double = 0
        var encoderMs: Double = 0
        /// `generate` call -> first token: the S=1 prefill of the whole prompt.
        var prefillMs: Double
        /// First token -> last token.
        var decodeMs: Double
    }

    public let arch: FunASRArchitecture
    let engine: any InferenceEngine
    public let tokenizer: any Tokenizer

    private let encoder: GraphModel
    private let frontEnd: FunASRFbankPreprocessor
    private let audioBuffer: any MTLBuffer
    private let attachedTokenCount = Mutex<Int>(0)

    public init(
        decoderBundleAt decoderURL: URL,
        encoderModelAt encoderURL: URL,
        arch: FunASRArchitecture = .funASRNano2512,
        encoderComputeUnits: GraphModel.ComputeUnits = .gpu,
        engineVariant: EngineVariant = .pipelined
    ) async throws {
        // The shippable bundle is the static-[1,1]-query twin; chunked prefill stays off so prefill
        // runs as pipelined S=1 steps — the Qwen3-ASR / omni-proven config.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil { setenv("COREAI_CHUNK_THRESHOLD", "1", 1) }
        self.arch = arch
        self.frontEnd = FunASRFbankPreprocessor()

        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        guard let device = MTLCreateSystemDefaultDevice() else { throw KitASRError.noMetalDevice }
        let byteCount = arch.audioEmbedsCount * MemoryLayout<Float16>.size
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw KitASRError.bufferAllocationFailed
        }
        memset(buffer.contents(), 0, byteCount)
        self.audioBuffer = buffer

        let bundle = try LanguageBundle(at: decoderURL)
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
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }

    public var audioAttached: Bool { attachedTokenCount.withLock { $0 > 0 } }
    public var attachedAudioTokens: Int { attachedTokenCount.withLock { $0 } }

    // MARK: - Audio attach (front end -> encoder -> static buffer)

    /// Encode LFR features `[L, 560]` (row-major, L <= 500) through the audio encoder and write the
    /// first N = ceil(L / 8) rows of its output into the decoder's static buffer (the rest is zeroed;
    /// the decoder gathers `0..<N`).
    public func attach(feats: [Float], lfrFrames l: Int) async throws {
        let n = FunASRFbankPreprocessor.audioTokenCount(lfrFrames: l)
        guard l <= arch.lfrMax, n <= arch.maxAudioTokens else {
            throw KitASRError.audioTooLong(tokens: n, max: arch.maxAudioTokens)
        }
        guard n > 0 else {
            detach()
            return
        }
        let (x, mask, _) = arch.encoderInputs(feats: feats, lfrFrames: l)
        let outputs = try await encoder.run([
            "feats": .float32(x, shape: [1, arch.lfrMax, arch.featureDim]),
            "mask": .float32(mask, shape: [1, arch.lfrMax]),
        ])
        guard let embeds = outputs["audio_embeds"] else {
            throw KitASRError.encoderOutputMissing("audio_embeds")
        }
        writeEmbeds(embeds.floats(), audioTokenCount: n)
    }

    /// Encode one window of raw 16 kHz mono audio — at most `arch.windowSamples` (30 s): fbank + LFR
    /// -> audio encoder -> static buffer. `KitFunASRModel.transcribe(samples:)` cuts longer audio.
    public func attach(samples: [Float], sampleRate: Int = 16000) async throws {
        guard sampleRate == 16000 else { throw KitASRError.unsupportedSampleRate(sampleRate) }
        _ = try await attachTimed(samples: samples)
    }

    /// `attach(samples:)`, returning the front-end and encoder wall times in ms.
    func attachTimed(samples: [Float]) async throws -> (frontendMs: Double, encoderMs: Double) {
        let l = FunASRFbankPreprocessor.lfrFrames(samples: samples.count)
        guard l <= arch.lfrMax else {
            throw KitASRError.audioTooLong(
                tokens: FunASRFbankPreprocessor.audioTokenCount(lfrFrames: l), max: arch.maxAudioTokens)
        }
        let clock = ContinuousClock()
        let start = clock.now
        let (feats, frames) = frontEnd.features(samples)
        let encoded = clock.now
        try await attach(feats: feats, lfrFrames: frames)
        return (Self.milliseconds(encoded - start), Self.milliseconds(clock.now - encoded))
    }

    public func detach() {
        memset(audioBuffer.contents(), 0, audioBuffer.length)
        attachedTokenCount.withLock { $0 = 0 }
    }

    private func writeEmbeds(_ values: [Float], audioTokenCount n: Int) {
        memset(audioBuffer.contents(), 0, audioBuffer.length)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let pointer = audioBuffer.contents().assumingMemoryBound(to: Float16.self)
        let valid = min(n * arch.hidden, values.count, arch.audioEmbedsCount)
        for i in 0..<valid { pointer[i] = Float16(values[i]) }
        attachedTokenCount.withLock { $0 = n }
        #else
        fatalError("Float16 is not supported on this platform")
        #endif
    }

    // MARK: - Transcribe (one attached window)

    /// Transcribe the attached window: funasr's prompt, greedy decode to `<|im_end|>` /
    /// `<|endoftext|>` (or `maxTokens`, funasr's cap of 512), decode, clean up. `hotwords` is a
    /// context list the model may prefer; `language` a hint such as "中文" (nil = auto); `itn`
    /// off asks for unnormalized text. `onPartial` is called with the running transcript after each
    /// decoded token.
    public func transcribe(
        hotwords: [String] = [], language: String? = nil, itn: Bool = true, maxTokens: Int = 512,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        let decoded = try await decode(
            hotwords: hotwords, language: language, itn: itn, maxTokens: maxTokens,
            onPartial: onPartial)
        return Transcription(language: "", text: decoded.text)
    }

    func decode(
        hotwords: [String], language: String?, itn: Bool, maxTokens: Int,
        onPartial: (@Sendable (String) -> Void)?
    ) async throws -> Decoded {
        let n = attachedAudioTokens
        guard n > 0 else { throw KitASRError.noAudioAttached }
        let prompt = FunASRPromptRenderer.render(
            tokenizer: tokenizer, arch: arch, audioTokenCount: n, hotwords: hotwords,
            language: language, itn: itn)

        try await engine.reset()
        let clock = ContinuousClock()
        var ids: [Int32] = []
        var textIDs: [Int] = []
        var emitted = ""
        var stopped = false
        let start = clock.now
        var first = start
        var last = start
        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        for try await output in stream {
            last = clock.now
            if ids.isEmpty { first = last }
            ids.append(output.tokenId)
            if arch.eosTokenIDs.contains(output.tokenId) {
                stopped = true
                break
            }
            textIDs.append(Int(output.tokenId))
            guard let onPartial else { continue }
            // Re-decode the running transcript; skip emits that land mid-multibyte (`\u{FFFD}`).
            let partial = Self.clean(tokenizer.decode(tokens: textIDs, skipSpecialTokens: true))
            if partial != emitted && !partial.unicodeScalars.contains("\u{FFFD}") {
                emitted = partial
                onPartial(partial)
            }
        }
        return Decoded(
            text: Self.clean(tokenizer.decode(tokens: textIDs, skipSpecialTokens: true)),
            tokenIDs: ids, hitCap: !stopped && ids.count >= maxTokens, promptLength: prompt.count,
            prefillMs: ids.isEmpty ? 0 : Self.milliseconds(first - start),
            decodeMs: Self.milliseconds(last - first))
    }

    /// funasr's clean-up of the decoded text, `re.sub(r"\s+", " ", text.replace("/sil", " "))`, then
    /// trimmed. Whitespace is Python's `str.isspace` set, so the text matches funasr's character for
    /// character.
    static func clean(_ text: String) -> String {
        var out = String.UnicodeScalarView()
        var gap = false
        for scalar in text.replacingOccurrences(of: "/sil", with: " ").unicodeScalars {
            if isPythonWhitespace(scalar) {
                gap = true
                continue
            }
            if gap && !out.isEmpty { out.append(" ") }
            gap = false
            out.append(scalar)
        }
        return String(out)
    }

    private static func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F,
            0x205F, 0x3000:
            return true
        default:
            return false
        }
    }

    static func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
    }
}
