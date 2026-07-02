// KitASRModel.swift — a Core AI Qwen3-ASR transcription bundle. The zoo's first ASR model.
//
// Direct API (the natural transcriber path):
// ```swift
// let asr = try await KitASRModel(model: .qwen3ASR1_7B)
// try await asr.attach(samples: pcm16kMono)            // mel -> AuT encoder -> static buffer
// let result = try await asr.transcribe()              // -> Transcription(language, text)
// ```
//
// It ALSO conforms to FoundationModels' `LanguageModel`, so a `LanguageModelSession` can drive it
// (attach the clip out-of-band, then `respond(to:)` streams the transcript). v1: one clip at a time.

import CoreAILanguageModels
import CoreAIKitVision
import Foundation
import FoundationModels
import Tokenizers

/// A downloadable ASR model = a decoder bundle + its paired AuT encoder, addressed as paths inside
/// one HF repo, plus the architecture geometry.
public struct ASRModelID: Sendable, Hashable {
    public let decoder: ModelID
    public let encoder: ModelID
    public let arch: ASRArchitecture

    public init(decoder: ModelID, encoder: ModelID, arch: ASRArchitecture) {
        self.decoder = decoder
        self.encoder = encoder
        self.arch = arch
    }

    /// Qwen3-ASR-1.7B transcription (Apache-2.0) — the zoo's first ASR model. ≤30 s clips, 52 langs.
    public static let qwen3ASR1_7B = ASRModelID(
        decoder: ModelID(
            "mlboydaisuke/Qwen3-ASR-1.7B-CoreAI",
            path: "gpu-pipelined/qwen3_asr_1.7b_decode_int8hu_n390_s1"),
        encoder: ModelID(
            "mlboydaisuke/Qwen3-ASR-1.7B-CoreAI",
            path: "gpu-pipelined/qwen3_asr_1.7b_audio_encoder_fp16_k30"),
        arch: .qwen3ASR1_7B)

    /// Presets by catalog id. The catalog carries one variant path per entry, while an ASR
    /// model is a decoder + paired AuT encoder + geometry — so every `asr` catalog entry
    /// driven by `KitASRModel` pairs with a preset here, keyed by the id its card shows.
    static let byCatalogID: [String: ASRModelID] = [
        "qwen3-asr-1.7b": .qwen3ASR1_7B
    ]
}

/// A Core AI Qwen3-ASR bundle. Primary use is the direct `transcribe()`; `LanguageModel`
/// conformance is for `LanguageModelSession` integration.
public struct KitASRModel: LanguageModel {
    public typealias Executor = KitASRExecutor

    let runtime: ASRRuntime
    let modelID: String

    public var capabilities: LanguageModelCapabilities { LanguageModelCapabilities(capabilities: []) }

    public var executorConfiguration: KitASRExecutor.Configuration {
        KitASRExecutor.Configuration(runtime: runtime, modelID: modelID)
    }

    /// Loads an ASR model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let asr = try await KitASRModel(catalog: "qwen3-asr-1.7b")
    /// ```
    ///
    /// Resolves the repo, decoder/encoder pair, and geometry internally (no network needed
    /// for the lookup), then downloads if needed.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        guard let model = ASRModelID.byCatalogID[id] else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(model: model, store: store, downloadProgress: downloadProgress)
    }

    /// Downloads the decoder + encoder bundles from the Hub (if needed) and loads them.
    public init(
        model: ASRModelID,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        let encoderRoot = try await store.download(model.encoder, progress: downloadProgress)
        try await self.init(
            decoderBundleAt: decoderURL,
            encoderModelAt: Self.resolveEncoderModel(in: encoderRoot),
            arch: model.arch)
    }

    /// Loads a local decoder bundle directory + a local encoder `.aimodel`.
    public init(
        decoderBundleAt decoderURL: URL, encoderModelAt encoderURL: URL, arch: ASRArchitecture,
        encoderComputeUnits: GraphModel.ComputeUnits = .gpu, modelID: String? = nil
    ) async throws {
        let runtime = try await ASRRuntime(
            decoderBundleAt: decoderURL, encoderModelAt: encoderURL, arch: arch,
            encoderComputeUnits: encoderComputeUnits)
        self.init(runtime: runtime, modelID: modelID ?? decoderURL.standardizedFileURL.path)
    }

    public init(runtime: ASRRuntime, modelID: String) {
        self.runtime = runtime
        self.modelID = modelID
    }

    // MARK: - Direct transcriber API

    /// Encode a raw 16 kHz mono waveform into the decoder's audio buffer (mel → AuT encoder).
    public func attach(samples: [Float], sampleRate: Int = 16000) async throws {
        try await runtime.attach(samples: samples, sampleRate: sampleRate)
    }

    /// Transcribe the attached clip. `language` nil = auto-detect (52 languages). Pass `onPartial`
    /// to stream the running transcript (called after each decoded text token).
    public func transcribe(
        language: String? = nil, maxTokens: Int = 448,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        try await runtime.transcribe(language: language, maxTokens: maxTokens, onPartial: onPartial)
    }

    /// One-shot convenience: attach + transcribe. Pass `onPartial` to stream the running transcript.
    public func transcribe(
        samples: [Float], language: String? = nil,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        try await attach(samples: samples)
        return try await transcribe(language: language, onPartial: onPartial)
    }

    public func detachAudio() { runtime.detach() }

    private static func resolveEncoderModel(in root: URL) throws -> URL {
        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil),
            let aimodel = entries.first(where: { $0.pathExtension == "aimodel" })
        {
            return aimodel
        }
        return root.appendingPathComponent("\(root.lastPathComponent).aimodel")
    }
}
