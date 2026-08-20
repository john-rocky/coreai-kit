// KitAudioModel.swift — a Core AI audio-understanding bundle behind FoundationModels'
// `LanguageModel` protocol, so a local audio model answers "what do you hear?" through a plain
// `LanguageModelSession`.
//
// ```swift
// let model = try await KitAudioModel(decoderBundleAt: dec, encoderModelAt: enc, arch: .qwen2_5Omni3B)
// try await model.attach(samples: pcm16kMono)          // mel -> encoder -> static buffer
// let answer = try await LanguageModelSession(model: model).respond(to: "What do you hear?")
// ```
//
// FoundationModels has no audio attachment segment, so the audio is attached to the runtime
// out-of-band (the call above) before the session asks the question; the executor renders the
// audio block from the attached count. The audio path is its own executor (`KitAudioExecutor`) and
// runtime (`AudioRuntime`): the text `KitLanguageModel` is untouched. v1: one clip per session.

import CoreAILanguageModels
import CoreAIKitVision
import Foundation
import FoundationModels
import Tokenizers

/// A downloadable audio model = a decoder bundle + its paired audio encoder, addressed as paths
/// inside one HF repo, plus the architecture geometry.
public struct AudioModelID: Sendable, Hashable {
    public let decoder: ModelID
    public let encoder: ModelID
    public let arch: AudioArchitecture

    public init(decoder: ModelID, encoder: ModelID, arch: AudioArchitecture) {
        self.decoder = decoder
        self.encoder = encoder
        self.arch = arch
    }

    /// Qwen2.5-Omni-3B audio understanding (Apache-2.0) — the zoo's first on-device audio-QA
    /// model. Mac-class. (HF paths populated when the bundles are published.)
    public static let qwen2_5Omni3B = AudioModelID(
        decoder: ModelID(
            "mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI",
            path: "gpu-pipelined/qwen2_5_omni_3b_thinker_int8lin_n750_s1"),
        encoder: ModelID(
            "mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI",
            path: "gpu-pipelined/qwen2_5_omni_3b_audio_encoder_fp16_k15"),
        arch: .qwen2_5Omni3B)

    /// Presets by catalog id. An audio model is a decoder + encoder pair plus graph geometry
    /// — none of which ride catalog.json — so every `audio` catalog entry pairs with a preset
    /// here; the id is the one the model's card shows.
    static let byCatalogID: [String: AudioModelID] = [
        "qwen2.5-omni-3b-audio": .qwen2_5Omni3B
    ]

    /// A copy with both sub-bundle ids pinned to a Hub revision (nil = unchanged), so a
    /// catalog entry's pin covers the decoder and its paired audio encoder alike.
    func pinned(_ revision: String?) -> AudioModelID {
        guard let revision else { return self }
        return AudioModelID(
            decoder: decoder.pinned(revision), encoder: encoder.pinned(revision), arch: arch)
    }
}

/// A Core AI audio-understanding bundle as a `LanguageModelSession` provider.
public struct KitAudioModel: LanguageModel {
    public typealias Executor = KitAudioExecutor

    let runtime: AudioRuntime
    let modelID: String
    let profile: OutputProfile

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        if profile.rules.contains(where: { $0.kind == .thinking }) {
            capabilities.append(.reasoning)
        }
        return LanguageModelCapabilities(capabilities)
    }

    public var executorConfiguration: KitAudioExecutor.Configuration {
        KitAudioExecutor.Configuration(runtime: runtime, modelID: modelID, profile: profile)
    }

    /// Loads an audio-understanding model by its catalog id — the id shown on the model's
    /// card:
    ///
    /// ```swift
    /// let audio = try await KitAudioModel(catalog: "qwen2.5-omni-3b-audio")
    /// ```
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .audio)
        guard let model = AudioModelID.byCatalogID[entry.id] else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(
            model: model.pinned(entry.revision), store: store, downloadProgress: downloadProgress)
    }

    /// Downloads the decoder + encoder bundles from the Hub (if needed) and loads them.
    public init(
        model: AudioModelID,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        let encoderRoot = try await store.download(model.encoder, progress: downloadProgress)
        try await self.init(
            decoderBundleAt: decoderURL, encoderModelAt: encoderRoot, arch: model.arch)
    }

    /// Loads a local decoder bundle directory + a local encoder graph (the `.aimodel`/`.aimodelc`
    /// itself, or the bundle directory holding it).
    public init(
        decoderBundleAt decoderURL: URL, encoderModelAt encoderURL: URL, arch: AudioArchitecture,
        encoderComputeUnits: GraphModel.ComputeUnits = .gpu, modelID: String? = nil
    ) async throws {
        let runtime = try await AudioRuntime(
            decoderBundleAt: decoderURL, encoderModelAt: encoderURL, arch: arch,
            encoderComputeUnits: encoderComputeUnits)
        self.init(runtime: runtime, modelID: modelID ?? decoderURL.standardizedFileURL.path)
    }

    /// Wraps an already-created runtime. `modelID` keys the session's executor cache.
    public init(runtime: AudioRuntime, modelID: String) {
        self.runtime = runtime
        self.modelID = modelID
        self.profile = OutputProfile.detect(probing: runtime.tokenizer)
    }

    // MARK: - Audio attach (out-of-band; call before asking the question)

    /// Encode a raw 16 kHz mono waveform into the decoder's audio buffer (mel → encoder).
    public func attach(samples: [Float], sampleRate: Int = 16000) async throws {
        try await runtime.attach(samples: samples, sampleRate: sampleRate)
    }

    /// Clears the resident audio (the next turn is text-only).
    public func detachAudio() { runtime.detach() }
}
