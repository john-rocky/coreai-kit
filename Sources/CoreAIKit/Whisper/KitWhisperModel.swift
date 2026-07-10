// KitWhisperModel.swift — a downloadable Core AI Whisper transcription bundle. The zoo's first
// official ASR (Apple-recipe export on the stock runtime), here wrapped as a one-call kit model.
//
// ```swift
// let whisper = try await KitWhisperModel(model: .largeV3Turbo)
// let result = try await whisper.transcribe(samples: pcm16kMono)   // -> Transcription(language, text)
// ```
//
// Unlike `KitASRModel` (Qwen3-ASR on the pipelined LLM engine) this rides one stateless graph via
// `GraphModel`, so it needs no decoder bundle/metadata — just the `.aimodel` + `tokenizer/`.

import CoreAIKitVision
import Foundation

/// A hosted Whisper model = the graph `.aimodel` (by name inside the repo) + its `tokenizer/`,
/// plus the fixed-window geometry.
public struct WhisperModelID: Sendable, Hashable {
    public let model: ModelID
    public let aimodelName: String
    public let arch: WhisperArchitecture

    public init(model: ModelID, aimodelName: String, arch: WhisperArchitecture) {
        self.model = model
        self.aimodelName = aimodelName
        self.arch = arch
    }

    /// Whisper large-v3-turbo (OpenAI, MIT) — the zoo's first official ASR, fp16 fixed-128 window,
    /// ≤30 s clips, 100 languages. Flat repo (`.aimodel` + `tokenizer/` at root).
    public static let largeV3Turbo = WhisperModelID(
        model: .whisperLargeV3Turbo,
        aimodelName: "whisper-large-v3-turbo_float16_fixed128.aimodel",
        arch: .largeV3Turbo)

    /// Presets by catalog id. Whisper geometry (window/vocab/special tokens) can't ride
    /// catalog.json, so every `asr` catalog entry driven by `KitWhisperModel` pairs with a
    /// preset here — the id is the one the model's card shows.
    static let byCatalogID: [String: WhisperModelID] = [
        "whisper-large-v3-turbo": .largeV3Turbo
    ]

    /// A copy with the graph id pinned to a Hub revision (nil = unchanged).
    func pinned(_ revision: String?) -> WhisperModelID {
        guard let revision else { return self }
        return WhisperModelID(
            model: model.pinned(revision), aimodelName: aimodelName, arch: arch)
    }
}

/// A Core AI Whisper transcription bundle. Primary use is the direct `transcribe()`.
/// `Sendable` is explicit: a public struct doesn't infer it across modules, and the app's
/// `@MainActor` view model sends the value into `transcribe`'s non-isolated async context.
public struct KitWhisperModel: Sendable {
    public let runtime: WhisperRuntime

    /// Loads a Whisper model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let whisper = try await KitWhisperModel(catalog: "whisper-large-v3-turbo")
    /// ```
    ///
    /// Resolves the graph geometry from the preset table, and the platform availability +
    /// revision pin from the live catalog (built-in snapshot offline), then downloads if
    /// needed.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .asr)
        guard let model = WhisperModelID.byCatalogID[entry.id] else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(
            model: model.pinned(entry.revision), store: store, computeUnits: computeUnits,
            downloadProgress: downloadProgress)
    }

    /// Downloads the graph + tokenizer from the Hub (if needed) and loads them.
    public init(
        model: WhisperModelID = .largeV3Turbo,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let root = try await store.download(model.model, progress: downloadProgress)
        try await self.init(bundleRootAt: root, model: model, computeUnits: computeUnits)
    }

    /// Loads from a local bundle root that holds `<aimodelName>` and `tokenizer/`.
    public init(
        bundleRootAt root: URL,
        model: WhisperModelID = .largeV3Turbo,
        computeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        self.runtime = try await WhisperRuntime(
            modelAt: root.appendingPathComponent(model.aimodelName),
            tokenizerAt: root.appendingPathComponent("tokenizer"),
            arch: model.arch,
            computeUnits: computeUnits)
    }

    /// Transcribe a raw 16 kHz mono waveform. `language` nil = auto-detect (100 languages). Pass
    /// `onPartial` to stream the running transcript (called after each decoded token).
    public func transcribe(
        samples: [Float], sampleRate: Int = 16000, language: String? = nil, translate: Bool = false,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        try await runtime.transcribe(
            samples: samples, sampleRate: sampleRate, language: language, translate: translate,
            onPartial: onPartial)
    }
}
