// KitTranscriber.swift — one speech-to-text entry point for any `asr` catalog id. The catalog
// speaks ids while each speech family has its own driving class (Whisper = stateless graph,
// Qwen3-ASR = LLM engine + AuT encoder, Parakeet = TDT transducer), so somebody must own the
// id → class dispatch. It lives here, next to the per-class id registries, so consumers and
// example apps only ever hold a catalog id — a picker or CLI flag away from the model card —
// never an engine name.
//
// ```swift
// let transcriber = try await KitTranscriber(catalog: "whisper-large-v3-turbo")
// let result = try await transcriber.transcribe(samples: samples)   // -> Transcription
// ```

import CoreAIKitVision
import Foundation

/// Any speech-to-text model in the catalog (`ModelCatalog.builtin.available(.asr)`) behind one
/// `transcribe(samples:)` call. Use the concrete types (`KitWhisperModel`, `KitASRModel`,
/// `KitParakeetModel`) when you need family-specific control (translate, compute units, …).
public struct KitTranscriber: Sendable {
    enum Engine: Sendable {
        case whisper(KitWhisperModel)
        case qwenASR(KitASRModel)
        case parakeet(KitParakeetModel)
    }

    let engine: Engine

    /// The catalog id this transcriber was loaded from.
    public let id: String

    /// Loads a speech-to-text model by its catalog id — the id shown on the model's card.
    /// Downloads on first use (progress via `downloadProgress`), then loads from the local cache.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        if WhisperModelID.byCatalogID[id] != nil {
            engine = .whisper(
                try await KitWhisperModel(
                    catalog: id, store: store, downloadProgress: downloadProgress))
        } else if ASRModelID.byCatalogID[id] != nil {
            engine = .qwenASR(
                try await KitASRModel(
                    catalog: id, store: store, downloadProgress: downloadProgress))
        } else if id == "parakeet-tdt-0.6b-v3" {
            engine = .parakeet(
                try await KitParakeetModel(
                    catalog: id, store: store, downloadProgress: downloadProgress))
        } else if let entry = ModelCatalog.builtin.models.first(where: { $0.id == id }),
            entry.kind != .asr
        {
            throw CoreAIKitError.catalogKindMismatch(
                id: id, expected: "asr", found: entry.kind.rawValue)
        } else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        self.id = id
    }

    /// Transcribe a raw 16 kHz mono waveform. `language` nil = auto-detect (Parakeet has no
    /// language control and ignores it). Pass `onPartial` to stream the running transcript.
    public func transcribe(
        samples: [Float], language: String? = nil,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        switch engine {
        case .whisper(let model):
            return try await model.transcribe(
                samples: samples, language: language, onPartial: onPartial)
        case .qwenASR(let model):
            return try await model.transcribe(
                samples: samples, language: language, onPartial: onPartial)
        case .parakeet(let model):
            return try await model.transcribe(samples: samples, onPartial: onPartial)
        }
    }
}
