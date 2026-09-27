// QuickStart.swift — the take-home core of this runner: speech in → text out, no UI. Both shells
// call this file for every piece of model work: the GUI view model keeps a `SpeechModel` loaded
// and transcribes clip after clip with it, the CLI (`CLI/main.swift`) runs
// `transcribe(audio:model:)` once. Want transcription in your own app? This file is the part you
// copy; the model card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation

/// Transcribe an audio file with any speech-to-text model in the catalog
/// (`ModelCatalog.builtin.available(.asr)`: Whisper large-v3-turbo, Qwen3-ASR, Parakeet-TDT,
/// Fun-ASR-Nano), loading the model for this one call. First use downloads the model (progress
/// via `downloadProgress`), later runs load from the local cache. `hotwords` — names and terms to
/// spell right — needs a model that takes them (`SpeechModel.takesHotwords`). `onPartial` streams
/// the running transcript while the clip decodes.
func transcribe(
    audio url: URL,
    model id: String = "whisper-large-v3-turbo",
    hotwords: [String] = [],
    language: String? = nil,
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil,
    onPartial: (@Sendable (String) -> Void)? = nil
) async throws -> Transcription {
    // A hotword model goes through `SpeechModel`, the type the GUI holds; every other id is the
    // marked block, the card's snippet, which the card generator compiles on its own.
    if SpeechModel.takesHotwords(id) {
        let model = try await SpeechModel(catalog: id, downloadProgress: downloadProgress)
        return try await model.transcribe(
            samples: AudioFile.pcm16kMono(url), hotwords: hotwords, language: language,
            onPartial: onPartial)
    }
    guard hotwords.isEmpty else { throw SpeechModelError.noHotwords(model: id) }
    // CARD-SNIPPET-BEGIN
    let transcriber = try await KitTranscriber(catalog: id, downloadProgress: downloadProgress)
    let samples = try AudioFile.pcm16kMono(url)  // any wav/m4a/mp3 → 16 kHz mono Float
    return try await transcriber.transcribe(samples: samples, language: language, onPartial: onPartial)
    // CARD-SNIPPET-END
}

/// A speech-to-text model loaded once and kept for every clip after it — what an app that
/// transcribes more than once holds. `KitTranscriber` drives every `asr` id in the catalog;
/// Fun-ASR-Nano is held as `KitFunASRModel` instead, because it alone takes a hotword list and
/// `KitTranscriber` has no parameter to pass one through.
enum SpeechModel: Sendable {
    case transcriber(KitTranscriber)
    case funASR(KitFunASRModel)

    /// Whether the model with this catalog id takes a hotword list.
    static func takesHotwords(_ id: String) -> Bool { id == "fun-asr-nano-2512" }

    /// Loads a model by its catalog id. First use downloads it (progress via `downloadProgress`),
    /// later runs load from the local cache.
    init(
        catalog id: String, downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        if Self.takesHotwords(id) {
            self = .funASR(try await KitFunASRModel(catalog: id, downloadProgress: downloadProgress))
        } else {
            self = .transcriber(
                try await KitTranscriber(catalog: id, downloadProgress: downloadProgress))
        }
    }

    /// Transcribe a 16 kHz mono waveform. `language` nil = auto-detect; `onPartial` streams the
    /// running transcript.
    func transcribe(
        samples: [Float], hotwords: [String] = [], language: String? = nil,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        switch self {
        case .funASR(let model):
            return try await model.transcribe(
                samples: samples, hotwords: hotwords, language: language, onPartial: onPartial)
        case .transcriber(let transcriber):
            guard hotwords.isEmpty else { throw SpeechModelError.noHotwords(model: transcriber.id) }
            return try await transcriber.transcribe(
                samples: samples, language: language, onPartial: onPartial)
        }
    }
}

enum SpeechModelError: LocalizedError {
    case noHotwords(model: String)

    var errorDescription: String? {
        switch self {
        case .noHotwords(let id):
            return "\(id) takes no hotword list — fun-asr-nano-2512 does."
        }
    }
}

/// "Tavenmoor, Anwen Thorsby" → ["Tavenmoor", "Anwen Thorsby"]: split at commas, trimmed, blanks
/// dropped — the hotword field and `--hotwords` read the same way.
func hotwordList(_ text: String) -> [String] {
    text.split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
}
