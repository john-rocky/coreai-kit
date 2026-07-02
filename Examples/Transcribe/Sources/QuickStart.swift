// QuickStart.swift — the take-home core of this runner: speech in → text out, one typed
// function, no UI. Both the GUI app and the CLI call exactly this function (the view model is
// a display shell, `CLI/main.swift` an argument shell). Want transcription in your own app?
// This file is the part you copy; the model card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation

/// Transcribe an audio file with any speech-to-text model in the catalog
/// (`ModelCatalog.builtin.available(.asr)`: Whisper large-v3-turbo, Qwen3-ASR, Parakeet-TDT).
/// First use downloads the model (progress via `downloadProgress`), later runs load from the
/// local cache. `onPartial` streams the running transcript while the clip decodes.
func transcribe(
    audio url: URL,
    model id: String = "whisper-large-v3-turbo",
    language: String? = nil,
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil,
    onPartial: (@Sendable (String) -> Void)? = nil
) async throws -> Transcription {
    // CARD-SNIPPET-BEGIN
    let transcriber = try await KitTranscriber(catalog: id, downloadProgress: downloadProgress)
    let samples = try AudioFile.pcm16kMono(url)  // any wav/m4a/mp3 → 16 kHz mono Float
    return try await transcriber.transcribe(samples: samples, language: language, onPartial: onPartial)
    // CARD-SNIPPET-END
}
