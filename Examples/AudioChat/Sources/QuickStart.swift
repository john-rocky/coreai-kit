// QuickStart.swift — the take-home core of this runner: audio + question in → answer out, one
// typed function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this
// function; the GUI drives the same `KitAudioModel(catalog:)` behind a FoundationModels
// `LanguageModelSession`. Want on-device audio understanding in your own app? This file is
// the part you copy; the model card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation
import FoundationModels

/// Ask any audio-understanding model in the catalog about a clip
/// (`ModelCatalog.builtin.available(.audio)`). First use downloads the decoder + encoder
/// (progress via `downloadProgress`), later runs load from the local cache. Live mic?
/// `MicRecorder` (kit API) captures 16 kHz mono `[Float]` — attach that instead.
func askAboutAudio(
    _ question: String,
    audioAt audioURL: URL,
    model id: String = "qwen2.5-omni-3b-audio",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    // CARD-SNIPPET-BEGIN
    let audio = try await KitAudioModel(catalog: id, downloadProgress: downloadProgress)
    try await audio.attach(samples: AudioFile.pcm16kMono(audioURL))  // clip → encoder buffer
    let session = LanguageModelSession(model: audio)
    let reply = try await session.respond(to: question)
    // CARD-SNIPPET-END
    return reply.content
}
