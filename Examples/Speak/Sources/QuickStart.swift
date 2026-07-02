// QuickStart.swift — the take-home core of this runner: text in → speech out, one typed
// function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this function;
// the GUI drives the same `KitSpeaker(catalog:)` and plays the samples. Want on-device
// text-to-speech in your own app? This file is the part you copy; the model card's 💻 snippet
// is the marked block below.

import CoreAIKit
import Foundation

/// Speak one utterance with any text-to-speech model in the catalog
/// (`ModelCatalog.builtin.available(.tts)`). First use downloads the voice (progress via
/// `downloadProgress`), later runs load from the local cache. Live playback? Use
/// `synthesizeStreaming(_:onChunk:)` — chunks arrive as they decode.
func say(
    _ text: String,
    model id: String = "voxcpm-0.5b",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> SpokenAudio {
    // CARD-SNIPPET-BEGIN
    let speaker = try await KitSpeaker(catalog: id, downloadProgress: downloadProgress)
    let audio = try await speaker.synthesize(text)
    // CARD-SNIPPET-END
    return audio
}
