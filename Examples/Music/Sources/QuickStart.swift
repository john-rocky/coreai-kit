// QuickStart.swift — the take-home core of this runner: prompt in → music out, one typed
// function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this function;
// the GUI drives the same `KitMusician(catalog:)` and plays the result. Want on-device music
// generation in your own app? This file is the part you copy; the model card's 💻 snippet is
// the marked block below.

import CoreAIKit
import Foundation

/// Generate audio for a prompt with any text-to-music model in the catalog
/// (`ModelCatalog.builtin.available(.music)`). First use downloads the model (progress via
/// `downloadProgress`), later runs load from the local cache.
func compose(
    _ prompt: String,
    seconds: Float = 11,
    model id: String = "stable-audio-open-small",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> SpokenAudio {
    // CARD-SNIPPET-BEGIN
    let musician = try await KitMusician(catalog: id, downloadProgress: downloadProgress)
    let audio = try await musician.generate(prompt, seconds: seconds)
    // CARD-SNIPPET-END
    return audio
}
