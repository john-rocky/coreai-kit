// QuickStart.swift — the take-home core of this runner: video clip in → ranked actions out,
// one typed function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly
// this function; the GUI drives the same `ActionRecognizer(catalog:)` over a rolling buffer
// of live camera frames. Want on-device action recognition in your own app? This file is the
// part you copy; the model card's 💻 snippet is the marked block below.

import CoreAIKitVision
import Foundation

/// Name the action in a video clip with any video model in the catalog
/// (`ModelCatalog.builtin.available(.video)`). First use downloads the model (progress via
/// `downloadProgress`), later runs load from the local cache. Live camera? Keep the last 16
/// `CameraFeed` frames and call `classify(frames:)` — see the GUI app.
func recognizeAction(
    in videoURL: URL,
    model id: String = "vjepa2-vitl-ssv2",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> [ActionRecognizer.Prediction] {
    // CARD-SNIPPET-BEGIN
    let recognizer = try await ActionRecognizer(catalog: id, downloadProgress: downloadProgress)
    let actions = try await recognizer.classify(videoAt: videoURL, topK: 3)
    // CARD-SNIPPET-END
    return actions
}
