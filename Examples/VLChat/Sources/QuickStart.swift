// QuickStart.swift — the take-home core of this runner: image + prompt in → answer out, one
// typed function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this
// function; the GUI drives the same `KitVisionModel(catalog:)` gesture, held across turns for
// its transcript. Want an on-device VLM in your own app? This file is the part you copy; the
// model card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation
import FoundationModels

/// Ask any vision-language model in the catalog about an image
/// (`ModelCatalog.builtin.available(.vlm)`: Qwen3-VL 2B/4B/8B). First use downloads the
/// decoder + vision bundles (progress via `downloadProgress`), later runs load from the local
/// cache. Multi-turn about the same image? Hold the `LanguageModelSession` and call
/// `respond(to:)` per turn — it keeps the conversation history.
func ask(
    _ prompt: String,
    aboutImageAt imageURL: URL,
    model id: String = "qwen3-vl-2b",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    // CARD-SNIPPET-BEGIN
    let vlm = try await KitVisionModel(catalog: id, downloadProgress: downloadProgress)
    let session = LanguageModelSession(model: vlm)
    let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
    let reply = try await session.respond(to: Prompt {
        prompt
        Attachment(image.cgImage, orientation: image.orientation)
    })
    // CARD-SNIPPET-END
    return reply.content
}
