// QuickStart.swift — the take-home core of Look: a photo + a question in → the answer out, one
// typed function, no UI. The screen (LookModel) makes the same calls with the model held across
// questions and the answer streamed; this file is the part you copy into your own app.

import CoreAIKit
import Foundation
import FoundationModels

/// Ask the on-device vision-language model about a photo. First use downloads the decoder +
/// vision bundles (progress via `downloadProgress`), later runs load from the local cache.
/// Nothing leaves the device: the photo and the question go to the model in this process.
func ask(
    _ question: String,
    aboutImageAt imageURL: URL,
    model id: String = "minicpm-v-4.7-1b",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    let vlm = try await KitVisionModel(catalog: id, downloadProgress: downloadProgress)
    let session = LanguageModelSession(model: vlm)
    let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
    let reply = try await session.respond(to: Prompt {
        Attachment(image.cgImage, orientation: image.orientation)  // the image, then the question
        question
    })
    return reply.content
}
