// QuickStart.swift — the take-home core of this runner: image in → ×4 image out, one typed
// function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this function;
// the GUI drives the same `SuperResolver(catalog:)` on the photo you pick. Want on-device
// super-resolution in your own app? This file is the part you copy; the model card's 💻
// snippet is the marked block below.

import CoreAIKitVision
import CoreGraphics
import Foundation

/// Upscale one image file ×4 with any super-resolution model in the catalog
/// (`ModelCatalog.builtin.available(.superResolution)`). First use downloads the model
/// (progress via `downloadProgress`), later runs load from the local cache. Large inputs are
/// tiled and feather-blended internally; `maxInputSide` caps the input first.
func upscale(
    at imageURL: URL,
    model id: String = "adcsr-x4",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> CGImage {
    // CARD-SNIPPET-BEGIN
    let resolver = try await SuperResolver(catalog: id, downloadProgress: downloadProgress)
    let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
    let upscaled = try await resolver.upscale(image.cgImage)
    // CARD-SNIPPET-END
    return upscaled
}
