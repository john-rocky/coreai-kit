// QuickStart.swift — the take-home core of this runner: image in → depth map out, one typed
// function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this function;
// the GUI drives the same `DepthEstimator(catalog:)` per camera frame (the `CameraFeed`
// wiring is the app's ~10 lines). Want on-device depth in your own app? This file is the part
// you copy; the model card's 💻 snippet is the marked block below.

import CoreAIKitVision
import CoreGraphics
import Foundation

/// Estimate relative depth for one image file with any depth model in the catalog
/// (`ModelCatalog.builtin.available(.depth)`). First use downloads the model (progress via
/// `downloadProgress`), later runs load from the local cache. Live camera? Feed each
/// `CameraFeed` frame to `estimateDepth(for:)` — see the GUI app.
func depthMap(
    at imageURL: URL,
    model id: String = "depth-anything-3-small",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> CGImage {
    // CARD-SNIPPET-BEGIN
    let estimator = try await DepthEstimator(catalog: id, downloadProgress: downloadProgress)
    let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
    let depth = try await estimator.estimateDepth(for: image.cgImage)
    // CARD-SNIPPET-END
    guard let map = depth.cgImage() else {
        throw VisionError.bundleLayout("depth map could not be rendered")
    }
    return map
}
