// QuickStart.swift — the take-home core of this runner: image in → detections out, one typed
// function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this function;
// the GUI drives the same detector per camera frame on the zero-copy capture path (the
// `CameraFeed` wiring is the app's chrome). Want on-device detection in your own app? This
// file is the part you copy; the model card's 💻 snippet is the marked block below.

import CoreAIKitVision
import Foundation

/// Detect objects in one image file with any detection model in the catalog
/// (`ModelCatalog.builtin.available(.detection)`). First use downloads the model (progress
/// via `downloadProgress`), later runs load from the local cache. Real time? Feed each
/// camera frame to `detect(in:)` — see the GUI app's pixel-buffer fast path.
func detect(
    at imageURL: URL,
    model id: String = "rf-detr",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> [Detection] {
    // CARD-SNIPPET-BEGIN
    let detector = try await ObjectDetector(catalog: id, downloadProgress: downloadProgress)
    let image = try ImageFile.load(imageURL)  // any image file → CGImage + EXIF orientation
    let detections = try await detector.detect(in: image.cgImage)
    // CARD-SNIPPET-END
    return detections
}
