// KitDetector.swift — one object-detection entry point for any `detection` catalog id. The
// DETR family (no NMS, dets/labels graph) and YOLOX (dense head, host-side NMS) have different
// graph contracts; the catalog id picks the right driver so a caller never knows which.

import CoreGraphics
import Foundation

/// Any detection model in the catalog (`ModelCatalog.builtin.available(.detection)`) behind
/// one `detect(in:)`.
///
/// ```swift
/// let detector = try await KitDetector(catalog: "rf-detr")
/// let boxes = try await detector.detect(in: image)
/// ```
public struct KitDetector: Sendable {
    enum Engine: Sendable {
        case detr(ObjectDetector)
        case yolox(YOLOXDetector)
    }

    private let engine: Engine
    /// The catalog id this detector was loaded from.
    public let catalogID: String

    /// Loads a detection model by its catalog id — the id shown on the model's card.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .detection)
        switch entry.id {
        case "yolox-s":
            self.engine = .yolox(
                try await YOLOXDetector(
                    model: entry.modelID!, store: store, computeUnits: computeUnits,
                    downloadProgress: downloadProgress))
        default:
            self.engine = .detr(
                try await ObjectDetector(
                    catalog: id, store: store, computeUnits: computeUnits,
                    downloadProgress: downloadProgress))
        }
        self.catalogID = entry.id
    }

    /// Detects objects in one image (preprocessing included). Results are sorted by
    /// descending confidence; `box` is normalized with origin at the top-left.
    public func detect(
        in image: CGImage, scoreThreshold: Float = 0.5, maxDetections: Int = 50
    ) async throws -> [Detection] {
        switch engine {
        case .detr(let d):
            return try await d.detect(
                in: image, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
        case .yolox(let d):
            return try await d.detect(
                in: image, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
        }
    }
}
