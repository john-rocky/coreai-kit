// KitDetector.swift — one object-detection entry point for any `detection` catalog id. The
// DETR family (no NMS, dets/labels graph) and YOLOX (dense head, host-side NMS) have different
// graph contracts; the catalog id picks the right driver so a caller never knows which.

import CoreGraphics
import CoreVideo
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

    /// Square side the model consumes. Deliver capture buffers near this size — both
    /// drivers rescale, so a 4K frame is throughput spent on a resize.
    public var inputSize: Int {
        switch engine {
        case .detr(let d): d.inputSize
        case .yolox(let d): d.inputSize
        }
    }

    // MARK: - Real-time path

    /// A frame preprocessed for whichever driver is behind this detector. The two graph
    /// contracts differ (DETR takes square-resized RGB and needs no NMS; YOLOX takes
    /// letterboxed BGR and carries the geometry to undo it), which is why a live app that
    /// wants both otherwise ends up writing this enum itself — `Examples/DetectCamera` did.
    public struct PreparedFrame: Sendable {
        enum Storage: Sendable {
            case detr(ObjectDetector.PreparedInput)
            case yolox(YOLOXDetector.PreparedInput)
        }
        let storage: Storage
    }

    /// CPU half of the real-time path: scale and channel-split a 32BGRA capture buffer,
    /// with no CGImage round trip. Thread-safe, so it can run on frame N+1 while
    /// `detect(_:)` is still on the GPU with frame N.
    public func prepare(_ pixelBuffer: CVPixelBuffer) throws -> PreparedFrame {
        switch engine {
        case .detr(let d): PreparedFrame(storage: .detr(try d.prepare(pixelBuffer)))
        case .yolox(let d): PreparedFrame(storage: .yolox(try d.prepare(pixelBuffer)))
        }
    }

    /// GPU half of the real-time path.
    public func detect(
        _ frame: PreparedFrame, scoreThreshold: Float = 0.5, maxDetections: Int = 50
    ) async throws -> [Detection] {
        switch frame.storage {
        case .detr(let input):
            guard case .detr(let d) = engine else { throw Self.frameFromAnotherDetector }
            return try await d.detect(
                input, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
        case .yolox(let input):
            guard case .yolox(let d) = engine else { throw Self.frameFromAnotherDetector }
            return try await d.detect(
                input, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
        }
    }

    /// Only reachable by handing a frame prepared by one detector to a detector of the
    /// other family — the two graph contracts are not interchangeable.
    private static let frameFromAnotherDetector = VisionError.bundleLayout(
        "prepared frame belongs to a different detector family; prepare it on the "
            + "detector you are about to call")

    /// Both halves, for a caller that is not pipelining.
    public func detect(
        in pixelBuffer: CVPixelBuffer, scoreThreshold: Float = 0.5, maxDetections: Int = 50
    ) async throws -> [Detection] {
        try await detect(
            prepare(pixelBuffer), scoreThreshold: scoreThreshold,
            maxDetections: maxDetections)
    }
}
