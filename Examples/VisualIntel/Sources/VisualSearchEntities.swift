// VisualSearchEntities.swift — the App Intents entities the system Visual Intelligence UI
// renders. An entity is the only thing the system sees: it has no idea a converted CLIP /
// RF-DETR model produced it. Each entity needs an `OpenIntent` (see VisualSearchIntents.swift)
// or the app will not surface in Visual Intelligence at all.

import AppIntents
import Foundation

/// An object RF-DETR found in the captured frame. `id` is the COCO class label, so taps route
/// to a per-class detail screen.
@available(iOS 27.0, macOS 27.0, *)
struct DetectedObjectEntity: AppEntity {
    var id: String
    var label: String
    /// Confidence in [0, 1].
    var score: Double
    /// Small JPEG crop of the detected region.
    var thumbnail: Data

    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Detected Object"

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(
            title: "\(label.capitalized)",
            subtitle: "\(Int((score * 100).rounded()))% confidence",
            image: thumbnail.isEmpty ? nil : .init(data: thumbnail))
    }

    static let defaultQuery = DetectedObjectQuery()
}

@available(iOS 27.0, macOS 27.0, *)
struct DetectedObjectQuery: EntityQuery {
    func entities(for identifiers: [String]) async throws -> [DetectedObjectEntity] {
        // Detections are transient (recomputed per capture), so there is nothing to rehydrate
        // from an id alone. Return a minimal stand-in so a late lookup still resolves to the
        // class rather than vanishing; the live entity carries the thumbnail and score.
        identifiers.map { DetectedObjectEntity(id: $0, label: $0, score: 0, thumbnail: Data()) }
    }
}

/// A photo from the user's library that CLIP judged visually similar to the captured frame.
/// `id` is the `PHAsset.localIdentifier`.
@available(iOS 27.0, macOS 27.0, *)
struct PhotoMatchEntity: AppEntity {
    var id: String
    /// Cosine similarity in [0, 1].
    var score: Double
    var thumbnail: Data?

    static let typeDisplayRepresentation: TypeDisplayRepresentation = "Matching Photo"

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(
            title: "Similar photo",
            subtitle: "\(Int((score * 100).rounded()))% similar",
            image: thumbnail.map { .init(data: $0) })
    }

    static let defaultQuery = PhotoMatchQuery()
}

@available(iOS 27.0, macOS 27.0, *)
struct PhotoMatchQuery: EntityQuery {
    func entities(for identifiers: [String]) async throws -> [PhotoMatchEntity] {
        // Rehydrate from the persisted CLIP index's cached thumbnails (synchronous,
        // PhotoKit-free — important when this runs in a background launch).
        let index = VisualIntelEngine.shared.index
        return identifiers.map {
            PhotoMatchEntity(id: $0, score: 0, thumbnail: index.thumbnailData(for: $0))
        }
    }
}

/// What `IntentValueQuery` returns to Visual Intelligence: a detected object or a matching photo.
/// The `@UnionValue` macro lets one query vend more than one entity type (WWDC26 pattern; iOS 27 /
/// macOS 27).
@available(iOS 27.0, macOS 27.0, *)
@UnionValue
enum VisualSearchResult {
    case detectedObject(DetectedObjectEntity)
    case photo(PhotoMatchEntity)
}
