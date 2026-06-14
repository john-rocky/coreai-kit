// VisualSearchIntents.swift — the system-facing surface of the example.
//
// This is the whole Visual Intelligence integration, and it is entirely model-agnostic: the
// system hands us a `SemanticContentDescriptor` (a captured camera frame on iOS, a screenshot
// on iPad/Mac) and renders whatever `AppEntity`s we return. Our converted CLIP / RF-DETR models
// run inside `values(for:)`; the system never knows or cares what produced the results.
//
// Three pieces — all required for the app to appear in Visual Intelligence:
//   1. `IntentValueQuery`  — receives the pixels, returns entities.
//   2. `OpenIntent` per entity type — taps in the VI result list route here. WITHOUT an
//      OpenIntent the app does not surface in Visual Intelligence at all.
//   3. `@AppIntent(schema: .visualIntelligence.semanticContentSearch)` — the optional
//      "Continue in app" / "More results" action — lives in its own file
//      (ContinueInAppIntent.swift) so it can be removed independently if the beta macro
//      misbehaves, leaving the gate (pieces 1 + 2) intact.

import AppIntents
import CoreVideo
import Foundation
import VisualIntelligence

// MARK: - 1. The query Visual Intelligence calls with the captured pixels

@available(iOS 27.0, macOS 27.0, *)
struct VisualSearchValueQuery: IntentValueQuery {
    @Dependency var engine: VisualIntelEngine

    func values(for input: SemanticContentDescriptor) async throws -> [VisualSearchResult] {
        guard let pixelBuffer = input.pixelBuffer,
            let cgImage = VisualIntelEngine.cgImage(from: pixelBuffer)
        else { return [] }

        // Run the light, proven-safe surface (RF-DETR nano + CLIP) — the shipped demo.
        let analysis = try await engine.analyze(cgImage)

        // This tab is RF-DETR detections + CLIP photo matches only — fast, so the detection tab
        // never waits on a model.
        var results: [VisualSearchResult] = []

        // Unique id per detection (label + index) so multiple same-class objects (e.g. several
        // cups) each surface instead of collapsing on a shared id.
        results += analysis.detections.enumerated().map { index, d in
            .detectedObject(
                DetectedObjectEntity(
                    id: "\(d.label)#\(index)", label: d.label, score: Double(d.score),
                    thumbnail: d.thumbnailJPEG))
        }
        results += analysis.photos.map {
            .photo(
                PhotoMatchEntity(
                    id: $0.id, score: Double($0.score), thumbnail: $0.thumbnailJPEG))
        }
        return results
    }
}

// MARK: - 2. OpenIntents — required, or the app is invisible to Visual Intelligence

@available(iOS 27.0, macOS 27.0, *)
struct OpenDetectedObjectIntent: OpenIntent {
    static let title: LocalizedStringResource = "Open Detected Object"

    @Parameter(title: "Object")
    var target: DetectedObjectEntity

    @Dependency var router: AppRouter

    func perform() async throws -> some IntentResult {
        await router.open(detectedLabel: target.label.isEmpty ? target.id : target.label)
        return .result()
    }
}

@available(iOS 27.0, macOS 27.0, *)
struct OpenPhotoMatchIntent: OpenIntent {
    static let title: LocalizedStringResource = "Open Matching Photo"

    @Parameter(title: "Photo")
    var target: PhotoMatchEntity

    @Dependency var router: AppRouter

    func perform() async throws -> some IntentResult {
        await router.open(photoID: target.id)
        return .result()
    }
}
