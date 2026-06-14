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

        MemoryProbe.mark("vi-query-enter")

        // Always run the light, proven-safe surface (RF-DETR nano + CLIP). This is the shipped demo
        // and the graceful fallback: if the VLM below is jetsam-killed in the background launch,
        // these results still answer — the Visual Intelligence surface never goes empty.
        let analysis = try await engine.analyze(cgImage)
        MemoryProbe.mark("vi-after-rfdetr-clip")

        var results: [VisualSearchResult] = []

        // EXPERIMENTAL (G1 measurement; default OFF, toggled by the in-app switch). Also run the
        // ~2 GB VLM here, INSIDE the background Visual Intelligence launch, to measure whether it
        // survives the (undocumented) launch memory budget. `caption(instrumented:)` logs
        // footprint + headroom around the load and inference, so a jetsam is pinned by the last
        // breadcrumb in the unified log. `try?` so a throw/kill degrades to RF-DETR/CLIP, never a
        // crash. The VLM must already be downloaded (no bandwidth in the background) — the in-app
        // "Ask the VLM" button is the foreground step that caches it.
        if VLMVISettings.runInVisualIntelligence {
            if let vlm = try? await engine.caption(cgImage, instrumented: true),
                !vlm.text.isEmpty
            {
                MemoryProbe.log.log(
                    "vi-vlm SURVIVED latency=\(vlm.milliseconds, format: .fixed(precision: 0))ms answer=\(vlm.text, privacy: .public)"
                )
                results.append(
                    .answer(
                        VisualAnswerEntity(
                            id: "vlm-answer", answer: vlm.text,
                            thumbnail: VisualIntelEngine.previewJPEG(cgImage))))
            } else {
                MemoryProbe.log.log(
                    "vi-vlm NO-ANSWER (threw or process killed) — Visual Intelligence falls back to RF-DETR/CLIP"
                )
            }
        }

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

@available(iOS 27.0, macOS 27.0, *)
struct OpenVisualAnswerIntent: OpenIntent {
    static let title: LocalizedStringResource = "Open On-device Answer"

    @Parameter(title: "Answer")
    var target: VisualAnswerEntity

    @Dependency var router: AppRouter

    func perform() async throws -> some IntentResult {
        await router.open(answer: target.answer)
        return .result()
    }
}
