// ContinueInAppIntent.swift — the optional "Continue in app" / "More results" surface.
//
// This conforms to Apple's `semanticContentSearch` schema via `@AppIntent(schema:)`. When the
// user taps "more results" in Visual Intelligence, the app opens and shows the full result set
// (run through the same engine). It is deliberately ISOLATED in its own file: the
// `@AppIntent(schema: .visualIntelligence.semanticContentSearch)` macro is the least-stable part
// against the 27 beta SDK, so if it fails to build you can delete just this file and the core
// Visual Intelligence integration (IntentValueQuery + entities + OpenIntents) still stands.

import AppIntents
import Foundation
import VisualIntelligence

@available(iOS 26.0, macOS 27.0, *)
@AppIntent(schema: .visualIntelligence.semanticContentSearch)
struct ContinueVisualSearchInAppIntent {
    static let title: LocalizedStringResource = "Search in VisualIntel"
    static let openAppWhenRun: Bool = true

    var semanticContent: SemanticContentDescriptor

    @Dependency var engine: VisualIntelEngine
    @Dependency var router: AppRouter

    func perform() async throws -> some IntentResult {
        guard let pixelBuffer = semanticContent.pixelBuffer,
            let cgImage = VisualIntelEngine.cgImage(from: pixelBuffer)
        else {
            await router.showResults(detections: [], photos: [])
            return .result()
        }
        let analysis = try await engine.analyze(cgImage, topPhotos: 12, minimumPhotoScore: 0)
        await router.showResults(detections: analysis.detections, photos: analysis.photos)
        return .result()
    }
}
