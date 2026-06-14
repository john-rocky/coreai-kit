// VisualIntelApp.swift — app entry + navigation router.
//
// The router is the bridge from the system-facing intents back into the UI: an `OpenIntent`
// (tap a Visual Intelligence result) or the "Continue in app" intent runs `perform()`, which
// drives this `@Observable` router, which the views present. Both the engine and the router are
// registered as App Intents dependencies in `init()` so `@Dependency` resolves them — crucially
// also during the background launch the system uses to run the Visual Intelligence query.

import AppIntents
import SwiftUI

@MainActor
@Observable
final class AppRouter {
    static let shared = AppRouter()

    /// Full results to present after "Continue in app".
    struct Snapshot {
        var detections: [DetectionResult]
        var photos: [PhotoResult]
        /// The on-device VLM's answer, computed in the FOREGROUND here (full memory budget) — the
        /// robust path that sidesteps the background-launch budget.
        var vlmAnswer: String?
        var vlmMilliseconds: Double?
    }

    var presentedDetectionLabel: String?
    var presentedPhotoID: String?
    /// The VLM answer routed from an `OpenVisualAnswerIntent` tap in the Visual Intelligence list.
    var presentedAnswer: String?
    var results: Snapshot?
    var showResults = false

    func open(detectedLabel label: String) {
        presentedDetectionLabel = label
    }

    func open(photoID id: String) {
        presentedPhotoID = id
    }

    func open(answer: String) {
        presentedAnswer = answer
    }

    func showResults(
        detections: [DetectionResult], photos: [PhotoResult],
        vlmAnswer: String? = nil, vlmMilliseconds: Double? = nil
    ) {
        results = Snapshot(
            detections: detections, photos: photos,
            vlmAnswer: vlmAnswer, vlmMilliseconds: vlmMilliseconds)
        showResults = true
    }
}

@main
struct VisualIntelApp: App {
    init() {
        // S=1 single-buffer bundles need this set before any Core AI engine is created (known
        // trap); the VL decode bundle is one. `init()` runs first even in a background Visual
        // Intelligence launch, so this lands before any model loads.
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)

        // Make our model engine and router resolvable via `@Dependency` in the intents. App
        // Intents may launch the app in the background to answer a Visual Intelligence query;
        // `init()` still runs first, so the registrations are in place before `perform()`.
        // Resolve the singletons here on the main actor, then register the (Sendable) values —
        // referencing the main-actor static inside the registration closure would not be Sendable.
        let engine = VisualIntelEngine.shared
        let router = AppRouter.shared
        AppDependencyManager.shared.add(dependency: engine)
        AppDependencyManager.shared.add(dependency: router)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(AppRouter.shared)
        }
    }
}
