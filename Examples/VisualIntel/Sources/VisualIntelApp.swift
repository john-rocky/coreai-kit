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
    }

    var presentedDetectionLabel: String?
    var presentedPhotoID: String?
    var results: Snapshot?
    var showResults = false

    func open(detectedLabel label: String) {
        presentedDetectionLabel = label
    }

    func open(photoID id: String) {
        presentedPhotoID = id
    }

    func showResults(detections: [DetectionResult], photos: [PhotoResult]) {
        results = Snapshot(detections: detections, photos: photos)
        showResults = true
    }
}

@main
struct VisualIntelApp: App {
    init() {
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
