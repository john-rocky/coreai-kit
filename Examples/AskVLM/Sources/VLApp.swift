// VLApp.swift — app entry + navigation router for the on-device VLM Visual Intelligence app.
//
// App Intents may launch this app in the background to answer a Visual Intelligence query; `init()`
// still runs first, so the engine + router dependency registrations are in place before the query
// or an `OpenIntent`'s `perform()` runs.

import AppIntents
import SwiftUI

@MainActor
@Observable
final class VLRouter {
    static let shared = VLRouter()

    /// The VLM answer routed from an `OpenVLAnswerIntent` tap in the Visual Intelligence list.
    var presentedAnswer: String?

    func open(answer: String) {
        presentedAnswer = answer
    }
}

@main
struct VLApp: App {
    init() {
        // S=1 VL bundles need this set before any Core AI engine is created (known trap). `init()`
        // runs first even in a background Visual Intelligence launch.
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)

        let engine = VLEngine.shared
        let router = VLRouter.shared
        AppDependencyManager.shared.add(dependency: engine)
        AppDependencyManager.shared.add(dependency: router)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(VLRouter.shared)
        }
    }
}
