// TextClassify — a support inbox sorted on device: one tap, and every message gets an intent, an
// urgency and a sentiment from GLiNER2.5-Decide on Core AI. The inbox is synthetic
// (../Sources/InboxCore), the same one `textclassify-cli --inbox` sorts headless.

import SwiftUI
#if os(iOS)
import UIKit
#endif

@main
struct TextClassifyApp: App {
    @State private var model: InboxModel
    private let autoplay: Autoplay

    init() {
        let autoplay = Autoplay()
        self.autoplay = autoplay
        _model = State(initialValue: InboxModel(count: autoplay.count, seed: autoplay.seed))
    }

    var body: some Scene {
        WindowGroup {
            InboxScreen(model: model)
                .task { await model.load(bundle: autoplay.bundle) }
                .task { await autoplay.run(model) }
                #if os(iOS)
                .statusBarHidden(true)
                .persistentSystemOverlays(.hidden)
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }  // the screen stays on
                #else
                // a phone-shaped window
                .frame(minWidth: 360, idealWidth: 420, minHeight: 720, idealHeight: 910)
                #endif
        }
        #if os(macOS)
        .defaultSize(width: 420, height: 910)
        .windowResizability(.contentMinSize)
        .windowStyle(.hiddenTitleBar)
        #endif
    }
}
