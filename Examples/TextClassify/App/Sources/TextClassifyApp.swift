// TextClassify — your messages sorted on device: every message gets one of the categories you name,
// an urgency and a sentiment from GLiNER2.5-Decide on Core AI, and By category lists each category
// with its most urgent messages first. The inbox is pasted, imported from a file, or the sample
// inbox (../Sources/InboxCore), the same one `textclassify-cli --inbox` sorts headless.

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
        _model = State(initialValue: InboxModel(
            source: autoplay.source, sampleCount: autoplay.count, seed: autoplay.seed,
            categories: autoplay.categories, showing: autoplay.showing))
    }

    var body: some Scene {
        WindowGroup {
            InboxScreen(model: model)
                .task { await model.load(bundle: autoplay.bundle) }
                .task { model.begin() }
                .task { await autoplay.run(model) }
                .preferredColorScheme(.dark)
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
