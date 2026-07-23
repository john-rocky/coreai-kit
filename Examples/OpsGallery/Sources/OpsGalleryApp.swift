// OpsGalleryApp.swift — every task op as a card: pick an input, run the one-line call,
// see the result. The whole gallery renders from `CoreAI.Op.allCases`; the only other
// wiring is the process-wide download hook installed here.

import CoreAIOps
import SwiftUI

@main
struct OpsGalleryApp: App {
    init() {
        CoreAI.onDownload { progress in
            Task { @MainActor in DownloadHub.shared.latest = progress }
        }
    }

    var body: some Scene {
        WindowGroup {
            GalleryView()
        }
    }
}
