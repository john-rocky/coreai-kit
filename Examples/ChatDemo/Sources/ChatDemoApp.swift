import SwiftUI

@main
struct ChatDemoApp: App {
    var body: some Scene {
        WindowGroup {
            if EntrySmokeView.isRequested {
                EntrySmokeView()
            } else if FalconBench.isRequested {
                FalconBenchView()
            } else {
                ChatView()
            }
        }
    }
}
