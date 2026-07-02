import SwiftUI

@main
struct ChatDemoApp: App {
    var body: some Scene {
        WindowGroup {
            if FalconBench.isRequested {
                FalconBenchView()
            } else {
                ChatView()
            }
        }
    }
}
