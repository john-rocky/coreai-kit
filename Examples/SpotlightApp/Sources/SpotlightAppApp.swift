// SpotlightAppApp.swift — entry point. Normally shows the chat UI; with SPOTLIGHT_SELFTEST=1 it
// runs the exact same RagEngine path once, headless, and exits with a pass/fail code. The
// self-test is the Mac gate (no UI automation needed) — it proves the 2-tool RAG grounds an
// answer through the very session code the UI drives.

import SwiftUI

struct SpotlightAppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

@main
struct AppEntry {
    static func main() {
        if ProcessInfo.processInfo.environment["SPOTLIGHT_INDEXPROBE"] != nil {
            SelfTest.probeBlocking()  // does not return — index-settle diagnostic, no model
        }
        if ProcessInfo.processInfo.environment["SPOTLIGHT_SELFTEST"] != nil {
            SelfTest.runBlocking()  // does not return
        }
        SpotlightAppApp.main()
    }
}
