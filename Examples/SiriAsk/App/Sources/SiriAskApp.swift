// SiriAskApp — entry point. The app is "Gemma" (CFBundleDisplayName). On appear, ContentView kicks
// off model load + warm-up (ModelStatus.prepare), so a subsequent "Hey Siri, Ask Gemma" reuses the
// warm session instead of cold-loading the ~4.4 GB model inside Siri's intent budget. Registering
// AppShortcuts is automatic via the AppShortcutsProvider (GemmaShortcuts).

import SwiftUI

@main
struct SiriAskApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                #if os(macOS)
                .frame(minWidth: 460, minHeight: 600)
                #endif
        }
    }
}
