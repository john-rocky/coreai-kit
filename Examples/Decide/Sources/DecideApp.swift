import SwiftUI

@main
struct DecideApp: App {
    @State private var runtime = DecideRuntime()

    var body: some Scene {
        WindowGroup {
            TabView {
                SpeechGateView()
                    .tabItem { Label("Speech gate", systemImage: "waveform") }
                ClipboardView()
                    .tabItem { Label("Clipboard", systemImage: "doc.on.clipboard") }
                SearchView()
                    .tabItem { Label("Search", systemImage: "magnifyingglass") }
            }
            .environment(runtime)
            #if os(macOS)
                .frame(minWidth: 640, minHeight: 640)
            #endif
        }
    }
}
