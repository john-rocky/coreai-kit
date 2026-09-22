import SwiftUI

@main
struct DecideApp: App {
    @State private var runtime = DecideRuntime()
    /// Shared with the menu bar on the Mac, so a watched copy's verdict shows without the window.
    @State private var clipboard = ClipboardModel()
    @State private var autoplay = Autoplay()
    @State private var tab: Autoplay.Screen = Autoplay().screen ?? .speech

    var body: some Scene {
        WindowGroup {
            TabView(selection: $tab) {
                SpeechGateView()
                    .tabItem { Label("Speech gate", systemImage: "waveform") }
                    .tag(Autoplay.Screen.speech)
                ClipboardView()
                    .tabItem { Label("Clipboard", systemImage: "doc.on.clipboard") }
                    .tag(Autoplay.Screen.clipboard)
                SearchView()
                    .tabItem { Label("Search", systemImage: "magnifyingglass") }
                    .tag(Autoplay.Screen.search)
                ChecklistView()
                    .tabItem { Label("Checklist", systemImage: "checklist") }
                    .tag(Autoplay.Screen.checklist)
                SorterView()
                    .tabItem { Label("Sorter", systemImage: "folder") }
                    .tag(Autoplay.Screen.sorter)
            }
            .environment(runtime)
            .environment(clipboard)
            .environment(autoplay)
            #if os(macOS)
                .frame(minWidth: 640, minHeight: 640)
            #endif
        }
        #if os(macOS)
        .defaultSize(width: 900, height: 900)
        .defaultPosition(.topTrailing)
        #endif
        #if os(macOS)
        MenuBarExtra {
            if clipboard.history.isEmpty {
                Text(clipboard.watching ? "Copy something to see its verdict here." : "Turn on Watch in the Clipboard tab.")
            } else {
                ForEach(clipboard.history.prefix(6)) { verdict in
                    Text("\(verdict.line) — \(verdict.text.prefix(32))… (\(ms(verdict.milliseconds)))")
                }
            }
            Divider()
            Toggle("Watch clipboard", isOn: Binding(
                get: { clipboard.watching },
                set: { clipboard.setWatching($0, runtime: runtime) }))
                .disabled(!runtime.isReady)
        } label: {
            // Text only: a Label collapses to its icon in the menu bar, and the verdict is the point.
            Text(clipboard.menuBarLabel)
        }
        .environment(runtime)
        .environment(clipboard)
        #endif
    }
}
