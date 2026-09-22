import SwiftUI

@main
struct DecideApp: App {
    @State private var runtime = DecideRuntime()
    @State private var autoplay = Autoplay()
    @State private var tab: Autoplay.Screen = Autoplay().screen ?? .form

    var body: some Scene {
        WindowGroup {
            TabView(selection: $tab) {
                FormView()
                    .tabItem { Label("Autofill", systemImage: "rectangle.and.pencil.and.ellipsis") }
                    .tag(Autoplay.Screen.form)
                ChecklistView()
                    .tabItem { Label("Checklist", systemImage: "checklist") }
                    .tag(Autoplay.Screen.checklist)
                SorterView()
                    .tabItem { Label("Sorter", systemImage: "folder") }
                    .tag(Autoplay.Screen.sorter)
                SearchView()
                    .tabItem { Label("Search", systemImage: "magnifyingglass") }
                    .tag(Autoplay.Screen.search)
                SpeechGateView()
                    .tabItem { Label("Speech gate", systemImage: "waveform") }
                    .tag(Autoplay.Screen.speech)
            }
            .environment(runtime)
            .environment(autoplay)
            #if os(macOS)
                .frame(minWidth: 640, minHeight: 640)
            #endif
        }
        #if os(macOS)
        .defaultSize(width: 900, height: 900)
        .defaultPosition(.topTrailing)
        #endif
    }
}
