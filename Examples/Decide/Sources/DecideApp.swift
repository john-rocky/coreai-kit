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
                DriveView()
                    .tabItem { Label("Drive", systemImage: "car") }
                    .tag(Autoplay.Screen.drive)
                ColumnsView()
                    .tabItem { Label("Columns", systemImage: "tablecells") }
                    .tag(Autoplay.Screen.columns)
                GuardView()
                    .tabItem { Label("Guard", systemImage: "hand.raised") }
                    .tag(Autoplay.Screen.guard)
                ContextView()
                    .tabItem { Label("Context", systemImage: "arrow.down.right.and.arrow.up.left") }
                    .tag(Autoplay.Screen.context)
                TypingView()
                    .tabItem { Label("Typing", systemImage: "keyboard") }
                    .tag(Autoplay.Screen.typing)
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
