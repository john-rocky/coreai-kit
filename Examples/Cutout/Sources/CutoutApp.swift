// CutoutApp — one screen (CutoutView) over one loaded model (CutoutModel), and the hands-off driver (Autoplay).
//
// The band: the screen's width × 16/9 (402 × 714 pt on an iPhone 18 Pro); the screen lays its rows out inside it.

import SwiftUI

@main
struct CutoutApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

struct RootView: View {
    @State private var model = CutoutModel()
    @State private var autoplay = Autoplay()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        GeometryReader { geo in
            let band = min(geo.size.height, (geo.size.width * 16 / 9).rounded(.down))
            VStack(spacing: 0) {
                CutoutView(model: model, band: band)
                    .frame(width: geo.size.width, height: band)
                Spacer(minLength: 0)
            }
        }
        .background(Palette.background.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onChange(of: scenePhase, initial: true) { _, phase in autoplay.scene(phase) }
        .task { await model.load() }
        .task { await autoplay.run(model) }
        #if os(macOS)
        .frame(minWidth: 420, minHeight: 760)
        #endif
    }
}
