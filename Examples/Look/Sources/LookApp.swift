// LookApp — two screens over one loaded model: Ask (a photo and a question → an answer,
// LookView.swift) and Watch (rules in words → reactions to what the camera sees, WatchView.swift).
// The switch between them sits above the screen's band; `-take 1` hides it, so the band starts
// right under the status bar for a recording. `-watch 1` opens on Watch.
//
// The band: the screen's width × 16/9 (402 × 714 pt on an iPhone 18 Pro). Both screens lay their
// rows out inside it, and both records keep its pixels as `take_rect_px` — what a recording crops.

import SwiftUI

@main
struct LookApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

struct RootView: View {
    enum Screen: String, CaseIterable, Identifiable {
        case ask = "Ask"
        case watch = "Watch"

        var id: String { rawValue }
    }

    /// The Ask | Watch switch, in points.
    static let switchHeight: CGFloat = 32
    static let switchGap: CGFloat = 8

    @State private var look = LookModel()
    @State private var watch = WatchModel()
    @State private var autoplay = Autoplay()
    @State private var watchAutoplay = WatchAutoplay()
    @State private var screen: Screen = UserDefaults.standard.bool(forKey: "watch") ? .watch : .ask
    @Environment(\.scenePhase) private var scenePhase
    private let take = UserDefaults.standard.bool(forKey: "take")

    /// Ask and Watch use the same model: no switching while either is using it.
    private var busy: Bool { look.phase == .thinking || watch.phase == .watching }

    var body: some View {
        GeometryReader { geo in
            let above = take ? 0 : Self.switchHeight + Self.switchGap
            let band = min(geo.size.height - above, (geo.size.width * 16 / 9).rounded(.down))
            VStack(spacing: 0) {
                if !take {
                    Picker("Screen", selection: $screen) {
                        ForEach(Screen.allCases) { item in Text(item.rawValue).tag(item) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .disabled(busy)
                    .frame(height: Self.switchHeight)
                    .padding(.horizontal, 16)
                    .padding(.bottom, Self.switchGap)
                    .accessibilityIdentifier("screen.switch")
                }
                Group {
                    switch screen {
                    case .ask: LookView(model: look, band: band)
                    case .watch: WatchView(model: watch, band: band, take: take)
                    }
                }
                .frame(width: geo.size.width, height: band)
                .onGeometryChange(for: CGRect.self) { $0.frame(in: .global) } action: { rect in
                    look.bandRect = rect
                    watch.bandRect = rect
                }
                Spacer(minLength: 0)
            }
        }
        .background(Palette.background.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onChange(of: scenePhase, initial: true) { _, phase in
            autoplay.scene(phase)
            watch.scene(phase)
        }
        .onChange(of: screen) { _, shown in watch.setVisible(shown == .watch) }
        .task { await look.load() }
        .task { await autoplay.run(look) }
        .task { await watch.attach(look, visible: screen == .watch) }
        .task { await watchAutoplay.run(watch) }
        #if os(macOS)
        .frame(minWidth: 420, minHeight: 800)
        #endif
    }
}
