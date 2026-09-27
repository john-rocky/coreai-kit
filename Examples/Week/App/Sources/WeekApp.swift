// Week — a calendar week planned on device: one tap, and every event gets "what does it need
// before it?" from decider-0.8b on Core AI, then the events that need something become reminders.
// The week is synthetic (../Sources/WeekCore), written into the app's own Demo week calendar, the
// same one `week-cli run` plans headless.

import SwiftUI
#if os(iOS)
import UIKit
#endif

@main
struct WeekApp: App {
    @State private var model: WeekModel
    @State private var grant = GrantModel()
    private let autoplay: Autoplay

    init() {
        let autoplay = Autoplay()
        self.autoplay = autoplay
        _model = State(initialValue: WeekModel(
            count: autoplay.count, seed: autoplay.seed, store: autoplay.store, synced: autoplay.syncedStore))
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if autoplay.grantOnly {
                    GrantScreen(model: grant)
                        .task { await grant.run(synced: autoplay.syncedStore, log: autoplay.write) }
                } else {
                    WeekScreen(model: model)
                        .task { await model.load(bundle: autoplay.bundle) }
                        .task { await autoplay.run(model) }
                }
            }
            #if os(iOS)
            .statusBarHidden(true)
            .persistentSystemOverlays(.hidden)
            .onAppear { UIApplication.shared.isIdleTimerDisabled = true }  // the screen stays on
            #else
            // a phone-shaped window
            .frame(minWidth: 360, idealWidth: 420, minHeight: 720, idealHeight: 910)
            #endif
        }
        #if os(macOS)
        .defaultSize(width: 420, height: 910)
        .windowResizability(.contentMinSize)
        .windowStyle(.hiddenTitleBar)
        #endif
    }
}

/// `-grantOnly 1`: asks for Calendar then Reminders access, shows the answer (the UI test reads it)
/// and writes Documents/access.json (`devicectl device copy from` reads it), with the source the
/// calendar would go to; loads no model and writes no calendar.
@MainActor
@Observable
final class GrantModel {
    private(set) var line = "access: asking"

    func run(synced: Bool, log: @MainActor (String) -> Void) async {
        let store = CalendarStore(synced: synced)
        let granted = await store.requestAccess()
        let events = CalendarStore.status(.event), reminders = CalendarStore.status(.reminder)
        line = "events=\(events) reminders=\(reminders)"
        let source = events == "fullAccess" ? store.plannedCalendarSource : "none (calendar access \(events); in-app sample week)"
        log("grantOnly: \(line) · calendar_source \(source)")
        let json = JSONValue.object([
            "events": .string(events),
            "reminders": .string(reminders),
            "granted": .object(["events": .bool(granted.events), "reminders": .bool(granted.reminders)]),
            "calendar_source": .string(source),
            "synced_store": .bool(synced),
            "device": .string(Device.model),
            "os": .string(Device.os),
            "timestamp": .string(ISO8601DateFormatter().string(from: Date())),
        ])
        let url = URL.documentsDirectory.appending(path: "access.json")
        do {
            try Data((json.json(pretty: true) + "\n").utf8).write(to: url)
            log("access \(url.path)")
        } catch {
            log("access.json not written: \(error.localizedDescription)")
        }
    }
}

struct GrantScreen: View {
    let model: GrantModel

    var body: some View {
        VStack(spacing: 12) {
            Text("Calendar and Reminders access")
                .font(.headline)
                .foregroundStyle(.white)
            Text(model.line)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(WeekStyle.latency)
                .accessibilityIdentifier("access-status")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(WeekStyle.background.ignoresSafeArea())
    }
}
