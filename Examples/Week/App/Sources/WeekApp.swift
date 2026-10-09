// Week — your week planned on device: every event gets "what does it need before it?" from
// decider-0.8b on Core AI, and the events that need something are listed first. The week is read
// from your calendar (never written), pasted as lines, or the sample week (../Sources/WeekCore), the
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
        _model = State(initialValue: WeekModel(source: autoplay.source))
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if autoplay.grantOnly {
                    GrantScreen(model: grant)
                        .task { await grant.run(out: autoplay.out, log: autoplay.write) }
                } else {
                    WeekScreen(model: model)
                        .task { await model.load(bundle: autoplay.bundle) }
                        .task { await model.begin() }
                        .task { await autoplay.run(model) }
                }
            }
            .preferredColorScheme(.dark)
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
/// and writes access.json to the output directory (`devicectl device copy from` reads Documents);
/// loads no model, reads no event, writes nothing else.
@MainActor
@Observable
final class GrantModel {
    private(set) var line = "access: asking"

    func run(out: URL, log: @MainActor (String) -> Void) async {
        let store = CalendarStore()
        let events = await store.requestCalendarAccess()
        let reminders = await store.requestRemindersAccess()
        let eventsStatus = CalendarStore.status(.event), remindersStatus = CalendarStore.status(.reminder)
        line = "events=\(eventsStatus) reminders=\(remindersStatus)"
        log("grantOnly: \(line)")
        let json = JSONValue.object([
            "events": .string(eventsStatus),
            "reminders": .string(remindersStatus),
            "granted": .object(["events": .bool(events), "reminders": .bool(reminders)]),
            "device": .string(Device.model),
            "os": .string(Device.os),
            "timestamp": .string(ISO8601DateFormatter().string(from: Date())),
        ])
        let url = out.appending(path: "access.json")
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
