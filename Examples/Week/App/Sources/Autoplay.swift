// Autoplay — presses the buttons hands-off, for a recording or a smoke run from a script:
//
//   Week.app/Contents/MacOS/Week -autoplay 1 -delay 3 -count 20 -seed 7 -log 1 \
//       [-trigger <file>] [-reminders 1] [-store 0] [-bundle <dir>]
//   Week -grantOnly 1        # ask for Calendar and Reminders access, write Documents/access.json, idle
//
// The calendar and the reminders list go into the device's local source only; `-syncedStore 1`
// (never passed here, only by hand) lets them go to the default source, which may be iCloud.
//
// The app reads its week and loads the model; once it is READY, autoplay waits `delay` seconds
// (after `trigger` exists, when given: a recorder creates it once the capture is rolling; a
// relative path is in Documents, where `devicectl device copy to` can put it) and presses Plan my
// week. With `-reminders 1` it also presses Add reminders two seconds after DONE. `-count` /
// `-seed` pick the generated week, `-store 0` keeps it in the app (no calendar read, no reminders
// written) and `-bundle` picks the model directory, with or without autoplay. With `-log 1` the
// screen's status line is appended to Documents/week-autoplay.log whenever it changes (checked
// four times a second), and every DONE writes Documents/week-result-<epoch>.json — rewritten
// when the reminders are added — which is how the iPhone's numbers are read back over
// `devicectl device copy from`, no UI in the loop. Nothing else changes: this presses the screen's
// own buttons.

import Foundation

@MainActor
final class Autoplay {
    let enabled: Bool
    let delay: Double
    let count: Int
    let seed: UInt64
    let trigger: String?
    let log: Bool
    let bundle: String?
    let reminders: Bool
    let store: Bool
    let syncedStore: Bool
    let grantOnly: Bool
    private var started = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        enabled = defaults.bool(forKey: "autoplay")
        delay = defaults.object(forKey: "delay") != nil ? max(0, defaults.double(forKey: "delay")) : 3
        let n = defaults.integer(forKey: "count")
        count = n > 0 ? n : 20
        seed = defaults.object(forKey: "seed") != nil ? UInt64(clamping: defaults.integer(forKey: "seed")) : 7
        trigger = defaults.string(forKey: "trigger").map {
            $0.hasPrefix("/") ? $0 : URL.documentsDirectory.appending(path: $0).path
        }
        log = defaults.bool(forKey: "log")
        bundle = defaults.string(forKey: "bundle")
        reminders = defaults.bool(forKey: "reminders")
        store = defaults.object(forKey: "store") != nil ? defaults.bool(forKey: "store") : true
        syncedStore = defaults.bool(forKey: "syncedStore")
        grantOnly = defaults.bool(forKey: "grantOnly")
    }

    /// Runs once per launch: starts the log mirror, then (with `-autoplay 1`) presses Plan my week
    /// and, with `-reminders 1`, Add reminders.
    func run(_ model: WeekModel) async {
        guard !started else { return }
        started = true
        if log {
            write("launch · \(model.count) events · seed \(model.seed) · store \(store ? 1 : 0) · syncedStore \(syncedStore ? 1 : 0)"
                + " · \(Device.model) · \(Device.os)")
            Task { await mirror(model) }
        }
        guard enabled else { return }
        while model.phase == .loading { try? await Task.sleep(for: .milliseconds(100)) }
        guard model.phase == .ready else { return }
        if let trigger {
            write("waiting for \(trigger)")
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        write("autoplay: Plan my week")
        model.planWeek()
        guard reminders else { return }
        while model.phase == .planning { try? await Task.sleep(for: .milliseconds(100)) }
        guard model.phase == .done else { return }
        try? await Task.sleep(for: .seconds(2))
        if model.canAddReminders {
            write("autoplay: Add \(model.plan.count) reminders")
            await model.addReminders()
        } else {
            write("autoplay: no reminders to add (\(model.weekSource))")
        }
    }

    /// Appends the status line whenever it changes, and a result file at every DONE (rewritten when
    /// the reminders are added).
    private func mirror(_ model: WeekModel) async {
        var last = ""
        var wasDone = false
        var resultURL: URL?
        var lastReminders: Int?
        while true {
            let line = model.statusLine
            if line != last {
                last = line
                write(line)
            }
            let isDone = model.phase == .done
            if isDone, !wasDone {
                resultURL = URL.documentsDirectory.appending(path: "week-result-\(Int(Date().timeIntervalSince1970)).json")
                lastReminders = model.remindersAdded
                writeResult(model, to: resultURL!)
            } else if isDone, model.remindersAdded != lastReminders, let resultURL {
                lastReminders = model.remindersAdded
                writeResult(model, to: resultURL)
            }
            wasDone = isDone
            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    private func writeResult(_ model: WeekModel, to url: URL) {
        guard let json = model.resultJSON() else { return }
        do {
            try Data((json.json(pretty: true) + "\n").utf8).write(to: url)
            write("result \(url.path)")
        } catch {
            write("result not written: \(error.localizedDescription)")
        }
    }

    /// One timestamped line to Documents/week-autoplay.log (only with `-log 1`) and to stdout.
    func write(_ line: String) {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let text = "\(formatter.string(from: Date())) \(line)\n"
        print("[week] \(line)")
        guard log else { return }
        let url = URL.documentsDirectory.appending(path: "week-autoplay.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
