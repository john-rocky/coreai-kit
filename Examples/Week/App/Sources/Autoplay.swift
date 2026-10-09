// Autoplay — presses the buttons hands-off, for a recording or a smoke run from a script:
//
//   Week.app/Contents/MacOS/Week -autoplay 1 -source calendar|paste|sample [-events <.txt or .json>] \
//       -delay 3 -log 1 [-out <dir>] [-trigger <file>] [-reminders 1] [-bundle <dir>]
//   Week -grantOnly 1        # ask for Calendar and Reminders access, write access.json, idle
//
// `-source` is the week the app opens on (your calendar when it is not given). The app loads the
// model and reads that week; once it is READY, autoplay waits for `trigger` when given (a recorder
// creates it once the capture is rolling; a relative path is in the output directory), then `delay`
// seconds, and presses Plan my week. With `-source paste -events <file>` it first opens the Paste
// sheet on the file through the sheet's own Import (a `.json` file in `week-cli dump --out`'s form
// becomes one line per event), and the button it presses after the wait is the sheet's Plan: the
// reading a person's paste gets, refusals included. With `-reminders 1` it also presses Add
// reminders two seconds after DONE. `-bundle` picks the model directory, with or without autoplay.
//
// With `-log 1` the screen's status line is appended to week-autoplay.log whenever it changes
// (checked four times a second), and every DONE writes week-result-<epoch>.json, rewritten when the
// reminders are added. Both go to `-out`, else Documents: on an iPhone that is what `devicectl device
// copy from` reads back; on a Mac, Documents asks for folder access and may sync to iCloud, so a
// script passes `-out`. Nothing else changes: this presses the screen's own buttons.

import Foundation

@MainActor
final class Autoplay {
    let enabled: Bool
    let delay: Double
    let source: WeekModel.Source
    let events: String?
    let trigger: String?
    let log: Bool
    let out: URL
    let bundle: String?
    let reminders: Bool
    let grantOnly: Bool
    private var started = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        enabled = defaults.bool(forKey: "autoplay")
        delay = defaults.object(forKey: "delay") != nil ? max(0, defaults.double(forKey: "delay")) : 3
        source = defaults.string(forKey: "source").flatMap(WeekModel.Source.init(rawValue:)) ?? .calendar
        let out = defaults.string(forKey: "out").map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true)
        } ?? URL.documentsDirectory
        self.out = out
        events = defaults.string(forKey: "events").map { ($0 as NSString).expandingTildeInPath }
        trigger = defaults.string(forKey: "trigger").map {
            $0.hasPrefix("/") ? $0 : out.appending(path: $0).path
        }
        log = defaults.bool(forKey: "log")
        bundle = defaults.string(forKey: "bundle")
        reminders = defaults.bool(forKey: "reminders")
        grantOnly = defaults.bool(forKey: "grantOnly")
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    }

    /// Runs once per launch: starts the log mirror, then (with `-autoplay 1`) presses Plan my week,
    /// or the Paste sheet's Plan, and with `-reminders 1` Add reminders.
    func run(_ model: WeekModel) async {
        guard !started else { return }
        started = true
        if log {
            write("launch · source \(source.rawValue) · events \(events.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "-")"
                + " · \(Device.model) · \(Device.os)")
            Task { await mirror(model) }
        }
        guard enabled else { return }
        while model.phase == .loading || model.readingWeek { try? await Task.sleep(for: .milliseconds(100)) }
        guard model.phase == .ready else { return }
        if source == .paste, let events {
            await model.select(.paste)
            model.importFile(URL(fileURLWithPath: events))
            write("autoplay: Paste sheet with \(URL(fileURLWithPath: events).lastPathComponent)")
        }
        if let trigger {
            write("waiting for \(trigger)")
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        if model.showingPaste {
            write("autoplay: Plan (Paste sheet)")
            guard model.submitPaste() else {
                write("autoplay: the sheet refused it · \(model.pasteRefusal ?? "")")
                return
            }
        } else {
            guard !model.events.isEmpty else {
                write("autoplay: nothing to plan · \(model.note ?? "")")
                return
            }
            write("autoplay: Plan my week")
            model.planWeek()
        }
        guard reminders else { return }
        while model.phase == .planning { try? await Task.sleep(for: .milliseconds(100)) }
        guard model.phase == .done else { return }
        try? await Task.sleep(for: .seconds(2))
        if model.canAddReminders {
            write("autoplay: Add \(model.upcoming.count) reminders")
            await model.addReminders()
        } else {
            write("autoplay: no reminders to add (\(model.source.weekName), \(model.upcoming.count) upcoming)")
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
                resultURL = out.appending(path: "week-result-\(Int(Date().timeIntervalSince1970)).json")
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

    /// One timestamped line to week-autoplay.log in the output directory (only with `-log 1`) and
    /// to stdout.
    func write(_ line: String) {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let text = "\(formatter.string(from: Date())) \(line)\n"
        print("[week] \(line)")
        guard log else { return }
        let url = out.appending(path: "week-autoplay.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
