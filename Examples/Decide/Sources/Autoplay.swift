// Autoplay — drives one screen hands-off, for a recording or a smoke run from a script:
//
//   Decide.app/Contents/MacOS/Decide -autoplay checklist -model minicpm5-2b -delay 1.5
//
// opens that tab, loads the model, waits `delay` seconds after Ready, then presses the
// screen's own sample button. With `-trigger <path>` it also waits, after Ready, until that
// file exists — a recorder creates it once the capture is rolling. With `-log 1` it mirrors
// the screen's status line into Documents/autoplay-<screen>.log as it changes (how the iPhone
// numbers are read back over `devicectl device copy from`, no UI in the loop). Nothing else
// changes: the screens are the same code with the same buttons; this only presses them.
//
// Room check loads its own image model, not the shared runtime (`-model` does not apply):
//
//   Decide.app/Contents/MacOS/Decide -autoplay room -rooms <dir> -grid 448 -delay 3 -log 1 [-detail 1]
//
// presses Check all once the rooms and the model are ready; `-detail <n>` then opens room n
// three seconds after DONE, as a tap on it would. `-rooms` / `-grid` are the screen's own
// options (RoomCheckModel), with or without autoplay.

import Foundation
import Observation

@MainActor
@Observable
final class Autoplay {
    enum Screen: String, CaseIterable {
        case speech, search, checklist, sorter, form, drive, columns, `guard`, context, typing, room
    }

    let screen: Screen?
    let model: String?
    let delay: Double
    let trigger: String?
    let log: Bool
    /// `-feed 1`: a screen that watches the pasteboard feeds itself the sample copies (the
    /// iPhone has no `pbcopy`; on the Mac the recorder feeds them from outside).
    let feed: Bool
    /// `-detail <n>`: Room check opens room n (1-based) after DONE.
    let detail: Int?
    private var fired = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        screen = defaults.string(forKey: "autoplay").flatMap(Screen.init(rawValue:))
        model = defaults.string(forKey: "model")
        let d = defaults.double(forKey: "delay")
        delay = d > 0 ? d : 1.5
        // A relative trigger path lives in Documents, where `devicectl device copy to` can put it.
        trigger = defaults.string(forKey: "trigger").map {
            $0.hasPrefix("/") ? $0 : URL.documentsDirectory.appending(path: $0).path
        }
        log = defaults.bool(forKey: "log")
        feed = defaults.bool(forKey: "feed")
        let n = defaults.integer(forKey: "detail")
        detail = n > 0 ? n : nil
    }

    /// Runs `action` once, on the autoplayed screen only, after the model is ready. `status`
    /// is the screen's status line; with `-log 1` every change of it is appended to the log for
    /// 90 seconds after the action, then a `DONE` line.
    func run(
        _ target: Screen, runtime: DecideRuntime, status: @escaping @MainActor () -> String = { "" },
        action: @MainActor () async -> Void
    ) async {
        await run(target, status: status, action: action) {
            if let model { runtime.selectedID = model }
            let loadStart = Date()
            await runtime.load()
            let line = "load \(runtime.status.label) in \(Int(Date().timeIntervalSince(loadStart) * 1000)) ms · \(runtime.loadedID ?? "-")"
            return (line, runtime.isReady)
        }
    }

    /// The same run for a screen that loads its own model: `load` readies the screen and returns
    /// its line for the log and whether it is ready. With `finished`, the log mirrors the status
    /// until it is true, however long the run takes, instead of for 90 seconds.
    func run(
        _ target: Screen, status: @escaping @MainActor () -> String = { "" },
        finished: (@MainActor () -> Bool)? = nil, action: @MainActor () async -> Void,
        load: @MainActor () async -> (line: String, ready: Bool)
    ) async {
        guard screen == target, !fired else { return }
        fired = true
        let loaded = await load()
        write(target, loaded.line)
        guard loaded.ready else { return }
        if let trigger {
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        await action()
        guard log else { return }
        var last = ""
        let end = Date().addingTimeInterval(90)
        while true {
            let now = status()
            if now != last {
                last = now
                write(target, now)
            }
            if let finished { if finished() { break } } else if Date() >= end { break }
            try? await Task.sleep(for: .milliseconds(250))
        }
        write(target, "DONE")
    }

    /// Appends one timestamped line to Documents/autoplay-<screen>.log (only with `-log 1`).
    private func write(_ target: Screen, _ line: String) {
        guard log else { return }
        let url = URL.documentsDirectory.appending(path: "autoplay-\(target.rawValue).log")
        let stamp = ISO8601DateFormatter().string(from: Date())
        let text = "\(stamp) \(line)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
