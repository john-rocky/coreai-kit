// Autoplay — presses Sort hands-off, for a recording or a smoke run from a script:
//
//   TextClassify.app/Contents/MacOS/TextClassify -autoplay 1 -delay 3 -count 1000 -seed 7 -log 1 \
//       [-trigger <file>] [-bundle <dir>]
//
// The app loads the model; once it is READY, autoplay waits `delay` seconds (after `trigger`
// exists, when given: a recorder creates it once the capture is rolling; a relative path is in
// Documents, where `devicectl device copy to` can put it) and presses Sort. `-count` / `-seed`
// pick the inbox and `-bundle` the model directory, with or without autoplay. With `-log 1` the
// screen's status line is appended to Documents/inbox-autoplay.log whenever it changes (checked
// four times a second), and every DONE writes Documents/inbox-result-<epoch>.json — how the
// iPhone's numbers are read back over `devicectl device copy from`, no UI in the loop. Nothing
// else changes: this presses the screen's own button.

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
    private var started = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        enabled = defaults.bool(forKey: "autoplay")
        delay = defaults.object(forKey: "delay") != nil ? max(0, defaults.double(forKey: "delay")) : 3
        let n = defaults.integer(forKey: "count")
        count = n > 0 ? n : 1000
        seed = defaults.object(forKey: "seed") != nil ? UInt64(clamping: defaults.integer(forKey: "seed")) : 7
        trigger = defaults.string(forKey: "trigger").map {
            $0.hasPrefix("/") ? $0 : URL.documentsDirectory.appending(path: $0).path
        }
        log = defaults.bool(forKey: "log")
        bundle = defaults.string(forKey: "bundle")
    }

    /// Runs once per launch: starts the log mirror, then (with `-autoplay 1`) presses Sort.
    func run(_ model: InboxModel) async {
        guard !started else { return }
        started = true
        if log {
            write("launch · \(model.count) messages · seed \(model.seed) · \(Device.model) · \(Device.os)")
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
        write("autoplay: Sort")
        model.sort()
    }

    /// Appends the status line whenever it changes, and a result file at every DONE.
    private func mirror(_ model: InboxModel) async {
        var last = ""
        var wasDone = false
        while true {
            let line = model.statusLine
            if line != last {
                last = line
                write(line)
            }
            let isDone = model.phase == .done
            if isDone, !wasDone { writeResult(model) }
            wasDone = isDone
            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    private func writeResult(_ model: InboxModel) {
        guard let json = model.resultJSON() else { return }
        let url = URL.documentsDirectory.appending(path: "inbox-result-\(Int(Date().timeIntervalSince1970)).json")
        do {
            try Data((json.json(pretty: true) + "\n").utf8).write(to: url)
            write("result \(url.path)")
        } catch {
            write("result not written: \(error.localizedDescription)")
        }
    }

    /// One timestamped line to Documents/inbox-autoplay.log (only with `-log 1`) and to stdout.
    private func write(_ line: String) {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let text = "\(formatter.string(from: Date())) \(line)\n"
        print("[inbox] \(line)")
        guard log else { return }
        let url = URL.documentsDirectory.appending(path: "inbox-autoplay.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
