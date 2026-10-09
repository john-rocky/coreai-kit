// Autoplay — presses the buttons hands-off, for a recording or a smoke run from a script:
//
//   TextClassify.app/Contents/MacOS/TextClassify -autoplay 1 -source paste|file|sample [-text <file>] \
//       [-categories "work, family, bills"] [-view inbox|category] -count 100 -seed 7 -delay 3 -log 1 \
//       [-out <dir>] [-trigger <file>] [-bundle <dir>]
//
// `-source` is where the inbox comes from (the sample inbox when it is not given), `-categories` the
// categories field's text and `-view` the view the app opens on. The app loads the model; once it is
// READY, autoplay hands the inbox over the way a person would: with `-source paste -text <file>` it
// opens the Paste sheet holding the file's text (as if pasted), with `-source file -text <file>` it
// imports the file through Import file's own reading. Then it waits for `trigger` when given (a
// recorder creates it once the capture is rolling), then `delay` seconds, and presses Sort: the
// sheet's Sort for a paste, else Sort inbox. Refusals are the screen's own (an empty paste, more than
// 2,000 messages, a file that is not UTF-8, categories that cannot be read) and are logged.
// `-count` / `-seed` pick the sample inbox and `-bundle` the model directory, with or without autoplay.
//
// With `-log 1` the screen's status line is appended to inbox-autoplay.log whenever it changes
// (checked four times a second), and every DONE writes inbox-result-<epoch>.json. Both go to `-out`,
// else Documents: on an iPhone that is what `devicectl device copy from` reads back (and where
// `devicectl device copy to` puts a `-text` file); on a Mac, Documents asks for folder access and may
// sync to iCloud, so a script passes `-out`. A relative `-text` or `-trigger` is in that directory.
// Nothing else changes: this presses the screen's own buttons.

import Foundation

@MainActor
final class Autoplay {
    let enabled: Bool
    let delay: Double
    let source: InboxModel.Source
    let text: String?
    let categories: String?
    let showing: InboxModel.Showing
    let count: Int
    let seed: UInt64
    let trigger: String?
    let log: Bool
    let out: URL
    let bundle: String?
    private var started = false

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        enabled = defaults.bool(forKey: "autoplay")
        delay = defaults.object(forKey: "delay") != nil ? max(0, defaults.double(forKey: "delay")) : 3
        source = defaults.string(forKey: "source").flatMap(InboxModel.Source.init(rawValue:)) ?? .sample
        categories = defaults.string(forKey: "categories")
        showing = defaults.string(forKey: "view").flatMap(InboxModel.Showing.init(rawValue:)) ?? .inbox
        let n = defaults.integer(forKey: "count")
        count = n > 0 ? n : 100
        seed = defaults.object(forKey: "seed") != nil ? UInt64(clamping: defaults.integer(forKey: "seed")) : 7
        let out = defaults.string(forKey: "out").map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true)
        } ?? URL.documentsDirectory
        self.out = out
        func inOut(_ path: String) -> String {
            let p = (path as NSString).expandingTildeInPath
            return p.hasPrefix("/") ? p : out.appending(path: p).path
        }
        text = defaults.string(forKey: "text").map(inOut)
        trigger = defaults.string(forKey: "trigger").map(inOut)
        log = defaults.bool(forKey: "log")
        bundle = defaults.string(forKey: "bundle")
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    }

    /// Runs once per launch: starts the log mirror, then (with `-autoplay 1`) hands the inbox over
    /// and presses Sort.
    func run(_ model: InboxModel) async {
        guard !started else { return }
        started = true
        if log {
            write("launch · source \(source.rawValue) · text \(text.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "-")"
                + " · categories \(model.categories.labels.count) · view \(showing.rawValue) · count \(count) · seed \(seed)"
                + " · \(Device.model) · \(Device.os)")
            Task { await mirror(model) }
        }
        guard enabled else { return }
        while model.phase == .loading { try? await Task.sleep(for: .milliseconds(100)) }
        guard model.phase == .ready else { return }
        if let text {
            let url = URL(fileURLWithPath: text)
            switch source {
            case .paste:
                guard let data = try? Data(contentsOf: url), let pasted = InboxInput.text(utf8: data) else {
                    write("autoplay: \(url.lastPathComponent) cannot be read as UTF-8 text, nothing pasted")
                    return
                }
                model.pasteText = pasted
                model.showingPaste = true
                write("autoplay: Paste sheet with \(url.lastPathComponent) · \(model.pasteStatus)")
            case .file:
                model.importFile(url)
                write("autoplay: Import file \(url.lastPathComponent) · "
                    + (model.messages.isEmpty ? model.note ?? "" : InboxModel.messagesLabel(model.count)))
            case .sample:
                break
            }
        }
        if let trigger {
            write("waiting for \(trigger)")
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        if model.showingPaste {
            write("autoplay: Sort (Paste sheet)")
            guard model.submitPaste() else {
                write("autoplay: the sheet refused it · \(model.pasteRefusal ?? "")")
                return
            }
            if model.phase != .sorting { write("autoplay: not sorted · \(model.sortRefusal ?? model.statusLine)") }
        } else {
            guard model.canSort else {
                write("autoplay: Sort refused · \(model.sortRefusal ?? model.statusLine)")
                return
            }
            write("autoplay: Sort")
            model.sort()
        }
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
        let url = out.appending(path: "inbox-result-\(Int(Date().timeIntervalSince1970)).json")
        do {
            try Data((json.json(pretty: true) + "\n").utf8).write(to: url)
            write("result \(url.path)")
        } catch {
            write("result not written: \(error.localizedDescription)")
        }
    }

    /// One timestamped line to inbox-autoplay.log in the output directory (only with `-log 1`) and
    /// to stdout.
    func write(_ line: String) {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let text = "\(formatter.string(from: Date())) \(line)\n"
        print("[inbox] \(line)")
        guard log else { return }
        let url = out.appending(path: "inbox-autoplay.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
