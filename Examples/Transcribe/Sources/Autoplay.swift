// Autoplay — drives the screen hands-off, for a recording or a smoke run from a script:
//
//   Transcribe.app/Contents/MacOS/Transcribe -autoplay 1 -model fun-asr-nano-2512 \
//     -clip ~/clips/memo.wav -hotwords "Tavenmoor, Anwen Thorsby" -play 1 -delay 2 -gap 4 -log 1
//
// picks the model and waits for it to load, then — once `-trigger <path>` exists (a recorder
// creates it when the capture is rolling) and `delay` seconds have passed — sets the clip, plays
// it (`-play 1`) to its end, transcribes it with the hotword field empty, waits `gap` seconds,
// puts `-hotwords` in the field and transcribes again (without `-hotwords`, the first run only).
// A relative `-clip` or `-trigger` is a file in Documents, where `devicectl device copy to` puts
// it. With `-log 1` it appends READY / PLAY / RESULT / DONE lines (ERROR when a step fails) to
// Documents/transcribe-autoplay.log and writes the runs to
// Documents/transcribe-autoplay-<epoch>.json — how iPhone numbers are read back over
// `devicectl device copy from`. Nothing else changes: the screen is the same code with the same
// buttons; this only presses them.

import Foundation

@MainActor
final class Autoplay {
    let enabled: Bool
    let model: String?
    let clip: String?
    let hotwords: String
    let play: Bool
    let delay: Double
    let gap: Double
    let trigger: String?
    let log: Bool
    private var fired = false

    init() {
        let defaults = UserDefaults.standard  // -key value launch arguments land here
        enabled = defaults.bool(forKey: "autoplay")
        model = defaults.string(forKey: "model")
        clip = defaults.string(forKey: "clip").map(Self.inDocuments)
        hotwords = defaults.string(forKey: "hotwords") ?? ""
        play = defaults.bool(forKey: "play")
        delay = defaults.object(forKey: "delay") != nil ? defaults.double(forKey: "delay") : 2
        gap = defaults.object(forKey: "gap") != nil ? defaults.double(forKey: "gap") : 4
        trigger = defaults.string(forKey: "trigger").map(Self.inDocuments)
        log = defaults.bool(forKey: "log")
    }

    /// Runs the sequence once per launch, with `-autoplay 1` only.
    func run(_ screen: TranscribeModel) async {
        guard enabled, !fired else { return }
        fired = true
        let epoch = Int(Date().timeIntervalSince1970)
        if let model { screen.selectedID = model }
        guard let id = screen.selectedID else { return finish("ERROR no model: pass -model <catalog id>") }
        let clock = ContinuousClock()
        let start = clock.now
        await screen.load()
        let loadMs = Int((Self.seconds(clock.now - start) * 1000).rounded())
        guard screen.isReady else { return finish("ERROR load: \(screen.statusLabel)") }
        write("READY model=\(id) load_ms=\(loadMs)")

        if let trigger {
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        try? await Task.sleep(for: .seconds(delay))
        guard let clip, FileManager.default.fileExists(atPath: clip) else {
            return finish("ERROR no clip at \(clip ?? "-"): pass -clip <path, or a file in Documents>")
        }
        screen.loadFile(URL(fileURLWithPath: clip))
        var played: Double?
        if play {
            guard let seconds = screen.togglePlay() else {
                return finish("ERROR play: \(screen.statusLabel)")
            }
            write("PLAY seconds=\(Self.decimals(seconds))")
            await screen.playbackFinished()
            played = seconds
        }

        var runs: [TranscribeModel.Run] = []
        screen.hotwordsText = ""
        await screen.transcribe()
        guard let plain = result(screen, "plain") else { return }
        runs.append(plain)
        if !hotwords.isEmpty {
            if screen.takesHotwords {
                try? await Task.sleep(for: .seconds(gap))
                screen.hotwordsText = hotwords
                await screen.transcribe()
                guard let hinted = result(screen, "hotwords") else { return }
                runs.append(hinted)
            } else {
                write("ERROR \(id) takes no hotword list: ran without -hotwords")
            }
        }
        record(epoch: epoch, model: id, loadMs: loadMs, clip: clip, played: played, runs: runs)
        write("DONE")
    }

    /// The run Transcribe just finished, logged as one RESULT line; nil (logged, DONE) on failure.
    private func result(_ screen: TranscribeModel, _ tag: String) -> TranscribeModel.Run? {
        guard let run = screen.lastRun else {
            finish("ERROR run=\(tag): \(screen.statusLabel)")
            return nil
        }
        write(
            "RESULT run=\(tag) audio_s=\(Self.decimals(run.audioSeconds))"
                + " wall_s=\(Self.decimals(run.seconds)) text=\(run.text)")
        return run
    }

    private func finish(_ line: String) {
        write(line)
        write("DONE")
    }

    /// The runs as JSON, for the record: Documents/transcribe-autoplay-<epoch>.json.
    private func record(
        epoch: Int, model: String, loadMs: Int, clip: String, played: Double?,
        runs: [TranscribeModel.Run]
    ) {
        guard log else { return }
        var object: [String: Any] = [
            "model": model, "load_ms": loadMs, "clip": clip, "device": Self.device,
            "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "runs": runs.map { run -> [String: Any] in
                [
                    "hotwords": run.hotwords, "audio_s": run.audioSeconds, "wall_s": run.seconds,
                    "text": run.text,
                ]
            },
        ]
        if let played { object["played_s"] = played }
        let url = URL.documentsDirectory.appending(path: "transcribe-autoplay-\(epoch).json")
        if let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        {
            try? data.write(to: url)
        }
    }

    /// Appends one timestamped line to Documents/transcribe-autoplay.log (only with `-log 1`).
    private func write(_ line: String) {
        guard log else { return }
        let url = URL.documentsDirectory.appending(path: "transcribe-autoplay.log")
        let text = "\(ISO8601DateFormatter().string(from: Date())) \(line)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private static func inDocuments(_ path: String) -> String {
        path.hasPrefix("/") ? path : URL.documentsDirectory.appending(path: path).path
    }

    private static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }

    private static func decimals(_ value: Double) -> String { String(format: "%.2f", value) }

    /// The hardware model the numbers came from ("iPhone18,1", "Mac16,5").
    private static var device: String {
        #if os(macOS)
        let key = "hw.model"
        #else
        let key = "hw.machine"
        #endif
        var size = 0
        sysctlbyname(key, nil, &size, nil, 0)
        var bytes = [UInt8](repeating: 0, count: size)
        sysctlbyname(key, &bytes, &size, nil, 0)
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
}
