import SwiftUI
import UIKit

// Scripted runs: launch arguments that do what a person does on the screen, so the app can be checked on a device
// without touching it (the screenshots are taken from the Mac). Without these arguments the app is the plain UI.
//
//   -model timestral|timesfm  -source sample|paste|csv  -sample <index>  -input <file in Documents>
//   -horizon <steps>  -band 80|90|98  -compute gpu|ane  -bundle <folder in Documents>  -repeat <forecasts>
//   -delay <seconds>  -gap <seconds>  -fixture <folder in Documents>  -out <name>  -log <file in Documents>
//
// -input fills the paste box (source paste) or stands in for the file picker (source csv) with the file's text.
// -delay keeps the opened screen without the input for that many seconds and -gap waits that many more before the
// first forecast, so a recording shows each step. Both count from the run's start on one clock (sleeps in a row
// drift); 0, the default, or a value that is not a number means no wait.
// -compute gives every Timestral graph that unit; without it each graph runs on its form's unit.
// -fixture replays a fixture pack instead (windows.json + contexts.f32.bin, as the model zoo's sample_ref.py writes
// them): every window through the Timestral forecaster the screen uses, the forecasts to <folder>/<out>/ (default
// swift) as forecast.f32.bin + manifest.json, the layout the zoo's sample_score.py compares with its Python host.
// Every forecast appends one JSON line to the log (the outcome, its milliseconds, the thermal state); the first line
// also has the free disk space, the last line is {"event":"done"}.
//
//   xcrun devicectl device process launch --device <udid> com.coreaikit.forecast -- \
//     -model timestral -source csv -input series.csv -horizon 48 -log run.jsonl

extension View {
    func scriptedRun(_ vm: ForecastViewModel) -> some View { modifier(ScriptedRun(vm: vm)) }
}

struct ScriptedRun: ViewModifier {
    @ObservedObject var vm: ForecastViewModel

    func body(content: Content) -> some View {
        content.task { await run() }
    }

    @MainActor private func run() async {
        let args = UserDefaults.standard  // `-key value` launch arguments land in the argument domain
        guard let model = args.string(forKey: "model") else { return }
        let clock = ContinuousClock()
        let start = clock.now  // -delay and -gap count from here
        UIApplication.shared.isIdleTimerDisabled = true
        let docs = URL.documentsDirectory
        let log = args.string(forKey: "log").map { docs.appending(path: $0) }
        func write(_ object: [String: Any]) {
            guard let log else { return }
            var line = object
            line["t"] = Date().timeIntervalSince1970
            line["thermal"] = Self.thermal()
            guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
            if !FileManager.default.fileExists(atPath: log.path) {
                try? FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
                FileManager.default.createFile(atPath: log.path, contents: nil)
            }
            if let h = try? FileHandle(forWritingTo: log) {
                h.seekToEndOfFile()
                h.write(data + Data("\n".utf8))
                try? h.close()
            }
        }

        vm.model = model == "timesfm" ? .timesfm : .timestral
        vm.source = SeriesSource.allCases.first { $0.rawValue.lowercased().hasPrefix(args.string(forKey: "source") ?? "sample") }
            ?? .sample
        vm.pick = args.integer(forKey: "sample")
        if let h = args.string(forKey: "horizon") { vm.horizonText = h }
        vm.band = Band.allCases.first { $0.rawValue == (args.string(forKey: "band") ?? "80") + "%" } ?? .p80
        vm.computeOverride = args.string(forKey: "compute").flatMap { $0 == "auto" ? nil : $0 }
        if let bundle = args.string(forKey: "bundle") { vm.localBundle = docs.appending(path: bundle) }
        let total = max(1, args.integer(forKey: "repeat"))
        var index = 0
        vm.onResult = { outcome in
            index += 1
            var o = outcome
            o["event"] = "forecast"
            o["index"] = index
            write(o)
        }
        let free = (try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
        write(["event": "start", "model": vm.model.rawValue, "source": vm.source.rawValue, "horizon": vm.horizonText,
               "repeat": total, "device": Self.machine(), "os": ProcessInfo.processInfo.operatingSystemVersionString,
               "compute": vm.computeOverride ?? "auto", "free_disk_mb": free.map { Int($0 / 1_000_000) as Any } ?? NSNull()])
        let delay = Self.seconds(args, "delay")
        let gap = Self.seconds(args, "gap")
        if delay > 0 { try? await Task.sleep(until: start + .seconds(delay), tolerance: .milliseconds(10), clock: clock) }
        if let name = args.string(forKey: "input") {
            let text = (try? String(contentsOf: docs.appending(path: name), encoding: .utf8)) ?? ""
            if vm.source == .paste {
                vm.pasted = text
            } else {
                vm.csvText = text
                vm.csvName = (name as NSString).lastPathComponent
            }
        }
        if let pack = args.string(forKey: "fixture") {
            let line = await replay(docs.appending(path: pack), out: args.string(forKey: "out") ?? "swift")
            write(line)
            write(["event": "done", "forecasts": 0])
            return
        }
        if gap > 0 { try? await Task.sleep(until: start + .seconds(delay + gap), tolerance: .milliseconds(10), clock: clock) }
        for _ in 0..<total { await vm.forecast() }
        write(["event": "done", "forecasts": index])
    }

    /// -fixture: every window of the pack through the screen's Timestral forecaster; -> the log line.
    @MainActor private func replay(_ pack: URL, out name: String) async -> [String: Any] {
        struct Window: Decodable {
            let id: String
            let horizon: Int
            let context_length: Int
            let context_offset: Int
        }
        struct Pack: Decodable {
            let windows: [Window]
            let contexts_file: String
        }
        do {
            let p = try JSONDecoder().decode(Pack.self, from: Data(contentsOf: pack.appending(path: "windows.json")))
            let contexts = try Data(contentsOf: pack.appending(path: p.contexts_file))
                .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let t0 = Date()
            let forecaster = try await vm.timestralForecaster()
            let loadMs = Date().timeIntervalSince(t0) * 1000
            var floats: [Float] = []
            var entries: [[String: Any]] = []
            for w in p.windows {
                guard w.context_offset >= 0, w.context_offset + w.context_length <= contexts.count else {
                    entries.append(["id": w.id, "error": "context outside \(p.contexts_file)"])
                    continue
                }
                do {
                    let series = Array(contexts[w.context_offset..<(w.context_offset + w.context_length)])
                    let q = try await forecaster.forecast(series, horizon: w.horizon)
                    let flat = q.quantiles.flatMap { $0 }
                    let d = q.details
                    entries.append([
                        "id": w.id, "horizon": w.horizon, "forecast_offset": floats.count, "forecast_floats": flat.count,
                        "ms": d.totalMilliseconds, "graph_ms": d.graphMilliseconds,
                        "bucket": [d.bucket.contextPatches, d.bucket.horizonPatches], "rows": d.bucket.rows,
                        "compute": ForecastViewModel.unitName(d), "thermal": Self.thermal(),
                    ])
                    floats += flat
                } catch {
                    entries.append(["id": w.id, "error": error.localizedDescription])
                }
            }
            let dir = pack.appending(path: name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try floats.withUnsafeBufferPointer { try Data(buffer: $0).write(to: dir.appending(path: "forecast.f32.bin")) }
            let errors = entries.filter { $0["error"] != nil }.count
            let manifest: [String: Any] = [
                "load_ms": loadMs, "compute": vm.computeOverride ?? "auto", "device": Self.machine(), "windows": entries,
            ]
            try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                .write(to: dir.appending(path: "manifest.json"))
            return ["event": "fixture", "windows": entries.count, "errors": errors, "load_ms": loadMs, "out": name]
        } catch {
            return ["event": "fixture", "error": error.localizedDescription]
        }
    }

    /// -delay / -gap in seconds: 0 when absent, not a number, negative or infinite.
    static func seconds(_ args: UserDefaults, _ key: String) -> Double {
        guard let value = args.string(forKey: key).flatMap({ Double($0) }), value.isFinite else { return 0 }
        return max(0, value)
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func machine() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }
}
