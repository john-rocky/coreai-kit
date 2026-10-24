// Autoplay — drives the screen hands-off, for a recording or a check from a script, and keeps a record of every request:
//
//   xcrun devicectl device process launch --device <udid> com.coreaikit.cutout -- \
//     -autoplay 1 -example lighthouse -save 1 -frames 1 -log 1
//
// `-autoplay 1` waits for READY (the model loads when the app starts), then — once `-trigger <file>` exists and `delay`
// seconds have passed — taps an example photo (`-example lighthouse|chess|airplane`, or opens `-image <file>`), puts
// `-words "…"` in the box (the example's own words without it), picks `-task boxes` when asked, taps Cut out and waits
// for the answer; `-view photo` leaves the boxes on the screen instead of the cut-out; `-save 1` taps Save and waits
// for the outcome. Each press calls the function its button calls, so the run is the screen's own path.
//
// `-log 1` appends START / READY / PICK / WORDS / RUN / DONE / SAVE lines (ERROR when a step fails) to
// Documents/cutout-autoplay.log and writes each request to Documents/cutout-run-<epoch>.json: the step times, the answer,
// the boxes, the memory footprint and the thermal state. `-frames 1` writes the screen as PNGs to
// Documents/frames/<epoch>/ at READY, PICK, the answer and SAVE. `-quit 1` exits after the record.
//
// Checks that use the same code without the buttons (Engine/Fixture.swift, `WeVisOneDecoder.tailBench`):
//   -fixture <rows.json in Documents>   every row through the request code; outputs in Documents/fixture/<form>/<id>/
//   -tail 1                             the prompt's tail fed both ways (64-pad prefill vs S=1), -reps 8
//   -bench 3                            the example request once cold and 3 times warm, through Cut out
//   -bbench 8 -bform <graph B>          the loaded graph B against another (K instances per call) on the example's own
//                                       graph A outputs, alternating 8 rounds (CutoutPipeline.benchGraphB)
//   -tbench 8 -tform <tower>            the loaded tower against another (the fp16w32 tower), alternating 8 calls
//   -abench 8 -aform <graph A> [-aunit ane]  the loaded graph A against another (on the Neural Engine with -aunit ane)
// The environment variables CUTOUT_TOWER / CUTOUT_GRAPH_B load another tower / graph B for the whole launch (ModelFolder).
// `-photosauth denied` makes Save behave as if Photos access were off (the screen's sentence for that case; the record
// says it was injected).

import CoreAI
import Foundation
import ImageIO
import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

@MainActor
final class Autoplay {
    let defaults = UserDefaults.standard  // -key value launch arguments land here
    private var fired = false
    private var epoch = 0
    private var frameFiles: [String] = []
    private var inactive: [(Double, Double?)] = []

    var enabled: Bool { defaults.bool(forKey: "autoplay") }
    var log: Bool { defaults.bool(forKey: "log") }
    var frames: Bool { defaults.bool(forKey: "frames") }

    func run(_ screen: CutoutModel) async {
        guard !fired else { return }
        fired = true
        epoch = Int(Date().timeIntervalSince1970)
        let thermal = Self.thermal(), free = Self.freeBytes()
        let mode = defaults.string(forKey: "fixture") != nil ? "fixture" : defaults.bool(forKey: "tail") ? "tail"
            : defaults.integer(forKey: "bbench") > 0 ? "bbench" : defaults.integer(forKey: "tbench") > 0 ? "tbench"
            : defaults.integer(forKey: "abench") > 0 ? "abench"
            : defaults.integer(forKey: "bench") > 0 ? "bench" : enabled ? "autoplay" : "hands"
        let freeText: String = free.map { String($0) } ?? "-"
        var start: String = "START epoch=\(epoch) mode=\(mode) models=\(screen.modelRoot.path) thermal=\(thermal)"
        start += " free_bytes=\(freeText) footprint=\(memoryFootprint().now)"
        start += " device=\(CutoutDevice.machine) os=\(CutoutDevice.osBuild)"
        write(start)
        screen.onFinished = { [weak self] run in self?.record(run, screen: screen) }
        screen.onSaved = { [weak self] line in self?.write("SAVE \(line)") }
        guard mode != "hands" else { return }
        while screen.phase == .loading { try? await Task.sleep(for: .milliseconds(50)) }
        guard screen.phase == .ready else {
            write("ERROR load: \(screen.detail)")
            frame("failed")
            writeLoadRecord(screen, mode: mode, extra: ["error": screen.detail])
            return finish()
        }
        let parts: [String: Double] = screen.loadRecord.map { $0.seconds.mapValues { Self.r3($0) ?? 0 } } ?? [:]
        var ready: String = "READY t=\(Self.r3(screen.readySinceStart) ?? 0) load_s=\(Self.r3(screen.loadSeconds) ?? 0)"
        ready += " parts=\(parts) footprint=\(memoryFootprint().now)"
        write(ready)
        frame("ready")
        switch mode {
        case "fixture": await fixture(screen)
        case "tail": await tail(screen)
        case "bench": await bench(screen)
        case "bbench", "tbench", "abench": await partBench(screen, mode: mode)
        default: await sequence(screen)
        }
        finish()
    }

    func scene(_ phase: ScenePhase) {
        let t = CutoutDevice.sinceStart()
        switch phase {
        case .active:
            if let last = inactive.last, last.1 == nil {
                inactive[inactive.count - 1].1 = t
                write("ACTIVE t=\(Self.r3(t))")
            }
        case .inactive, .background:
            if inactive.last.map({ $0.1 != nil }) ?? true {
                inactive.append((t, nil))
                write("INACTIVE t=\(Self.r3(t)) phase=\(phase == .background ? "background" : "inactive")")
            }
        @unknown default: break
        }
    }

    // MARK: - the screen's path

    private func sequence(_ screen: CutoutModel) async {
        await waitTrigger()
        try? await Task.sleep(for: .seconds(defaults.object(forKey: "delay") != nil ? defaults.double(forKey: "delay") : 1))
        let gap = defaults.object(forKey: "gap") != nil ? defaults.double(forKey: "gap") : 1
        if let image = defaults.string(forKey: "image") {
            screen.pick(fileAt: URL(fileURLWithPath: Self.inDocuments(image)))
            write("PICK image=\(image) ok=\(screen.photo != nil)")
        } else {
            let id = defaults.string(forKey: "example") ?? "lighthouse"
            let ok = screen.pickExample(id)
            write("PICK example=\(id) ok=\(ok)")
        }
        frame("pick")
        try? await Task.sleep(for: .seconds(gap))
        if let w = defaults.string(forKey: "words") { screen.setWords(w) }
        if defaults.string(forKey: "task") == "boxes" { screen.task = .boxes }
        write("WORDS \(screen.words) task=\(screen.task.rawValue)")
        frame("words")
        try? await Task.sleep(for: .seconds(gap))
        write("RUN")
        await screen.cutOut()?.value
        if defaults.string(forKey: "view") == "photo" { screen.showCutout = false }
        try? await Task.sleep(for: .milliseconds(400))
        frame("answer")
        if defaults.bool(forKey: "save") {
            try? await Task.sleep(for: .seconds(gap))
            write("SAVE tap")
            await screen.save()?.value
            try? await Task.sleep(for: .milliseconds(400))
            frame("save")
        }
        if defaults.object(forKey: "hold") != nil { try? await Task.sleep(for: .seconds(defaults.double(forKey: "hold"))) }
    }

    // MARK: - checks

    private func fixture(_ screen: CutoutModel) async {
        guard let p = screen.pipeline, let rel = defaults.string(forKey: "fixture") else { return }
        let rowsURL = URL(fileURLWithPath: Self.inDocuments(rel))
        do {
            let rows = try Fixture.rows(at: rowsURL)
            let out = Self.docs.appending(path: "fixture")
            let t0 = ContinuousClock.now
            let summary = try await Fixture.run(pipeline: p, rows: rows, imagesBase: rowsURL.deletingLastPathComponent(), out: out,
                                                forms: [(.pad, p.masks, "pad-fp16")], writeAlpha: false, log: { self.write($0) })
            let seconds = CutoutModel.seconds(ContinuousClock.now - t0)
            write("DONE fixture rows=\(rows.count) seconds=\(Self.r3(seconds)) footprint_peak=\(memoryFootprint().peak)")
            writeLoadRecord(screen, mode: "fixture", extra: ["rows": summary, "seconds": seconds])
        } catch {
            write("ERROR fixture \(error)")
        }
    }

    private func tail(_ screen: CutoutModel) async {
        guard let p = screen.pipeline else { return }
        let image = Self.inDocuments(defaults.string(forKey: "tailimage") ?? "fixtures/img/syn_a.png")
        let prompt = defaults.string(forKey: "tailprompt")
            ?? "Locate all the instances that match the following description: red rectangle. blue circle. green circle."
        do {
            let file = try ImageHost.load(url: URL(fileURLWithPath: image))
            let rgb = try ImageHost.uprightRGB8(file.image, orientation: file.orientation)
            let v = try await p.tower.encode(patches: try ImageHost.patches(ImageHost.pillowBicubic(rgb, width: 448, height: 448)))
            let row = try p.prompt.row(prompt: prompt)
            let reps = defaults.object(forKey: "reps") != nil ? defaults.integer(forKey: "reps") : 8
            let runs = try await p.decoder.tailBench(prompt: row.ids, inputs: p.decoder.staticInputs(v, row: row),
                                                     tails: [13, 45, 63], reps: reps)
            var report = TailReport.make(runs, promptLength: row.ids.count)
            report["thermal_end"] = Self.thermal()
            report["footprint"] = ["now": memoryFootprint().now, "peak": memoryFootprint().peak]
            Self.writeJSON(report, Self.docs.appending(path: "tail-\(epoch).json"))
            for s in (report["summary"] as? [[String: Any]]) ?? [] {
                let s1 = (s["s1_ms"] as? [String: Double])?["median"] ?? 0, pad = (s["pad_ms"] as? [String: Double])?["median"] ?? 0
                write("TAIL r=\(s["r"] ?? "?") s1_ms=\(Self.r3(s1)) pad_ms=\(Self.r3(pad)) agree=\(s["agree"] ?? [:])")
            }
            write("DONE tail json=tail-\(epoch).json")
        } catch {
            write("ERROR tail \(error)")
        }
    }

    private func bench(_ screen: CutoutModel) async {
        let n = defaults.integer(forKey: "bench")
        let id = defaults.string(forKey: "example") ?? "lighthouse"
        guard screen.pickExample(id) else { return write("ERROR bench: no example \(id)") }
        if let w = defaults.string(forKey: "words") { screen.setWords(w) }
        for i in 0...n {
            write("BENCH \(i) \(i == 0 ? "cold" : "warm") words=\(screen.words)")
            await screen.cutOut()?.value
            try? await Task.sleep(for: .milliseconds(300))
        }
        write("DONE bench n=\(n)")
    }

    /// `-bbench` / `-tbench`: another graph B or tower from the model folder against the loaded one (Engine, Pipeline.swift).
    private func partBench(_ screen: CutoutModel, mode: String) async {
        guard let p = screen.pipeline else { return }
        let reps = defaults.integer(forKey: mode)
        let id = defaults.string(forKey: "example") ?? (mode == "bbench" ? "chess" : "lighthouse")
        guard screen.pickExample(id), let photo = screen.photo else { return write("ERROR \(mode): no example \(id)") }
        if let w = defaults.string(forKey: "words") { screen.setWords(w) }
        do {
            let key = mode == "bbench" ? "bform" : mode == "abench" ? "aform" : "tform"
            guard let form = defaults.string(forKey: key) else { return write("ERROR \(mode): no -\(key)") }
            let url = try p.folder.asset(named: form)
            var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
            options.expectFrequentReshapes = false
            if url.pathExtension == "aimodelc" { options = .default }
            let t0 = ContinuousClock.now
            write("\(mode.uppercased()) load \(url.lastPathComponent) footprint=\(memoryFootprint().now)")
            var rec: [String: Any]
            if mode == "abench" {
                // -aunit ane: the other graph A's .aimodel specialized for the Neural Engine (graph B as loaded)
                var aOptions = options
                if defaults.string(forKey: "aunit") == "ane", url.pathExtension != "aimodelc" {
                    aOptions = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
                }
                let other = try await MaskGraphs(graphA: url, graphB: p.folder.graphB, options: options, graphAOptions: aOptions)
                write("ABENCH loaded load_s=\(Self.r3(secondsSince(t0)) ?? 0) unit=\(defaults.string(forKey: "aunit") ?? "gpu") footprint=\(memoryFootprint().now)")
                rec = try await p.benchGraphA(other: other, image: photo.rgb, words: screen.words, reps: reps)
                rec["other_unit"] = defaults.string(forKey: "aunit") ?? "gpu"
                let b = (rec["base_ms"] as? [String: Any])?["median"] as? Double, o = (rec["other_ms"] as? [String: Any])?["median"] as? Double
                write("ABENCH base_ms=\(Self.r3(b) ?? 0) other_ms=\(Self.r3(o) ?? 0) agreement=\(rec["agreement"] ?? [:])")
            } else if mode == "bbench" {
                let other = try await MaskGraphB(contentsOf: url, options: options)
                write("\(mode.uppercased()) loaded rows=\(other.rows) load_s=\(Self.r3(secondsSince(t0)) ?? 0) footprint=\(memoryFootprint().now)")
                rec = try await p.benchGraphB(other: other, image: photo.rgb, words: screen.words, reps: reps)
                let per = rec["per_instance_ms"] as? [String: Double] ?? [:]
                write("BBENCH rows=\(other.rows) base_ms_per_instance=\(Self.r3(per["base"]) ?? 0) other_ms_per_instance=\(Self.r3(per["other"]) ?? 0) bit_equal=\(rec["logits_bit_equal_instances"] ?? "?") max_abs_d=\(rec["logits_max_abs_d"] ?? "?")")
            } else {
                let other = try await VisionTaps(contentsOf: url, options: options)
                write("\(mode.uppercased()) loaded load_s=\(Self.r3(secondsSince(t0)) ?? 0) footprint=\(memoryFootprint().now)")
                rec = try await p.benchTower(other: other, image: photo.rgb, reps: reps)
                let b = (rec["base_ms"] as? [String: Any])?["median"] as? Double, o = (rec["other_ms"] as? [String: Any])?["median"] as? Double
                write("TBENCH base_ms=\(Self.r3(b) ?? 0) other_ms=\(Self.r3(o) ?? 0) agreement=\(rec["agreement"] ?? [:])")
            }
            rec["other_load_wall_s"] = secondsSince(t0)
            rec["thermal_end"] = Self.thermal()
            rec["load_options"] = screen.loadRecord?.options ?? [:]
            rec["inactive"] = inactive.map { [$0.0, $0.1 ?? -1] }
            rec["device"] = CutoutDevice.machine
            rec["os"] = CutoutDevice.osBuild
            Self.writeJSON(rec, Self.docs.appending(path: "\(mode)-\(epoch).json"))
            write("DONE \(mode) json=\(mode)-\(epoch).json footprint_peak=\(memoryFootprint().peak)")
        } catch {
            write("ERROR \(mode) \(error)")
        }
    }

    // MARK: - records

    private func record(_ run: CutoutModel.Run, screen: CutoutModel) {
        let s: [String: Double] = run.seconds.mapValues { Self.r3($0) ?? 0 }
        let total: Double = s["total"] ?? 0
        let compose: Double = Self.r3(run.composeSeconds) ?? 0
        var line: String = "DONE stop=\(run.stop) pieces=\(run.pieces) tokens=\(run.tokens) total_s=\(total) compose_s=\(compose)"
        line += " footprint=\(run.footprint.now) peak=\(run.footprint.peak) thermal=\(Self.thermal()) answer=\(run.answer)"
        write(line)
        guard log else { return }
        var rec: [String: Any] = [
            "schema": "cutout-run/1", "epoch": epoch, "words": run.words, "task": run.task.rawValue, "photo": run.photoSource,
            "photo_pixels": run.photoPixels, "started_since_start_s": run.startedSinceStart, "answer": run.answer,
            "tokens": run.tokens, "pieces": run.pieces, "boxes": screen.pieces.map(\.box), "labels": screen.pieces.map(\.label),
            "seconds": run.seconds, "compose_s": run.composeSeconds ?? NSNull(), "stop": run.stop, "error": run.error ?? NSNull(),
            "footprint": ["now": run.footprint.now, "peak": run.footprint.peak], "thermal": Self.thermal(),
            "device": CutoutDevice.machine, "os": CutoutDevice.osBuild, "load_s": screen.loadSeconds ?? NSNull(),
            "load_parts": screen.loadRecord?.seconds ?? [:], "load_footprint": screen.loadRecord?.footprint ?? [:],
            "load_options": screen.loadRecord?.options ?? [:], "ready_since_start_s": screen.readySinceStart ?? NSNull(),
            "numbers": "\(screen.numbersHead) · \(screen.numbersTail)", "detail": screen.detail,
            "photos_auth_injected": screen.injectedPhotosDenied, "frames": frameFiles,
            "inactive": inactive.map { [$0.0, $0.1 ?? -1] },
        ]
        rec["screen_px"] = Self.screenPixels()
        Self.writeJSON(rec, Self.docs.appending(path: "cutout-run-\(epoch)-\(Int(run.startedSinceStart * 1000)).json"))
    }

    private func writeLoadRecord(_ screen: CutoutModel, mode: String, extra: [String: Any]) {
        guard log else { return }
        var rec: [String: Any] = ["schema": "cutout-load/1", "epoch": epoch, "mode": mode, "device": CutoutDevice.machine,
                                  "os": CutoutDevice.osBuild, "thermal": Self.thermal(),
                                  "footprint": ["now": memoryFootprint().now, "peak": memoryFootprint().peak],
                                  "load_s": screen.loadSeconds ?? NSNull(), "load_parts": screen.loadRecord?.seconds ?? [:],
                                  "load_footprint": screen.loadRecord?.footprint ?? [:], "detail": screen.detail]
        for (k, v) in extra { rec[k] = v }
        Self.writeJSON(rec, Self.docs.appending(path: "cutout-\(mode)-\(epoch).json"))
    }

    private func finish() {
        write("END footprint_peak=\(memoryFootprint().peak) thermal=\(Self.thermal())")
        if defaults.bool(forKey: "quit") {
            Task { try? await Task.sleep(for: .seconds(1)); exit(0) }
        }
    }

    private func waitTrigger() async {
        guard let t = defaults.string(forKey: "trigger") else { return }
        let path = Self.inDocuments(t)
        while !FileManager.default.fileExists(atPath: path) { try? await Task.sleep(for: .milliseconds(100)) }
        write("TRIGGER")
    }

    // MARK: - frames and the log

    private func frame(_ name: String) {
        guard frames, let image = Self.captureWindow() else { return }
        let path = "frames/\(epoch)/\(name).png"
        frameFiles.append(path)
        let url = Self.docs.appending(path: path)
        Task.detached { Self.writePNG(image, to: url) }
    }

    nonisolated static func writePNG(_ image: CGImage, to url: URL) {
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let file = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(file, image, nil)
        CGImageDestinationFinalize(file)
    }

    static func captureWindow() -> CGImage? {
        #if os(iOS)
        guard let window = keyWindow else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = window.traitCollection.displayScale
        return UIGraphicsImageRenderer(bounds: window.bounds, format: format).image { _ in
            _ = window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
        }.cgImage
        #elseif os(macOS)
        guard let view = NSApplication.shared.windows.first?.contentView,
              let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds)
        else { return nil }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        return bitmap.cgImage
        #endif
    }

    static func screenPixels() -> [Int] {
        #if os(iOS)
        guard let window = keyWindow else { return [] }
        let s = window.traitCollection.displayScale
        return [Int((window.bounds.width * s).rounded()), Int((window.bounds.height * s).rounded())]
        #elseif os(macOS)
        guard let w = NSApplication.shared.windows.first, let v = w.contentView else { return [] }
        return [Int(v.bounds.width * w.backingScaleFactor), Int(v.bounds.height * w.backingScaleFactor)]
        #endif
    }

    #if os(iOS)
    private static var keyWindow: UIWindow? {
        UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.flatMap(\.windows).first { $0.isKeyWindow }
    }
    #endif

    /// One timestamped line to Documents/cutout-autoplay.log (with `-log 1`), and to stdout.
    private func write(_ line: String) {
        print(line)
        guard log else { return }
        let url = Self.docs.appending(path: "cutout-autoplay.log")
        let text = "\(Self.stamp.string(from: Date())) \(line)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    static let stamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    nonisolated static func writeJSON(_ object: Any, _ url: URL) {
        guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys]) else { return }
        try? data.write(to: url)
    }

    /// Where the log, the records, the frames and relative paths live: the app's Documents, or `-docs <dir>` (a Mac run
    /// from a script, so it does not write into the user's own Documents folder).
    nonisolated static var docs: URL {
        if let d = UserDefaults.standard.string(forKey: "docs") { return URL(fileURLWithPath: d) }
        return URL.documentsDirectory
    }

    static func inDocuments(_ path: String) -> String {
        path.hasPrefix("/") ? path : Self.docs.appending(path: path).path
    }

    static func r3(_ v: Double?) -> Double? { v.map { ($0 * 1000).rounded() / 1000 } }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    static func freeBytes() -> Int64? {
        (try? URL.documentsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
    }
}
