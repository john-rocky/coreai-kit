// Autoplay — drives the screen hands-off, for a recording or a check from a script, and keeps a
// record of every answer:
//
//   xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
//     -autoplay 1 -example sign -chip 0 -delay 3 -gap 2 -frames 1 -log 1
//
// `-autoplay 1` waits for READY (the model loads when the app starts; `-model <catalog id>` picks
// another), then — once `-trigger <file>` exists (a recorder creates it when the capture is rolling)
// and `delay` seconds have passed — taps an example photo (`-example sign|object|scene`), waits `gap`,
// taps an example question (`-chip 0|1|2`, or types `-question "…"`; neither = the example's own),
// waits `gap`, taps Ask and waits for the answer. Each press calls the function its button calls
// (pickExample / applyChip / ask), so the run is the screen's own path. The steps keep one timeline
// from the trigger (`Task.sleep(until:)`), so a recording does not drift. A relative `-trigger` is a
// file in Documents, where `devicectl device copy to` puts it.
//
// `-log 1` appends START / READY / PICK / CHIP / ASK / FIRST / DONE lines (ERROR when a step fails)
// to Documents/look-autoplay.log and writes the run to Documents/look-run-<epoch>.json — how iPhone
// numbers are read back over `devicectl device copy from`. Without `-autoplay 1`, `-log 1` records
// the questions asked on the screen by hand (or by the UI test) the same way: ASK / FIRST / DONE and
// one record per answer. `-frames 1` also writes the screen as PNGs to Documents/frames/<epoch>/: at
// READY, PICK, CHIP, the first token and DONE (by hand: the first token and DONE). Both log the
// stretches when the app was not the active app — a system alert over it, Control Center, another
// app in front: INACTIVE / ACTIVE lines, and `inactive_count`, `inactive_s`, `inactive_at` in the
// record. `-quit 1` exits after the autoplay record; `-probe 1` logs START (the thermal state, the
// free bytes) and DONE, nothing else. Nothing else changes: the screen is the same code with the
// same buttons, and the layout is the same with or without `-take 1` (kept in the record as `take`).

import CoreAIKit
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
    let enabled: Bool
    let example: String
    let chip: Int?
    let question: String?
    let delay: Double
    let gap: Double
    let trigger: String?
    let take: Bool
    let frames: Bool
    let log: Bool
    let quit: Bool
    let probe: Bool
    private var fired = false
    private var epoch = 0
    private var frameFiles: [String] = []
    private var frameMillis: [String: Int] = [:]
    private var pendingWrites: [Task<Void, Never>] = []
    private var firstFrame: Task<Void, Never>?
    private var sceneLog = SceneLog()
    /// A question asked by hand: the thermal state and free bytes at its Ask.
    private var handStart: (thermal: String, free: Int64?)?

    init() {
        let defaults = UserDefaults.standard  // -key value launch arguments land here
        enabled = defaults.bool(forKey: "autoplay")
        example = defaults.string(forKey: "example") ?? "sign"
        chip = defaults.object(forKey: "chip") != nil ? defaults.integer(forKey: "chip") : nil
        question = defaults.string(forKey: "question")
        delay = defaults.object(forKey: "delay") != nil ? defaults.double(forKey: "delay") : 3
        gap = defaults.object(forKey: "gap") != nil ? defaults.double(forKey: "gap") : 2
        trigger = defaults.string(forKey: "trigger").map(Self.inDocuments)
        take = defaults.bool(forKey: "take")
        frames = defaults.bool(forKey: "frames")
        log = defaults.bool(forKey: "log")
        quit = defaults.bool(forKey: "quit")
        probe = defaults.bool(forKey: "probe")
    }

    /// Runs once per launch: the sequence with `-autoplay 1`; with `-log 1` alone, the record of
    /// each question asked on the screen.
    func run(_ screen: LookModel) async {
        guard !fired else { return }
        fired = true
        if enabled {
            await sequence(screen)
        } else if log {
            followHands(screen)
        }
    }

    /// The scene's phase changed (LookView): keep the stretches when the app was not active.
    func scene(_ phase: ScenePhase) {
        if let line = sceneLog.changed(to: phase, at: LookDevice.sinceStart()) { write(line) }
    }

    // MARK: - -autoplay 1

    private func sequence(_ screen: LookModel) async {
        epoch = Int(Date().timeIntervalSince1970)
        let thermalStart = Self.thermal()
        let freeStart = Self.freeBytes()
        let asked = question.map { "question=\($0)" } ?? "chip=\(chip.map(String.init) ?? "-")"
        write(
            "START epoch=\(epoch) mode=autoplay model=\(screen.modelID) example=\(example) \(asked) delay=\(delay)"
                + " gap=\(gap) take=\(take) frames=\(frames) thermal=\(thermalStart)"
                + " free_bytes=\(freeStart.map(String.init) ?? "-")")
        if probe {  // -probe 1: the thermal state and free bytes now, nothing else
            write("DONE probe")
            if quit { exit(0) }
            return
        }

        while screen.phase == .loading { try? await Task.sleep(for: .milliseconds(50)) }
        guard screen.phase == .ready else {
            return await finish(screen, "load: \(screen.detail)", thermalStart, freeStart)
        }
        write(
            "READY model=\(screen.modelID) engine_ready_s=\(Self.decimals(screen.engineReadySeconds))"
                + " t_ready_s=\(Self.decimals(screen.readySinceStart))")
        frame("ready")

        if let trigger {
            while !FileManager.default.fileExists(atPath: trigger) {
                try? await Task.sleep(for: .milliseconds(100))
            }
            write("TRIGGER \(trigger)")
        }
        let clock = ContinuousClock()
        var at = clock.now + .seconds(delay)
        try? await Task.sleep(until: at, tolerance: .milliseconds(5), clock: clock)
        guard screen.pickExample(example) else {
            return await finish(screen, "pick \(example): \(screen.detail)", thermalStart, freeStart)
        }
        write("PICK \(example)")
        await Task.yield()
        frame("pick")

        at = at + .seconds(gap)
        try? await Task.sleep(until: at, tolerance: .milliseconds(5), clock: clock)
        if let question {
            screen.setQuestion(question)
            write("QUESTION \(question)")
        } else {
            let index = chip ?? LookModel.examples.firstIndex(where: { $0.id == example }) ?? 0
            screen.applyChip(index)
            write("CHIP \(index)")
        }
        await Task.yield()
        frame("chip")

        at = at + .seconds(gap)
        try? await Task.sleep(until: at, tolerance: .milliseconds(5), clock: clock)
        screen.onFirstToken = { [weak self, weak screen] in
            guard let self, let screen else { return }
            self.firstToken(screen)
        }
        write("ASK")
        guard let answering = screen.ask() else {
            return await finish(screen, "ask: \(screen.detail)", thermalStart, freeStart)
        }
        await answering.value
        screen.onFirstToken = nil
        await firstFrame?.value
        await Task.yield()
        frame("done")
        try? await Task.sleep(for: .seconds(3))  // the answer stays on screen, as a take holds it
        let box = screen.answerBox  // read after the hold: the box as the screen shows it at DONE

        let file = await record(
            screen, mode: "autoplay", thermalStart: thermalStart, freeStart: freeStart, box: box, failure: nil)
        done(screen, file)
        if quit { exit(0) }
    }

    /// A step failed before an answer: the record with the reason, then one ERROR line.
    private func finish(_ screen: LookModel, _ reason: String, _ thermalStart: String, _ freeStart: Int64?) async {
        let file = await record(
            screen, mode: "autoplay", thermalStart: thermalStart, freeStart: freeStart, box: screen.answerBox,
            failure: reason)
        write("ERROR \(reason) json=\(file)")
        if quit { exit(1) }
    }

    // MARK: - -log 1 alone: questions asked by hand

    /// Every Ask on the screen (a finger, or the UI test) gets ASK / FIRST / DONE and its record.
    private func followHands(_ screen: LookModel) {
        write(
            "START epoch=\(Int(Date().timeIntervalSince1970)) mode=hand model=\(screen.modelID) take=\(take)"
                + " frames=\(frames) thermal=\(Self.thermal()) free_bytes=\(Self.freeBytes().map(String.init) ?? "-")")
        screen.onAsked = { [weak self, weak screen] in
            guard let self, let screen else { return }
            self.epoch = Int(Date().timeIntervalSince1970)
            self.frameFiles = []
            self.frameMillis = [:]
            self.firstFrame = nil
            self.handStart = (thermal: Self.thermal(), free: Self.freeBytes())
            self.write("ASK epoch=\(self.epoch) photo=\(screen.photo?.source ?? "-") question=\(screen.question)")
        }
        screen.onFirstToken = { [weak self, weak screen] in
            guard let self, let screen else { return }
            self.firstToken(screen)
        }
        screen.onFinished = { [weak self, weak screen] _ in
            guard let self, let screen else { return }
            Task { await self.recordHand(screen) }
        }
    }

    private func recordHand(_ screen: LookModel) async {
        await firstFrame?.value
        await Task.yield()
        frame("done")
        try? await Task.sleep(for: .milliseconds(500))  // the box settles on its last line
        let start = handStart ?? (thermal: Self.thermal(), free: Self.freeBytes())
        let file = await record(
            screen, mode: "hand", thermalStart: start.thermal, freeStart: start.free, box: screen.answerBox,
            failure: nil)
        done(screen, file)
    }

    // MARK: - Both

    private func firstToken(_ screen: LookModel) {
        write("FIRST first_token_s=\(Self.decimals(screen.firstTokenSeconds))")
        firstFrame = Task {
            await Task.yield()
            self.frame("first")
        }
    }

    private func done(_ screen: LookModel, _ file: String) {
        if let run = screen.lastRun, run.stop == "done" {
            write(
                "DONE stop=done first_token_s=\(Self.decimals(run.firstToken)) tok_s=\(Self.decimals(run.tokensPerSecond))"
                    + " output_tokens=\(run.outputTokens) snapshots=\(run.snapshots) json=\(file)")
        } else {
            let run = screen.lastRun
            write("ERROR stop=\(run?.stop ?? "-") \(run?.error ?? screen.detail) json=\(file)")
        }
    }

    // MARK: - The record

    /// Documents/look-run-<epoch>.json: the run, the load before it, the phone, and the screen
    /// (the band, the answer box, the frames, the stretches when the app was not active). Written
    /// only with `-log 1`.
    private func record(
        _ screen: LookModel, mode: String, thermalStart: String, freeStart: Int64?, box: LookModel.AnswerBox,
        failure: String?
    ) async -> String {
        let name = "look-run-\(epoch).json"
        guard log else { return name }
        for write in pendingWrites { await write.value }
        let run = screen.lastRun
        let (screenPixels, scale) = Self.screenFacts()
        let band = screen.bandRect
        let now = LookDevice.sinceStart()
        var object: [String: Any] = [:]
        func put(_ key: String, _ value: Any?) { object[key] = value ?? NSNull() }

        put("schema", "look-run/2")
        put("mode", mode)
        put("epoch", epoch)
        put("args", Array(CommandLine.arguments.dropFirst()))
        put("model", screen.modelID)
        put("repo", screen.entry?.repo)
        put("revision", screen.entry?.revision)
        put("form", Self.form(screen))
        put("device", LookDevice.machine)
        put("device_name", LookDevice.name)
        put("os", "\(LookDevice.osVersion) (\(LookDevice.osBuild))")
        put("app", "Look \(LookDevice.appVersion) (\(LookDevice.appBuild))")
        put("kit", nil)
        put("thermal_start", thermalStart)
        put("thermal_end", Self.thermal())
        put("free_bytes_start", freeStart)
        put("offline", screen.offline)
        put("example", enabled ? example : nil)
        put("photo_source", run?.photoSource ?? screen.photo?.source)
        put("photo_bytes", run?.photoBytes ?? screen.photo?.bytes)
        put("photo_px", run?.photoPixels ?? screen.photo.map { [$0.cgImage.width, $0.cgImage.height] })
        put("question", run?.question ?? screen.question)
        put("catalog_s", Self.round3(screen.catalogSeconds))
        put("engine_ready_s", Self.round3(screen.engineReadySeconds))
        put("prewarm_s", Self.round3(screen.prewarmSeconds))
        put("t_ready_s", Self.round3(screen.readySinceStart))
        put("t_pick_s", Self.round3(screen.pickedSinceStart))
        put("t_chip_s", Self.round3(screen.questionSinceStart))
        put("t_ask_s", Self.round3(run?.askedSinceStart))
        put("first_token_s", Self.round3(run?.firstToken))
        put("last_text_s", Self.round3(run?.lastText))
        put("generate_s", Self.round3(run?.generate))
        put("input_tokens", run?.inputTokens)
        put("output_tokens", run?.outputTokens)
        put("snapshots", run?.snapshots)
        put("text_updates", run?.textUpdates)
        put("tok_s", Self.round3(run?.tokensPerSecond))
        put("tok_s_to_last_text", Self.round3(run?.tokensPerSecondToLastText))
        put("answer", run?.answer)
        put("answer_shown", run.map { AnswerText.plain($0.answer) })
        put("numbers_line", screen.numbersLine)
        put("footer", screen.footer)
        put("stop", run?.stop ?? "error")
        put("error", failure ?? run?.error)
        put("take", take)
        put("screen_px", screenPixels)
        put("scale", scale)
        put("take_rect_px", [band.minX, band.minY, band.width, band.height].map { Int(($0 * scale).rounded()) })
        put(
            "answer_box_pt",
            [
                "content": (box.content * 10).rounded() / 10, "container": (box.container * 10).rounded() / 10,
                "offset": (box.offset * 10).rounded() / 10, "overflow": box.content > box.container + 0.5,
                "first_line_in_view": box.offset <= 0.5,
            ] as [String: Any])
        let stretches = sceneLog.summary(now: now)
        put("inactive_count", stretches.count)
        put("inactive_s", Self.round3(stretches.seconds))
        put("inactive_at", stretches.rows)
        put("scene_first_active_s", Self.round3(sceneLog.firstActive))
        put("scene_at_record", sceneLog.current)
        put("t_record_s", Self.round3(now))
        put("frames", frameFiles)
        put("frame_ms", frameMillis)

        let url = URL.documentsDirectory.appending(path: name)
        if let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        {
            try? data.write(to: url)
        }
        return name
    }

    /// The decoder bundle the catalog row names, and the vision tower the kit pairs with it.
    private static func form(_ screen: LookModel) -> String {
        let decoder = screen.entry?.variant?.path ?? "?"
        let vision = screen.modelID == "minicpm-v-4.7-1b" ? VLModelID.miniCPMV47.vision.path ?? "?" : "?"
        return "\(decoder) + \(vision)"
    }

    // MARK: - Frames

    /// The screen as it is now, as a PNG at the screen's scale (only with `-frames 1`). The window
    /// draws itself, so the text field and the scrolling answer are the pixels on the glass; the
    /// PNG is encoded off the main actor.
    private func frame(_ name: String) {
        guard frames else { return }
        let clock = ContinuousClock()
        let start = clock.now
        guard let image = Self.captureWindow() else {
            write("ERROR frame \(name): no window to draw")
            return
        }
        frameMillis[name] = Int((LookModel.seconds(clock.now - start) * 1000).rounded())
        let path = "frames/\(epoch)/\(name).png"
        frameFiles.append(path)
        let url = URL.documentsDirectory.appending(path: path)
        pendingWrites.append(Task.detached { Autoplay.writePNG(image, to: url) })
    }

    nonisolated private static func writePNG(_ image: CGImage, to url: URL) {
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let file = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(file, image, nil)
        CGImageDestinationFinalize(file)
    }

    private static func captureWindow() -> CGImage? {
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

    /// The screen in pixels and its scale (1206 × 2622 at 3 on an iPhone 18 Pro).
    private static func screenFacts() -> ([Int], Double) {
        #if os(iOS)
        guard let window = keyWindow else { return ([], 0) }
        let scale = window.traitCollection.displayScale
        return ([Int((window.bounds.width * scale).rounded()), Int((window.bounds.height * scale).rounded())], scale)
        #elseif os(macOS)
        guard let window = NSApplication.shared.windows.first, let view = window.contentView else { return ([], 0) }
        let scale = window.backingScaleFactor
        return ([Int((view.bounds.width * scale).rounded()), Int((view.bounds.height * scale).rounded())], scale)
        #endif
    }

    #if os(iOS)
    private static var keyWindow: UIWindow? {
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap(\.windows)
            .first { $0.isKeyWindow }
    }
    #endif

    // MARK: - Log

    /// Appends one timestamped line to Documents/look-autoplay.log (only with `-log 1`).
    private func write(_ line: String) {
        guard log else { return }
        let url = URL.documentsDirectory.appending(path: "look-autoplay.log")
        let text = "\(Self.stamp.string(from: Date())) \(line)\n"
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(text.utf8))
            try? handle.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    private static let stamp: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static func inDocuments(_ path: String) -> String {
        path.hasPrefix("/") ? path : URL.documentsDirectory.appending(path: path).path
    }

    private static func decimals(_ value: Double?) -> String {
        value.map { String(format: "%.3f", $0) } ?? "-"
    }

    nonisolated static func round3(_ value: Double?) -> Double? {
        value.map { ($0 * 1000).rounded() / 1000 }
    }

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

/// The stretches when the app was not the active app, from the scene phase, in seconds since the
/// process started. A system alert over the app (another app's permission request, a low-battery
/// warning) or Control Center makes the scene inactive; another app in front puts it in the
/// background. Counted from the first time the app was active: the launch is not a stretch. An
/// alert already on the screen when the app starts keeps it from ever becoming active, so the record
/// also keeps when it first did (`scene_first_active_s`, null = never) and the phase at the record.
struct SceneLog {
    struct Stretch {
        var start: Double
        var end: Double?
        var phase: String
    }

    private(set) var stretches: [Stretch] = []
    private(set) var firstActive: Double?
    private(set) var current = "inactive"

    /// Returns the log line for a change, nil when there is nothing to log.
    mutating func changed(to phase: ScenePhase, at time: Double) -> String? {
        current = phase == .active ? "active" : phase == .background ? "background" : "inactive"
        if phase == .active {
            if firstActive == nil {
                firstActive = time
                return String(format: "ACTIVE first t_s=%.3f", time)
            }
            guard let last = stretches.indices.last, stretches[last].end == nil else { return nil }
            stretches[last].end = time
            return String(format: "ACTIVE t_s=%.3f after_s=%.3f", time, time - stretches[last].start)
        }
        guard firstActive != nil else { return nil }
        let name = phase == .background ? "background" : "inactive"
        if let last = stretches.indices.last, stretches[last].end == nil {
            guard stretches[last].phase != name else { return nil }
            if name == "background" { stretches[last].phase = name }  // the same stretch, deeper
            return String(format: "INACTIVE t_s=%.3f phase=%@", time, name)
        }
        stretches.append(Stretch(start: time, end: nil, phase: name))
        return String(format: "INACTIVE t_s=%.3f phase=%@", time, name)
    }

    /// For the record: how many stretches, their seconds (an open one counts up to `now`), and each.
    func summary(now: Double) -> (count: Int, seconds: Double, rows: [[String: Any]]) {
        let rows: [[String: Any]] = stretches.map { stretch in
            [
                "at_s": Autoplay.round3(stretch.start) ?? 0, "s": Autoplay.round3((stretch.end ?? now) - stretch.start) ?? 0,
                "phase": stretch.phase, "open": stretch.end == nil,
            ]
        }
        let seconds = stretches.reduce(0) { $0 + (($1.end ?? now) - $1.start) }
        return (stretches.count, seconds, rows)
    }
}
