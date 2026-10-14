// Autoplay — drives the screen hands-off, for a recording or a check from a script:
//
//   xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
//     -autoplay 1 -example sign -chip 0 -delay 3 -gap 2 -take 1 -frames 1 -log 1
//
// waits for READY (the model loads when the app starts; `-model <catalog id>` picks another), then
// — once `-trigger <file>` exists (a recorder creates it when the capture is rolling) and `delay`
// seconds have passed — taps an example photo (`-example sign|object|scene`), waits `gap`, taps an
// example question (`-chip 0|1|2`, or types `-question "…"`; neither = the example's own), waits
// `gap`, taps Ask and waits for the answer. Each press calls the function its button calls
// (pickExample / applyChip / ask), so the run is the screen's own path. The steps keep one timeline
// from the trigger (`Task.sleep(until:)`), so a recording does not drift. A relative `-trigger` is a
// file in Documents, where `devicectl device copy to` puts it.
//
// With `-log 1` it appends START / READY / PICK / CHIP / ASK / FIRST / DONE lines (ERROR when a step
// fails) to Documents/look-autoplay.log and writes the run to Documents/look-run-<epoch>.json — how
// iPhone numbers are read back over `devicectl device copy from`. `-frames 1` also writes the screen
// as PNGs at READY, PICK, CHIP, the first token and DONE to Documents/frames/<epoch>/. `-quit 1`
// exits after the record; `-probe 1` logs START (the thermal state, the free bytes) and DONE, nothing
// else. Nothing else changes: the screen is the same code with the same buttons.

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

    /// Runs the sequence once per launch, with `-autoplay 1` only.
    func run(_ screen: LookModel) async {
        guard enabled, !fired else { return }
        fired = true
        epoch = Int(Date().timeIntervalSince1970)
        let thermalStart = Self.thermal()
        let freeStart = Self.freeBytes()
        let asked = question.map { "question=\($0)" } ?? "chip=\(chip.map(String.init) ?? "-")"
        write(
            "START epoch=\(epoch) model=\(screen.modelID) example=\(example) \(asked) delay=\(delay) gap=\(gap)"
                + " take=\(take) frames=\(frames) thermal=\(thermalStart) free_bytes=\(freeStart.map(String.init) ?? "-")")
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
            self.write("FIRST first_token_s=\(Self.decimals(screen.firstTokenSeconds))")
            self.firstFrame = Task {
                await Task.yield()
                self.frame("first")
            }
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
        let box = screen.answerBox
        try? await Task.sleep(for: .seconds(3))  // the answer stays on screen, as a take holds it

        let file = await record(screen, thermalStart: thermalStart, freeStart: freeStart, box: box, failure: nil)
        if let run = screen.lastRun, run.stop == "done" {
            write(
                "DONE stop=done first_token_s=\(Self.decimals(run.firstToken)) tok_s=\(Self.decimals(run.tokensPerSecond))"
                    + " output_tokens=\(run.outputTokens) snapshots=\(run.snapshots) json=\(file)")
        } else {
            let run = screen.lastRun
            write("ERROR stop=\(run?.stop ?? "-") \(run?.error ?? screen.detail) json=\(file)")
        }
        if quit { exit(0) }
    }

    /// A step failed before an answer: the record with the reason, then one ERROR line.
    private func finish(_ screen: LookModel, _ reason: String, _ thermalStart: String, _ freeStart: Int64?) async {
        let file = await record(screen, thermalStart: thermalStart, freeStart: freeStart, box: screen.answerBox, failure: reason)
        write("ERROR \(reason) json=\(file)")
        if quit { exit(1) }
    }

    // MARK: - The record

    /// Documents/look-run-<epoch>.json: the run, the load before it, the phone, and the screen
    /// (the take band, the answer box, the frames). Written only with `-log 1`.
    private func record(
        _ screen: LookModel, thermalStart: String, freeStart: Int64?, box: LookModel.AnswerBox, failure: String?
    ) async -> String {
        let name = "look-run-\(epoch).json"
        guard log else { return name }
        for write in pendingWrites { await write.value }
        let run = screen.lastRun
        let (screenPixels, scale) = Self.screenFacts()
        let band = screen.bandRect
        var object: [String: Any] = [:]
        func put(_ key: String, _ value: Any?) { object[key] = value ?? NSNull() }
        func round3(_ value: Double?) -> Any? { value.map { ($0 * 1000).rounded() / 1000 } }

        put("schema", "look-run/1")
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
        put("example", example)
        put("photo_source", run?.photoSource ?? screen.photo?.source)
        put("photo_bytes", run?.photoBytes ?? screen.photo?.bytes)
        put("photo_px", run?.photoPixels ?? screen.photo.map { [$0.cgImage.width, $0.cgImage.height] })
        put("question", run?.question ?? screen.question)
        put("catalog_s", round3(screen.catalogSeconds))
        put("engine_ready_s", round3(screen.engineReadySeconds))
        put("prewarm_s", round3(screen.prewarmSeconds))
        put("t_ready_s", round3(screen.readySinceStart))
        put("t_pick_s", round3(screen.pickedSinceStart))
        put("t_chip_s", round3(screen.questionSinceStart))
        put("t_ask_s", round3(run?.askedSinceStart))
        put("first_token_s", round3(run?.firstToken))
        put("last_text_s", round3(run?.lastText))
        put("generate_s", round3(run?.generate))
        put("input_tokens", run?.inputTokens)
        put("output_tokens", run?.outputTokens)
        put("snapshots", run?.snapshots)
        put("text_updates", run?.textUpdates)
        put("tok_s", round3(run?.tokensPerSecond))
        put("tok_s_to_last_text", round3(run?.tokensPerSecondToLastText))
        put("answer", run?.answer)
        put("answer_shown", run.map { String(LookModel.shown($0.answer).characters) })
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
