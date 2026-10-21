// Watch — Look's watch mode, for a check from a script: frames from the back camera (or the three
// example photos in turn) go to the model one at a time with the conditions given at launch, and the
// model answers yes or no for each of them in one turn. A condition that turns yes fires its
// reaction: a system sound, its SF Symbol over the picture for 1.5 s, and a haptic.
//
//   xcrun devicectl device process launch --device <udid> com.coreaikit.look -- \
//     -watch 1 -source camera -seconds 60 \
//     -conditions "a chessboard|a lighthouse|a sign with words on it|a person" -log 1 -quit 1
//
// One frame at a time: when an answer is in, the newest camera frame is taken (the frames that came
// meanwhile are dropped), made a CGImage with its orientation, and sent to a new session (no history,
// greedy, at most `-tokens` answer tokens, default 32) with one prompt: "Answer each line with yes or
// no, nothing else." and a line "<i>. Is there <condition>?" per condition (`-form single`: a session
// per condition, "Is there <condition>? Answer yes or no."). The answer is read line by line
// (WatchParse): yes, no, or ? when a line says neither. `-source photos` sends the examples sign,
// object, scene, sign … instead, each as a tap on its thumbnail reads it; with the four default
// conditions every frame is scored against what its photo shows (`-truth "sign:no,no,yes,no;…"`
// names the answers for other conditions).
//
// `-log 1` appends START / READY / CAMERA / WATCH (one per frame) / DONE lines (ERROR when it fails)
// to Documents/look-autoplay.log, the log Autoplay writes, and writes Documents/watch-run-<epoch>.json
// (schema watch-run/1): each frame's times, tokens, answers, raw text, thermal state and memory
// footprint, and the run's summary. `-quit 1` exits after the record. The first camera run asks for
// the camera permission; `-permwait` caps the wait for the answer (default 600 s).

import AVFoundation
import AudioToolbox
import CoreAIKit
import CoreMedia
import Foundation
import FoundationModels
import ImageIO
import Observation
import SwiftUI
import VideoToolbox
import os
#if canImport(Darwin)
import Darwin
#endif
#if os(iOS)
import UIKit
#endif

/// The watch mode's launch arguments (`-key value`, read from UserDefaults as Autoplay reads its own).
struct WatchSettings {
    /// The conditions when `-conditions` names none.
    static let defaultConditions = ["a chessboard", "a lighthouse", "a sign with words on it", "a person"]
    /// What the three example photos show, for the default conditions in their order.
    static let defaultTruth: [String: [String]] = [
        "sign": ["no", "no", "yes", "no"],
        "object": ["yes", "no", "no", "no"],
        "scene": ["no", "yes", "no", "no"],
    ]
    /// The SF Symbol of each condition by its place; `-symbols` names others.
    static let defaultSymbols = ["checkerboard.rectangle", "light.beacon.max", "signpost.right", "person"]

    let enabled: Bool
    let source: String
    let seconds: Double
    let conditions: [String]
    let symbols: [String]
    let form: String
    let maxTokens: Int
    let truth: [String: [String]]?
    let permissionWait: Double
    let log: Bool
    let quit: Bool
    let modelID: String

    init(_ defaults: UserDefaults = .standard) {
        enabled = defaults.bool(forKey: "watch")
        #if os(iOS)
        source = defaults.string(forKey: "source") ?? "camera"
        #else
        source = defaults.string(forKey: "source") ?? "photos"
        #endif
        seconds = defaults.object(forKey: "seconds") != nil ? defaults.double(forKey: "seconds") : 60
        let listed = Self.list(defaults.string(forKey: "conditions"))
        let conditions = listed.isEmpty ? Self.defaultConditions : listed
        self.conditions = conditions
        let named = Self.list(defaults.string(forKey: "symbols"))
        symbols = conditions.indices.map { index in
            if named.indices.contains(index) { return named[index] }
            return Self.defaultSymbols.indices.contains(index) ? Self.defaultSymbols[index] : "checkmark.circle"
        }
        form = defaults.string(forKey: "form") == "single" ? "single" : "batch"
        maxTokens = defaults.object(forKey: "tokens") != nil ? max(1, defaults.integer(forKey: "tokens")) : 32
        if let text = defaults.string(forKey: "truth") {
            truth = Self.parseTruth(text)
        } else {
            truth = conditions == Self.defaultConditions ? Self.defaultTruth : nil
        }
        permissionWait = defaults.object(forKey: "permwait") != nil ? defaults.double(forKey: "permwait") : 600
        log = defaults.bool(forKey: "log")
        quit = defaults.bool(forKey: "quit")
        modelID = defaults.string(forKey: "model") ?? "minicpm-v-4.7-1b"
    }

    /// "a|b|c" → ["a", "b", "c"], each trimmed, the empty ones dropped.
    private static func list(_ text: String?) -> [String] {
        (text ?? "").split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    /// "sign:no,no,yes,no;object:yes,no,no,no" → ["sign": ["no", "no", "yes", "no"], …]
    private static func parseTruth(_ text: String) -> [String: [String]] {
        var out: [String: [String]] = [:]
        for item in text.split(separator: ";") {
            let parts = item.split(separator: ":", maxSplits: 1)
            guard parts.count == 2 else { continue }
            out[parts[0].trimmingCharacters(in: .whitespaces)] = parts[1].split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces).lowercased()
            }
        }
        return out
    }
}

/// The prompts, built from the conditions.
enum WatchPrompt {
    static let instruction = "Answer each line with yes or no, nothing else."

    /// One prompt for all the conditions: the instruction, then "<i>. Is there <condition>?" per line.
    static func batch(_ conditions: [String]) -> String {
        ([instruction] + conditions.enumerated().map { "\($0.offset + 1). Is there \($0.element)?" })
            .joined(separator: "\n")
    }

    /// One prompt per condition (`-form single`).
    static func single(_ condition: String) -> String { "Is there \(condition)? Answer yes or no." }
}

/// Reads yes / no out of an answer.
enum WatchParse {
    /// The batch answer, one value per condition. A line that starts with a number k answers
    /// condition k; a line without a number answers the first condition still open; a line's value is
    /// its first word "yes" or "no". If conditions are still open and the whole answer holds exactly
    /// one such word per condition, those words answer them in order. Open = "?".
    static func batch(_ text: String, count: Int) -> [String] {
        var out = Array(repeating: "?", count: count)
        for raw in text.lowercased().split(whereSeparator: \.isNewline) {
            var line = Substring(raw.trimmingCharacters(in: .whitespacesAndNewlines))
            if line.isEmpty { continue }
            var index: Int?
            let digits = line.prefix(while: { $0.isASCII && $0.isNumber })
            if !digits.isEmpty {
                if let k = Int(digits), k >= 1, k <= count { index = k - 1 }
                line = line.dropFirst(digits.count).drop(while: { $0 == " " || $0 == "\t" })
                if let mark = line.first, ".):-".contains(mark) { line = line.dropFirst() }
                line = line.drop(while: { $0 == " " || $0 == "\t" })
            }
            guard let word = words(in: line).first else { continue }
            if index == nil { index = out.firstIndex(of: "?") }
            if let index, out[index] == "?" { out[index] = word }
        }
        if out.contains("?") {
            let all = words(in: Substring(text.lowercased()))
            if all.count == count { out = all }
        }
        return out
    }

    /// One condition's answer: its first word "yes" or "no", else "?".
    static func single(_ text: String) -> String {
        words(in: Substring(text.lowercased())).first ?? "?"
    }

    /// The words "yes" and "no" of a lowercased text, in order (a word = a run of letters, digits and _).
    static func words(in text: Substring) -> [String] {
        var found: [String] = []
        var word = ""
        for character in text {
            if character.isLetter || character.isNumber || character == "_" {
                word.append(character)
            } else {
                if word == "yes" || word == "no" { found.append(word) }
                word = ""
            }
        }
        if word == "yes" || word == "no" { found.append(word) }
        return found
    }
}

enum WatchError: LocalizedError {
    case notLoaded
    case camera(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded: "the model is not loaded"
        case .camera(let reason): reason
        }
    }
}

/// The memory the record keeps per frame.
enum WatchMemory {
    /// `phys_footprint` in MB, the value jetsam counts; -1 when unreadable.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    /// MB left before this process reaches its memory limit (iOS); -1 elsewhere.
    static func availableMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / 1_048_576
        #else
        return -1
        #endif
    }
}

@MainActor
@Observable
final class WatchModel {
    enum Phase: Equatable {
        case loading, ready, watching, done, failed
    }

    struct Row {
        let condition: String
        let symbol: String
        var answer = "–"
    }

    /// One frame for the model: the image as the model gets it, what the screen shows, and its times.
    struct Frame {
        let image: CGImage
        let orientation: CGImagePropertyOrientation
        let display: CGImage?
        let name: String
        let ageMs: Double
        let convertMs: Double
        let captureWall: Double
    }

    /// One answer from one session.
    struct Reply {
        let text: String
        let firstMs: Double?
        let modelMs: Double
        let inputTokens: Int
        let outputTokens: Int
    }

    let settings: WatchSettings
    var phase: Phase = .loading
    var detail = "loading…"
    var rows: [Row]
    /// The picture of the last answered frame (photos); the camera shows its live preview instead.
    var shownImage: CGImage?
    /// The conditions whose reaction is on the screen, by index.
    var overlay: [Int] = []
    var msLine = "capture→input – ms · first token – ms · total – ms"
    var stateLine = "frame – · thermal –"
    private(set) var entry: CatalogEntry?
    private(set) var camera: CameraFeed?

    @ObservationIgnored private var model: KitVisionModel?
    @ObservationIgnored private var started = false
    @ObservationIgnored private var epoch = 0
    @ObservationIgnored private var thermalStart = "-"
    @ObservationIgnored private var freeStart: Int64?
    @ObservationIgnored private var catalogSeconds: Double?
    @ObservationIgnored private var engineReadySeconds: Double?
    @ObservationIgnored private var prewarmSeconds: Double?
    @ObservationIgnored private var readySinceStart: Double?
    @ObservationIgnored private var photos: [(id: String, photo: PickedPhoto)] = []
    @ObservationIgnored private var cameraInfo: [String: Any] = [:]
    @ObservationIgnored private var permission: Bool?
    @ObservationIgnored private var frames: [[String: Any]] = []
    @ObservationIgnored private var loopSeconds: Double?
    @ObservationIgnored private var failure: String?
    @ObservationIgnored private var overlayID = 0
    @ObservationIgnored private var sceneLog = SceneLog()

    init() {
        let settings = WatchSettings()
        self.settings = settings
        rows = zip(settings.conditions, settings.symbols).map { Row(condition: $0.0, symbol: $0.1) }
    }

    var prompt: String { WatchPrompt.batch(settings.conditions) }

    var footer: String {
        let name = entry.map { "\($0.repo) @\(($0.revision ?? "main").prefix(8))" } ?? settings.modelID
        let subtree = entry?.variant?.path.split(separator: "/").first.map(String.init) ?? "–"
        return [name, subtree, LookDevice.name, LookDevice.os, "Look \(LookDevice.appVersion)"].joined(separator: " · ")
    }

    // MARK: - The run

    /// Load, open the source, watch for `-seconds`, write the record. Called once, when the screen appears.
    func start() async {
        guard !started else { return }
        started = true
        epoch = Int(Date().timeIntervalSince1970)
        thermalStart = Autoplay.thermal()
        freeStart = Autoplay.freeBytes()
        let startFields: [String] = [
            "START epoch=\(epoch) mode=watch source=\(settings.source)",
            "seconds=\(Self.decimals(settings.seconds)) form=\(settings.form) tokens=\(settings.maxTokens)",
            "conditions=\(settings.conditions.joined(separator: "|")) thermal=\(thermalStart)",
            "free_bytes=\(freeStart.map { String($0) } ?? "-")",
        ]
        write(startFields.joined(separator: " "))
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true  // a minute without a touch must not lock the screen
        #endif
        if let error = await load() { return finish("load: \(error)") }
        phase = .ready
        detail = "Ready"
        write(
            "READY model=\(settings.modelID) engine_ready_s=\(Self.decimals(engineReadySeconds))"
                + " t_ready_s=\(Self.decimals(readySinceStart))")
        switch settings.source {
        case "photos":
            if let error = loadPhotos() { return finish(error) }
        case "camera":
            #if os(iOS)
            if let error = await openCamera() { return finish(error) }
            #else
            return finish("camera: iOS only")
            #endif
        default:
            return finish("source \(settings.source): photos or camera")
        }
        await watch()
        finish(failure)
    }

    /// The scene's phase changed (WatchView): keep the stretches when the app was not active.
    func scene(_ phase: ScenePhase) {
        if let line = sceneLog.changed(to: phase, at: LookDevice.sinceStart()) { write(line) }
    }

    private func load() async -> String? {
        let clock = ContinuousClock()
        do {
            var start = clock.now
            entry = try await ModelCatalog.entry(forID: settings.modelID, expecting: .vlm)
            catalogSeconds = LookModel.seconds(clock.now - start)
            start = clock.now
            let vlm = try await KitVisionModel(catalog: settings.modelID) { progress in
                let line = progress.fraction < 1 ? "Downloading \(Int(progress.fraction * 100)) %" : "loading…"
                Task { @MainActor in self.detail = line }
            }
            engineReadySeconds = LookModel.seconds(clock.now - start)
            start = clock.now
            LanguageModelSession(model: vlm).prewarm()
            prewarmSeconds = LookModel.seconds(clock.now - start)
            model = vlm
            readySinceStart = LookDevice.sinceStart()
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    /// The three examples, read as a tap on their thumbnails reads them (the file's first image and
    /// its EXIF orientation).
    private func loadPhotos() -> String? {
        for example in LookModel.examples {
            guard let url = example.url, let data = try? Data(contentsOf: url),
                let photo = PickedPhoto(data: data, source: "example \(example.id)")
            else { return "photos: \(example.file) is not in this build" }
            photos.append((id: example.id, photo: photo))
        }
        return nil
    }

    #if os(iOS)
    private func openCamera() async -> String? {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        write("CAMERA status=\(Self.name(status))")
        switch status {
        case .authorized:
            break
        case .notDetermined:
            detail = "Allow the camera to start watching"
            let clock = ContinuousClock()
            let asked = clock.now
            Task { self.permission = await AVCaptureDevice.requestAccess(for: .video) }
            while permission == nil, clock.now - asked < .seconds(settings.permissionWait) {
                try? await Task.sleep(for: .milliseconds(200))
            }
            let answered = permission.map { String($0) } ?? "-"
            write("CAMERA granted=\(answered) after_s=\(Self.decimals(LookModel.seconds(clock.now - asked)))")
            guard let permission else {
                return "camera: no answer to the permission request in \(Int(settings.permissionWait)) s"
            }
            guard permission else { return "camera: permission denied" }
        default:
            return "camera: permission \(Self.name(status))"
        }
        let feed = CameraFeed()
        do {
            cameraInfo = try feed.configure()
        } catch {
            return "camera: \(error.localizedDescription)"
        }
        camera = feed
        feed.start()
        let clock = ContinuousClock()
        let opened = clock.now
        while feed.deliveredCount == 0, clock.now - opened < .seconds(5) {
            try? await Task.sleep(for: .milliseconds(20))
        }
        guard feed.deliveredCount > 0 else { return "camera: no frame in 5 s" }
        let firstFrame = Self.round3(LookModel.seconds(clock.now - opened)) ?? 0
        cameraInfo["first_frame_s"] = firstFrame
        let fps = cameraInfo["fps"] as? Double ?? 0
        write("CAMERA running preset=hd1280x720 fps=\(fps) first_frame_s=\(firstFrame)")
        return nil
    }

    /// The newest camera frame (waits up to 3 s for one), as a CGImage that reads its pixel buffer.
    private func cameraFrame(_ feed: CameraFeed) async -> Frame? {
        let clock = ContinuousClock()
        let waited = clock.now
        var taken = feed.take()
        while taken == nil, clock.now - waited < .seconds(3) {
            try? await Task.sleep(for: .milliseconds(5))
            taken = feed.take()
        }
        guard let sample = taken?.sample else { return nil }
        let now = CMClockGetTime(feed.session.synchronizationClock ?? CMClockGetHostTimeClock())
        let ageMs = max(0, (now - CMSampleBufferGetPresentationTimeStamp(sample)).seconds * 1000)
        let start = clock.now
        guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return nil }
        if cameraInfo["buffer_px"] == nil {
            cameraInfo["buffer_px"] = [CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer)]
        }
        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &image)
        guard let image else { return nil }
        // The phone upright: the back camera's sensor lies sideways, so the picture turns right.
        return Frame(
            image: image, orientation: .right, display: nil, name: "camera", ageMs: ageMs,
            convertMs: LookModel.seconds(clock.now - start) * 1000,
            captureWall: Date().timeIntervalSince1970 - ageMs / 1000)
    }
    #endif

    private func photoFrame(_ index: Int) -> Frame? {
        guard !photos.isEmpty else { return nil }
        let clock = ContinuousClock()
        let start = clock.now
        let item = photos[(index - 1) % photos.count]
        return Frame(
            image: item.photo.cgImage, orientation: item.photo.orientation, display: item.photo.display,
            name: item.id, ageMs: 0, convertMs: LookModel.seconds(clock.now - start) * 1000,
            captureWall: Date().timeIntervalSince1970)
    }

    private func nextFrame(_ index: Int) async -> Frame? {
        #if os(iOS)
        if settings.source == "camera", let camera { return await cameraFrame(camera) }
        #endif
        return photoFrame(index)
    }

    /// The loop: one frame at a time until `-seconds` have passed (a frame started is finished).
    private func watch() async {
        let clock = ContinuousClock()
        let start = clock.now
        let count = settings.conditions.count
        phase = .watching
        detail = "\(settings.source == "camera" ? "Camera" : "Example photos") · \(count) conditions"
        var previous = Array(repeating: "?", count: count)
        var index = 0
        while clock.now - start < .seconds(settings.seconds) {
            index += 1
            let grabbed = clock.now
            guard let frame = await nextFrame(index) else {
                failure = "frame \(index): no frame from the \(settings.source)"
                break
            }
            var replies: [Reply] = []
            do {
                if settings.form == "single" {
                    for condition in settings.conditions {
                        replies.append(try await answer(frame, WatchPrompt.single(condition)))
                    }
                } else {
                    replies.append(try await answer(frame, prompt))
                }
            } catch {
                failure = "frame \(index): \(error.localizedDescription)"
                break
            }
            let parseStart = clock.now
            let answers: [String]
            if settings.form == "single" {
                answers = replies.map { WatchParse.single($0.text) }
            } else {
                answers = WatchParse.batch(replies.first?.text ?? "", count: count)
            }
            let parsed = clock.now
            let fired = answers.indices.filter { answers[$0] == "yes" && previous[$0] != "yes" }
            previous = answers
            react(fired)
            show(frame, answers)
            record(frame, index, replies, answers, fired, start: start, grabbed: grabbed, parseStart: parseStart, parsed: parsed)
        }
        loopSeconds = LookModel.seconds(clock.now - start)
    }

    /// One prompt about one frame in a new session: the answer, its first token and its end in ms
    /// from the call, and the session's token counts.
    private func answer(_ frame: Frame, _ text: String) async throws -> Reply {
        guard let model else { throw WatchError.notLoaded }
        let session = LanguageModelSession(model: model)
        let prompt = Prompt {
            Attachment(frame.image, orientation: frame.orientation)  // the image, then the prompt
            text
        }
        let clock = ContinuousClock()
        let start = clock.now
        var first: Double?
        var content = ""
        let stream = session.streamResponse(
            to: prompt, options: GenerationOptions(maximumResponseTokens: settings.maxTokens))
        for try await snapshot in stream {
            if snapshot.content.isEmpty { continue }
            if first == nil { first = LookModel.seconds(clock.now - start) * 1000 }
            content = snapshot.content
        }
        let modelMs = LookModel.seconds(clock.now - start) * 1000
        return Reply(
            text: content, firstMs: first, modelMs: modelMs, inputTokens: session.usage.input.totalTokenCount,
            outputTokens: session.usage.output.totalTokenCount)
    }

    /// The conditions that turned yes: a system sound and a haptic each, and their symbols on the
    /// picture for 1.5 s.
    private func react(_ fired: [Int]) {
        guard !fired.isEmpty else { return }
        #if os(iOS)
        let haptics = UINotificationFeedbackGenerator()
        for _ in fired {
            AudioServicesPlaySystemSound(SystemSoundID(1057))
            haptics.notificationOccurred(.success)
        }
        #endif
        withAnimation(.easeOut(duration: 0.15)) { overlay = fired }
        overlayID += 1
        let id = overlayID
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            if self.overlayID == id { withAnimation(.easeIn(duration: 0.2)) { self.overlay = [] } }
        }
    }

    /// The rows, the picture (photos) and the numbers lines for the frame just answered.
    private func show(_ frame: Frame, _ answers: [String]) {
        for k in rows.indices where answers.indices.contains(k) { rows[k].answer = answers[k] }
        if let display = frame.display { shownImage = display }
    }

    // MARK: - The record

    /// The frame's row of the record, its WATCH line, and the numbers lines on the screen.
    private func record(
        _ frame: Frame, _ index: Int, _ replies: [Reply], _ answers: [String], _ fired: [Int],
        start: ContinuousClock.Instant, grabbed: ContinuousClock.Instant, parseStart: ContinuousClock.Instant,
        parsed: ContinuousClock.Instant
    ) {
        let count = settings.conditions.count
        let tGrab = LookModel.seconds(grabbed - start)
        let tDone = LookModel.seconds(parsed - start)
        let totalMs = LookModel.seconds(parsed - grabbed) * 1000
        let toInputMs = frame.ageMs + frame.convertMs
        let modelMs = replies.reduce(0.0) { $0 + $1.modelMs }
        let inputTokens = replies.reduce(0) { $0 + $1.inputTokens }
        let outputTokens = replies.reduce(0) { $0 + $1.outputTokens }
        let firstMs = replies.first?.firstMs
        let thermal = Autoplay.thermal()
        let footprint = WatchMemory.footprintMB()

        var row: [String: Any] = [:]
        row["i"] = index
        row["source"] = frame.name
        row["t_capture_s"] = Self.round3(tGrab - frame.ageMs / 1000) ?? 0
        row["t_grab_s"] = Self.round3(tGrab) ?? 0
        row["t_done_s"] = Self.round3(tDone) ?? 0
        row["t_wall"] = Self.round3(frame.captureWall) ?? 0
        row["age_ms"] = Self.round1(frame.ageMs)
        row["convert_ms"] = Self.round1(frame.convertMs)
        row["capture_to_input_ms"] = Self.round1(toInputMs)
        if let firstMs { row["first_token_ms"] = Self.round1(firstMs) } else { row["first_token_ms"] = NSNull() }
        row["model_ms"] = Self.round1(modelMs)
        row["parse_ms"] = Self.round1(LookModel.seconds(parsed - parseStart) * 1000)
        row["total_ms"] = Self.round1(totalMs)
        row["capture_to_answer_ms"] = Self.round1(frame.ageMs + totalMs)
        row["input_tokens"] = inputTokens
        row["output_tokens"] = outputTokens
        row["answers"] = answers
        if settings.form == "single" { row["raw"] = replies.map(\.text) } else { row["raw"] = replies.first?.text ?? "" }
        row["reacted"] = fired.map { $0 + 1 }
        row["thermal"] = thermal
        row["footprint_mb"] = Self.round1(footprint)
        row["available_mb"] = Self.round1(WatchMemory.availableMB())
        var scored = "-"
        if let truth = settings.truth?[frame.name], truth.count == count {
            let correct = zip(answers, truth).filter { $0.0 == $0.1 }.count
            row["truth"] = truth
            row["correct"] = correct
            scored = "\(correct)/\(count)"
        }
        frames.append(row)

        let firstText = firstMs.map { String(Int($0.rounded())) } ?? "-"
        let reacted = fired.isEmpty ? "-" : fired.map { String($0 + 1) }.joined(separator: ",")
        let fields: [String] = [
            "WATCH frame=\(index) src=\(frame.name) t_s=\(Self.decimals(tGrab))",
            "capture_to_input_ms=\(Int(toInputMs.rounded())) first_token_ms=\(firstText)",
            "total_ms=\(Int(totalMs.rounded())) in=\(inputTokens) out=\(outputTokens)",
            "answers=\(answers.joined(separator: ",")) correct=\(scored) react=\(reacted)",
            "thermal=\(thermal) footprint_mb=\(Int(footprint.rounded()))",
        ]
        write(fields.joined(separator: " "))

        let firstShown = firstMs.map { String(Int($0.rounded())) } ?? "–"
        msLine = "capture→input \(Int(toInputMs.rounded())) ms · first token \(firstShown) ms · total \(Int(totalMs.rounded())) ms"
        let elapsed = String(format: "%.0f", tDone)
        let planned = String(format: "%.0f", settings.seconds)
        stateLine = "frame \(index) · \(elapsed) / \(planned) s · thermal \(thermal) · \(Int(footprint.rounded())) MB"
    }

    private func finish(_ failure: String?) {
        #if os(iOS)
        camera?.stop()
        UIApplication.shared.isIdleTimerDisabled = false
        #endif
        let file = writeRecord(failure)
        if let failure {
            phase = .failed
            detail = failure
            write("ERROR \(failure) json=\(file)")
        } else {
            phase = .done
            detail = "Done: \(frames.count) frames"
            let stats = Self.stats(frames.compactMap { $0["total_ms"] as? Double })
            let mean = stats.map { String(describing: $0["mean"] ?? "-") } ?? "-"
            let p50 = stats.map { String(describing: $0["p50"] ?? "-") } ?? "-"
            let maximum = stats.map { String(describing: $0["max"] ?? "-") } ?? "-"
            let correct = frames.compactMap { $0["correct"] as? Int }.reduce(0, +)
            let judged = frames.filter { $0["correct"] != nil }.count * settings.conditions.count
            let fields: [String] = [
                "DONE stop=done frames=\(frames.count)",
                "mean_total_ms=\(mean) p50_total_ms=\(p50) max_total_ms=\(maximum)",
                "correct=\(judged > 0 ? "\(correct)/\(judged)" : "-") json=\(file)",
            ]
            write(fields.joined(separator: " "))
        }
        if settings.quit { exit(failure == nil ? 0 : 1) }
    }

    /// Documents/watch-run-<epoch>.json (only with `-log 1`).
    private func writeRecord(_ failure: String?) -> String {
        let name = "watch-run-\(epoch).json"
        guard settings.log else { return name }
        var object: [String: Any] = [:]
        func put(_ key: String, _ value: Any?) { object[key] = value ?? NSNull() }
        let count = settings.conditions.count
        let now = LookDevice.sinceStart()

        put("schema", "watch-run/1")
        put("mode", "watch")
        put("epoch", epoch)
        put("args", Array(CommandLine.arguments.dropFirst()))
        put("model", settings.modelID)
        put("repo", entry?.repo)
        put("revision", entry?.revision)
        let vision = settings.modelID == "minicpm-v-4.7-1b" ? VLModelID.miniCPMV47.vision.path ?? "?" : "?"
        put("bundles", "\(entry?.variant?.path ?? "?") + \(vision)")
        put("device", LookDevice.machine)
        put("device_name", LookDevice.name)
        put("os", "\(LookDevice.osVersion) (\(LookDevice.osBuild))")
        put("app", "Look \(LookDevice.appVersion) (\(LookDevice.appBuild))")
        put("source", settings.source)
        put("seconds", settings.seconds)
        put("conditions", settings.conditions)
        put("symbols", settings.symbols)
        put("prompt_form", settings.form)
        if settings.form == "single" {
            put("prompt", settings.conditions.map { WatchPrompt.single($0) })
        } else {
            put("prompt", prompt)
        }
        put("max_tokens", settings.maxTokens)
        put("truth", settings.truth)
        if settings.source == "camera" {
            put("camera", cameraInfo)
            put("camera_frames_delivered", camera?.deliveredCount)
        }
        put("thermal_start", thermalStart)
        put("thermal_end", Autoplay.thermal())
        put("free_bytes_start", freeStart)
        put("free_bytes_end", Autoplay.freeBytes())
        put("catalog_s", Self.round3(catalogSeconds))
        put("engine_ready_s", Self.round3(engineReadySeconds))
        put("prewarm_s", Self.round3(prewarmSeconds))
        put("t_ready_s", Self.round3(readySinceStart))
        put("loop_s", Self.round3(loopSeconds))
        put("stop", failure == nil ? "done" : "error")
        put("error", failure)
        put("frames", frames)

        var summary: [String: Any] = ["frames": frames.count]
        for key in ["total_ms", "first_token_ms", "model_ms", "capture_to_input_ms", "capture_to_answer_ms"] {
            let values = frames.compactMap { $0[key] as? Double }
            if let stats = Self.stats(values) { summary[key] = stats } else { summary[key] = NSNull() }
        }
        let answers = frames.compactMap { $0["answers"] as? [String] }
        summary["yes"] = (0..<count).map { k in answers.filter { $0.indices.contains(k) && $0[k] == "yes" }.count }
        summary["unparsed"] = answers.reduce(0) { total, row in total + row.filter { $0 == "?" }.count }
        summary["reactions"] = frames.reduce(0) { total, row in total + ((row["reacted"] as? [Int])?.count ?? 0) }
        let scored = frames.filter { $0["correct"] != nil }
        if !scored.isEmpty {
            summary["correct"] = scored.reduce(0) { total, row in total + (row["correct"] as? Int ?? 0) }
            summary["judged"] = scored.count * count
            // Each photo's first frame (each photo's answer once), and whether its later frames said the same.
            var firstSeen: [String: [String]] = [:]
            var once: [String: Int] = [:]
            var same = true
            for row in scored {
                guard let source = row["source"] as? String, let got = row["answers"] as? [String] else { continue }
                if let seen = firstSeen[source] {
                    if seen != got { same = false }
                } else {
                    firstSeen[source] = got
                    once[source] = row["correct"] as? Int ?? 0
                }
            }
            summary["correct_first_frame_of_each_photo"] = once
            summary["same_answers_for_the_same_photo"] = same
        }
        let footprints = frames.compactMap { $0["footprint_mb"] as? Double }.filter { $0 >= 0 }
        if let low = footprints.min(), let high = footprints.max() {
            summary["footprint_mb"] = ["min": low, "max": high]
        }
        var thermals: [String: Int] = [:]
        for row in frames { if let state = row["thermal"] as? String { thermals[state, default: 0] += 1 } }
        summary["thermal_frames"] = thermals
        put("summary", summary)

        let stretches = sceneLog.summary(now: now)
        put("inactive_count", stretches.count)
        put("inactive_s", Self.round3(stretches.seconds))
        put("inactive_at", stretches.rows)
        put("scene_first_active_s", Self.round3(sceneLog.firstActive))
        put("t_record_s", Self.round3(now))

        let url = URL.documentsDirectory.appending(path: name)
        if let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        {
            try? data.write(to: url)
        }
        return name
    }

    // MARK: - Helpers

    /// Appends one timestamped line to Documents/look-autoplay.log (only with `-log 1`).
    private func write(_ line: String) {
        guard settings.log else { return }
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

    /// n, mean, the median (p50), min and max; nil for no values.
    private static func stats(_ values: [Double]) -> [String: Any]? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let n = sorted.count
        let median = n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        let mean = values.reduce(0, +) / Double(n)
        return ["n": n, "mean": round1(mean), "p50": round1(median), "min": round1(sorted[0]), "max": round1(sorted[n - 1])]
    }

    private static func decimals(_ value: Double?) -> String {
        value.map { String(format: "%.3f", $0) } ?? "-"
    }

    private static func round1(_ value: Double) -> Double { (value * 10).rounded() / 10 }

    private static func round3(_ value: Double?) -> Double? { Autoplay.round3(value) }

    #if os(iOS)
    private static func name(_ status: AVAuthorizationStatus) -> String {
        switch status {
        case .authorized: "authorized"
        case .denied: "denied"
        case .notDetermined: "notDetermined"
        case .restricted: "restricted"
        @unknown default: "unknown"
        }
    }
    #endif
}

extension WatchModel.Phase {
    var label: String {
        switch self {
        case .loading: "LOADING"
        case .ready: "READY"
        case .watching: "WATCHING"
        case .done: "DONE"
        case .failed: "FAILED"
        }
    }

    var color: Color {
        switch self {
        case .loading: Color(white: 0.36)
        case .ready: Palette.accent
        case .watching: Color(red: 0.84, green: 0.50, blue: 0.08)
        case .done: Color(red: 0.16, green: 0.50, blue: 0.22)
        case .failed: Color(red: 0.78, green: 0.20, blue: 0.20)
        }
    }
}

/// The watch mode's screen, dark, top to bottom: the state, the picture (the live camera, or the
/// example photo last answered) with the reactions over it, one row per condition with its answer,
/// the latest frame's times, and the model and device it ran on.
struct WatchView: View {
    @State private var model = WatchModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            statusRow
            picture
            VStack(spacing: 4) {
                ForEach(model.rows.indices, id: \.self) { index in conditionRow(model.rows[index]) }
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(model.msLine)
                    .font(.system(size: 18, weight: .bold))
                    .foregroundStyle(.white)
                Text(model.stateLine)
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Palette.secondary)
            }
            .lineLimit(1)
            .minimumScaleFactor(0.6)
            .accessibilityIdentifier("watch.numbers")
            Text(model.footer)
                .font(.system(size: 11))
                .foregroundStyle(Palette.secondary)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Palette.background.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onChange(of: scenePhase, initial: true) { _, phase in model.scene(phase) }
        .task { await model.start() }
    }

    private var statusRow: some View {
        HStack(spacing: 10) {
            Text(model.phase.label)
                .font(.system(size: 15, weight: .heavy))
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 4)
                .background(Capsule().fill(model.phase.color))
            Text(model.detail)
                .font(.system(size: 16))
                .foregroundStyle(Palette.secondary)
                .lineLimit(1)
        }
        .frame(height: 28)
    }

    private var picture: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12).fill(Palette.surface)
            #if os(iOS)
            if let camera = model.camera {
                CameraPreview(session: camera.session)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            #endif
            if let image = model.shownImage {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            if !model.overlay.isEmpty {
                reaction
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: 330)
    }

    private var reaction: some View {
        VStack(spacing: 8) {
            HStack(spacing: 18) {
                ForEach(model.overlay, id: \.self) { index in
                    Image(systemName: model.rows[index].symbol)
                        .font(.system(size: 88, weight: .semibold))
                }
            }
            Text(model.overlay.map { model.rows[$0].condition }.joined(separator: " · "))
                .font(.system(size: 20, weight: .bold))
                .lineLimit(2)
                .multilineTextAlignment(.center)
        }
        .foregroundStyle(.white)
        .padding(22)
        .background(RoundedRectangle(cornerRadius: 22).fill(Color.black.opacity(0.6)))
        .transition(.opacity)
        .accessibilityIdentifier("watch.reaction")
    }

    private func conditionRow(_ row: WatchModel.Row) -> some View {
        HStack(spacing: 12) {
            Image(systemName: row.symbol)
                .font(.system(size: 22))
                .foregroundStyle(.white)
                .frame(width: 34)
            Text(row.condition)
                .font(.system(size: 19))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Spacer(minLength: 8)
            Text(row.answer.uppercased())
                .font(.system(size: 17, weight: .heavy))
                .foregroundStyle(.white)
                .frame(minWidth: 52)
                .padding(.vertical, 3)
                .background(Capsule().fill(color(row.answer)))
        }
        .frame(height: 40)
        .padding(.horizontal, 10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Palette.surface))
    }

    private func color(_ answer: String) -> Color {
        switch answer {
        case "yes": Color(red: 0.16, green: 0.55, blue: 0.24)
        case "?": Color(red: 0.84, green: 0.50, blue: 0.08)
        default: Palette.surfaceHigh
        }
    }
}

/// The back camera for the watch loop: the newest frame only (a frame not taken before the next one
/// arrives is dropped, and the output drops the frames that come while the delegate is busy).
final class CameraFeed: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "look.watch.camera")
    private let lock = NSLock()
    private var newest: CMSampleBuffer?
    private var delivered = 0

    /// The back wide-angle camera at 1280 × 720, 32-bit BGRA, into this feed; returns what it set.
    func configure() throws -> [String: Any] {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw WatchError.camera("no back camera")
        }
        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        guard session.canAddInput(input), session.canAddOutput(output) else {
            session.commitConfiguration()
            throw WatchError.camera("the capture session cannot take the back camera")
        }
        session.addInput(input)
        session.addOutput(output)
        session.commitConfiguration()
        let frame = device.activeVideoMinFrameDuration
        let fps = frame.seconds > 0 ? (10 / frame.seconds).rounded() / 10 : 0
        return [
            "camera": "back wide-angle", "preset": "hd1280x720", "pixel_format": "32BGRA", "fps": fps,
            "orientation": "right",
        ]
    }

    func start() { queue.async { self.session.startRunning() } }

    func stop() { queue.async { self.session.stopRunning() } }

    /// The newest frame since the last take, and how many frames the camera delivered so far.
    func take() -> (sample: CMSampleBuffer, delivered: Int)? {
        lock.lock()
        defer { lock.unlock() }
        guard let sample = newest else { return nil }
        newest = nil
        return (sample, delivered)
    }

    var deliveredCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return delivered
    }

    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection
    ) {
        lock.lock()
        newest = sampleBuffer
        delivered += 1
        lock.unlock()
    }
}

#if os(iOS)
/// The live camera behind the condition rows (the frames the model reads come from the same session).
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        override func layoutSubviews() {
            super.layoutSubviews()
            // The phone upright: turn the sideways sensor's picture to stand up.
            if let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90
            }
        }
    }
}
#endif
