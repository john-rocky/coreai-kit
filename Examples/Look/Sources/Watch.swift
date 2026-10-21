// Watch — Look's second screen: write what to look for, one rule per line, pick what happens when
// the model sees it, and Start. Frames from the back camera, a video you pick, or the three example
// photos go to the model one at a time; every rule is asked in the same turn, and a rule whose
// answer turns yes fires its reaction (WatchRules.swift) with its SF Symbol over the picture.
// The model is the one the Ask screen loaded (LookModel): it loads once per launch.
//
// One frame at a time: when an answer is in, the source's newest frame is taken (camera frames that
// came meanwhile are dropped), cropped to what the preview shows, and sent to a new session (no
// history, greedy, at most `-tokens` answer tokens, default 32) with one prompt: "Answer each line
// with yes or no, nothing else." and a line "<i>. Is there <condition>?" per rule. The answer is
// read line by line (WatchParse). A rule fires when its answer turns yes (it was not yes on the
// frame before) and it has not fired in the last 3 s; a yes that stays yes does not fire again. At
// the thermal state fair the loop starts a frame at most every 2 s, at serious every 4 s.
//
// The example photos come in turn — sign, object, scene, sign … — each on the screen for at least
// `-hold` seconds (default 4.5): the next one comes with the first frame taken after that. That is
// long enough for a reaction to show its 3 s before the picture changes. A video (picked, or
// `-file` for a script) waits on its first frame; Start plays it from its start, muted, and the run
// ends when the video does.
//
// Start and Stop are start() and stop(); Autoplay.swift (WatchAutoplay) calls the same two for a
// recording or a check. With `-log 1`, every run — Start to Stop — appends PRESS / WATCH (one per
// frame) / FIRE (one per reaction) / DONE lines to Documents/look-autoplay.log and writes
// Documents/watch-run-<epoch>.json (schema watch-run/2).

import AVFoundation
import AudioToolbox
import CoreAIKit
import CoreMedia
import Foundation
import FoundationModels
import ImageIO
import Observation
import PhotosUI
import SwiftUI
import VideoToolbox
import os
#if canImport(Darwin)
import Darwin
#endif
#if os(iOS)
import UIKit
#endif

/// The watch screen's launch arguments (`-key value`, read from UserDefaults as Autoplay reads its own).
struct WatchSettings {
    /// `-watch 1`: the app opens on the watch screen (and logs its START and READY lines).
    let opened: Bool
    let source: WatchModel.Source
    /// `-seconds`: WatchAutoplay presses Start, watches this long, then presses Stop.
    let seconds: Double?
    /// `-rules` as given (kept in the record).
    let rulesArgument: String?
    let rules: [WatchRule]
    /// Rules edited on the screen are kept between launches unless the launch named its own.
    let persist: Bool
    let maxTokens: Int
    /// `-truth "sign:no,no,yes;…"`: what each example photo should get, rule by rule.
    let truth: [String: [String]]?
    let hold: Double
    let cooldown: Double
    let permissionWait: Double
    let log: Bool
    let quit: Bool
    let take: Bool
    let frames: Bool
    let trigger: String?
    let delay: Double
    let gap: Double
    /// `-file <path>` (or `-video <name>`): a video file opened as the video source — a path in the
    /// app's container ("Documents/clip.mp4"), a name in Documents, or an absolute path.
    let video: String?
    /// `-edit new|<n>`: open the rule editor at READY; `-add "cond=reaction"`: a rule to add.
    let edit: String?
    let add: String?
    let modelID: String

    init(_ defaults: UserDefaults = .standard) {
        opened = defaults.bool(forKey: "watch")
        #if os(iOS)
        source = WatchModel.Source(rawValue: defaults.string(forKey: "source") ?? "") ?? .camera
        #else
        source = WatchModel.Source(rawValue: defaults.string(forKey: "source") ?? "") ?? .photos
        #endif
        seconds = defaults.object(forKey: "seconds") != nil ? defaults.double(forKey: "seconds") : nil
        let symbols = Self.list(defaults.string(forKey: "symbols"))
        let rulesText = defaults.string(forKey: "rules")
        rulesArgument = rulesText
        let conditions = Self.list(defaults.string(forKey: "conditions"))
        let given = rulesText.map { WatchRules.parse($0, symbols: symbols) } ?? []
        if !given.isEmpty {
            rules = given
            persist = false
        } else if !conditions.isEmpty {
            rules = WatchRules.fromConditions(conditions, symbols: symbols)
            persist = false
        } else {
            rules = WatchRules.saved() ?? WatchRules.defaults
            persist = true
        }
        maxTokens = defaults.object(forKey: "tokens") != nil ? max(1, defaults.integer(forKey: "tokens")) : 32
        truth = defaults.string(forKey: "truth").map(Self.parseTruth)
        hold = defaults.object(forKey: "hold") != nil ? max(0, defaults.double(forKey: "hold")) : 4.5
        cooldown = defaults.object(forKey: "cooldown") != nil ? max(0, defaults.double(forKey: "cooldown")) : 3
        permissionWait = defaults.object(forKey: "permwait") != nil ? defaults.double(forKey: "permwait") : 600
        log = defaults.bool(forKey: "log")
        quit = defaults.bool(forKey: "quit")
        take = defaults.bool(forKey: "take")
        frames = defaults.bool(forKey: "frames")
        trigger = defaults.string(forKey: "trigger")
        delay = defaults.object(forKey: "delay") != nil ? max(0, defaults.double(forKey: "delay")) : 1
        gap = defaults.object(forKey: "gap") != nil ? max(0, defaults.double(forKey: "gap")) : 2
        video = defaults.string(forKey: "file") ?? defaults.string(forKey: "video")
        edit = defaults.string(forKey: "edit")
        add = defaults.string(forKey: "add")
        modelID = defaults.string(forKey: "model") ?? "minicpm-v-4.7-1b"
    }

    /// `-file`'s path as a URL: "Documents/clip.mp4" in the app's container, "clip.mp4" in Documents,
    /// "/…" as it is.
    static func videoURL(_ path: String) -> URL {
        if path.hasPrefix("/") { return URL(fileURLWithPath: path) }
        if path.hasPrefix("Documents/") { return URL.documentsDirectory.deletingLastPathComponent().appending(path: path) }
        return URL.documentsDirectory.appending(path: path)
    }

    /// "a|b|c" → ["a", "b", "c"], each trimmed, the empty ones dropped.
    private static func list(_ text: String?) -> [String] {
        (text ?? "").split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    /// "sign:no,no,yes;object:yes,no,no" → ["sign": ["no", "no", "yes"], …]
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

/// The prompt, built from the rules' conditions.
enum WatchPrompt {
    static let instruction = "Answer each line with yes or no, nothing else."

    /// The instruction, then "<i>. Is there <condition>?" per line.
    static func batch(_ conditions: [String]) -> String {
        ([instruction] + conditions.enumerated().map { "\($0.offset + 1). Is there \($0.element)?" })
            .joined(separator: "\n")
    }
}

/// Reads yes / no out of an answer.
enum WatchParse {
    /// One value per condition. A line that starts with a number k answers condition k; a line
    /// without a number answers the first condition still open; a line's value is its first word
    /// "yes" or "no". If conditions are still open and the whole answer holds exactly one such word
    /// per condition, those words answer them in order. Open = "?".
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
    case video(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded: "the model is not loaded"
        case .camera(let reason): reason
        case .video(let reason): reason
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
        case loading, ready, watching, paused, failed
    }

    enum Source: String, CaseIterable, Identifiable {
        case camera, video, photos

        var id: String { rawValue }

        var title: String {
            switch self {
            case .camera: "Camera"
            case .video: "Video"
            case .photos: "Example photos"
            }
        }
    }

    /// One rule's state on the screen.
    struct RowState {
        var answer = "–"
        var count = 0
        var timerEnd: Date?
    }

    /// The reaction on the picture: the fired rules' symbols, a large line, a small line.
    struct Overlay: Equatable {
        let id: Int
        let symbols: [String]
        let title: String
        let subtitle: String
        let speech: Bool
    }

    /// The rule the editor sheet is open for.
    struct EditTarget: Identifiable {
        let id = UUID()
        let rule: WatchRule
        let isNew: Bool
    }

    /// One frame for the model: the image as the model gets it, what the screen shows, and its times.
    struct Frame {
        let image: CGImage
        let orientation: CGImagePropertyOrientation
        let name: String
        let ageMs: Double
        let convertMs: Double
        let captureWall: Double
        /// When the picture on the screen changed to this one (the example photos), on the loop's clock.
        let switchedAt: ContinuousClock.Instant?
        let crop: CGRect?
        let videoTime: Double?
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
    var rules: [WatchRule]
    var rowStates: [UUID: RowState] = [:]
    var source: Source
    /// The example photo on the screen (Example photos).
    var shownImage: CGImage?
    private(set) var camera: CameraFeed?
    private(set) var video: VideoFeed?
    var overlay: Overlay?
    var logLines: [String] = []
    var editing: EditTarget?
    var numbersHead = "first token – s · total – s"
    var numbersTail = "GPU · on-device · frame –"
    /// Stop was pressed; the frame in flight finishes first.
    private(set) var stopping = false

    // Read by WatchAutoplay and the record; the screen does not draw from them.
    @ObservationIgnored var bandRect: CGRect = .zero
    @ObservationIgnored var previewSize: CGSize = .zero
    @ObservationIgnored var onFire: (@MainActor (Int) -> Void)?
    @ObservationIgnored var beforeRecord: (@MainActor () async -> [String: Any])?
    @ObservationIgnored private(set) var launchEpoch = 0
    @ObservationIgnored private(set) var lastFailure: String?
    @ObservationIgnored private(set) var loopTask: Task<Void, Never>?
    @ObservationIgnored private(set) var opening = false

    @ObservationIgnored private var engine: LookModel?
    @ObservationIgnored private var model: KitVisionModel?
    @ObservationIgnored private var attached = false
    @ObservationIgnored private var visible = false
    @ObservationIgnored private var photos: [(id: String, photo: PickedPhoto)] = []
    @ObservationIgnored private var photoIndex = 0
    @ObservationIgnored private var photoSince: ContinuousClock.Instant?
    @ObservationIgnored private var cameraInfo: [String: Any] = [:]
    @ObservationIgnored private var cameraDenied = false
    @ObservationIgnored private var permission: Bool?
    @ObservationIgnored private var previous: [UUID: String] = [:]
    @ObservationIgnored private var lastFire: [UUID: ContinuousClock.Instant] = [:]
    @ObservationIgnored private var overlayID = 0
    @ObservationIgnored private var sceneLog = SceneLog()
    @ObservationIgnored private var run = Run()
    private let speech = AVSpeechSynthesizer()

    /// What one run (Start to Stop) keeps for its record.
    private struct Run {
        var epoch = 0
        var clock = ContinuousClock()
        var start: ContinuousClock.Instant?
        var startedSinceStart: Double?
        var thermalStart = "-"
        var freeStart: Int64?
        var rules: [WatchRule] = []
        var prompt = ""
        var truth: [String: [String]]?
        var frames: [[String: Any]] = []
        var fires: [[String: Any]] = []
        var suppressed: [[String: Any]] = []
        var timers: [[String: Any]] = []
        var presses: [[String: Any]] = []
        var loopSeconds: Double?
        var failure: String?
        var stopRequested = false
        /// The video source played to its end, which ended the run.
        var endedByVideo = false
    }

    init() {
        let settings = WatchSettings()
        self.settings = settings
        rules = settings.rules
        source = settings.source
    }

    var prompt: String { WatchPrompt.batch(rules.map(\.condition)) }

    var canStart: Bool {
        guard model != nil, !rules.isEmpty, phase == .ready || phase == .paused else { return false }
        switch source {
        case .camera: return camera != nil
        case .video: return video != nil
        case .photos: return !photos.isEmpty
        }
    }

    var showsLog: Bool { rules.contains { $0.reaction == .log } }

    var footer: String { engine?.footer ?? settings.modelID }

    /// Both numbers lines as one string, for the record and for VoiceOver.
    var numbersLine: String { "\(numbersHead) · \(numbersTail)" }

    // MARK: - Load and sources

    /// Waits for the model the Ask screen loads, then opens the source when the screen shows.
    /// Called once, when the app starts; `visible` = the app opened on this screen.
    func attach(_ engine: LookModel, visible: Bool) async {
        guard !attached else { return }
        attached = true
        self.engine = engine
        self.visible = visible
        keepAwake()
        launchEpoch = Int(Date().timeIntervalSince1970)
        if settings.opened {
            let fields: [String] = [
                "START epoch=\(launchEpoch) mode=watch source=\(source.rawValue)",
                "seconds=\(settings.seconds.map { Self.decimals($0) } ?? "-") rules=\(rules.count)",
                "take=\(settings.take) frames=\(settings.frames) thermal=\(Autoplay.thermal())",
                "free_bytes=\(Autoplay.freeBytes().map { String($0) } ?? "-")",
            ]
            note(fields.joined(separator: " "))
        }
        loadPhotos()
        while engine.phase == .loading {
            detail = engine.detail
            try? await Task.sleep(for: .milliseconds(50))
        }
        guard let vlm = engine.loadedModel else {
            phase = .failed
            detail = engine.detail
            note("ERROR load: \(engine.detail) json=-")
            return
        }
        model = vlm
        phase = .ready
        if settings.opened {
            note(
                "READY model=\(settings.modelID) engine_ready_s=\(Self.decimals(engine.engineReadySeconds))"
                    + " t_ready_s=\(Self.decimals(engine.readySinceStart))")
        }
        if source == .video, let file = settings.video {
            await openVideo(url: WatchSettings.videoURL(file))
        }
        if self.visible { await show() }
        readyDetail()
    }

    /// The screen came into view (or went out of it): the camera runs and the video plays only
    /// while it shows.
    func setVisible(_ shown: Bool) {
        guard shown != visible else { return }
        visible = shown
        keepAwake()
        if shown {
            Task { await show() }
        } else {
            camera?.stop()
            video?.pause()
        }
    }

    /// The screen stays on while this screen shows (you point the phone, you do not touch it) and
    /// while a run goes on.
    private func keepAwake() {
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = visible || phase == .watching
        #endif
    }

    /// A tap on a source.
    func select(_ next: Source) {
        guard phase != .watching, !opening else { return }
        if next != .camera { camera?.stop() }
        if next != .video { video?.pause() }
        source = next
        Task { await show() }
    }

    /// A video from Pick a video.
    func openVideo(_ item: PhotosPickerItem) async {
        guard phase != .watching else { return }
        detail = "Opening the video…"
        do {
            guard let movie = try await item.loadTransferable(type: PickedMovie.self) else {
                detail = "Could not read that video"
                return
            }
            await openVideo(url: movie.url)
        } catch {
            detail = "Could not read that video: \(error.localizedDescription)"
        }
    }

    private func openVideo(url: URL) async {
        opening = true
        defer { opening = false }
        do {
            let feed = try await VideoFeed.open(url)
            video?.close()
            video = feed
            camera?.stop()
            source = .video
            let size = "\(Int(feed.naturalSize.width))x\(Int(feed.naturalSize.height))"
            note("VIDEO file=\(feed.name) duration_s=\(Self.decimals(feed.duration)) size=\(size) angle=\(feed.angle)")
            if !visible { feed.pause() }
            readyDetail()
        } catch {
            detail = "Could not open that video: \(error.localizedDescription)"
            note("VIDEO error=\(error.localizedDescription)")
        }
    }

    /// Opens what the source needs and shows it.
    private func show() async {
        switch source {
        case .photos:
            if shownImage == nil, let first = photos.first { shownImage = first.photo.display }
        case .video:
            break  // paused on its first frame until Start plays it from the start
        case .camera:
            #if os(iOS)
            if let camera {
                camera.start()
            } else if !opening, !cameraDenied {
                opening = true
                if let error = await openCamera() {
                    detail = error
                    note("CAMERA error=\(error)")
                }
                opening = false
            }
            #else
            detail = "The camera source is on iPhone"
            #endif
        }
        if phase == .ready || phase == .paused { readyDetail() }
    }

    private func loadPhotos() {
        guard photos.isEmpty else { return }
        for example in LookModel.examples {
            guard let url = example.url, let data = try? Data(contentsOf: url),
                let photo = PickedPhoto(data: data, source: "example \(example.id)")
            else { continue }
            photos.append((id: example.id, photo: photo))
        }
        if source == .photos { shownImage = photos.first?.photo.display }
    }

    #if os(iOS)
    private func openCamera() async -> String? {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        note("CAMERA status=\(Self.name(status))")
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
            note("CAMERA granted=\(answered) after_s=\(Self.decimals(LookModel.seconds(clock.now - asked)))")
            guard let permission else {
                return "camera: no answer to the permission request in \(Int(settings.permissionWait)) s"
            }
            guard permission else {
                cameraDenied = true
                return "Camera access is off: Settings › Privacy › Camera"
            }
        default:
            cameraDenied = true
            return "Camera access is off: Settings › Privacy › Camera"
        }
        let feed = CameraFeed()
        do {
            cameraInfo = try feed.configure()
        } catch {
            return "camera: \(error.localizedDescription)"
        }
        feed.start()
        let clock = ContinuousClock()
        let opened = clock.now
        while feed.deliveredCount == 0, clock.now - opened < .seconds(5) {
            try? await Task.sleep(for: .milliseconds(20))
        }
        guard feed.deliveredCount > 0 else { return "camera: no frame in 5 s" }
        camera = feed
        if !visible || source != .camera { feed.stop() }
        let firstFrame = Self.round3(LookModel.seconds(clock.now - opened)) ?? 0
        cameraInfo["first_frame_s"] = firstFrame
        let fps = cameraInfo["fps"] as? Double ?? 0
        note("CAMERA running preset=hd1280x720 fps=\(fps) first_frame_s=\(firstFrame)")
        return nil
    }
    #endif

    private func readyDetail() {
        guard phase == .ready || phase == .paused else { return }
        if rules.isEmpty {
            detail = "Add a rule first"
            return
        }
        if phase == .paused, !run.frames.isEmpty {
            let fires = run.fires.count
            let how = run.endedByVideo ? "The video ended" : "Stopped"
            detail = "\(how) · \(run.frames.count) frames · \(fires) \(fires == 1 ? "reaction" : "reactions")"
            return
        }
        switch source {
        case .camera:
            detail = camera == nil ? (cameraDenied ? "Camera access is off: Settings › Privacy › Camera" : "Opening the camera…") : "Camera · Start to watch"
        case .video:
            detail = video == nil ? "Pick a video, then Start" : "Video · Start to watch"
        case .photos:
            detail = photos.isEmpty ? "The example photos are not in this build: see the README" : "Example photos · Start to watch"
        }
    }

    /// The scene's phase changed: keep the stretches when the app was not active.
    func scene(_ phase: ScenePhase) {
        if let line = sceneLog.changed(to: phase, at: LookDevice.sinceStart()) { note(line) }
    }

    // MARK: - Rules

    /// A tap on a rule's row: the editor for it.
    func edit(_ rule: WatchRule) {
        guard phase != .watching else { return }
        editing = EditTarget(rule: rule, isNew: false)
    }

    /// "+ Add a rule": the editor for a new one (`prefill` = autoplay's `-add`).
    func addRule(prefill: WatchRule? = nil) {
        guard phase != .watching, rules.count < WatchRules.maxCount else { return }
        let rule =
            prefill
            ?? WatchRule(condition: "", reaction: .sound(.chime), symbol: WatchRules.newSymbol)
        editing = EditTarget(rule: rule, isNew: true)
    }

    /// The editor's Done: the rule replaces the one with its id, or is added.
    func save(_ rule: WatchRule) {
        var rule = rule
        rule.condition = rule.condition.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rule.condition.isEmpty else { return }
        if let index = rules.firstIndex(where: { $0.id == rule.id }) {
            if rules[index].reaction != rule.reaction { rowStates[rule.id]?.count = 0 }
            rules[index] = rule
        } else if rules.count < WatchRules.maxCount {
            rules.append(rule)
        }
        editing = nil
        if settings.persist { WatchRules.save(rules) }
        note("RULES n=\(rules.count) saved=\(rule.condition)=\(rule.reaction.token) symbol=\(rule.symbol)")
        readyDetail()
    }

    /// The editor's Delete.
    func delete(_ id: UUID) {
        rules.removeAll { $0.id == id }
        rowStates[id] = nil
        editing = nil
        if settings.persist { WatchRules.save(rules) }
        note("RULES n=\(rules.count) deleted")
        readyDetail()
    }

    /// The editor's Play it.
    func preview(_ sound: WatchSound) { AudioServicesPlaySystemSound(sound.systemSound) }

    // MARK: - Start and Stop

    /// Start: the loop runs until stop(). Returns false (and says why) when it cannot start.
    @discardableResult
    func start() -> Bool {
        guard canStart else {
            if rules.isEmpty { detail = "Add a rule first" }
            note("PRESS start refused: \(detail)")
            return false
        }
        run = Run()
        run.epoch = Int(Date().timeIntervalSince1970)
        run.startedSinceStart = LookDevice.sinceStart()
        run.thermalStart = Autoplay.thermal()
        run.freeStart = Autoplay.freeBytes()
        run.rules = rules
        run.prompt = prompt
        run.truth = truth(for: rules)
        run.presses.append(["press": "start", "t_s": Self.round3(run.startedSinceStart) ?? 0])
        previous = [:]
        lastFire = [:]
        for rule in rules { rowStates[rule.id, default: RowState()].answer = "–" }
        stopping = false
        phase = .watching
        detail = "\(source.title) · \(rules.count) \(rules.count == 1 ? "rule" : "rules")"
        if source == .video { video?.restart() }
        note("PRESS start source=\(source.rawValue) rules=\(rules.map { "\($0.condition)=\($0.reaction.token)" }.joined(separator: "|"))")
        keepAwake()
        #if os(iOS)
        if rules.contains(where: { $0.reaction.kind == .say }) {
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .voicePrompt, options: [.mixWithOthers])
            try? AVAudioSession.sharedInstance().setActive(true)
        }
        #endif
        loopTask = Task { await self.loop() }
        return true
    }

    /// Stop: the frame in flight finishes, then the run ends (and its record is written).
    func stop() {
        guard phase == .watching, !run.stopRequested else { return }
        run.stopRequested = true
        stopping = true
        run.presses.append(["press": "stop", "t_s": Self.round3(LookDevice.sinceStart()) ?? 0])
        note("PRESS stop")
        detail = "Stopping after this frame…"
    }

    /// Waits for the run to end (its record written).
    func finished() async { await loopTask?.value }

    /// What each example photo should show, rule by rule (`-truth`, else the known conditions).
    private func truth(for rules: [WatchRule]) -> [String: [String]]? {
        if let given = settings.truth, given.values.allSatisfy({ $0.count == rules.count }) { return given }
        return WatchRules.truth(rules)
    }

    // MARK: - The loop

    private func loop() async {
        let clock = run.clock
        let start = clock.now
        run.start = start
        let count = run.rules.count
        var index = 0
        var lastGrab: ContinuousClock.Instant?
        photoSince = nil
        while !run.stopRequested {
            if videoEnded { break }
            // The thermal pace: at fair a frame starts at most every 2 s, at serious every 4 s.
            let pace = Self.pace(Autoplay.thermal())
            var waitedMs = 0.0
            if pace > 0, let lastGrab {
                let next = lastGrab + .seconds(pace)
                let before = clock.now
                while clock.now < next, !run.stopRequested { try? await Task.sleep(for: .milliseconds(20)) }
                waitedMs = LookModel.seconds(clock.now - before) * 1000
                if run.stopRequested || videoEnded { break }
            }
            index += 1
            let grabbed = clock.now
            lastGrab = grabbed
            guard let frame = await nextFrame(index, at: grabbed) else {
                run.failure = "frame \(index): no frame from the \(source.title.lowercased())"
                break
            }
            let reply: Reply
            do {
                reply = try await answer(frame, run.prompt)
            } catch {
                run.failure = "frame \(index): \(error.localizedDescription)"
                break
            }
            let parseStart = clock.now
            let answers = WatchParse.batch(reply.text, count: count)
            let parsed = clock.now
            let (fired, suppressed) = fires(answers, at: parsed)
            show(answers)
            let firedAt = clock.now
            react(fired)
            record(
                frame, index, reply, answers, fired, suppressed, pace: pace, waitedMs: waitedMs, start: start,
                grabbed: grabbed, parseStart: parseStart, parsed: parsed, firedAt: firedAt)
        }
        run.loopSeconds = LookModel.seconds(clock.now - start)
        await finishRun()
    }

    /// The video source has played to its end: the run ends with it.
    private var videoEnded: Bool {
        guard source == .video, video?.ended == true else { return false }
        run.endedByVideo = true
        return true
    }

    /// The seconds between frame starts at a thermal state (0 = the next frame right away).
    static func pace(_ thermal: String) -> Double {
        switch thermal {
        case "fair": 2
        case "serious", "critical": 4
        default: 0
        }
    }

    private func nextFrame(_ index: Int, at grabbed: ContinuousClock.Instant) async -> Frame? {
        switch source {
        case .camera:
            #if os(iOS)
            if let camera { return await cameraFrame(camera) }
            #endif
            return nil
        case .video:
            return await videoFrame()
        case .photos:
            return photoFrame(at: grabbed)
        }
    }

    private var aspect: Double {
        previewSize.height > 0 ? Double(previewSize.width / previewSize.height) : 0
    }

    #if os(iOS)
    /// The newest camera frame (waits up to 3 s for one), cropped to what the preview shows.
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
        let (cropped, box) = WatchCrop.crop(image, orientation: .right, aspect: aspect)
        return Frame(
            image: cropped, orientation: .right, name: "camera", ageMs: ageMs,
            convertMs: LookModel.seconds(clock.now - start) * 1000,
            captureWall: Date().timeIntervalSince1970 - ageMs / 1000, switchedAt: nil, crop: box, videoTime: nil)
    }
    #endif

    /// The frame the video shows now (waits up to 2 s for the first one after Start), cropped to what
    /// the preview shows.
    private func videoFrame() async -> Frame? {
        guard let video else { return nil }
        let clock = ContinuousClock()
        let waited = clock.now
        var taken = video.take()
        while taken == nil, clock.now - waited < .seconds(2) {
            try? await Task.sleep(for: .milliseconds(20))
            taken = video.take()
        }
        let start = clock.now
        guard let taken else { return nil }
        let (cropped, box) = WatchCrop.crop(taken.image, orientation: video.orientation, aspect: aspect)
        return Frame(
            image: cropped, orientation: video.orientation, name: "video", ageMs: 0,
            convertMs: LookModel.seconds(clock.now - start) * 1000, captureWall: Date().timeIntervalSince1970,
            switchedAt: nil, crop: box, videoTime: taken.time)
    }

    /// The example photo on the screen, whole (as a tap on its thumbnail reads it); the next one
    /// comes when this one has been on the screen for `-hold` seconds.
    private func photoFrame(at grabbed: ContinuousClock.Instant) -> Frame? {
        guard !photos.isEmpty else { return nil }
        if let since = photoSince {
            if LookModel.seconds(grabbed - since) >= settings.hold {
                photoIndex = (photoIndex + 1) % photos.count
                photoSince = grabbed
            }
        } else {
            photoSince = grabbed
        }
        let item = photos[photoIndex]
        shownImage = item.photo.display
        return Frame(
            image: item.photo.cgImage, orientation: item.photo.orientation, name: item.id, ageMs: 0, convertMs: 0,
            captureWall: Date().timeIntervalSince1970, switchedAt: photoSince, crop: nil, videoTime: nil)
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

    /// The rules that fire on these answers: yes now, not yes on the frame before, and no fire in
    /// the last `cooldown` seconds (those are `suppressed`).
    private func fires(_ answers: [String], at now: ContinuousClock.Instant) -> (fired: [Int], suppressed: [Int]) {
        var fired: [Int] = []
        var suppressed: [Int] = []
        for (k, rule) in run.rules.enumerated() where answers.indices.contains(k) {
            let before = previous[rule.id] ?? "–"
            previous[rule.id] = answers[k]
            guard answers[k] == "yes", before != "yes" else { continue }
            if let last = lastFire[rule.id], LookModel.seconds(now - last) < settings.cooldown {
                suppressed.append(k)
                continue
            }
            lastFire[rule.id] = now
            fired.append(k)
        }
        return (fired, suppressed)
    }

    private func show(_ answers: [String]) {
        for (k, rule) in run.rules.enumerated() where answers.indices.contains(k) {
            rowStates[rule.id, default: RowState()].answer = answers[k]
        }
    }

    // MARK: - Reactions

    /// The fired rules' reactions, and their symbols on the picture for at least 3 s.
    private func react(_ fired: [Int]) {
        guard !fired.isEmpty else { return }
        var title = ""
        var subtitle = ""
        var speech = false
        for (n, k) in fired.enumerated() {
            let rule = run.rules[k]
            let line: String
            switch rule.reaction {
            case .sound(let sound):
                playSound(sound, fire: run.fires.count + n)
                line = "♪ \(sound.title)"
            case .haptic:
                haptic()
                line = "Buzz"
            case .say(let text):
                say(text, fire: run.fires.count + n)
                line = rule.condition
                if n == 0 {
                    speech = true
                    title = "“\(text)”"
                }
            case .timer(let minutes):
                line = startTimer(rule, minutes: minutes)
            case .count:
                rowStates[rule.id, default: RowState()].count += 1
                line = "Count \(rowStates[rule.id]?.count ?? 0)"
            case .log:
                let time = Self.clockTime.string(from: Date())
                logLines.insert("\(time)  \(rule.condition)", at: 0)
                if logLines.count > 50 { logLines.removeLast() }
                line = "Logged \(time)"
            }
            if n == 0 {
                if title.isEmpty { title = rule.condition }
                subtitle = line
            } else {
                subtitle += " · \(rule.condition)"
            }
        }
        showOverlay(Overlay(id: 0, symbols: fired.map { run.rules[$0].symbol }, title: title, subtitle: subtitle, speech: speech))
    }

    private func showOverlay(_ next: Overlay) {
        overlayID += 1
        let id = overlayID
        withAnimation(.easeOut(duration: 0.15)) {
            overlay = Overlay(id: id, symbols: next.symbols, title: next.title, subtitle: next.subtitle, speech: next.speech)
        }
        Task {
            let clock = ContinuousClock()
            let shown = clock.now
            try? await Task.sleep(for: .seconds(3))
            // A spoken line stays until the voice is done (at most 8 s).
            while next.speech, self.speech.isSpeaking, clock.now - shown < .seconds(8) {
                try? await Task.sleep(for: .milliseconds(100))
            }
            if self.overlayID == id { withAnimation(.easeIn(duration: 0.2)) { self.overlay = nil } }
        }
    }

    /// A system sound; the record keeps how long until the system said it had played.
    private func playSound(_ sound: WatchSound, fire: Int?) {
        let clock = ContinuousClock()
        let started = clock.now
        Self.play(sound.systemSound) { [weak self] in
            let ms = LookModel.seconds(clock.now - started) * 1000
            guard let model = self else { return }
            Task { @MainActor in model.noteFire(fire, "sound_ms", Self.round1(ms)) }
        }
    }

    /// Plays a system sound and calls `done` when it has played (on another thread).
    nonisolated private static func play(_ id: SystemSoundID, done: @escaping @Sendable () -> Void) {
        AudioServicesPlaySystemSoundWithCompletion(id, done)
    }

    private func haptic() {
        #if os(iOS)
        UINotificationFeedbackGenerator().notificationOccurred(.success)
        #endif
    }

    /// The line in the system voice (English); the record keeps when the voice started and ended.
    private func say(_ text: String, fire: Int?) {
        guard !text.isEmpty else { return }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speech.speak(utterance)
        let clock = ContinuousClock()
        let started = clock.now
        Task {
            var began: Double?
            while clock.now - started < .seconds(10) {
                if began == nil, self.speech.isSpeaking { began = LookModel.seconds(clock.now - started) * 1000 }
                if began != nil, !self.speech.isSpeaking { break }
                try? await Task.sleep(for: .milliseconds(50))
            }
            self.noteFire(fire, "speech_start_ms", began.map(Self.round1))
            self.noteFire(fire, "speech_end_ms", Self.round1(LookModel.seconds(clock.now - started) * 1000))
        }
    }

    /// Starts the rule's timer (a running one keeps running); returns the overlay's small line.
    private func startTimer(_ rule: WatchRule, minutes: Double) -> String {
        if let end = rowStates[rule.id]?.timerEnd, end > Date() {
            return "Timer running · \(WatchReaction.clock(end.timeIntervalSinceNow)) left"
        }
        let seconds = minutes * 60
        rowStates[rule.id, default: RowState()].timerEnd = Date().addingTimeInterval(seconds)
        let startedSinceStart = LookDevice.sinceStart()
        run.timers.append([
            "rule": (run.rules.firstIndex { $0.id == rule.id } ?? -1) + 1, "condition": rule.condition,
            "minutes": minutes, "started_s": Self.round3(startedSinceStart) ?? 0,
        ])
        let id = rule.id
        Task {
            try? await Task.sleep(for: .seconds(seconds))
            self.timerDone(id)
        }
        return "Timer \(WatchReaction.clock(seconds)) started"
    }

    private func timerDone(_ id: UUID) {
        guard let rule = rules.first(where: { $0.id == id }), rowStates[id]?.timerEnd != nil else { return }
        rowStates[id]?.timerEnd = nil
        playSound(.chime, fire: nil)
        haptic()
        showOverlay(Overlay(id: 0, symbols: [rule.symbol], title: "Time’s up", subtitle: rule.condition, speech: false))
        note("TIMER done rule=\(rule.condition) t_s=\(Self.decimals(LookDevice.sinceStart()))")
    }

    /// A value learned after the fire (a sound's or a voice's times) into its row of the record.
    private func noteFire(_ fire: Int?, _ key: String, _ value: Any?) {
        guard let fire, run.fires.indices.contains(fire) else { return }
        run.fires[fire][key] = value ?? NSNull()
    }

    // MARK: - The record

    /// The frame's row and fires in the record, its WATCH and FIRE lines, and the numbers lines.
    private func record(
        _ frame: Frame, _ index: Int, _ reply: Reply, _ answers: [String], _ fired: [Int], _ suppressed: [Int],
        pace: Double, waitedMs: Double, start: ContinuousClock.Instant, grabbed: ContinuousClock.Instant,
        parseStart: ContinuousClock.Instant, parsed: ContinuousClock.Instant, firedAt: ContinuousClock.Instant
    ) {
        let count = run.rules.count
        let tGrab = LookModel.seconds(grabbed - start)
        let tDone = LookModel.seconds(parsed - start)
        let tFire = LookModel.seconds(firedAt - start)
        let totalMs = LookModel.seconds(parsed - grabbed) * 1000
        let toInputMs = frame.ageMs + frame.convertMs
        let thermal = Autoplay.thermal()
        let footprint = WatchMemory.footprintMB()
        let tCapture = tGrab - frame.ageMs / 1000
        let tSwitch = frame.switchedAt.map { LookModel.seconds($0 - start) }

        var row: [String: Any] = [:]
        row["i"] = index
        row["source"] = frame.name
        row["t_capture_s"] = Self.round3(tCapture) ?? 0
        row["t_grab_s"] = Self.round3(tGrab) ?? 0
        row["t_done_s"] = Self.round3(tDone) ?? 0
        row["t_wall"] = Self.round3(frame.captureWall) ?? 0
        if let tSwitch { row["t_switch_s"] = Self.round3(tSwitch) ?? 0 }
        if let videoTime = frame.videoTime { row["video_t_s"] = Self.round3(videoTime) ?? 0 }
        if let crop = frame.crop { row["crop_px"] = [crop.minX, crop.minY, crop.width, crop.height].map { Int($0) } }
        row["image_px"] = [frame.image.width, frame.image.height]
        row["pace_s"] = pace
        row["waited_ms"] = Self.round1(waitedMs)
        row["age_ms"] = Self.round1(frame.ageMs)
        row["convert_ms"] = Self.round1(frame.convertMs)
        row["capture_to_input_ms"] = Self.round1(toInputMs)
        if let firstMs = reply.firstMs { row["first_token_ms"] = Self.round1(firstMs) } else { row["first_token_ms"] = NSNull() }
        row["model_ms"] = Self.round1(reply.modelMs)
        row["parse_ms"] = Self.round1(LookModel.seconds(parsed - parseStart) * 1000)
        row["total_ms"] = Self.round1(totalMs)
        row["capture_to_answer_ms"] = Self.round1(frame.ageMs + totalMs)
        row["input_tokens"] = reply.inputTokens
        row["output_tokens"] = reply.outputTokens
        row["answers"] = answers
        row["raw"] = reply.text
        row["fired"] = fired.map { $0 + 1 }
        row["suppressed"] = suppressed.map { $0 + 1 }
        row["thermal"] = thermal
        row["footprint_mb"] = Self.round1(footprint)
        row["available_mb"] = Self.round1(WatchMemory.availableMB())
        var scored = "-"
        if let truth = run.truth?[frame.name], truth.count == count {
            let correct = zip(answers, truth).filter { $0.0 == $0.1 }.count
            row["truth"] = truth
            row["correct"] = correct
            scored = "\(correct)/\(count)"
        }
        run.frames.append(row)

        let firstText = reply.firstMs.map { String(Int($0.rounded())) } ?? "-"
        let fields: [String] = [
            "WATCH frame=\(index) src=\(frame.name) t_s=\(Self.decimals(tGrab))",
            "capture_to_input_ms=\(Int(toInputMs.rounded())) first_token_ms=\(firstText)",
            "total_ms=\(Int(totalMs.rounded())) in=\(reply.inputTokens) out=\(reply.outputTokens)",
            "answers=\(answers.joined(separator: ",")) correct=\(scored)",
            "fire=\(fired.isEmpty ? "-" : fired.map { String($0 + 1) }.joined(separator: ","))",
            "thermal=\(thermal) pace_s=\(Self.decimals(pace)) footprint_mb=\(Int(footprint.rounded()))",
        ]
        note(fields.joined(separator: " "))

        for k in fired + suppressed {
            let rule = run.rules[k]
            let isFire = fired.contains(k)
            var item: [String: Any] = [
                "rule": k + 1, "condition": rule.condition, "reaction": rule.reaction.name, "symbol": rule.symbol,
                "frame": index, "source": frame.name, "t_capture_s": Self.round3(tCapture) ?? 0,
                "t_grab_s": Self.round3(tGrab) ?? 0, "t_fire_s": Self.round3(tFire) ?? 0,
                "wall": Self.round3(Date().timeIntervalSince1970) ?? 0,
                "from_capture_ms": Self.round1((tFire - tCapture) * 1000),
            ]
            if let detail = rule.reaction.detail { item["detail"] = detail }
            if let tSwitch { item["from_switch_ms"] = Self.round1((tFire - tSwitch) * 1000) }
            if let videoTime = frame.videoTime { item["video_t_s"] = Self.round3(videoTime) ?? 0 }
            if rule.reaction == .count { item["count"] = rowStates[rule.id]?.count ?? 0 }
            if isFire {
                run.fires.append(item)
                let latency = Int(((tFire - (tSwitch ?? tCapture)) * 1000).rounded())
                note(
                    "FIRE rule=\(k + 1) cond=\(rule.condition) reaction=\(rule.reaction.name) frame=\(index)"
                        + " t=\(Self.decimals(tFire)) latency_ms=\(latency)")
                onFire?(run.fires.count)
            } else {
                item["reason"] = "cooldown"
                run.suppressed.append(item)
            }
        }

        numbersHead = "first token \(Self.seconds2(reply.firstMs)) s · total \(Self.seconds2(totalMs)) s"
        let backend = engine?.backend ?? "GPU"
        let offline = engine?.offline == true ? ", offline" : ""
        let paced = pace > 0 ? " · every \(Self.decimals1(pace)) s" : ""
        numbersTail = "\(backend) · on-device\(offline) · frame \(index) · \(thermal)\(paced) · \(Int(footprint.rounded())) MB"
    }

    private func finishRun() async {
        #if os(iOS)
        if run.rules.contains(where: { $0.reaction.kind == .say }) {
            try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        }
        #endif
        lastFailure = run.failure
        stopping = false
        phase = .paused
        keepAwake()
        if run.endedByVideo {
            note("VIDEO ended frames=\(run.frames.count) t_s=\(Self.decimals(run.loopSeconds))")
        }
        if let failure = run.failure {
            detail = failure
        } else {
            readyDetail()
        }
        await Task.yield()
        var extra: [String: Any] = [:]
        if let beforeRecord { extra = await beforeRecord() }
        let file = writeRecord(extra)
        if let failure = run.failure {
            note("ERROR \(failure) json=\(file)")
        } else {
            let stats = Self.stats(run.frames.compactMap { $0["total_ms"] as? Double })
            let correct = run.frames.compactMap { $0["correct"] as? Int }.reduce(0, +)
            let judged = run.frames.filter { $0["correct"] != nil }.count * run.rules.count
            let fields: [String] = [
                "DONE stop=done frames=\(run.frames.count) fires=\(run.fires.count)",
                "mean_total_ms=\(stats.map { String(describing: $0["mean"] ?? "-") } ?? "-")",
                "p50_total_ms=\(stats.map { String(describing: $0["p50"] ?? "-") } ?? "-")",
                "max_total_ms=\(stats.map { String(describing: $0["max"] ?? "-") } ?? "-")",
                "correct=\(judged > 0 ? "\(correct)/\(judged)" : "-") json=\(file)",
            ]
            note(fields.joined(separator: " "))
        }
    }

    /// Documents/watch-run-<epoch>.json (only with `-log 1`).
    private func writeRecord(_ extra: [String: Any]) -> String {
        let name = "watch-run-\(run.epoch).json"
        guard settings.log else { return name }
        var object: [String: Any] = [:]
        func put(_ key: String, _ value: Any?) { object[key] = value ?? NSNull() }
        let count = run.rules.count
        let now = LookDevice.sinceStart()
        let (screenPixels, scale) = Autoplay.screenFacts()

        put("schema", "watch-run/2")
        put("mode", "watch")
        put("epoch", run.epoch)
        put("args", Array(CommandLine.arguments.dropFirst()))
        put("model", settings.modelID)
        put("repo", engine?.entry?.repo)
        put("revision", engine?.entry?.revision)
        let vision = settings.modelID == "minicpm-v-4.7-1b" ? VLModelID.miniCPMV47.vision.path ?? "?" : "?"
        put("bundles", "\(engine?.entry?.variant?.path ?? "?") + \(vision)")
        put("device", LookDevice.machine)
        put("device_name", LookDevice.name)
        put("os", "\(LookDevice.osVersion) (\(LookDevice.osBuild))")
        put("app", "Look \(LookDevice.appVersion) (\(LookDevice.appBuild))")
        put("source", source.rawValue)
        put("seconds", settings.seconds)
        put("rules_arg", settings.rulesArgument)
        put("rules", run.rules.indices.map { Self.ruleRecord(run.rules[$0], number: $0 + 1) })
        put("conditions", run.rules.map(\.condition))
        put("symbols", run.rules.map(\.symbol))
        put("prompt_form", "batch")
        put("prompt", run.prompt)
        put("max_tokens", settings.maxTokens)
        put("truth", run.truth)
        put("cooldown_s", settings.cooldown)
        switch source {
        case .photos:
            put("hold_s", settings.hold)
        case .camera:
            put("camera", cameraInfo)
            put("camera_frames_delivered", camera?.deliveredCount)
        case .video:
            if let video {
                put(
                    "video",
                    [
                        "file": video.name, "duration_s": Self.round3(video.duration) ?? 0,
                        "size": [Int(video.naturalSize.width), Int(video.naturalSize.height)], "angle": video.angle,
                        "loops": false, "run_ended_with_the_video": run.endedByVideo,
                    ] as [String: Any])
            }
        }
        put("preview_pt", [Self.round1(Double(previewSize.width)), Self.round1(Double(previewSize.height))])
        put("thermal_start", run.thermalStart)
        put("thermal_end", Autoplay.thermal())
        put("free_bytes_start", run.freeStart)
        put("free_bytes_end", Autoplay.freeBytes())
        put("catalog_s", Self.round3(engine?.catalogSeconds))
        put("engine_ready_s", Self.round3(engine?.engineReadySeconds))
        put("prewarm_s", Self.round3(engine?.prewarmSeconds))
        put("t_ready_s", Self.round3(engine?.readySinceStart))
        put("t_start_s", Self.round3(run.startedSinceStart))
        put("loop_s", Self.round3(run.loopSeconds))
        put("stop", run.failure == nil ? "done" : "error")
        put("error", run.failure)
        put("presses", run.presses)
        put("frames", run.frames)
        put("fires", run.fires)
        put("suppressed", run.suppressed)
        put("timers", run.timers)
        put("numbers_line", numbersLine)
        put("footer", footer)
        put("take", settings.take)
        put("screen_px", screenPixels)
        put("scale", scale)
        put("take_rect_px", [bandRect.minX, bandRect.minY, bandRect.width, bandRect.height].map { Int(($0 * scale).rounded()) })

        var summary: [String: Any] = ["frames": run.frames.count]
        for key in ["total_ms", "first_token_ms", "model_ms", "capture_to_input_ms", "capture_to_answer_ms"] {
            let values = run.frames.compactMap { $0[key] as? Double }
            if let stats = Self.stats(values) { summary[key] = stats } else { summary[key] = NSNull() }
        }
        let answers = run.frames.compactMap { $0["answers"] as? [String] }
        summary["yes"] = (0..<count).map { k in answers.filter { $0.indices.contains(k) && $0[k] == "yes" }.count }
        summary["unparsed"] = answers.reduce(0) { total, row in total + row.filter { $0 == "?" }.count }
        summary["fires"] = run.fires.count
        summary["suppressed"] = run.suppressed.count
        var byReaction: [String: Int] = [:]
        for fire in run.fires { if let name = fire["reaction"] as? String { byReaction[name, default: 0] += 1 } }
        summary["fires_by_reaction"] = byReaction
        summary["fires_by_rule"] = (1...max(count, 1)).map { k in run.fires.filter { ($0["rule"] as? Int) == k }.count }
        for key in ["from_switch_ms", "from_capture_ms"] {
            let values = run.fires.compactMap { $0[key] as? Double }
            if let stats = Self.stats(values) { summary["fire_\(key)"] = stats }
        }
        let scored = run.frames.filter { $0["correct"] != nil }
        if !scored.isEmpty {
            summary["correct"] = scored.reduce(0) { total, row in total + (row["correct"] as? Int ?? 0) }
            summary["judged"] = scored.count * count
        }
        let footprints = run.frames.compactMap { $0["footprint_mb"] as? Double }.filter { $0 >= 0 }
        if let low = footprints.min(), let high = footprints.max() {
            summary["footprint_mb"] = ["min": low, "max": high]
        }
        var thermals: [String: Int] = [:]
        for row in run.frames { if let state = row["thermal"] as? String { thermals[state, default: 0] += 1 } }
        summary["thermal_frames"] = thermals
        summary["paced_frames"] = run.frames.filter { ($0["pace_s"] as? Double ?? 0) > 0 }.count
        put("summary", summary)

        let stretches = sceneLog.summary(now: now)
        put("inactive_count", stretches.count)
        put("inactive_s", Self.round3(stretches.seconds))
        put("inactive_at", stretches.rows)
        put("scene_first_active_s", Self.round3(sceneLog.firstActive))
        put("scene_at_record", sceneLog.current)
        put("t_record_s", Self.round3(now))
        for (key, value) in extra { object[key] = value }

        let url = URL.documentsDirectory.appending(path: name)
        if let data = try? JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        {
            try? data.write(to: url)
        }
        return name
    }

    // MARK: - Helpers

    /// A rule as the record keeps it.
    private static func ruleRecord(_ rule: WatchRule, number: Int) -> [String: Any] {
        var item: [String: Any] = [
            "i": number, "condition": rule.condition, "reaction": rule.reaction.name, "symbol": rule.symbol,
        ]
        if let detail = rule.reaction.detail { item["detail"] = detail }
        return item
    }

    /// Appends one timestamped line to Documents/look-autoplay.log (only with `-log 1`).
    func note(_ line: String) {
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

    private static let clockTime: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
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

    private static func decimals1(_ value: Double) -> String { String(format: "%.1f", value) }

    /// Milliseconds as seconds with two decimals ("1.36"); "–" for none.
    private static func seconds2(_ milliseconds: Double?) -> String {
        milliseconds.map { String(format: "%.2f", $0 / 1000) } ?? "–"
    }

    nonisolated private static func round1(_ value: Double) -> Double { (value * 10).rounded() / 10 }

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
        case .paused: "PAUSED"
        case .failed: "FAILED"
        }
    }

    var color: Color {
        switch self {
        case .loading: Color(white: 0.36)
        case .ready: Palette.accent
        case .watching: Color(red: 0.84, green: 0.50, blue: 0.08)
        case .paused: Color(white: 0.36)
        case .failed: Color(red: 0.78, green: 0.20, blue: 0.20)
        }
    }
}
