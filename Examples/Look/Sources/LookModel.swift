// LookModel — one photo + one typed question → one streamed answer from a vision-language model
// on this device, timed by the app itself. The buttons in LookView and the recording harness in
// Autoplay call the same functions: pickExample(_:), pick(_:), applyChip(_:), ask(), stop().
//
// The model loads when the app starts (READY) through its catalog id — the id on the model's card —
// exactly as QuickStart.swift does. Every question is a new `LanguageModelSession` (no history):
// the photo, then the question, in one turn, greedy, at most 256 answer tokens.

import CoreAIKit
import Foundation
import FoundationModels
import ImageIO
import Network
import Observation
import PhotosUI
import SwiftUI
#if os(iOS)
import UIKit
#endif

/// A photo ready for the model: its pixels as decoded plus the EXIF orientation (camera photos
/// arrive rotated; both go to the model, as `ImageFile.load` hands them over), an upright copy for
/// the screen, and where it came from.
struct PickedPhoto {
    let cgImage: CGImage
    let orientation: CGImagePropertyOrientation
    let display: CGImage
    let source: String
    let bytes: Int?

    init(cgImage: CGImage, orientation: CGImagePropertyOrientation, source: String, bytes: Int?) {
        self.cgImage = cgImage
        self.orientation = orientation
        self.display = cgImage.upright(orientation)
        self.source = source
        self.bytes = bytes
    }

    /// Decodes an image file's bytes (JPEG, HEIC, PNG, WebP …) the way `ImageFile.load` reads a
    /// file: the first image and its EXIF orientation. A photo with a long side over 4096 px is
    /// subsampled while it decodes; the model draws every photo into its own small input anyway.
    init?(data: Data, source: String) {
        guard let file = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let properties = CGImageSourceCopyPropertiesAtIndex(file, 0, nil) as? [CFString: Any]
        let raw = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        let side = max(
            properties?[kCGImagePropertyPixelWidth] as? Int ?? 0,
            properties?[kCGImagePropertyPixelHeight] as? Int ?? 0)
        let image =
            side > 4096
            ? CGImageSourceCreateThumbnailAtIndex(
                file, 0,
                [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceThumbnailMaxPixelSize: 4096,
                    kCGImageSourceCreateThumbnailWithTransform: false,
                ] as CFDictionary)
            : CGImageSourceCreateImageAtIndex(file, 0, nil)
        guard let image else { return nil }
        self.init(
            cgImage: image, orientation: CGImagePropertyOrientation(rawValue: raw) ?? .up,
            source: source, bytes: data.count)
    }
}

@MainActor
@Observable
final class LookModel {
    enum Phase: Equatable {
        case loading, ready, thinking, done, stopped, failed
    }

    /// One of the three bundled example photos (Look.app/Examples/<file>).
    struct Example: Identifiable {
        let id: String
        let file: String
        var url: URL? { Bundle.main.url(forResource: file, withExtension: nil, subdirectory: "Examples") }
    }

    static let examples = [
        Example(id: "sign", file: "sign.webp"),
        Example(id: "object", file: "object.jpg"),
        Example(id: "scene", file: "scene.jpg"),
    ]

    /// Example questions: a tap puts one in the question box, which stays editable.
    static let chips = [
        "What does this sign say?",
        "What is this and how do I use it?",
        "Describe this scene in one sentence",
    ]

    /// One answered (or stopped) question, as the record keeps it. Times in seconds.
    struct Run {
        var question: String
        var photoSource: String
        var photoBytes: Int?
        var photoPixels: [Int]
        /// When Ask was tapped, from the process start.
        var askedSinceStart: Double
        /// From the Ask to the first snapshot that carried text.
        var firstToken: Double?
        /// From the Ask to the last snapshot that changed the text.
        var lastText: Double?
        /// From the Ask to the end of the stream.
        var generate: Double
        var inputTokens: Int
        var outputTokens: Int
        /// Snapshots that carried text, and those of them that changed it.
        var snapshots: Int
        var textUpdates: Int
        var answer: String
        var stop: String
        var error: String?

        /// (tokens − 1) / (end of the stream − first token): the rate on the numbers line.
        var tokensPerSecond: Double {
            guard let firstToken, outputTokens > 1, generate > firstToken else { return 0 }
            return Double(outputTokens - 1) / (generate - firstToken)
        }

        /// The same count over (last text − first token), without the stream's tail.
        var tokensPerSecondToLastText: Double? {
            guard let firstToken, let lastText, outputTokens > 1, lastText > firstToken else { return nil }
            return Double(outputTokens - 1) / (lastText - firstToken)
        }
    }

    /// The answer box's geometry in points, read by the record at DONE: whether the answer
    /// overflowed its box and whether its first line was still in view.
    struct AnswerBox: Equatable, Sendable {
        var content: Double = 0
        var container: Double = 0
        var offset: Double = 0
    }

    let modelID: String
    var phase: Phase = .loading
    /// The line beside the state pill.
    var detail = "loading…"
    var photo: PickedPhoto?
    var question = ""
    /// The answer as the model wrote it (Markdown); the screen shows `shown(answer)`.
    var answer = ""
    var firstTokenSeconds: Double?
    var tokensPerSecond: Double?
    var offline = false
    var thumbnails: [String: CGImage] = [:]
    private(set) var entry: CatalogEntry?
    private(set) var lastRun: Run?

    // Load and step times for the record.
    private(set) var catalogSeconds: Double?
    private(set) var engineReadySeconds: Double?
    private(set) var prewarmSeconds: Double?
    private(set) var readySinceStart: Double?
    private(set) var pickedSinceStart: Double?
    private(set) var questionSinceStart: Double?

    // Read by Autoplay only; the screen does not draw from them.
    @ObservationIgnored var bandRect: CGRect = .zero
    @ObservationIgnored var answerBox = AnswerBox()
    @ObservationIgnored var onFirstToken: (@MainActor () -> Void)?

    @ObservationIgnored private var model: KitVisionModel?
    @ObservationIgnored private var generationTask: Task<Void, Never>?
    private let monitor = NWPathMonitor()

    init() {
        modelID = UserDefaults.standard.string(forKey: "model") ?? "minicpm-v-4.7-1b"
    }

    var canAsk: Bool { model != nil && phase != .thinking && phase != .loading }

    // MARK: - Model

    /// Loads the model by its catalog id, then warms it: READY. Called once, when the screen appears.
    func load() async {
        guard model == nil, phase == .loading else { return }
        watchNetwork()
        loadThumbnails()
        let clock = ContinuousClock()
        do {
            var start = clock.now
            // The catalog row names the repo, the revision and the bundle path the footer shows.
            entry = try await ModelCatalog.entry(forID: modelID, expecting: .vlm)
            catalogSeconds = Self.seconds(clock.now - start)
            start = clock.now
            let vlm = try await KitVisionModel(catalog: modelID) { progress in
                let line = progress.fraction < 1 ? "Downloading \(Int(progress.fraction * 100)) %" : "loading…"
                Task { @MainActor in self.detail = line }
            }
            engineReadySeconds = Self.seconds(clock.now - start)
            start = clock.now
            LanguageModelSession(model: vlm).prewarm()
            prewarmSeconds = Self.seconds(clock.now - start)
            model = vlm
            readySinceStart = LookDevice.sinceStart()
            phase = .ready
            detail = "Pick a photo, ask a question"
        } catch {
            phase = .failed
            detail = error.localizedDescription
        }
    }

    // MARK: - Photo and question

    /// A tap on an example photo.
    @discardableResult
    func pickExample(_ id: String) -> Bool {
        guard phase != .thinking else { return false }
        guard let example = Self.examples.first(where: { $0.id == id }), let url = example.url,
            let data = try? Data(contentsOf: url),
            let photo = PickedPhoto(data: data, source: "example \(id)")
        else {
            detail = "The example photos are not in this build: see the README"
            return false
        }
        set(photo)
        return true
    }

    /// A photo from Pick a photo (the system picker runs outside the app: no permission needed).
    func pick(_ item: PhotosPickerItem) async {
        guard phase != .thinking else { return }
        guard let data = try? await item.loadTransferable(type: Data.self),
            let photo = PickedPhoto(data: data, source: "library")
        else {
            detail = "Could not read that photo"
            return
        }
        set(photo)
    }

    #if os(iOS)
    /// A photo from Take a photo.
    func pick(cameraImage image: UIImage) {
        guard phase != .thinking else { return }
        guard let cgImage = image.cgImage else {
            detail = "Could not read that photo"
            return
        }
        set(PickedPhoto(cgImage: cgImage, orientation: .init(image.imageOrientation), source: "camera", bytes: nil))
    }
    #endif

    /// A tap on an example question: it goes into the box, which stays editable.
    func applyChip(_ index: Int) {
        guard Self.chips.indices.contains(index) else { return }
        setQuestion(Self.chips[index])
    }

    func setQuestion(_ text: String) {
        guard phase != .thinking else { return }
        question = text
        questionSinceStart = LookDevice.sinceStart()
    }

    private func set(_ photo: PickedPhoto) {
        self.photo = photo
        pickedSinceStart = LookDevice.sinceStart()
        answer = ""
        firstTokenSeconds = nil
        tokensPerSecond = nil
        if model != nil {
            phase = .ready
            detail = "Pick a photo, ask a question"
        }
    }

    // MARK: - Ask

    /// Ask: a new session gets the photo, then the question, and the answer streams onto the
    /// screen. The clock starts here; the first token is the first snapshot that carries text.
    @discardableResult
    func ask() -> Task<Void, Never>? {
        guard let model, canAsk else { return nil }
        guard let photo else {
            detail = "Choose a photo first: tap an example, pick or take one"
            return nil
        }
        let question = self.question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else {
            detail = "Type a question, or tap one under the box"
            return nil
        }
        let clock = ContinuousClock()
        let asked = clock.now
        let askedSinceStart = LookDevice.sinceStart()
        phase = .thinking
        detail = "Reading the photo and the question"
        answer = ""
        firstTokenSeconds = nil
        tokensPerSecond = nil

        let session = LanguageModelSession(model: model)
        let prompt = Prompt {
            Attachment(photo.cgImage, orientation: photo.orientation)  // the image, then the question
            question
        }
        let task = Task {
            var first: Duration?
            var lastText: Duration?
            var snapshots = 0
            var textUpdates = 0
            var failure: String?
            do {
                let stream = session.streamResponse(
                    to: prompt, options: GenerationOptions(maximumResponseTokens: 256))
                for try await snapshot in stream {
                    let now = clock.now
                    if Task.isCancelled { break }
                    let text = snapshot.content
                    if text.isEmpty { continue }
                    snapshots += 1
                    if text != answer {
                        textUpdates += 1
                        lastText = now - asked
                        answer = text
                    }
                    if first == nil {
                        first = now - asked
                        firstTokenSeconds = Self.seconds(now - asked)
                        onFirstToken?()
                    }
                }
            } catch is CancellationError {
            } catch {
                failure = error.localizedDescription
            }
            let generate = Self.seconds(clock.now - asked)
            let usage = session.usage
            let stop = failure != nil ? "error" : (Task.isCancelled ? "stopped" : "done")
            let run = Run(
                question: question, photoSource: photo.source, photoBytes: photo.bytes,
                photoPixels: [photo.cgImage.width, photo.cgImage.height],
                askedSinceStart: askedSinceStart, firstToken: first.map(Self.seconds),
                lastText: lastText.map(Self.seconds), generate: generate,
                inputTokens: usage.input.totalTokenCount, outputTokens: usage.output.totalTokenCount,
                snapshots: snapshots, textUpdates: textUpdates, answer: answer, stop: stop,
                error: failure)
            lastRun = run
            switch stop {
            case "done":
                tokensPerSecond = run.tokensPerSecond
                phase = .done
                detail = "The photo and the question in one turn"
            case "stopped":
                phase = .stopped
                detail = "Stopped"
            default:
                phase = .failed
                detail = failure ?? "The answer failed"
            }
        }
        generationTask = task
        return task
    }

    func stop() {
        generationTask?.cancel()
    }

    // MARK: - What the screen shows

    /// The bundle subtree the catalog row names ("gpu-pipelined").
    var subtree: String {
        entry?.variant?.path.split(separator: "/").first.map(String.init) ?? "–"
    }

    /// Where the model runs, from that subtree: a `gpu-…` bundle runs on the GPU.
    var backend: String { subtree.hasPrefix("gpu") ? "GPU" : subtree }

    var numbersHead: String {
        let first = firstTokenSeconds.map { String(format: "%.2f s", $0) } ?? "– s"
        let rate = tokensPerSecond.map { $0 >= 100 ? String(format: "%.0f", $0) : String(format: "%.1f", $0) } ?? "–"
        return "first token \(first) · \(rate) tok/s"
    }

    /// "offline" only while the phone has no network path at all.
    var numbersTail: String { "\(backend) · on-device\(offline ? ", offline" : "")" }

    var numbersLine: String { "\(numbersHead) · \(numbersTail)" }

    var footer: String {
        let model = entry.map { "\($0.repo) @\(($0.revision ?? "main").prefix(8))" } ?? modelID
        return [model, subtree, LookDevice.name, LookDevice.os, "Look \(LookDevice.appVersion)"]
            .joined(separator: " · ")
    }

    /// The answer as the screen shows it. The model writes Markdown: `**bold**` is set in bold,
    /// heading `#` and list `- ` marks are dropped, and a `**` still open while the answer streams
    /// is closed so the marks never flash. The record keeps the text as written.
    nonisolated static func shown(_ raw: String) -> AttributedString {
        var lines: [String] = []
        for line in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            var text = line.drop { $0 == " " }
            if text.hasPrefix("#") {
                text = text.drop { $0 == "#" }.drop { $0 == " " }
            } else if text.hasPrefix("- ") || text.hasPrefix("* ") {
                text = text.dropFirst(2)
            }
            lines.append(String(text))
        }
        var text = lines.joined(separator: "\n")
        if text.components(separatedBy: "**").count % 2 == 0 { text += "**" }
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }

    // MARK: - Helpers

    private func loadThumbnails() {
        for example in Self.examples {
            guard let url = example.url, let file = CGImageSourceCreateWithURL(url as CFURL, nil),
                let thumb = CGImageSourceCreateThumbnailAtIndex(
                    file, 0,
                    [
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                        kCGImageSourceThumbnailMaxPixelSize: 240,
                        kCGImageSourceCreateThumbnailWithTransform: true,
                    ] as CFDictionary)
            else { continue }
            thumbnails[example.id] = thumb
        }
    }

    private func watchNetwork() {
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            let offline = path.status != .satisfied
            Task { @MainActor in self.offline = offline }
        }
        monitor.start(queue: DispatchQueue(label: "look.network"))
    }

    nonisolated static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

/// The phone (or Mac) the numbers came from, for the footer and the record.
enum LookDevice {
    /// The hardware model: "iPhone19,2", "Mac16,5".
    static let machine: String = {
        #if os(macOS)
        return sysctlString("hw.model")
        #else
        return sysctlString("hw.machine")
        #endif
    }()

    /// Hardware models this sample was run on, by name; any other shows its model id.
    static let knownNames = ["iPhone19,2": "iPhone 18 Pro"]

    static var name: String { knownNames[machine] ?? machine }

    static let osVersion: String = {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return v.patchVersion == 0
            ? "\(v.majorVersion).\(v.minorVersion)" : "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }()

    /// The OS build ("24B5099f").
    static let osBuild = sysctlString("kern.osversion")

    static var os: String {
        #if os(macOS)
        return "macOS \(osVersion)"
        #else
        return "iOS \(osVersion)"
        #endif
    }

    static var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
    }

    static var appBuild: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
    }

    /// When this process started (the kernel's record of it); the app's own start if unreadable.
    static let processStart: Date = {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&name, 4, &info, &size, nil, 0) == 0 else { return Date() }
        let start = info.kp_proc.p_un.__p_starttime
        return Date(timeIntervalSince1970: Double(start.tv_sec) + Double(start.tv_usec) / 1e6)
    }()

    static func sinceStart() -> Double { Date().timeIntervalSince(processStart) }

    private static func sysctlString(_ key: String) -> String {
        var size = 0
        sysctlbyname(key, nil, &size, nil, 0)
        var bytes = [UInt8](repeating: 0, count: size)
        sysctlbyname(key, &bytes, &size, nil, 0)
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }
}

#if os(iOS)
extension CGImagePropertyOrientation {
    init(_ orientation: UIImage.Orientation) {
        switch orientation {
        case .up: self = .up
        case .down: self = .down
        case .left: self = .left
        case .right: self = .right
        case .upMirrored: self = .upMirrored
        case .downMirrored: self = .downMirrored
        case .leftMirrored: self = .leftMirrored
        case .rightMirrored: self = .rightMirrored
        @unknown default: self = .up
        }
    }
}
#endif
