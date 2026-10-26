// CutoutModel — one photo + the words for what to cut out → the boxes the model writes, a matte per box, and the cut-out
// saved as a PNG with transparency. The buttons in CutoutView and the recording harness in Autoplay call the same
// functions: pickExample(_:), pick(_:), setWords(_:), cutOut(), toggle(_:), save(), stop().
//
// The model loads when the app starts (READY) from the model folder (README: Run it): WeVisOne-VL-2B's decoder, its
// vision tower with the mask decoder's taps, and the two mask graphs, on the system CoreAI framework (Sources/Engine).
// Every request is one turn: the photo, then the words in the package's segmentation prompt, greedy.

import CoreGraphics
import Foundation
import ImageIO
import Observation
import Photos
import PhotosUI
import SwiftUI
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
import UniformTypeIdentifiers
#endif

/// A photo ready for the model: upright 8-bit RGB (the model's input and the cut-out's pixels), a copy for the screen,
/// and where it came from.
struct PickedPhoto {
    let rgb: RGB8Image
    let display: CGImage
    let source: String
    let bytes: Int?

    /// Decodes an image file's bytes (JPEG, HEIC, PNG, WebP …): the first image, turned upright by its EXIF orientation;
    /// a long side over 4096 px is subsampled while it decodes.
    init?(data: Data, source: String) {
        guard let file = try? ImageHost.load(data: data),
              let rgb = try? ImageHost.uprightRGB8(file.image, orientation: file.orientation),
              let display = ImageHost.cgImage(rgb)
        else { return nil }
        self.rgb = rgb
        self.display = display
        self.source = source
        self.bytes = data.count
    }

    init?(cgImage: CGImage, orientation: CGImagePropertyOrientation, source: String) {
        guard let rgb = try? ImageHost.uprightRGB8(cgImage, orientation: orientation), let display = ImageHost.cgImage(rgb) else {
            return nil
        }
        self.rgb = rgb
        self.display = display
        self.source = source
        self.bytes = nil
    }
}

@MainActor
@Observable
final class CutoutModel {
    enum Phase: Equatable {
        case loading, ready, working, done, nothing, stopped, failed
    }

    /// One of the three bundled example photos (Cutout.app/Examples/<file>) and the words that go with it.
    struct Example: Identifiable {
        let id: String
        let file: String
        let words: String
        /// where the thumbnail's crop centers, as a fraction of the photo's height (the airplane is small, high up)
        var focusY: Double = 0.5
        var url: URL? { Bundle.main.url(forResource: file, withExtension: nil, subdirectory: "Examples") }
    }

    static let examples = [
        Example(id: "lighthouse", file: "lighthouse.jpg", words: "the lighthouse"),
        Example(id: "chess", file: "chess.jpg", words: "the light chess pieces"),
        Example(id: "airplane", file: "airplane.jpg", words: "the airplane", focusY: 0.28),
    ]

    /// Words to tap into the box (it stays editable).
    static let chips = ["the lighthouse", "the light chess pieces", "the airplane", "the sky"]

    /// One thing the model found.
    struct Piece: Identifiable {
        let id: Int
        let label: String
        /// left, top, right, bottom in per mille of the photo
        let box: [Int]
        /// mask_logits [1024 · 1024] (nil for the boxes task)
        let logits: [Float]?
        var selected = true
    }

    /// One request, as the record keeps it.
    struct Run {
        var words: String
        var task: CutoutTask
        var photoSource: String
        var photoPixels: [Int]
        var startedSinceStart: Double
        var answer: String
        var tokens: Int
        var pieces: Int
        /// the pipeline's step times in seconds (Engine/Pipeline.swift)
        var seconds: [String: Double]
        var composeSeconds: Double?
        var stop: String
        var error: String?
        var footprint: (now: UInt64, peak: UInt64)
    }

    let modelRoot: URL
    var phase: Phase = .loading
    /// The line beside the state pill.
    var detail = "loading…"
    var photo: PickedPhoto?
    var words = ""
    var task: CutoutTask = .cutout
    var pieces: [Piece] = []
    /// The answer as the model wrote it.
    var answer = ""
    var tokens = 0
    /// The cut-out of the chosen pieces over transparency, at most 1024 px on its long side (the screen's copy).
    var cutoutImage: CGImage?
    var showCutout = true
    var saveLine: String?
    var boxSeconds: Double?
    var matteSeconds: Double?
    var thumbnails: [String: CGImage] = [:]
    private(set) var lastRun: Run?
    private(set) var loadRecord: CutoutPipeline.Load?
    private(set) var readySinceStart: Double?
    private(set) var loadSeconds: Double?

    @ObservationIgnored var onFinished: (@MainActor (Run) -> Void)?
    @ObservationIgnored var onSaved: (@MainActor (String) -> Void)?
    @ObservationIgnored var pipeline: CutoutPipeline?
    @ObservationIgnored private var runTask: Task<Void, Never>?
    /// `-photosauth denied`: the save path as if Photos access were off (a check of the screen's sentence; the record says so).
    @ObservationIgnored let injectedPhotosDenied = UserDefaults.standard.string(forKey: "photosauth") == "denied"

    init() {
        if let path = UserDefaults.standard.string(forKey: "models") {
            modelRoot = URL(fileURLWithPath: path.hasPrefix("/") ? path : Autoplay.docs.appending(path: path).path)
        } else {
            #if os(iOS)
            modelRoot = URL.documentsDirectory.appending(path: "WeVisOne-VL-2B-CoreAI")
            #else
            modelRoot = URL.applicationSupportDirectory.appending(path: "Cutout/WeVisOne-VL-2B-CoreAI")
            #endif
        }
    }

    var canRun: Bool { pipeline != nil && phase != .working && phase != .loading }

    // MARK: - Model

    /// Loads the four model files: READY. Called once, when the screen appears.
    func load() async {
        guard pipeline == nil, phase == .loading else { return }
        loadThumbnails()
        let clock = ContinuousClock()
        let start = clock.now
        do {
            #if os(macOS)
            let folder = try ModelFolder(root: modelRoot, preferAOT: true)
            #else
            let folder = try ModelFolder(root: modelRoot, preferAOT: false)
            #endif
            let p = try await CutoutPipeline(folder: folder) { part in
                Task { @MainActor in self.detail = "loading the \(part)…" }
            }
            pipeline = p
            loadRecord = p.load
            loadSeconds = Self.seconds(clock.now - start)
            readySinceStart = CutoutDevice.sinceStart()
            phase = .ready
            detail = photo == nil ? "Pick a photo, then say what to cut out" : "Say what to cut out"
        } catch {
            phase = .failed
            detail = (error as? CutoutError)?.description ?? error.localizedDescription
        }
    }

    // MARK: - Photo and words

    @discardableResult
    func pickExample(_ id: String) -> Bool {
        guard phase != .working else { return false }
        guard let example = Self.examples.first(where: { $0.id == id }), let url = example.url,
              let data = try? Data(contentsOf: url), let photo = PickedPhoto(data: data, source: "example \(id)")
        else {
            detail = "The example photos are not in this build: see the README"
            return false
        }
        set(photo)
        words = example.words
        return true
    }

    /// A photo from Pick a photo (the system picker runs outside the app: no permission needed).
    func pick(_ item: PhotosPickerItem) async {
        guard phase != .working else { return }
        guard let data = try? await item.loadTransferable(type: Data.self), let photo = PickedPhoto(data: data, source: "library")
        else {
            detail = "Could not read that photo"
            return
        }
        set(photo)
    }

    /// A photo file (Documents on the phone, any path on the Mac): `-image` and the Mac's Open.
    func pick(fileAt url: URL) {
        guard let data = try? Data(contentsOf: url), let photo = PickedPhoto(data: data, source: "file \(url.lastPathComponent)") else {
            detail = "Could not read that photo"
            return
        }
        set(photo)
    }

    #if os(iOS)
    func pick(cameraImage image: UIImage) {
        guard phase != .working, let cg = image.cgImage,
              let photo = PickedPhoto(cgImage: cg, orientation: .init(image.imageOrientation), source: "camera")
        else {
            detail = "Could not read that photo"
            return
        }
        set(photo)
    }
    #endif

    func setWords(_ text: String) {
        guard phase != .working else { return }
        words = text
    }

    private func set(_ photo: PickedPhoto) {
        self.photo = photo
        clearResult()
        if pipeline != nil {
            phase = .ready
            detail = "Say what to cut out"
        }
    }

    private func clearResult() {
        pieces = []
        answer = ""
        tokens = 0
        cutoutImage = nil
        saveLine = nil
        boxSeconds = nil
        matteSeconds = nil
    }

    // MARK: - Cut out

    @discardableResult
    func cutOut() -> Task<Void, Never>? {
        guard let pipeline, canRun else { return nil }
        guard let photo else {
            detail = "Choose a photo first: tap an example, pick or take one"
            return nil
        }
        let words = self.words.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !words.isEmpty else {
            detail = "Type what to cut out, or tap a word under the box"
            return nil
        }
        let task = self.task
        let started = CutoutDevice.sinceStart()
        clearResult()
        phase = .working
        detail = "Looking for \(words)…"
        let image = photo.rgb
        let t = Task {
            var failure: String?
            var result: CutoutPipeline.Result?
            do {
                result = try await pipeline.run(image: image, words: words, task: task, onToken: { _ in
                    Task { @MainActor in self.tokens += 1 }
                })
            } catch is CancellationError {
            } catch {
                failure = (error as? CutoutError)?.description ?? error.localizedDescription
            }
            var composeSeconds: Double? = nil
            if let r = result {
                answer = r.response
                tokens = r.generation.ids.count
                pieces = r.instances.enumerated().map { Piece(id: $0.offset, label: $0.element.label, box: $0.element.box, logits: $0.element.logits) }
                let s = r.seconds
                boxSeconds = ["resize448", "patches", "tower", "prompt", "decoder_prompt", "decode"].compactMap { s[$0] }.reduce(0, +)
                matteSeconds = task == .cutout && !pieces.isEmpty
                    ? ["resize1024", "canvas", "mask_inputs", "graph_a", "graph_b"].compactMap { s[$0] }.reduce(0, +) : nil
                if r.generation.stop == "cancelled" {
                    phase = .stopped
                    detail = "Stopped"
                } else if pieces.isEmpty {
                    phase = .nothing
                    detail = "Nothing in this photo matched “\(words)”"
                } else {
                    let tc = ContinuousClock.now
                    await composeCutout()
                    composeSeconds = Self.seconds(ContinuousClock.now - tc)
                    phase = .done
                    let labels = Set(pieces.map(\.label))
                    let what = labels.count == 1 ? "\(pieces.count) × \(labels.first!)" : "\(pieces.count) pieces"
                    detail = task == .cutout ? (pieces.count > 1 ? "\(what): tap one to leave it out" : "Save keeps the cut-out")
                        : "\(what), boxed"
                }
            } else if failure == nil {
                phase = .stopped
                detail = "Stopped"
            } else {
                phase = .failed
                detail = failure ?? "The request failed"
            }
            let run = Run(words: words, task: task, photoSource: photo.source, photoPixels: [image.width, image.height],
                          startedSinceStart: started, answer: answer, tokens: tokens, pieces: pieces.count,
                          seconds: result?.seconds ?? [:], composeSeconds: composeSeconds,
                          stop: failure != nil ? "error" : (result?.generation.stop ?? "stopped"), error: failure,
                          footprint: memoryFootprint())
            lastRun = run
            onFinished?(run)
        }
        runTask = t
        return t
    }

    func stop() { runTask?.cancel() }

    /// A tap on a piece: in or out of the cut-out.
    func toggle(_ id: Int) {
        guard phase == .done, let i = pieces.firstIndex(where: { $0.id == id }) else { return }
        pieces[i].selected.toggle()
        saveLine = nil
        Task { await composeCutout() }
    }

    /// The screen's cut-out: the chosen pieces' matte over the photo, at most 1024 px on the long side, premultiplied.
    private func composeCutout() async {
        guard let photo, task == .cutout else { cutoutImage = nil; return }
        let chosen = pieces.filter(\.selected).compactMap(\.logits)
        guard !chosen.isEmpty else { cutoutImage = nil; return }
        let display = photo.display
        cutoutImage = await Task.detached(priority: .userInitiated) {
            Self.preview(display, logits: chosen)
        }.value
    }

    nonisolated static func preview(_ display: CGImage, logits: [[Float]], maxSide: Int = 1024) -> CGImage? {
        let scale = min(1, Double(maxSide) / Double(max(display.width, display.height)))
        let w = max(1, Int((Double(display.width) * scale).rounded())), h = max(1, Int((Double(display.height) * scale).rounded()))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(display, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { return nil }
        let alpha = Alpha.combined8(logits, width: w, height: h)
        let px = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        for i in 0..<(w * h) {
            let a = UInt16(alpha[i])
            px[i * 4] = UInt8((UInt16(px[i * 4]) * a + 127) / 255)
            px[i * 4 + 1] = UInt8((UInt16(px[i * 4 + 1]) * a + 127) / 255)
            px[i * 4 + 2] = UInt8((UInt16(px[i * 4 + 2]) * a + 127) / 255)
            px[i * 4 + 3] = UInt8(a)
        }
        return ctx.makeImage()
    }

    // MARK: - Save

    /// The PNG of the chosen pieces at the photo's full size: Photos on the phone, a file on the Mac.
    @discardableResult
    func save() -> Task<Void, Never>? {
        guard phase == .done, task == .cutout, let photo else { return nil }
        let chosen = pieces.filter(\.selected).compactMap(\.logits)
        guard !chosen.isEmpty else {
            saveLine = "Choose at least one piece to save"
            return nil
        }
        saveLine = "Saving…"
        let rgb = photo.rgb
        let injected = injectedPhotosDenied
        return Task {
            let tp = ContinuousClock.now
            let png: Data
            do {
                png = try await Task.detached(priority: .userInitiated) {
                    try CutoutImage.png(rgb, alpha: Alpha.combined8(chosen, width: rgb.width, height: rgb.height))
                }.value
            } catch {
                saveLine = "Could not make the PNG: \(error.localizedDescription)"
                onSaved?("error png \(error.localizedDescription)")
                return
            }
            let pngSeconds = Self.seconds(ContinuousClock.now - tp)
            #if os(iOS)
            let outcome = await Self.saveToPhotos(png, injectedDenied: injected)
            saveLine = outcome.line
            onSaved?("\(outcome.record) bytes=\(png.count) png_s=\(String(format: "%.3f", pngSeconds))")
            #else
            let url = await Self.chooseFile()
            if let url {
                do {
                    try png.write(to: url)
                    saveLine = "Saved \(url.lastPathComponent) (\(png.count / 1024) KB)"
                    onSaved?("saved file=\(url.path) bytes=\(png.count) png_s=\(String(format: "%.3f", pngSeconds))")
                } catch {
                    saveLine = "Could not write the file: \(error.localizedDescription)"
                    onSaved?("error file \(error.localizedDescription)")
                }
            } else {
                saveLine = nil
                onSaved?("cancelled")
            }
            #endif
        }
    }

    struct SaveOutcome {
        let line: String
        let record: String
    }

    /// Adds the PNG to Photos (add-only access: the app never reads the library).
    nonisolated static func saveToPhotos(_ png: Data, injectedDenied: Bool) async -> SaveOutcome {
        var status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        if injectedDenied { status = .denied }
        if status == .notDetermined { status = await PHPhotoLibrary.requestAuthorization(for: .addOnly) }
        guard status == .authorized || status == .limited else {
            let how = injectedDenied ? "injected" : "\(status.rawValue)"
            return SaveOutcome(line: "Photos access is off: Settings › Apps › Cutout › Photos", record: "denied status=\(how)")
        }
        final class Box: @unchecked Sendable { var id: String? }
        let box = Box()
        do {
            try await PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                let options = PHAssetResourceCreationOptions()
                options.originalFilename = "cutout.png"
                request.addResource(with: .photo, data: png, options: options)
                box.id = request.placeholderForCreatedAsset?.localIdentifier
            }
            return SaveOutcome(line: "Saved to Photos", record: "saved photos_local_id=\(box.id ?? "-")")
        } catch {
            return SaveOutcome(line: "Could not save to Photos: \(error.localizedDescription)", record: "error photos \(error.localizedDescription)")
        }
    }

    #if os(macOS)
    /// The Mac's save panel; `-saveto <dir>` writes there instead (a script).
    static func chooseFile() async -> URL? {
        let name = "cutout-\(Int(Date().timeIntervalSince1970)).png"
        if let dir = UserDefaults.standard.string(forKey: "saveto") {
            return URL(fileURLWithPath: dir).appending(path: name)
        }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = name
        return await panel.begin() == .OK ? panel.url : nil
    }
    #endif

    // MARK: - What the screen shows

    var numbersHead: String {
        guard let b = boxSeconds else { return phase == .working ? "\(tokens) tokens…" : "– s" }
        var parts = [String(format: "boxes %.2f s", b)]
        if let m = matteSeconds { parts.append(String(format: "matte %.2f s", m)) }
        parts.append(pieces.count == 1 ? "1 found" : "\(pieces.count) found")
        return parts.joined(separator: " · ")
    }

    var numbersTail: String { "GPU · on-device" }

    var footer: String {
        ["WeVisOne-VL-2B", pipeline.map { $0.folder.decoder.deletingPathExtension().lastPathComponent } ?? "–",
         CutoutDevice.name, CutoutDevice.os, "Cutout \(CutoutDevice.appVersion)"].joined(separator: " · ")
    }

    // MARK: - Helpers

    private func loadThumbnails() {
        for example in Self.examples {
            guard let url = example.url, let file = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let thumb = CGImageSourceCreateThumbnailAtIndex(file, 0, [
                      kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 480,
                      kCGImageSourceCreateThumbnailWithTransform: true,
                  ] as CFDictionary)
            else { continue }
            // a wide strip around the example's focus (the tile is about 2.5 : 1)
            let h = min(Double(thumb.height), Double(thumb.width) / 2.5)
            let y = min(max(0, Double(thumb.height) * example.focusY - h / 2), Double(thumb.height) - h)
            thumbnails[example.id] = thumb.cropping(to: CGRect(x: 0, y: y, width: Double(thumb.width), height: h)) ?? thumb
        }
    }

    nonisolated static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
    }
}

/// The phone (or Mac) the numbers came from, for the footer and the record.
enum CutoutDevice {
    static let machine: String = {
        #if os(macOS)
        return sysctlString("hw.model")
        #else
        return sysctlString("hw.machine")
        #endif
    }()

    static let knownNames = ["iPhone19,2": "iPhone 18 Pro"]
    static var name: String { knownNames[machine] ?? machine }

    static let osVersion: String = {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return v.patchVersion == 0 ? "\(v.majorVersion).\(v.minorVersion)" : "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }()

    static let osBuild = sysctlString("kern.osversion")

    static var os: String {
        #if os(macOS)
        return "macOS \(osVersion)"
        #else
        return "iOS \(osVersion)"
        #endif
    }

    static var appVersion: String { Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?" }

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
