import CoreAIKitVision
import SwiftUI

enum DetectVariant: String, CaseIterable, Identifiable {
    case nano, medium
    var id: String { rawValue }
    var modelID: ModelID { self == .nano ? .rfdetrNano : .rfdetrMedium }
    var bundledName: String { "rfdetr-\(rawValue)_float32" }
}

@MainActor
@Observable
final class DetectCameraModel {
    var cameraImage: CGImage?
    var detections: [Detection] = []
    var status = "Loading model…"
    var inferenceMS: Double?
    var fps: Double?
    var variant: DetectVariant = .nano {
        didSet { if oldValue != variant { restart() } }
    }

    private var feed: CameraFeed?
    private var runTask: Task<Void, Never>?

    init() {
        // Bench hook: DETECT_VARIANT=nano|medium selects the initial model
        // (didSet does not fire during init, so this won't trigger a restart).
        if let raw = ProcessInfo.processInfo.environment["DETECT_VARIANT"],
            let v = DetectVariant(rawValue: raw)
        {
            variant = v
        }
    }

    func start() {
        guard runTask == nil else { return }
        runTask = Task { await run() }
    }

    func stop() {
        feed?.stop()
        runTask?.cancel()
        runTask = nil
    }

    private func restart() {
        stop()
        cameraImage = nil
        detections = []
        inferenceMS = nil
        fps = nil
        status = "Loading model…"
        start()
    }

    private func run() async {
        do {
            let detector = try await loadDetector()
            try await gate(detector)
            status = "Starting camera…"
            let feed = CameraFeed(framesPerSecond: 60)
            self.feed = feed
            var window: [Double] = []
            var lastFrame = SuspendingClock.now
            for await frame in try await feed.start() {
                guard !Task.isCancelled else { break }
                let start = SuspendingClock.now
                let dets = try await detector.detect(in: frame, scoreThreshold: 0.5)
                let ms = millis(since: start)
                let frameMS = millis(since: lastFrame)
                lastFrame = SuspendingClock.now
                cameraImage = frame
                detections = dets
                inferenceMS = ms
                fps = frameMS > 0 ? min(1000 / frameMS, 60) : nil
                status = "Live · \(variant.rawValue)"
                window.append(ms)
                if window.count == 30 {
                    let sorted = window.sorted()
                    NSLog(
                        "STATS variant=%@ median=%.1fms mean=%.1fms fps=%.1f",
                        variant.rawValue, sorted[15],
                        window.reduce(0, +) / 30, 1000 / sorted[15])
                    window.removeAll()
                }
            }
        } catch is CancellationError {
        } catch {
            status = "Error: \(error.localizedDescription)"
            NSLog("ERROR %@", String(describing: error))
        }
    }

    /// Model resolution order: app Documents (dev sideload via devicectl), else
    /// Hugging Face download. (.aimodel directories cannot ship inside the app
    /// bundle — the installer mistakes extension-suffixed root folders for
    /// nested bundles and rejects the app.)
    private func loadDetector() async throws -> ObjectDetector {
        let sideloaded = URL.documentsDirectory
            .appending(path: "Models/\(variant.bundledName).aimodel")
        if FileManager.default.fileExists(atPath: sideloaded.path) {
            NSLog("MODEL sideloaded %@", variant.bundledName)
            return try await ObjectDetector(bundleAt: sideloaded)
        }
        NSLog("MODEL downloading %@", variant.bundledName)
        return try await ObjectDetector(model: variant.modelID) { progress in
            Task { @MainActor in
                self.status = "Downloading… \(Int(progress.fraction * 100))%"
            }
        }
    }

    /// On-device numerics gate: detect on the bundled reference photo and log every
    /// confident detection (compare against the Mac fp32 oracle from the console).
    private func gate(_ detector: ObjectDetector) async throws {
        guard
            let url = Bundle.main.url(
                forResource: "gate_image", withExtension: "jpg", subdirectory: "Resources")
                ?? Bundle.main.url(forResource: "gate_image", withExtension: "jpg"),
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            NSLog("GATE skipped: no gate_image in bundle")
            return
        }
        status = "Gate…"
        let warm = SuspendingClock.now
        _ = try await detector.detect(in: image, scoreThreshold: 0.3)
        NSLog("GATE warmup %.0fms", millis(since: warm))
        let start = SuspendingClock.now
        let dets = try await detector.detect(in: image, scoreThreshold: 0.3)
        let ms = millis(since: start)
        for d in dets {
            NSLog(
                "GATE det id=%d %@ score=%.3f box=[%.3f,%.3f,%.3f,%.3f]",
                d.classID, d.label, d.score,
                d.box.origin.x, d.box.origin.y, d.box.width, d.box.height)
        }
        NSLog("GATE done variant=%@ n=%d time=%.1fms", variant.rawValue, dets.count, ms)
    }

    private func millis(since start: SuspendingClock.Instant) -> Double {
        let c = (SuspendingClock.now - start).components
        return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
    }
}

struct DetectCameraView: View {
    @State private var model = DetectCameraModel()

    var body: some View {
        VStack(spacing: 10) {
            ZStack(alignment: .topLeading) {
                if let image = model.cameraImage {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFit()
                        .overlay { DetectionOverlay(detections: model.detections) }
                } else {
                    Rectangle()
                        .fill(Color.gray.opacity(0.15))
                        .aspectRatio(3 / 4, contentMode: .fit)
                        .overlay { ProgressView() }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .padding(.horizontal, 8)

            Picker("Model", selection: $model.variant) {
                ForEach(DetectVariant.allCases) { v in
                    Text(v.rawValue.capitalized).tag(v)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 12)

            HStack {
                Text(model.status)
                Spacer()
                if let ms = model.inferenceMS {
                    Text(String(format: "%.0f ms", ms))
                }
                if let fps = model.fps {
                    Text(String(format: "· %.0f FPS", fps))
                }
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 14)
        }
        .padding(.vertical, 8)
        .task { model.start() }
        .onDisappear { model.stop() }
    }
}

struct DetectionOverlay: View {
    let detections: [Detection]

    private static let palette: [Color] = [
        .red, .green, .blue, .orange, .purple, .cyan, .yellow, .mint, .pink,
    ]

    var body: some View {
        GeometryReader { geo in
            ForEach(detections) { det in
                let rect = CGRect(
                    x: det.box.origin.x * geo.size.width,
                    y: det.box.origin.y * geo.size.height,
                    width: det.box.width * geo.size.width,
                    height: det.box.height * geo.size.height)
                let color = Self.palette[det.classID % Self.palette.count]
                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 4)
                        .stroke(color, lineWidth: 3)
                    Text("\(det.label) \(String(format: "%.2f", det.score))")
                        .font(.caption2.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(color, in: RoundedRectangle(cornerRadius: 3))
                        .offset(y: -16)
                }
                .frame(width: rect.width, height: rect.height)
                .offset(x: rect.origin.x, y: rect.origin.y)
            }
        }
    }
}
