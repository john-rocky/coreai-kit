import AVFoundation
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
    var detections: [Detection] = []
    /// Capture frame size (post-rotation, portrait) — the space detections live in.
    var frameSize = CGSize(width: 720, height: 1280)
    var session: AVCaptureSession?
    var status = "Loading model…"
    var inferenceMS: Double?
    var wallFPS: Double?
    var variant: DetectVariant = .nano {
        didSet { if oldValue != variant { restart() } }
    }

    private var feed: CameraFeed?
    private var runTask: Task<Void, Never>?
    private var prepTask: Task<Void, Never>?
    private var inferTask: Task<Void, Never>?

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
        prepTask?.cancel()
        prepTask = nil
        inferTask?.cancel()
        inferTask = nil
        runTask?.cancel()
        runTask = nil
    }

    private func restart() {
        stop()
        detections = []
        inferenceMS = nil
        wallFPS = nil
        session = nil
        status = "Loading model…"
        start()
    }

    private func run() async {
        do {
            let detector = try await loadDetector()
            try await gate(detector)
            status = "Starting camera…"
            // AVCaptureVideoPreviewLayer renders the preview directly; we only pay
            // for inference. bufferingNewest(1) drops stale frames while the model
            // runs. Capture rate trades against inference latency (the ISP/preview
            // competes with the GPU): ~40 fps is the measured sweet spot for nano
            // (60 fps capture slows inference 25->39 ms and LOWERS throughput).
            let targetFPS = Double(ProcessInfo.processInfo.environment["DETECT_FPS"] ?? "") ?? 60
            // Data-output buffers scaled to ~model size in hardware; the preview
            // layer still shows the full 720p feed at the capture rate.
            let feed = CameraFeed(
                framesPerSecond: targetFPS, preset: .hd1280x720,
                dataOutputSize: CGSize(width: 384, height: 512))
            self.feed = feed
            let frames = try await feed.startPixelBuffers()
            session = feed.captureSession
            status = "Live · \(variant.rawValue)"

            // Two-stage pipeline OFF the main actor: stage 1 preprocesses frames on
            // the CPU while stage 2 runs the previous frame on the GPU; UI updates
            // are fire-and-forget hops so they never sit on the inference path.
            let (prepared, preparedCont) = AsyncStream<ObjectDetector.PreparedInput>
                .makeStream(bufferingPolicy: .bufferingNewest(1))
            prepTask = Task.detached { [weak self] in
                var first = true
                for await frame in frames {
                    if Task.isCancelled { break }
                    if first {
                        first = false
                        let size = CGSize(
                            width: CVPixelBufferGetWidth(frame.pixelBuffer),
                            height: CVPixelBufferGetHeight(frame.pixelBuffer))
                        Task { @MainActor [weak self] in self?.frameSize = size }
                    }
                    if let input = try? detector.prepare(frame.pixelBuffer) {
                        preparedCont.yield(input)
                    }
                }
                preparedCont.finish()
            }

            let variantName = variant.rawValue
            inferTask = Task.detached { [weak self] in
                var inferWindow: [Double] = []
                var windowStart = SuspendingClock.now
                do {
                    for await input in prepared {
                        if Task.isCancelled { break }
                        let start = SuspendingClock.now
                        let dets = try await detector.detect(input, scoreThreshold: 0.5)
                        let c = (SuspendingClock.now - start).components
                        let ms = Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
                        Task { @MainActor [weak self] in self?.detections = dets }
                        inferWindow.append(ms)
                        if inferWindow.count == 30 {
                            let w = (SuspendingClock.now - windowStart).components
                            let wall = Double(w.seconds) * 1000 + Double(w.attoseconds) / 1e15
                            let sorted = inferWindow.sorted()
                            let fps = 30_000 / wall
                            NSLog(
                                "STATS variant=%@ median=%.1fms mean=%.1fms wallFPS=%.1f",
                                variantName, sorted[15],
                                inferWindow.reduce(0, +) / 30, fps)
                            Task { @MainActor [weak self] in
                                self?.inferenceMS = sorted[15]
                                self?.wallFPS = fps
                            }
                            inferWindow.removeAll()
                            windowStart = SuspendingClock.now
                        }
                    }
                } catch {
                    NSLog("ERROR %@", String(describing: error))
                    Task { @MainActor [weak self] in
                        self?.status = "Error: \(error.localizedDescription)"
                    }
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
            let url = Bundle.main.url(forResource: "gate_image", withExtension: "jpg"),
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
            ZStack {
                if let session = model.session {
                    CameraPreviewView(session: session)
                        .overlay {
                            DetectionOverlay(
                                detections: model.detections, frameSize: model.frameSize)
                        }
                } else {
                    Rectangle()
                        .fill(Color.gray.opacity(0.15))
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
                if let fps = model.wallFPS {
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

/// AVCaptureVideoPreviewLayer-backed view: the system compositor renders the camera
/// directly — zero per-frame app work, full capture frame rate.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        if uiView.previewLayer.session !== session {
            uiView.previewLayer.session = session
        }
    }
}

struct DetectionOverlay: View {
    let detections: [Detection]
    /// Size of the capture frame the normalized boxes refer to.
    let frameSize: CGSize

    private static let palette: [Color] = [
        .red, .green, .blue, .orange, .purple, .cyan, .yellow, .mint, .pink,
    ]

    var body: some View {
        GeometryReader { geo in
            // The preview uses aspect-FILL: the frame is scaled by max(...) and
            // center-cropped. Map normalized frame coords into view points.
            let scale = max(
                geo.size.width / frameSize.width, geo.size.height / frameSize.height)
            let offsetX = (geo.size.width - frameSize.width * scale) / 2
            let offsetY = (geo.size.height - frameSize.height * scale) / 2
            ForEach(detections) { det in
                let rect = CGRect(
                    x: det.box.origin.x * frameSize.width * scale + offsetX,
                    y: det.box.origin.y * frameSize.height * scale + offsetY,
                    width: det.box.width * frameSize.width * scale,
                    height: det.box.height * frameSize.height * scale)
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
            .clipped()
        }
    }
}
