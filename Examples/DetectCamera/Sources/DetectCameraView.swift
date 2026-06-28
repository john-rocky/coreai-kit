import AVFoundation
import CoreAIKitVision
import SwiftUI

enum DetectVariant: String, CaseIterable, Identifiable {
    case nano, medium
    case segNano = "seg-nano"
    case yolox
    var id: String { rawValue }
    var title: String {
        switch self {
        case .segNano: "Seg"
        case .yolox: "YOLOX"
        default: rawValue.capitalized
        }
    }
    /// YOLOX is a dense detector (host NMS); the others are RF-DETR (DETR, no NMS).
    var isYOLOX: Bool { self == .yolox }
    var modelID: ModelID {
        switch self {
        case .nano: .rfdetrNano
        case .medium: .rfdetrMedium
        case .segNano: .rfdetrSegNano
        case .yolox: .yoloxS
        }
    }
    /// Split (backbone/head) bundles exist for the RF-DETR detection variants only.
    var splitIDs: (backbone: ModelID, head: ModelID)? {
        switch self {
        case .nano: (.rfdetrNanoBackbone, .rfdetrNanoHead)
        case .medium: (.rfdetrMediumBackbone, .rfdetrMediumHead)
        case .segNano, .yolox: nil
        }
    }
    var bundledName: String {
        switch self {
        case .yolox: "yolox-s_float32"
        default: "rfdetr-\(rawValue)_float32"
        }
    }
    /// Camera data-output size feeding the model (portrait). YOLOX-S is 640², RF-DETR
    /// nano/medium ~384–576; the detectors letterbox/scale internally, so deliver near
    /// the model resolution.
    var dataOutputSize: CGSize {
        isYOLOX ? CGSize(width: 480, height: 640) : CGSize(width: 384, height: 512)
    }
}

/// A frame preprocessed for whichever detector is live. RF-DETR (square-resize RGB,
/// no NMS) and YOLOX (letterbox BGR, host NMS) have different graph contracts, so the
/// pipeline carries the prepared input tagged by kind.
enum PreparedFrame: Sendable {
    case detr(ObjectDetector.PreparedInput)
    case yolox(YOLOXDetector.PreparedInput)
}

/// The live detector, abstracting RF-DETR and YOLOX behind one prepare/detect surface.
enum LiveDetector: Sendable {
    case detr(ObjectDetector)
    case yolox(YOLOXDetector)

    var inputSize: Int {
        switch self {
        case .detr(let d): d.inputSize
        case .yolox(let d): d.inputSize
        }
    }

    func prepare(_ pixelBuffer: CVPixelBuffer) throws -> PreparedFrame {
        switch self {
        case .detr(let d): .detr(try d.prepare(pixelBuffer))
        case .yolox(let d): .yolox(try d.prepare(pixelBuffer))
        }
    }

    func detect(_ frame: PreparedFrame, scoreThreshold: Float) async throws -> [Detection] {
        switch (self, frame) {
        case let (.detr(d), .detr(input)):
            try await d.detect(input, scoreThreshold: scoreThreshold)
        case let (.yolox(d), .yolox(input)):
            try await d.detect(input, scoreThreshold: scoreThreshold)
        default:
            []
        }
    }

    func detect(in image: CGImage, scoreThreshold: Float) async throws -> [Detection] {
        switch self {
        case .detr(let d): try await d.detect(in: image, scoreThreshold: scoreThreshold)
        case .yolox(let d): try await d.detect(in: image, scoreThreshold: scoreThreshold)
        }
    }
}

enum DetectUnit: String, CaseIterable, Identifiable {
    case gpu, ane
    var id: String { rawValue }
    var computeUnits: GraphModel.ComputeUnits { self == .gpu ? .gpu : .neuralEngine }
}

@MainActor
@Observable
final class DetectCameraModel {
    var detections: [Detection] = []
    var maskImage: CGImage?
    /// Capture frame size (post-rotation, portrait) — the space detections live in.
    var frameSize = CGSize(width: 720, height: 1280)
    var session: AVCaptureSession?
    var status = "Loading model…"
    var inferenceMS: Double?
    var wallFPS: Double?
    /// Non-nil when the system reports thermal pressure (the GPU is throttled).
    var thermal: String?
    /// Live confidence cutoff fed to the detector — adjustable via the slider.
    var scoreThreshold: Float = 0.5
    var variant: DetectVariant = .nano {
        didSet { if oldValue != variant { restart() } }
    }
    var unit: DetectUnit = .gpu {
        didSet { if oldValue != unit { restart() } }
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
        if let raw = ProcessInfo.processInfo.environment["DETECT_UNIT"],
            let u = DetectUnit(rawValue: raw)
        {
            unit = u
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
                dataOutputSize: variant.dataOutputSize)
            self.feed = feed
            let frames = try await feed.startPixelBuffers()
            session = feed.captureSession
            status = "Live · \(variant.rawValue) · \(unit.rawValue)"

            // Two-stage pipeline OFF the main actor: stage 1 preprocesses frames on
            // the CPU while stage 2 runs the previous frame on the GPU; UI updates
            // are fire-and-forget hops so they never sit on the inference path.
            let (prepared, preparedCont) = AsyncStream<PreparedFrame>
                .makeStream(bufferingPolicy: .bufferingNewest(1))
            let dumpFrame = ProcessInfo.processInfo.environment["DETECT_DUMP"] == "1"
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
                        if dumpFrame { Self.dump(frame.pixelBuffer) }
                    }
                    if let input = try? detector.prepare(frame.pixelBuffer) {
                        preparedCont.yield(input)
                    }
                }
                preparedCont.finish()
            }

            let variantName = "\(variant.rawValue)/\(unit.rawValue)"
            let maskRenderer = MaskRenderer()
            inferTask = Task.detached { [weak self] in
                var inferWindow: [Double] = []
                var windowStart = SuspendingClock.now
                do {
                    for await input in prepared {
                        if Task.isCancelled { break }
                        let thr = await MainActor.run { self?.scoreThreshold ?? 0.5 }
                        let start = SuspendingClock.now
                        var dets = try await detector.detect(input, scoreThreshold: thr)
                        if ProcessInfo.processInfo.environment["DETECT_TESTBOX"] == "1" {
                            // debug: known normalized positions to validate overlay mapping
                            dets = [
                                (0.05, 0.05, "TL"), (0.75, 0.05, "TR"), (0.4, 0.4, "C"),
                                (0.05, 0.75, "BL"), (0.75, 0.75, "BR"),
                            ].enumerated().map { i, p in
                                Detection(
                                    classID: i + 1, label: p.2, score: 0.99,
                                    box: CGRect(x: p.0, y: p.1, width: 0.2, height: 0.2))
                            }
                        }
                        let c = (SuspendingClock.now - start).components
                        let ms = Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
                        let maskImage = maskRenderer.composite(dets)
                        Task { @MainActor [weak self] in
                            self?.detections = dets
                            self?.maskImage = maskImage
                        }
                        inferWindow.append(ms)
                        if inferWindow.count == 30 {
                            let w = (SuspendingClock.now - windowStart).components
                            let wall = Double(w.seconds) * 1000 + Double(w.attoseconds) / 1e15
                            let sorted = inferWindow.sorted()
                            let fps = 30_000 / wall
                            let proc = ProcessInfo.processInfo
                            let thermal = ["nominal", "fair", "serious", "critical"][
                                min(proc.thermalState.rawValue, 3)]
                            let lpm = proc.isLowPowerModeEnabled
                            NSLog(
                                "STATS variant=%@ median=%.1fms mean=%.1fms wallFPS=%.1f thermal=%@%@",
                                variantName, sorted[15],
                                inferWindow.reduce(0, +) / 30, fps, thermal,
                                lpm ? " lowPower=1" : "")
                            Task { @MainActor [weak self] in
                                self?.inferenceMS = sorted[15]
                                self?.wallFPS = fps
                                self?.thermal = proc.thermalState == .nominal ? nil : thermal
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
    private func loadDetector() async throws -> LiveDetector {
        let sideloaded = URL.documentsDirectory
            .appending(path: "Models/\(variant.bundledName).aimodel")

        // YOLOX: dense detector, no split — load the monolith on the chosen unit.
        if variant.isYOLOX {
            if FileManager.default.fileExists(atPath: sideloaded.path) {
                NSLog("MODEL sideloaded %@ unit=%@", variant.bundledName, unit.rawValue)
                return .yolox(
                    try await YOLOXDetector(bundleAt: sideloaded, computeUnits: unit.computeUnits))
            }
            NSLog("MODEL downloading %@ unit=%@", variant.bundledName, unit.rawValue)
            return .yolox(
                try await YOLOXDetector(model: variant.modelID, computeUnits: unit.computeUnits) {
                    progress in
                    Task { @MainActor in
                        self.status = "Downloading… \(Int(progress.fraction * 100))%"
                    }
                })
        }

        if unit == .ane, let split = variant.splitIDs {
            // ANE value needs the split deployment: a monolithic graph keeps the
            // whole model on the GPU delegate (the deformable head is not
            // ANE-lowerable). Backbone -> .neuralEngine, head -> .gpu.
            let bb = URL.documentsDirectory
                .appending(path: "Models/rfdetr-\(variant.rawValue)_backbone.aimodel")
            let head = URL.documentsDirectory
                .appending(path: "Models/rfdetr-\(variant.rawValue)_head.aimodel")
            if FileManager.default.fileExists(atPath: bb.path),
                FileManager.default.fileExists(atPath: head.path)
            {
                NSLog("MODEL split sideloaded %@ backbone=ane head=gpu", variant.rawValue)
                return .detr(try await ObjectDetector(backboneAt: bb, headAt: head))
            }
            NSLog("MODEL split downloading %@ backbone=ane head=gpu", variant.rawValue)
            return .detr(
                try await ObjectDetector(
                    backboneModel: split.backbone, headModel: split.head
                ) { progress in
                    Task { @MainActor in
                        self.status = "Downloading… \(Int(progress.fraction * 100))%"
                    }
                })
        }
        if FileManager.default.fileExists(atPath: sideloaded.path) {
            NSLog("MODEL sideloaded %@ unit=%@", variant.bundledName, unit.rawValue)
            return .detr(
                try await ObjectDetector(bundleAt: sideloaded, computeUnits: unit.computeUnits))
        }
        NSLog("MODEL downloading %@ unit=%@", variant.bundledName, unit.rawValue)
        return .detr(
            try await ObjectDetector(model: variant.modelID, computeUnits: unit.computeUnits) {
                progress in
                Task { @MainActor in
                    self.status = "Downloading… \(Int(progress.fraction * 100))%"
                }
            })
    }

    /// On-device numerics gate: detect on the bundled reference photo and log every
    /// confident detection (compare against the Mac fp32 oracle from the console).
    private func gate(_ detector: LiveDetector) async throws {
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
            let maskPx = d.mask.map(\.foregroundCount) ?? -1
            NSLog(
                "GATE det id=%d %@ score=%.3f box=[%.3f,%.3f,%.3f,%.3f] maskpx=%d",
                d.classID, d.label, d.score,
                d.box.origin.x, d.box.origin.y, d.box.width, d.box.height, maskPx)
        }
        NSLog(
            "GATE done variant=%@ unit=%@ n=%d time=%.1fms",
            variant.rawValue, unit.rawValue, dets.count, ms)
    }

    /// Debug: write the raw delivered buffer to Documents for host-side inspection.
    nonisolated static func dump(_ buffer: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var data = Data()
        var header = [Int32(w), Int32(h), Int32(stride)]
        data.append(Data(bytes: &header, count: 12))
        data.append(Data(bytes: base, count: stride * h))
        let url = URL.documentsDirectory.appending(path: "framedump.bin")
        try? data.write(to: url)
        NSLog("DUMP wrote %dx%d stride=%d to %@", w, h, stride, url.path)
    }

    private func millis(since start: SuspendingClock.Instant) -> Double {
        let c = (SuspendingClock.now - start).components
        return Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
    }
}

/// A compact confidence-threshold slider, shared by the camera and video views.
struct ThresholdSlider: View {
    @Binding var value: Float
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "slider.horizontal.3").font(.caption2).foregroundStyle(.secondary)
            Text("conf").font(.caption2).foregroundStyle(.secondary)
            Slider(value: $value, in: 0.05...0.95)
            Text(String(format: "%.2f", value))
                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                .frame(width: 32, alignment: .trailing)
        }
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
                                detections: model.detections, frameSize: model.frameSize,
                                maskImage: model.maskImage)
                        }
                } else {
                    Rectangle()
                        .fill(Color.gray.opacity(0.15))
                        .overlay { ProgressView() }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .padding(.horizontal, 8)

            HStack(spacing: 8) {
                Picker("Model", selection: $model.variant) {
                    ForEach(DetectVariant.allCases) { v in
                        Text(v.title).tag(v)
                    }
                }
                .pickerStyle(.segmented)
                Picker("Unit", selection: $model.unit) {
                    ForEach(DetectUnit.allCases) { u in
                        Text(u.rawValue.uppercased()).tag(u)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 140)
            }
            .padding(.horizontal, 12)

            ThresholdSlider(value: $model.scoreThreshold)
                .padding(.horizontal, 12)

            HStack {
                Text(model.status)
                if let thermal = model.thermal {
                    Text("🌡️ \(thermal)").foregroundStyle(.orange)
                }
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
    var maskImage: CGImage?

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
            // One top-leading ZStack with offset children; clip ONCE at the overlay
            // bounds. (.clipped() per ForEach child clips at the child's pre-offset
            // layout rect — every box away from the origin vanishes.)
            ZStack(alignment: .topLeading) {
                if let maskImage {
                    Image(decorative: maskImage, scale: 1)
                        .resizable()
                        .frame(
                            width: frameSize.width * scale, height: frameSize.height * scale)
                        .offset(x: offsetX, y: offsetY)
                }
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
            }
            .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
            .clipped()
        }
    }
}
