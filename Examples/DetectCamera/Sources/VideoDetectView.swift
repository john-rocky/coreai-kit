// VideoDetectView.swift — run the detector over a picked video file (in addition to the
// live camera). AVPlayer plays + displays the clip; AVPlayerItemVideoOutput hands each
// decoded frame (32BGRA) to the same LiveDetector used live, and the boxes are drawn with
// the shared DetectionOverlay. Frames are pulled on a @MainActor async loop (no
// CADisplayLink / Timer — Swift-6-clean), and detection itself throttles the loop: the
// next frame is only pulled once the previous detect() returns, so we always run on the
// latest available frame and drop the rest.

import AVFoundation
import CoreAIKitVision
import CoreTransferable
import CoreVideo
import PhotosUI
import QuartzCore
import SwiftUI
import UniformTypeIdentifiers

/// Loads a picked PhotosPicker video into a temp file we can hand to AVURLAsset.
struct PickedMovie: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let dst = URL.temporaryDirectory.appending(path: "detect_\(UUID().uuidString).mov")
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.copyItem(at: received.file, to: dst)
            return PickedMovie(url: dst)
        }
    }
}

/// Display rotation derived from a video track's preferredTransform.
enum VideoRotation {
    case none, cw90, cw180, cw270

    /// Maps a detection box (normalized in the RAW decoded frame) into the displayed,
    /// transform-applied frame so it lines up with what AVPlayerLayer shows.
    func map(_ r: CGRect) -> CGRect {
        switch self {
        case .none: r
        case .cw90: CGRect(x: 1 - r.minY - r.height, y: r.minX, width: r.height, height: r.width)
        case .cw180: CGRect(x: 1 - r.minX - r.width, y: 1 - r.minY - r.height, width: r.width, height: r.height)
        case .cw270: CGRect(x: r.minY, y: 1 - r.minX - r.width, width: r.height, height: r.width)
        }
    }
}

@MainActor
@Observable
final class VideoDetectModel {
    var detections: [Detection] = []
    /// Displayed (transform-applied) frame size — the space the normalized boxes map into.
    var frameSize = CGSize(width: 1280, height: 720)
    var status = "Pick a video to detect"
    var inferenceMS: Double?
    /// Live confidence cutoff fed to the detector — adjustable via the slider.
    var scoreThreshold: Float = 0.5
    var player: AVPlayer?
    var hasVideo = false

    var variant: DetectVariant = .yolox {
        didSet { if oldValue != variant { reloadDetector() } }
    }
    var unit: DetectUnit = .gpu {
        didSet { if oldValue != unit { reloadDetector() } }
    }
    var pickerItem: PhotosPickerItem? {
        didSet { if let pickerItem { Task { await open(pickerItem) } } }
    }

    private var detector: LiveDetector?
    private var output: AVPlayerItemVideoOutput?
    private var rotation: VideoRotation = .none
    private var loop: Task<Void, Never>?
    private var currentURL: URL?

    func stop() {
        loop?.cancel()
        loop = nil
        player?.pause()
    }

    private func reloadDetector() {
        detector = nil
        if let url = currentURL { Task { await openURL(url) } }
    }

    /// Loads the chosen model (monolith; the video path skips the camera's ANE split).
    private func loadDetector() async throws -> LiveDetector {
        if let detector { return detector }
        let sideloaded = URL.documentsDirectory
            .appending(path: "Models/\(variant.bundledName).aimodel")
        let exists = FileManager.default.fileExists(atPath: sideloaded.path)
        let progress: @Sendable (DownloadProgress) -> Void = { p in
            Task { @MainActor in self.status = "Downloading model… \(Int(p.fraction * 100))%" }
        }
        let made: LiveDetector
        if variant.isYOLOX {
            made = .yolox(
                exists
                    ? try await YOLOXDetector(bundleAt: sideloaded, computeUnits: unit.computeUnits)
                    : try await YOLOXDetector(model: variant.modelID, computeUnits: unit.computeUnits, downloadProgress: progress))
        } else {
            made = .detr(
                exists
                    ? try await ObjectDetector(bundleAt: sideloaded, computeUnits: unit.computeUnits)
                    : try await ObjectDetector(model: variant.modelID, computeUnits: unit.computeUnits, downloadProgress: progress))
        }
        detector = made
        return made
    }

    private func open(_ item: PhotosPickerItem) async {
        status = "Loading video…"
        do {
            guard let movie = try await item.loadTransferable(type: PickedMovie.self) else {
                status = "Could not load that video"
                return
            }
            await openURL(movie.url)
        } catch {
            status = "Load error: \(error.localizedDescription)"
        }
    }

    private func openURL(_ url: URL) async {
        stop()
        currentURL = url
        do {
            let detector = try await loadDetector()

            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                status = "No video track"
                return
            }
            let (transform, natural) = try await track.load(.preferredTransform, .naturalSize)
            (rotation, frameSize) = Self.orientation(transform: transform, natural: natural)

            let item = AVPlayerItem(asset: asset)
            let output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ])
            item.add(output)
            self.output = output
            let player = AVPlayer(playerItem: item)
            player.actionAtItemEnd = .none
            self.player = player
            hasVideo = true
            status = "Detecting · \(variant.rawValue) · \(unit.rawValue)"
            player.play()

            loop = Task { @MainActor [weak self] in await self?.run(output: output, detector: detector, player: player) }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    /// Frame pump + detection on the main actor. `detect` awaits the GPU (yielding the
    /// main thread), so the UI stays smooth and the loop self-throttles to the model rate.
    private func run(output: AVPlayerItemVideoOutput, detector: LiveDetector, player: AVPlayer) async {
        while !Task.isCancelled {
            // Seamless loop without a notification observer.
            if let dur = player.currentItem?.duration, dur.isNumeric, dur.seconds > 0,
                player.currentTime().seconds >= dur.seconds - 0.05
            {
                await player.seek(to: .zero)
                player.play()
            }
            let host = output.itemTime(forHostTime: CACurrentMediaTime())
            if output.hasNewPixelBuffer(forItemTime: host),
                let buffer = output.copyPixelBuffer(forItemTime: host, itemTimeForDisplay: nil)
            {
                do {
                    let prepared = try detector.prepare(buffer)  // sync CPU letterbox/scale
                    let start = SuspendingClock.now
                    let dets = try await detector.detect(prepared, scoreThreshold: scoreThreshold)
                    let c = (SuspendingClock.now - start).components
                    inferenceMS = Double(c.seconds) * 1000 + Double(c.attoseconds) / 1e15
                    detections = dets.map { Detection(classID: $0.classID, label: $0.label, score: $0.score, box: rotation.map($0.box)) }
                } catch {
                    NSLog("VIDEO detect error %@", String(describing: error))
                }
            } else {
                try? await Task.sleep(for: .milliseconds(8))
            }
        }
    }

    /// Classifies the preferred transform into a rotation + displayed size.
    private static func orientation(transform: CGAffineTransform, natural: CGSize) -> (VideoRotation, CGSize) {
        let deg = ((Int((atan2(transform.b, transform.a) * 180 / .pi).rounded()) % 360) + 360) % 360
        switch deg {
        case 90: return (.cw90, CGSize(width: natural.height, height: natural.width))
        case 180: return (.cw180, natural)
        case 270: return (.cw270, CGSize(width: natural.height, height: natural.width))
        default: return (.none, natural)
        }
    }
}

/// AVPlayerLayer-backed view. resizeAspectFill matches DetectionOverlay's aspect-fill
/// coordinate mapping (same as the camera preview).
struct PlayerLayerView: UIViewRepresentable {
    let player: AVPlayer

    final class LayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }

    func makeUIView(context: Context) -> LayerView {
        let view = LayerView()
        view.playerLayer.videoGravity = .resizeAspectFill
        view.playerLayer.player = player
        return view
    }

    func updateUIView(_ uiView: LayerView, context: Context) {
        if uiView.playerLayer.player !== player { uiView.playerLayer.player = player }
    }
}

struct VideoDetectView: View {
    @State private var model = VideoDetectModel()

    var body: some View {
        VStack(spacing: 10) {
            ZStack {
                if let player = model.player, model.hasVideo {
                    PlayerLayerView(player: player)
                        .overlay {
                            DetectionOverlay(detections: model.detections, frameSize: model.frameSize)
                        }
                } else {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.gray.opacity(0.15))
                        .overlay {
                            VStack(spacing: 10) {
                                Image(systemName: "film").font(.largeTitle).foregroundStyle(.secondary)
                                Text("Pick a video").foregroundStyle(.secondary)
                            }
                        }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .padding(.horizontal, 8)

            HStack(spacing: 8) {
                Picker("Model", selection: $model.variant) {
                    ForEach(DetectVariant.allCases) { v in Text(v.title).tag(v) }
                }
                .pickerStyle(.segmented)
                Picker("Unit", selection: $model.unit) {
                    ForEach(DetectUnit.allCases) { u in Text(u.rawValue.uppercased()).tag(u) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 140)
            }
            .padding(.horizontal, 12)

            ThresholdSlider(value: $model.scoreThreshold)
                .padding(.horizontal, 12)

            HStack {
                PhotosPicker(selection: $model.pickerItem, matching: .videos) {
                    Label(model.hasVideo ? "Change" : "Pick video", systemImage: "photo.on.rectangle")
                        .font(.caption.bold())
                }
                Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                if let ms = model.inferenceMS {
                    Text(String(format: "%.0f ms", ms)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 14)
        }
        .padding(.vertical, 8)
        .onDisappear { model.stop() }
    }
}

/// Top-level source switcher: live camera vs. a picked video, sharing the model/unit
/// pickers and the detection overlay.
struct RootView: View {
    enum Source: String, CaseIterable, Identifiable {
        case camera = "Camera", video = "Video"
        var id: String { rawValue }
    }
    @State private var source: Source = .camera

    var body: some View {
        VStack(spacing: 8) {
            Picker("Source", selection: $source) {
                ForEach(Source.allCases) { s in Text(s.rawValue).tag(s) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 12)
            .padding(.top, 6)

            switch source {
            case .camera: DetectCameraView()
            case .video: VideoDetectView()
            }
        }
    }
}
