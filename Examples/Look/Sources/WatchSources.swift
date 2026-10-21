// WatchSources — where the watch screen's frames come from: the back camera (CameraFeed, its
// preview), a video (VideoFeed: played from its start at Start, the frame on the screen taken when
// the loop needs one), and the crop that makes the model read what the preview shows.

import AVFoundation
import CoreGraphics
import CoreMedia
import CoreTransferable
import Foundation
import ImageIO
import QuartzCore
import SwiftUI
import UniformTypeIdentifiers
import VideoToolbox
#if os(iOS)
import UIKit
#elseif os(macOS)
import AppKit
#endif

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
/// The live camera, filling its box (the frames the model reads are cropped the same way).
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

/// A picked video, copied out of the photo library into a file AVFoundation can open (the system
/// picker runs outside the app: no permission needed).
struct PickedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = URL.temporaryDirectory.appending(path: "watch-\(UUID().uuidString).\(received.file.pathExtension)")
            try? FileManager.default.removeItem(at: copy)
            try FileManager.default.copyItem(at: received.file, to: copy)
            return PickedMovie(url: copy)
        }
    }
}

/// A video, muted (the reactions make the sounds): the preview shows it, and `take()` returns the
/// frame on the screen — the one the player shows at that moment (AVPlayerItemVideoOutput on the
/// player's own item, as DetectCamera reads a video: one decode for the screen and the model). It
/// opens paused on its first frame; Start plays it from the start (`restart()`), and a run over a video
/// ends with it (`ended`): it does not loop.
@MainActor
final class VideoFeed {
    let player: AVPlayer
    let name: String
    let duration: Double
    let naturalSize: CGSize
    /// The track's display rotation in degrees (an upright phone video: 90).
    let angle: Int
    let orientation: CGImagePropertyOrientation
    /// The video played to its end since the last `restart()`.
    private(set) var ended = false
    private let output: AVPlayerItemVideoOutput
    private var last: (buffer: CVPixelBuffer, time: Double)?
    private var endObserver: NSObjectProtocol?

    static func open(_ url: URL) async throws -> VideoFeed {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw WatchError.video("the file has no video track")
        }
        let (transform, natural) = try await track.load(.preferredTransform, .naturalSize)
        let duration = try await asset.load(.duration).seconds
        return VideoFeed(url: url, asset: asset, transform: transform, natural: natural, duration: duration)
    }

    private init(url: URL, asset: AVURLAsset, transform: CGAffineTransform, natural: CGSize, duration: Double) {
        let degrees = ((Int((atan2(transform.b, transform.a) * 180 / .pi).rounded()) % 360) + 360) % 360
        angle = degrees
        switch degrees {
        case 90: orientation = .right
        case 180: orientation = .down
        case 270: orientation = .left
        default: orientation = .up
        }
        name = url.lastPathComponent
        naturalSize = natural
        self.duration = duration.isFinite ? duration : 0
        let item = AVPlayerItem(asset: asset)
        output = AVPlayerItemVideoOutput(pixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        item.add(output)
        player = AVPlayer(playerItem: item)
        player.isMuted = true
        player.actionAtItemEnd = .pause
        endObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification, object: item, queue: .main
        ) { [weak self] _ in
            guard let feed = self else { return }
            Task { @MainActor in feed.ended = true }
        }
    }

    var isPlaying: Bool { player.rate != 0 }

    /// The play button: from the start again when it is at the end.
    func play() {
        if ended {
            ended = false
            player.seek(to: .zero)
        }
        player.play()
    }

    func pause() { player.pause() }

    /// Start: from the first frame, playing (the frames taken before are forgotten).
    func restart() {
        ended = false
        last = nil
        player.seek(to: .zero, toleranceBefore: .zero, toleranceAfter: .zero)
        player.play()
    }

    /// Stops it for good (another video, or another source).
    func close() {
        player.pause()
        if let endObserver { NotificationCenter.default.removeObserver(endObserver) }
        endObserver = nil
    }

    /// The frame on the screen now (the last one the output gave when no newer one is there), as
    /// stored in the file — `orientation` turns it upright — and its time in the video.
    func take() -> (image: CGImage, time: Double)? {
        let now = output.itemTime(forHostTime: CACurrentMediaTime())
        if output.hasNewPixelBuffer(forItemTime: now),
            let buffer = output.copyPixelBuffer(forItemTime: now, itemTimeForDisplay: nil)
        {
            last = (buffer, now.seconds)
        }
        guard let last else { return nil }
        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(last.buffer, options: nil, imageOut: &image)
        return image.map { ($0, last.time) }
    }
}

/// The video, filling its box (the frames the model reads are cropped the same way).
#if os(iOS)
struct PlayerPreview: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.playerLayer.videoGravity = .resizeAspectFill
        view.playerLayer.player = player
        return view
    }

    func updateUIView(_ view: PlayerView, context: Context) {
        if view.playerLayer.player !== player { view.playerLayer.player = player }
    }

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }

        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
}
#elseif os(macOS)
struct PlayerPreview: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = .resizeAspectFill
        view.layer = layer
        view.wantsLayer = true
        return view
    }

    func updateNSView(_ view: NSView, context: Context) {
        if let layer = view.layer as? AVPlayerLayer, layer.player !== player { layer.player = player }
    }
}
#endif

/// The part of a frame the preview shows. The camera and video previews fill their box, so they
/// show the middle of the picture; the model gets that same middle part, cropped before the kit
/// draws it into the model's input.
enum WatchCrop {
    /// The centered rectangle of an image as stored (before `orientation` turns it upright) that a
    /// box of `aspect` (width / height) shows when the upright picture fills it.
    static func rect(width: Int, height: Int, orientation: CGImagePropertyOrientation, aspect: Double) -> CGRect {
        let full = CGRect(x: 0, y: 0, width: width, height: height)
        let quarter = [.left, .right, .leftMirrored, .rightMirrored].contains(orientation)
        let uprightWidth = Double(quarter ? height : width)
        let uprightHeight = Double(quarter ? width : height)
        guard aspect > 0, uprightWidth > 0, uprightHeight > 0 else { return full }
        var cropWidth = uprightWidth
        var cropHeight = uprightHeight
        if uprightWidth / uprightHeight > aspect {
            cropWidth = uprightHeight * aspect
        } else {
            cropHeight = uprightWidth / aspect
        }
        let storedWidth = (quarter ? cropHeight : cropWidth).rounded(.down)
        let storedHeight = (quarter ? cropWidth : cropHeight).rounded(.down)
        return CGRect(
            x: ((Double(width) - storedWidth) / 2).rounded(.down), y: ((Double(height) - storedHeight) / 2).rounded(.down),
            width: storedWidth, height: storedHeight)
    }

    /// The image cropped to what a box of `aspect` shows (the whole image when the crop fails).
    static func crop(_ image: CGImage, orientation: CGImagePropertyOrientation, aspect: Double) -> (CGImage, CGRect) {
        let box = rect(width: image.width, height: image.height, orientation: orientation, aspect: aspect)
        guard box.width >= 16, box.height >= 16, let cropped = image.cropping(to: box) else {
            return (image, CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return (cropped, box)
    }
}
