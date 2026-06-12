// CameraFeed.swift — live camera frames as an AsyncStream, throttled to a target frame
// rate. Two flavors: CGImage frames (simple, render-ready) and raw CVPixelBuffer frames
// (zero-conversion fast path for real-time pipelines — pair with
// `ObjectDetector.detect(in: CVPixelBuffer)` and an AVCaptureVideoPreviewLayer attached
// to `captureSession` so display costs nothing). The app must declare
// NSCameraUsageDescription.

import AVFoundation
import CoreImage
import Foundation

/// One captured frame. CVPixelBuffer is not Sendable; the producer never touches a
/// buffer again after yielding it and the consumer only reads, so the transfer is safe.
public struct CameraFrame: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
}

public final class CameraFeed: NSObject, @unchecked Sendable {
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "coreai.kit.camera")
    private let ciContext = CIContext()
    private let position: AVCaptureDevice.Position
    private let preset: AVCaptureSession.Preset
    private let targetFPS: Double
    private let minFrameInterval: Double
    private var lastFrameTime: Double = 0
    private var continuation: AsyncStream<CGImage>.Continuation?
    private var pixelContinuation: AsyncStream<CameraFrame>.Continuation?

    private let dataOutputSize: CGSize?

    /// `dataOutputSize` asks the capture pipeline to deliver data-output buffers
    /// scaled down in hardware (the preview keeps the full session resolution) —
    /// big ISP/GPU saving for inference consumers that resize anyway.
    public init(
        position: AVCaptureDevice.Position = .back, framesPerSecond: Double = 10,
        preset: AVCaptureSession.Preset = .vga640x480, dataOutputSize: CGSize? = nil
    ) {
        self.position = position
        self.preset = preset
        self.targetFPS = framesPerSecond
        self.minFrameInterval = 1.0 / max(framesPerSecond, 0.1)
        self.dataOutputSize = dataOutputSize
        super.init()
    }

    /// The underlying session — attach an `AVCaptureVideoPreviewLayer` to display the
    /// live feed for free (the compositor renders it; no per-frame CPU work).
    public var captureSession: AVCaptureSession { session }

    /// Requests camera permission if needed, then starts streaming frames. Only the
    /// newest frame is buffered — slow consumers skip frames instead of lagging.
    public func start() async throws -> AsyncStream<CGImage> {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            throw VisionError.cameraAccessDenied
        }
        try configure()
        let (stream, continuation) = AsyncStream<CGImage>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        self.continuation = continuation
        continuation.onTermination = { [weak self] _ in self?.stop() }
        queue.async { self.session.startRunning() }  // startRunning blocks; keep off-main
        return stream
    }

    /// Same as `start()` but yields raw 32BGRA frames with no conversion —
    /// the fast path for real-time inference.
    public func startPixelBuffers() async throws -> AsyncStream<CameraFrame> {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            throw VisionError.cameraAccessDenied
        }
        try configure()
        let (stream, continuation) = AsyncStream<CameraFrame>.makeStream(
            bufferingPolicy: .bufferingNewest(1))
        self.pixelContinuation = continuation
        continuation.onTermination = { [weak self] _ in self?.stop() }
        queue.async { self.session.startRunning() }
        return stream
    }

    public func stop() {
        continuation?.finish()
        continuation = nil
        pixelContinuation?.finish()
        pixelContinuation = nil
        if session.isRunning { session.stopRunning() }
    }

    private func configure() throws {
        guard session.inputs.isEmpty else { return }
        session.beginConfiguration()
        session.sessionPreset = preset

        #if os(iOS)
        let device = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: position)
        #else
        let device = AVCaptureDevice.default(for: .video)
        #endif
        guard let device,
            let input = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            throw VisionError.cameraUnavailable
        }
        session.addInput(input)

        var settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if let size = dataOutputSize {
            settings[kCVPixelBufferWidthKey as String] = Int(size.width)
            settings[kCVPixelBufferHeightKey as String] = Int(size.height)
        }
        output.videoSettings = settings
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            throw VisionError.cameraUnavailable
        }
        session.addOutput(output)

        #if os(iOS)
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90  // portrait
        }
        #endif
        session.commitConfiguration()

        // Raise the device frame rate when the caller wants more than the preset's
        // default format provides (e.g. 60 fps capture so a ~25 ms model isn't capped
        // at 30). The preset's chosen format often tops out at 30 fps, so this may
        // require switching activeFormat to an equivalent-resolution high-fps format —
        // which must happen AFTER commitConfiguration (committing a sessionPreset
        // resets the device format and wipes custom frame durations).
        if targetFPS > 30, (try? device.lockForConfiguration()) != nil {
            defer { device.unlockForConfiguration() }
            let current = device.activeFormat
            let dims = CMVideoFormatDescriptionGetDimensions(current.formatDescription)
            if !current.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= targetFPS }) {
                // Same resolution, supports the target rate, prefer non-binned plain video.
                let candidate = device.formats.first(where: { f in
                    let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                    return d.width == dims.width && d.height == dims.height
                        && f.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= targetFPS })
                })
                guard let candidate else { return }
                device.activeFormat = candidate
            }
            let duration = CMTime(value: 1, timescale: CMTimeScale(targetFPS))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            NSLog("CameraFeed: %@ @ %.0f fps", device.activeFormat.description, targetFPS)
        }
    }
}

extension CameraFeed: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastFrameTime >= minFrameInterval else { return }
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        if let pixelContinuation {
            lastFrameTime = now
            pixelContinuation.yield(CameraFrame(pixelBuffer: buffer))
            return
        }
        guard let continuation else { return }
        lastFrameTime = now
        let ciImage = CIImage(cvPixelBuffer: buffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return
        }
        continuation.yield(cgImage)
    }
}
