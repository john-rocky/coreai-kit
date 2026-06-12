// CameraFeed.swift — live camera frames as an AsyncStream<CGImage>, throttled to a
// target frame rate. Pairs with GraphModel pipelines (depth, detection) so a live-camera
// ML view is a for-await loop. The app must declare NSCameraUsageDescription.

import AVFoundation
import CoreImage
import Foundation

public final class CameraFeed: NSObject, @unchecked Sendable {
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "coreai.kit.camera")
    private let ciContext = CIContext()
    private let position: AVCaptureDevice.Position
    private let minFrameInterval: Double
    private var lastFrameTime: Double = 0
    private var continuation: AsyncStream<CGImage>.Continuation?

    public init(
        position: AVCaptureDevice.Position = .back, framesPerSecond: Double = 10
    ) {
        self.position = position
        self.minFrameInterval = 1.0 / max(framesPerSecond, 0.1)
        super.init()
    }

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

    public func stop() {
        continuation?.finish()
        continuation = nil
        if session.isRunning { session.stopRunning() }
    }

    private func configure() throws {
        guard session.inputs.isEmpty else { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .vga640x480

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
            throw VisionError.cameraUnavailable
        }
        session.addInput(input)

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            throw VisionError.cameraUnavailable
        }
        session.addOutput(output)

        #if os(iOS)
        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90  // portrait
        }
        #endif
    }
}

extension CameraFeed: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastFrameTime >= minFrameInterval, let continuation else { return }
        lastFrameTime = now
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: buffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            return
        }
        continuation.yield(cgImage)
    }
}
