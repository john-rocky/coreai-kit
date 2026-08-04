// LiveVision.swift — camera in, model results out, with the loop already written.
//
// `CameraFeed` has vended frames since the beginning and nothing consumed them, so every
// live example in this package re-implemented the same consumer: a two-stage task pipeline,
// a stale-frame policy, a stats window, and a restart path. `Examples/DetectCamera` also
// hand-wrote an enum to put two detector families behind one prepare/detect surface — an
// abstraction leaking out of the package and into the app that adopted it.
//
// This is the consumer, once. The model is a parameter, so swapping it is an argument:
//
// ```swift
// let live = LiveVision()
// let detector = try await KitDetector(catalog: "yolox-s")     // or "rf-detr"
// for try await result in try await live.detections(with: detector) {
//     boxes = result.value                     // normalized, ready to draw over the preview
//     fps   = result.stats.framesPerSecond     // measured
// }
// ```
//
// Attach an `AVCaptureVideoPreviewLayer` to `captureSession` and the preview costs nothing:
// the compositor draws it, and the pipeline only pays for inference.

import AVFoundation
import CoreAIKitCore
import CoreGraphics
import CoreVideo
import Foundation

/// A live camera feeding a model, with the frame policy, the thermal governor and the
/// measurements already in place.
public final class LiveVision: @unchecked Sendable {
    /// Capture settings. The defaults are the ones the device examples converged on: a
    /// 720p session so the preview looks right, data-output buffers scaled down to model
    /// size in hardware, and a target rate the governor lowers when the phone gets hot.
    public struct Options: Sendable {
        /// Rate to run the model at while the device is cool. Capture is requested at the
        /// same rate; the pipeline drops whatever the model cannot keep up with.
        public var framesPerSecond: Double
        public var position: AVCaptureDevice.Position
        /// Session preset, which is what the *preview* shows — independent of the smaller
        /// buffers the model receives.
        public var preset: AVCaptureSession.Preset
        /// Rate multiplier once the device reaches a `.serious` thermal state, squared at
        /// `.critical`. `1` disables thermal backoff — appropriate only for a bench run.
        public var thermalBackoff: Double
        /// Size of the buffers handed to the model. `nil` derives it from the model's own
        /// input size, which is what a caller would otherwise have to look up per model.
        public var dataOutputSize: CGSize?

        public init(
            framesPerSecond: Double = 15,
            position: AVCaptureDevice.Position = .back,
            preset: AVCaptureSession.Preset = .hd1280x720,
            thermalBackoff: Double = 0.5,
            dataOutputSize: CGSize? = nil
        ) {
            self.framesPerSecond = framesPerSecond
            self.position = position
            self.preset = preset
            self.thermalBackoff = thermalBackoff
            self.dataOutputSize = dataOutputSize
        }
    }

    private let options: Options
    private let lock = NSLock()
    private var feed: CameraFeed?

    public init(options: Options = Options()) {
        self.options = options
    }

    /// The capture session, once a stream has started — attach an
    /// `AVCaptureVideoPreviewLayer` to show the live feed for free. `nil` before then: the
    /// session is configured from the model's input size, which is not known until a model
    /// is handed over.
    public var captureSession: AVCaptureSession? {
        lock.withLock { feed?.captureSession }
    }

    /// Stops the camera. The running stream finishes; calling a stream method again starts
    /// a fresh session.
    public func stop() {
        let running = lock.withLock { () -> CameraFeed? in
            defer { feed = nil }
            return feed
        }
        running?.stop()
    }

    /// Live object detection. Boxes are normalized (0…1, origin top-left) **in the frame the
    /// model was fed**, which is not the frame `captureSession` previews: the preview runs at
    /// the session preset's aspect and `.resizeAspectFill` crops it, while the model gets
    /// `captureSize(forModelInput:)`. Drawing a box therefore still needs the aspect-fill
    /// correction that `Examples/DetectCamera`'s `DetectionOverlay` does by hand.
    ///
    /// Closing that is a `TODO` on `LiveResult` — until it is closed, this call has not
    /// removed the coordinate work it was supposed to.
    public func detections(
        with detector: KitDetector, scoreThreshold: Float = 0.5, maxDetections: Int = 50
    ) async throws -> AsyncThrowingStream<LiveResult<[Detection]>, any Error> {
        let size =
            options.dataOutputSize ?? Self.captureSize(forModelInput: detector.inputSize)
        let frames = try await start(dataOutputSize: size)
        return LivePipeline.run(
            frames, governor: governor,
            prepare: { try detector.prepare($0.pixelBuffer) },
            infer: {
                try await detector.detect(
                    $0, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
            })
    }

    /// Live monocular depth — the capability Apple ships no answer for, at 54 MB.
    public func depth(
        with estimator: DepthEstimator, dataOutputSize: CGSize? = nil
    ) async throws -> AsyncThrowingStream<LiveResult<DepthMap>, any Error> {
        let size =
            dataOutputSize ?? options.dataOutputSize
            ?? CGSize(width: 480, height: 640)  // 4:3 portrait around the 224² input
        let frames = try await start(dataOutputSize: size)
        return LivePipeline.run(
            frames, governor: governor,
            prepare: { try estimator.prepare($0.pixelBuffer) },
            infer: { try await estimator.estimateDepth($0) })
    }

    /// Any per-frame model: hand it the two halves and it rides the same loop, governor and
    /// stats. This is the escape hatch that keeps the typed methods above from being a
    /// closed list — a segmentation model, a classifier, or your own `GraphModel` all fit.
    public func results<Staged: Sendable, Output: Sendable>(
        prepare: @escaping @Sendable (CameraFrame) throws -> Staged,
        infer: @escaping @Sendable (Staged) async throws -> Output,
        dataOutputSize: CGSize? = nil
    ) async throws -> AsyncThrowingStream<LiveResult<Output>, any Error> {
        let frames = try await start(
            dataOutputSize: dataOutputSize ?? options.dataOutputSize)
        return LivePipeline.run(
            frames, governor: governor, prepare: prepare, infer: infer)
    }

    /// The 4:3 portrait capture size to ask the ISP for around a model's square input:
    /// enough pixels that the model's own rescale is cheap, few enough that the hardware
    /// does the shrinking.
    ///
    /// **Both dimensions are rounded to even numbers**, because `AVCaptureVideoDataOutput`
    /// rejects an odd width or height with an `NSInvalidArgumentException` — it does not
    /// fail gracefully, it terminates the app. RF-DETR's 384 → 512 is even and YOLOX-S's
    /// 640 → 853 is not, so without this the crash appears only when the model is swapped,
    /// which is the swap this whole layer exists to make safe.
    public static func captureSize(forModelInput side: Int) -> CGSize {
        let width = max(2, side - side % 2)
        let height = max(2, (side * 4 / 3) - (side * 4 / 3) % 2)
        return CGSize(width: width, height: height)
    }

    private var governor: LiveGovernor {
        LiveGovernor(
            framesPerSecond: options.framesPerSecond, thermalBackoff: options.thermalBackoff)
    }

    private func start(dataOutputSize: CGSize?) async throws -> AsyncStream<CameraFrame> {
        let feed = makeFeed(dataOutputSize: dataOutputSize)
        return try await feed.startPixelBuffers()
    }

    private func makeFeed(dataOutputSize: CGSize?) -> CameraFeed {
        let feed = CameraFeed(
            position: options.position, framesPerSecond: options.framesPerSecond,
            preset: options.preset, dataOutputSize: dataOutputSize)
        let previous = lock.withLock { () -> CameraFeed? in
            defer { self.feed = feed }
            return self.feed
        }
        previous?.stop()  // one session at a time; a second stream replaces the first
        return feed
    }
}
