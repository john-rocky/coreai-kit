// CoreAI+Watch.swift — the camera as an op.
//
// Every other op takes the unit the *model* takes: one image, one clip, one string. An app
// does not hold one image, it holds a camera — and the distance between the two is the loop,
// the frame policy, the orientation, and the coordinate mapping that the adopter writes by
// hand. `watch` is that distance, removed.
//
// ```swift
// let watch = try await CoreAI.watch()                 // downloads RF-DETR on first use
// preview.session = watch.captureSession               // free: the compositor draws it
// for try await frame in watch {
//     boxes = frame.value                              // [Detection], normalized 0…1
// }
// ```
//
// And the two-stage form, which is the only shape in which a heavy model belongs anywhere
// near a live camera: a 36–103 MB detector runs continuously, and the expensive model runs
// only on the frames that matter.
//
// ```swift
// for try await moment in try await CoreAI.watch(for: .label("person")) {
//     let caption = try await CoreAI.caption(moment.image)     // ~seconds, not per-frame
// }
// ```

import AVFoundation
import CoreAIKit
import CoreAIKitVision
import CoreGraphics
import Foundation

/// A running camera-plus-model stream. Iterate it for results; hold it to reach the capture
/// session or to stop early. Dropping it stops the camera.
///
/// `@unchecked Sendable` for `AVCaptureSession`, which is not marked `Sendable` but is
/// internally synchronized and is only ever read here — handed to an
/// `AVCaptureVideoPreviewLayer` on the main actor. `CameraFeed`, which owns the session,
/// makes the same call.
public struct LiveWatch<Output: Sendable>: AsyncSequence, @unchecked Sendable {
    public typealias Element = LiveResult<Output>
    public typealias AsyncIterator = AsyncThrowingStream<LiveResult<Output>, any Error>
        .Iterator

    /// The capture session — attach an `AVCaptureVideoPreviewLayer` and the live preview
    /// costs nothing: it is drawn by the compositor, not by this pipeline.
    public let captureSession: AVCaptureSession?

    private let stream: AsyncThrowingStream<LiveResult<Output>, any Error>
    private let vision: LiveVision

    init(
        stream: AsyncThrowingStream<LiveResult<Output>, any Error>, vision: LiveVision
    ) {
        self.stream = stream
        self.vision = vision
        self.captureSession = vision.captureSession
    }

    public func makeAsyncIterator() -> AsyncIterator {
        stream.makeAsyncIterator()
    }

    /// Stops the camera. The stream finishes; the models stay loaded (and evictable).
    public func stop() {
        vision.stop()
    }
}

/// One frame the camera decided was worth the expensive model.
public struct WatchedMoment: Sendable {
    /// The frame itself, rendered once, at the moment the trigger fired.
    public let image: CGImage
    /// What fired the trigger.
    public let detections: [Detection]
}

/// A condition over the live stream, plus how long to wait before firing again.
///
/// The cooldown is the load-bearing part. A predicate over a 15 fps stream is true for as
/// long as the object is in shot, and running a vision-language model fifteen times a second
/// is a hot phone and a flat battery — so a trigger is a rate limit with a condition attached,
/// not the other way round.
public struct WatchTrigger: Sendable {
    let cooldown: Duration
    let predicate: @Sendable ([Detection]) -> Bool

    /// Fires when a label is present above `minimumScore` — the labels are the detector's
    /// own (`Detection.label`), matched case-insensitively.
    public static func label(
        _ label: String, minimumScore: Float = 0.5, cooldown: Duration = .seconds(3)
    ) -> WatchTrigger {
        let wanted = label.lowercased()
        return WatchTrigger(cooldown: cooldown) { detections in
            detections.contains {
                $0.score >= minimumScore && $0.label.lowercased() == wanted
            }
        }
    }

    /// Fires whenever anything is detected above `minimumScore`.
    public static func anything(
        minimumScore: Float = 0.5, cooldown: Duration = .seconds(3)
    ) -> WatchTrigger {
        WatchTrigger(cooldown: cooldown) { detections in
            detections.contains { $0.score >= minimumScore }
        }
    }

    /// Fires when your own condition holds — "more than three people", "nothing at all for
    /// the last few frames", whatever the app is actually watching for.
    public static func when(
        cooldown: Duration = .seconds(3),
        _ predicate: @escaping @Sendable ([Detection]) -> Bool
    ) -> WatchTrigger {
        WatchTrigger(cooldown: cooldown, predicate: predicate)
    }
}

extension CoreAI {
    /// Live camera → detections, per frame.
    ///
    /// Boxes are normalized (0…1, origin top-left) to the frame the model was fed, which is
    /// not the frame `captureSession` previews — drawing them over the preview still needs
    /// the aspect-fill correction (`TODO` on `LiveResult`). `result.stats` carries what the
    /// pipeline is actually achieving — measured frame rate, median latency, dropped frames,
    /// thermal state — because the requested rate is not the interesting number on a phone.
    public static func watch(
        camera: LiveVision.Options = LiveVision.Options(), scoreThreshold: Float = 0.5,
        options: OpOptions = OpOptions()
    ) async throws -> LiveWatch<[Detection]> {
        let id = options.model ?? defaultDetectionModel
        let detector = try await ImageOpModels.shared.detector(catalog: id)
        // Pinned for the life of the stream: a live pipeline is the one caller for which
        // "the model was evicted, reload it" is not a slower call but a stutter.
        await ModelResidency.shared.pin(ResidentModel(kind: ResidentKind.detector, id: id))
        let vision = LiveVision(options: camera)
        let stream = try await vision.detections(
            with: detector, scoreThreshold: scoreThreshold)
        return LiveWatch(stream: unpinning(stream, ResidentKind.detector, id), vision: vision)
    }

    /// Live camera → depth, per frame. Apple has no monocular depth API; this is a 54 MB
    /// model reading depth from the single wide camera, on any device.
    public static func watchDepth(
        camera: LiveVision.Options = LiveVision.Options(), options: OpOptions = OpOptions()
    ) async throws -> LiveWatch<DepthMap> {
        let id = options.model ?? defaultDepthModel
        let estimator = try await ImageOpModels.shared.depthEstimator(catalog: id)
        await ModelResidency.shared.pin(ResidentModel(kind: ResidentKind.depth, id: id))
        let vision = LiveVision(options: camera)
        let stream = try await vision.depth(with: estimator)
        return LiveWatch(stream: unpinning(stream, ResidentKind.depth, id), vision: vision)
    }

    /// Live camera → only the frames that satisfy `trigger`, rendered as images ready for an
    /// expensive model.
    ///
    /// This is the shape that makes an on-device VLM affordable on a phone: the small
    /// detector runs continuously and decides, and the model that costs gigabytes and
    /// seconds runs on a handful of frames. The default capture rate is deliberately lower
    /// than `watch()`'s — a trigger stage exists to spend less, not more.
    public static func watch(
        for trigger: WatchTrigger,
        camera: LiveVision.Options = LiveVision.Options(framesPerSecond: 5),
        scoreThreshold: Float = 0.5, options: OpOptions = OpOptions()
    ) async throws -> LiveWatch<WatchedMoment> {
        let id = options.model ?? defaultDetectionModel
        let detector = try await ImageOpModels.shared.detector(catalog: id)
        await ModelResidency.shared.pin(ResidentModel(kind: ResidentKind.detector, id: id))
        let vision = LiveVision(options: camera)

        // The frame rides through both stages so the render can happen after the decision:
        // converting every frame to a `CGImage` would cost more than the detector does.
        let detected = try await vision.results(
            prepare: { frame in (try detector.prepare(frame.pixelBuffer), frame) },
            infer: { staged in
                (
                    try await detector.detect(staged.0, scoreThreshold: scoreThreshold),
                    staged.1
                )
            },
            dataOutputSize: LiveVision.captureSize(forModelInput: detector.inputSize))

        let moments = AsyncThrowingStream<LiveResult<WatchedMoment>, any Error>(
            bufferingPolicy: .bufferingNewest(1)
        ) { out in
            let task = Task {
                let clock = ContinuousClock()
                var lastFired: ContinuousClock.Instant?
                do {
                    for try await result in detected {
                        let (detections, frame) = result.value
                        guard trigger.predicate(detections) else { continue }
                        if let lastFired, clock.now - lastFired < trigger.cooldown {
                            continue
                        }
                        guard let image = frame.cgImage() else { continue }
                        lastFired = clock.now
                        out.yield(
                            LiveResult(
                                value: WatchedMoment(
                                    image: image, detections: detections),
                                stats: result.stats))
                    }
                    out.finish()
                } catch {
                    out.finish(throwing: error)
                }
            }
            out.onTermination = { _ in task.cancel() }
        }
        return LiveWatch(stream: unpinning(moments, ResidentKind.detector, id), vision: vision)
    }

    /// Releases the model's residency pin when the stream ends, however it ends. Without
    /// this a watch that is started and stopped leaves its model unevictable for the life
    /// of the process — a leak of exactly the thing the budget exists to manage.
    private static func unpinning<Output: Sendable>(
        _ stream: AsyncThrowingStream<LiveResult<Output>, any Error>,
        _ kind: String, _ id: String
    ) -> AsyncThrowingStream<LiveResult<Output>, any Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(1)) { out in
            let task = Task {
                defer { ModelResidency.shared.unpinLater(ResidentModel(kind: kind, id: id)) }
                do {
                    for try await element in stream { out.yield(element) }
                    out.finish()
                } catch {
                    out.finish(throwing: error)
                }
            }
            out.onTermination = { _ in task.cancel() }
        }
    }
}
