// CoreAI+Scan.swift — a video file as an op.
//
// `recognizeAction` takes a clip and answers one question about the whole of it. An app
// holding a video usually has the other question: *where* in this hour of footage is the
// thing I care about. That answer is a timeline, and building it means sampling, decoding,
// running a model per sample, and stamping the results — the same four steps every time.
//
// ```swift
// for try await entry in CoreAI.scan(videoAt: clip) {
//     print(entry.time, entry.value.map(\.label))     // 0.5 s: ["person", "dog"]
// }
// ```
//
// The offline twin of `watch(for:)`, for the same reason it exists live: a cheap detector
// decides, an expensive model runs on what it picked.
//
// ```swift
// for try await moment in CoreAI.scan(videoAt: clip, for: .label("person")) {
//     let caption = try await CoreAI.caption(moment.value.image)
// }
// ```
//
// **Live drops, offline does not.** `watch()` throws frames away when the model falls
// behind, because a stale overlay is worse than a missing one. A scan is asked for a
// specific set of samples and delivers all of them — the cost of a slow model here is that
// the scan takes longer, which is the correct trade when nothing is on screen waiting.

import CoreAIKit
import CoreAIKitVision
import CoreGraphics
import Foundation

/// One result on a video's timeline.
public struct ScanEntry<Value: Sendable>: Sendable {
    /// Seconds from the start of the clip.
    public let time: TimeInterval
    public let value: Value

    public init(time: TimeInterval, value: Value) {
        self.time = time
        self.value = value
    }
}

/// How a scan picks the frames it runs the model on.
public struct ScanSampling: Sendable {
    /// Frames taken per second of footage. The default costs one detector call per second
    /// of video — an hour of footage in about a minute on a phone.
    public var framesPerSecond: Double

    /// Skip frames too similar to the last one kept, as mean absolute difference on a 32×32
    /// grey thumbnail (0…1). On fixed-camera footage this is the difference between paying
    /// for the whole clip and paying for the parts where something happened. `nil` keeps
    /// every sample.
    public var minimumChange: Float?

    /// Frame extraction strategy. `.automatic` chooses seeking or sequential decode from the
    /// rate — see `VideoFile`, where the crossover is measured.
    public var sampling: VideoFile.Sampling

    public init(
        framesPerSecond: Double = 1, minimumChange: Float? = nil,
        sampling: VideoFile.Sampling = .automatic
    ) {
        self.framesPerSecond = framesPerSecond
        self.minimumChange = minimumChange
        self.sampling = sampling
    }

    /// One sample per second, every one of them.
    public static let everySecond = ScanSampling()

    /// Dense enough to catch a brief event, and only where the picture actually moves.
    public static let changes = ScanSampling(framesPerSecond: 5, minimumChange: 0.02)
}

extension CoreAI {
    /// Video file → a time-stamped detection timeline, in order.
    ///
    /// Boxes are normalized to the frame, exactly as `detect` and `watch` return them.
    public static func scan(
        videoAt url: URL, sampling: ScanSampling = .everySecond, scoreThreshold: Float = 0.5,
        options: OpOptions = OpOptions()
    ) -> AsyncThrowingStream<ScanEntry<[Detection]>, any Error> {
        let id = options.model ?? defaultDetectionModel
        return AsyncThrowingStream(bufferingPolicy: .unbounded) { continuation in
            let task = Task {
                do {
                    let detector = try await ImageOpModels.shared.detector(catalog: id)
                    try await withPinnedModel(ResidentKind.detector, id) {
                        let frames = VideoFile.stream(
                            url, framesPerSecond: sampling.framesPerSecond,
                            sampling: sampling.sampling,
                            size: CGSize(
                                width: detector.inputSize, height: detector.inputSize),
                            minimumChange: sampling.minimumChange)
                        for try await frame in frames {
                            try Task.checkCancellation()
                            let detections = try await detector.detect(
                                in: frame.pixelBuffer, scoreThreshold: scoreThreshold)
                            continuation.yield(
                                ScanEntry(time: frame.time, value: detections))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Video file → only the moments matching `trigger`, each rendered as an image ready for
    /// an expensive model.
    ///
    /// The cooldown is in *video* time, not wall time: a person on screen for ten seconds
    /// yields one moment per cooldown of footage, however fast the scan happens to run.
    public static func scan(
        videoAt url: URL, for trigger: WatchTrigger,
        sampling: ScanSampling = .everySecond, scoreThreshold: Float = 0.5,
        options: OpOptions = OpOptions()
    ) -> AsyncThrowingStream<ScanEntry<WatchedMoment>, any Error> {
        let id = options.model ?? defaultDetectionModel
        return AsyncThrowingStream(bufferingPolicy: .unbounded) { continuation in
            let task = Task {
                do {
                    let detector = try await ImageOpModels.shared.detector(catalog: id)
                    let cooldown = trigger.cooldown.seconds
                    try await withPinnedModel(ResidentKind.detector, id) {
                        // Declared inside the sendable closure: the cooldown state belongs
                        // to this scan, and a `var` captured across the boundary is not
                        // something Swift 6 will let cross.
                        var lastFired: TimeInterval?
                        let frames = VideoFile.stream(
                            url, framesPerSecond: sampling.framesPerSecond,
                            sampling: sampling.sampling,
                            minimumChange: sampling.minimumChange)
                        for try await frame in frames {
                            try Task.checkCancellation()
                            let detections = try await detector.detect(
                                in: frame.pixelBuffer, scoreThreshold: scoreThreshold)
                            guard trigger.predicate(detections) else { continue }
                            if let lastFired, frame.time - lastFired < cooldown { continue }
                            // Rendered only here, on the frames that fired — the whole
                            // reason the trigger stage exists.
                            guard let image = frame.cgImage() else { continue }
                            lastFired = frame.time
                            continuation.yield(
                                ScanEntry(
                                    time: frame.time,
                                    value: WatchedMoment(
                                        image: image, detections: detections)))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
