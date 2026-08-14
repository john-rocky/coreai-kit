// QuickStart.swift — the take-home core of this runner: a live source and a model, four
// ways, as four typed functions with no UI imports. This is the file you copy.
//
// Every one of them is a stream you iterate. None of them contains a capture session, a
// frame-drop policy, a stats window, or a thermal check, because those live in the kit —
// which is the whole point of the example.

import CoreAIOps
import CoreGraphics
import Foundation

// MARK: - Live camera

/// Live camera → detections per frame. `model` is any `detection` catalog id, so swapping
/// RF-DETR for YOLOX is one string.
///
/// Iterate the result for `[Detection]` (normalized boxes) and `result.stats` (measured
/// frame rate, median latency, dropped frames, thermal state). Hold it for
/// `captureSession`, which an `AVCaptureVideoPreviewLayer` renders for free.
func liveDetections(
    model: String = "rf-detr", framesPerSecond: Double = 15, scoreThreshold: Float = 0.5
) async throws -> LiveWatch<[Detection]> {
    // CARD-SNIPPET-BEGIN
    let watch = try await CoreAI.watch(
        camera: LiveVision.Options(framesPerSecond: framesPerSecond),
        scoreThreshold: scoreThreshold, options: .model(model))
    // for try await frame in watch { frame.value /* [Detection] */ }
    // CARD-SNIPPET-END
    return watch
}

/// Live camera → a relative depth map per frame. Apple ships no monocular depth API; this
/// is a 54 MB model reading depth from the single wide camera.
func liveDepth(
    model: String = "depth-anything-3-small", framesPerSecond: Double = 10
) async throws -> LiveWatch<DepthMap> {
    try await CoreAI.watchDepth(
        camera: LiveVision.Options(framesPerSecond: framesPerSecond),
        options: .model(model))
}

/// Live camera → only the frames where `label` appears, at most one per `cooldown`.
///
/// The two-stage shape: a 36–103 MB detector runs continuously and decides; whatever you
/// do with the moment (a VLM caption, a saved still, a notification) runs on a handful of
/// frames instead of thirty a second.
func triggeredMoments(
    label: String, cooldown: Duration = .seconds(3), framesPerSecond: Double = 5
) async throws -> LiveWatch<WatchedMoment> {
    try await CoreAI.watch(
        for: .label(label, cooldown: cooldown),
        camera: LiveVision.Options(framesPerSecond: framesPerSecond))
}

/// One moment → a sentence describing it. Separate from the trigger on purpose: the VLM is
/// gigabytes and seconds, and the caller decides when to spend them.
func describe(_ moment: WatchedMoment) async throws -> String {
    try await CoreAI.caption(moment.image)
}

// MARK: - Video file

/// A video file → a time-stamped detection timeline.
///
/// `sampling` decides both how often the model runs and how the frames are read: at dense
/// rates the clip is decoded once, at sparse rates each sample is seeked to. Setting
/// `minimumChange` skips frames that look like the last one kept, which on fixed-camera
/// footage is most of them.
func scanVideo(
    at url: URL, framesPerSecond: Double = 1, minimumChange: Float? = nil
) -> AsyncThrowingStream<ScanEntry<[Detection]>, any Error> {
    CoreAI.scan(
        videoAt: url,
        sampling: ScanSampling(
            framesPerSecond: framesPerSecond, minimumChange: minimumChange))
}

/// A video file → only the moments where `label` appears, cooldown counted in video time.
func scanVideo(
    at url: URL, for label: String, framesPerSecond: Double = 2,
    cooldown: Duration = .seconds(3)
) -> AsyncThrowingStream<ScanEntry<WatchedMoment>, any Error> {
    CoreAI.scan(
        videoAt: url, for: .label(label, cooldown: cooldown),
        sampling: ScanSampling(framesPerSecond: framesPerSecond))
}
