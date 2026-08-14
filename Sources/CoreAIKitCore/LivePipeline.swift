// LivePipeline.swift — the loop every live-inference app writes by hand.
//
// A camera or a microphone produces items faster than a model consumes them, so the loop
// between the two always ends up doing the same four things: drop what went stale while the
// model was busy, overlap CPU preprocessing of item N+1 with inference on item N, back off
// when the device gets hot, and report what it is actually achieving. `Examples/DetectCamera`
// is ~120 lines of exactly that, written inside a SwiftUI model; so is every other live
// example, each slightly differently.
//
// This is that loop, once, with no camera and no model in it — Foundation only, so both the
// vision side (`LiveVision`) and anything audio can ride it.
//
// ```swift
// let results = LivePipeline.run(
//     frames, governor: .init(framesPerSecond: 30),
//     prepare: { try detector.prepare($0) },
//     infer:   { try await detector.detect($0) })
//
// for try await result in results {
//     draw(result.value)                     // [Detection]
//     hud(result.stats.framesPerSecond)      // measured, not requested
// }
// ```
//
// What it deliberately does not do: buffer. A live pipeline that queues is a live pipeline
// that lags, and a lagging camera overlay is worse than a slower correct one. Both stages
// keep exactly the newest item, so a slow model costs frames, never latency.

import Foundation

/// What a live pass is actually achieving, measured over a trailing window rather than
/// requested — the number a HUD should show and a benchmark should quote.
public struct LiveStats: Sendable {
    /// Median inference latency over the window.
    public let latency: Duration
    /// Results per second over the window, wall clock. Lower than `1 / latency` whenever
    /// preprocessing or capture is the bottleneck, which is the useful thing to know.
    public let framesPerSecond: Double
    /// Source items dropped since the previous result: the ones that arrived while the
    /// model was busy, plus the ones the governor skipped. Back-pressure, made visible.
    public let dropped: Int
    /// The device's thermal state when the result landed.
    public let thermalState: ProcessInfo.ThermalState
    /// The rate the governor is currently asking for. Below `LiveGovernor.framesPerSecond`
    /// means the device is hot and the pipeline is deliberately doing less work.
    public let targetFramesPerSecond: Double
}

/// How hard the pipeline is allowed to run, and what it does when the device says stop.
///
/// The thermal clause is not a nicety. A sustained camera-plus-model loop is the hottest
/// thing an app can do on a phone; iOS throttles the GPU under it, a long uninterrupted GPU
/// batch can be killed outright, and a feature that flattens the battery gets removed by
/// whoever ships it. Backing the frame rate off is the cheapest correct response, and doing
/// it here means every pipeline gets it without asking.
public struct LiveGovernor: Sendable {
    /// Target rate while the device is cool.
    public var framesPerSecond: Double

    /// Rate multiplier once the thermal state reaches `.serious`; squared at `.critical`.
    /// `1` disables thermal backoff entirely.
    public var thermalBackoff: Double

    public init(framesPerSecond: Double = 15, thermalBackoff: Double = 0.5) {
        self.framesPerSecond = framesPerSecond
        self.thermalBackoff = thermalBackoff
    }

    /// The rate to run at in `state`.
    public func targetRate(at state: ProcessInfo.ThermalState) -> Double {
        let backoff = max(0.05, min(1, thermalBackoff))
        switch state {
        case .serious: return max(1, framesPerSecond * backoff)
        case .critical: return max(1, framesPerSecond * backoff * backoff)
        default: return framesPerSecond
        }
    }
}

/// One model result plus what it cost.
///
/// **TODO — the coordinate contract is not closed.** A result carries no description of the
/// frame it came from, so a caller holding normalized boxes cannot map them onto a preview:
/// the preview shows the capture session's own aspect (16:9 at the default preset) while the
/// model was fed `LiveVision.captureSize(forModelInput:)` (3:4), and `.resizeAspectFill`
/// crops on top of that. Boxes drawn without correcting for both are visibly offset —
/// observed on device, 2026-08-03.
///
/// The fix is a frame size on this type plus the aspect-fill mapping shipped in
/// `CoreAIKitUI` rather than re-derived per app. `Examples/DetectCamera` already does it
/// correctly by hand (`DetectCameraView.swift`, `DetectionOverlay`: `scale = max(view/frame)`,
/// centre offsets) — that is the code to lift. Until then the two places that claim the
/// mapping is trivial (`LiveVision.detections`, `CoreAI.watch`) say so honestly instead.
public struct LiveResult<Output: Sendable>: Sendable {
    public let value: Output
    public let stats: LiveStats

    public init(value: Output, stats: LiveStats) {
        self.value = value
        self.stats = stats
    }
}

/// The two-stage, drop-stale, thermally-governed loop between a live source and a model.
public enum LivePipeline {
    /// Number of results averaged for `LiveStats`. Thirty is ~1–2 seconds of a live feed:
    /// long enough that one slow frame does not swing the readout, short enough that a
    /// thermal throttle shows up while the user is still looking.
    private static let statsWindow = 30

    /// Runs `prepare` then `infer` over `source`, yielding a result per item that survives.
    ///
    /// - `prepare` is the synchronous CPU half (resize, channel split, mel). It runs on its
    ///   own task so it overlaps with the model, and an item it throws on is skipped rather
    ///   than ending the stream — one malformed frame must not kill a live session.
    /// - `infer` is the model. A throw here *does* end the stream: the model failing is not
    ///   a transient condition, and silently yielding nothing forever is worse than an error.
    /// - Cancelling the returned stream (or leaving the `for await`) stops both stages.
    public static func run<Input: Sendable, Staged: Sendable, Output: Sendable>(
        _ source: AsyncStream<Input>,
        governor: LiveGovernor = LiveGovernor(),
        prepare: @escaping @Sendable (Input) throws -> Staged,
        infer: @escaping @Sendable (Staged) async throws -> Output
    ) -> AsyncThrowingStream<LiveResult<Output>, any Error> {
        // Newest-only on the way out as well as in: a consumer slower than the model gets
        // the current answer, never a backlog of answers about frames that have gone.
        AsyncThrowingStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            // Only the newest staged item is kept: while the model runs, later frames
            // replace each other instead of forming a queue.
            let (staged, stagedInput) = AsyncStream<Staged>.makeStream(
                bufferingPolicy: .bufferingNewest(1))
            let dropped = DroppedCounter()

            let prepareTask = Task.detached {
                let clock = ContinuousClock()
                var lastAccepted: ContinuousClock.Instant?
                for await input in source {
                    if Task.isCancelled { break }
                    let now = clock.now
                    let target = governor.targetRate(
                        at: ProcessInfo.processInfo.thermalState)
                    // The governor's own throttle, on top of whatever rate the source
                    // produces: a hot device is asked for fewer frames, not slower ones.
                    if let lastAccepted,
                        now - lastAccepted < .seconds(1 / max(target, 0.1))
                    {
                        await dropped.count()
                        continue
                    }
                    lastAccepted = now
                    guard let value = try? prepare(input) else {
                        await dropped.count()
                        continue
                    }
                    // `bufferingNewest(1)` means this may silently replace an item the
                    // model never got to — that item is dropped too.
                    if case .dropped = stagedInput.yield(value) { await dropped.count() }
                }
                stagedInput.finish()
            }

            let inferTask = Task.detached {
                let clock = ContinuousClock()
                var window: [Duration] = []
                var windowStart = clock.now
                var framesPerSecond: Double = 0
                var latency: Duration = .zero
                do {
                    for await item in staged {
                        if Task.isCancelled { break }
                        let started = clock.now
                        let value = try await infer(item)
                        window.append(clock.now - started)
                        if window.count >= statsWindow {
                            latency = window.sorted()[window.count / 2]
                            let elapsed = clock.now - windowStart
                            framesPerSecond =
                                Double(window.count) / max(elapsed.seconds, 0.001)
                            window.removeAll(keepingCapacity: true)
                            windowStart = clock.now
                        }
                        let thermalState = ProcessInfo.processInfo.thermalState
                        continuation.yield(
                            LiveResult(
                                value: value,
                                stats: LiveStats(
                                    latency: latency, framesPerSecond: framesPerSecond,
                                    dropped: await dropped.take(),
                                    thermalState: thermalState,
                                    targetFramesPerSecond: governor.targetRate(
                                        at: thermalState))))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                prepareTask.cancel()
                inferTask.cancel()
                stagedInput.finish()
            }
        }
    }
}

/// Items dropped since the last result was reported.
private actor DroppedCounter {
    private var pending = 0
    func count() { pending += 1 }
    /// Reads and resets — each result reports the drops that led to it.
    func take() -> Int {
        defer { pending = 0 }
        return pending
    }
}

extension Duration {
    /// Seconds as a `Double`, for rates and readouts.
    public var seconds: Double {
        let (whole, attoseconds) = components
        return Double(whole) + Double(attoseconds) / 1e18
    }
}
