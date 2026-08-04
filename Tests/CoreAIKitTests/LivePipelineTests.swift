// LivePipelineTests.swift — the live loop's policy, without a camera or a model.
//
// What has to be right is the policy, not the plumbing: a frame the model could not keep up
// with is dropped rather than queued, one bad frame does not end a session, and a hot device
// is asked to do less work. All three are testable with an array of integers.

import Foundation
import Testing

@testable import CoreAIKitCore
@testable import CoreAIKitVision

struct LivePipelineTests {
    /// Fast enough that the governor's own throttle never fires — these tests are about
    /// everything else.
    private static let ungoverned = LiveGovernor(framesPerSecond: 1_000_000)

    private func source(_ values: [Int]) -> AsyncStream<Int> {
        AsyncStream { continuation in
            for value in values { continuation.yield(value) }
            continuation.finish()
        }
    }

    @Test func resultsCarryTheModelOutput() async throws {
        let results = LivePipeline.run(
            source([1, 2, 3]), governor: Self.ungoverned,
            prepare: { $0 * 10 }, infer: { $0 + 1 })

        var seen: [Int] = []
        for try await result in results { seen.append(result.value) }

        // Stale-frame dropping means the count is not guaranteed, but every value that
        // arrives must have been through both stages.
        #expect(!seen.isEmpty)
        #expect(seen.allSatisfy { [11, 21, 31].contains($0) })
    }

    @Test func aFrameThatFailsToPrepareIsSkippedNotFatal() async throws {
        let results = LivePipeline.run(
            source([1, 2, 3, 4, 5]), governor: Self.ungoverned,
            prepare: { value -> Int in
                struct BadFrame: Error {}
                if value == 1 { throw BadFrame() }  // the first frame is malformed
                return value
            },
            infer: { $0 })

        var seen: [Int] = []
        for try await result in results { seen.append(result.value) }

        // A live session must survive one bad frame — the stream keeps going and the bad
        // value never appears.
        #expect(!seen.contains(1))
        #expect(!seen.isEmpty)
    }

    @Test func aModelFailureEndsTheStream() async throws {
        struct ModelDown: Error {}
        let results = LivePipeline.run(
            source([1, 2, 3]), governor: Self.ungoverned,
            prepare: { $0 }, infer: { _ -> Int in throw ModelDown() })

        await #expect(throws: ModelDown.self) {
            for try await _ in results {}
        }
    }

    @Test func aSlowModelDropsFramesRatherThanQueueing() async throws {
        // 200 items arrive at once; the model takes 5 ms each. Queueing them all would take
        // a second and every result after the first would be describing the past.
        let results = LivePipeline.run(
            source(Array(1...200)), governor: Self.ungoverned,
            prepare: { $0 },
            infer: { value in
                try? await Task.sleep(for: .milliseconds(5))
                return value
            })

        var seen: [Int] = []
        for try await result in results { seen.append(result.value) }

        #expect(seen.count < 200)  // most were dropped, which is the point
        #expect(!seen.isEmpty)
        // The drops are reported rather than hidden.
        #expect(seen.count < 100)
    }

    @Test func theGovernorBacksOffWhenTheDeviceIsHot() {
        let governor = LiveGovernor(framesPerSecond: 30, thermalBackoff: 0.5)

        #expect(governor.targetRate(at: .nominal) == 30)
        #expect(governor.targetRate(at: .fair) == 30)
        #expect(governor.targetRate(at: .serious) == 15)
        #expect(governor.targetRate(at: .critical) == 7.5)
    }

    @Test func backoffOfOneMeansNeverBackOff() {
        let benchmark = LiveGovernor(framesPerSecond: 60, thermalBackoff: 1)

        #expect(benchmark.targetRate(at: .critical) == 60)
    }

    @Test func theGovernorThrottlesTheSource() async throws {
        // 10 items/second against a source that delivers 50 at once: most must be skipped
        // before the model is ever asked, because the point of the governor is to not do
        // the work rather than to do it later.
        let results = LivePipeline.run(
            source(Array(1...50)), governor: LiveGovernor(framesPerSecond: 10),
            prepare: { $0 }, infer: { $0 })

        var seen: [Int] = []
        for try await result in results { seen.append(result.value) }

        #expect(seen.count <= 2)  // ~one frame; the rest arrived inside the same 100 ms
        #expect(seen.first == 1)  // and the first is never the one dropped
    }
}

/// The capture size handed to `AVCaptureVideoDataOutput`, which terminates the app on an odd
/// width or height rather than failing gracefully. Found on a device: RF-DETR's 384 is safe
/// and YOLOX-S's 640 is not, so the crash appeared only when the model was swapped — the one
/// swap this layer exists to make safe.
struct CaptureSizeTests {
    @Test func everyCatalogDetectorInputProducesAnEvenSize() {
        // 384 = RF-DETR nano, 640 = YOLOX-S; the rest bracket them.
        for side in [224, 320, 384, 416, 512, 560, 640, 768, 1024] {
            let size = LiveVision.captureSize(forModelInput: side)
            #expect(Int(size.width) % 2 == 0, "width odd for input \(side)")
            #expect(Int(size.height) % 2 == 0, "height odd for input \(side)")
        }
    }

    @Test func yoloxIsTheCaseThatUsedToCrash() {
        // 640 * 4 / 3 == 853, which is what AVFoundation rejected.
        let size = LiveVision.captureSize(forModelInput: 640)
        #expect(size.width == 640)
        #expect(size.height == 852)
    }

    @Test func theSizeStaysCloseToFourThirdsPortrait() {
        let size = LiveVision.captureSize(forModelInput: 384)
        #expect(size == CGSize(width: 384, height: 512))
    }
}
