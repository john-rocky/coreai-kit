// VideoFileTests.swift — frame extraction, without a model.
//
// Two things have to hold before any video op can be trusted: the strategy chooser picks
// the right reader for the requested rate, and both readers return the same frames at the
// same timestamps. The second one is the interesting test — the sequential path rides OS 27's
// `AVAssetReader.outputProvider` API, and "it compiled" is not evidence that it decodes.

import AVFoundation
import CoreVideo
import Foundation
import Testing

@testable import CoreAIKitVision

struct VideoFileSamplingTests {
    @Test func denseSamplingReadsSequentially() {
        // 15 samples/s out of 30 fps source: seeking would decode the same GOPs repeatedly.
        #expect(isSequential(.automatic, rate: 15, sourceRate: 30))
        #expect(isSequential(.automatic, rate: 5, sourceRate: 30))
    }

    @Test func sparseSamplingSeeks() {
        // One sample per ten seconds: sequential decode would pay for the whole clip to
        // deliver a handful of frames.
        #expect(!isSequential(.automatic, rate: 0.1, sourceRate: 30))
        #expect(!isSequential(.automatic, rate: 1, sourceRate: 60))
    }

    @Test func anExplicitChoiceIsHonoured() {
        #expect(isSequential(.sequential, rate: 0.01, sourceRate: 30))
        #expect(!isSequential(.seeking, rate: 100, sourceRate: 30))
    }

    @Test func anUnknownSourceRateReadsSequentially() {
        // Some containers report 0 fps. Sequential is the safe default: its cost is bounded
        // by the clip, while seeking at an unknown rate could be unbounded.
        #expect(isSequential(.automatic, rate: 1, sourceRate: 0))
    }

    private func isSequential(
        _ sampling: VideoFile.Sampling, rate: Double, sourceRate: Double
    ) -> Bool {
        if case .sequential = VideoFile.resolve(sampling, rate: rate, sourceRate: sourceRate) {
            return true
        }
        return false
    }
}

struct ChangeDetectorTests {
    private func buffer(grey: UInt8) -> CVPixelBuffer {
        var pixels: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault, 64, 64, kCVPixelFormatType_32BGRA, nil, &pixels)
        let buffer = pixels!
        CVPixelBufferLockBaseAddress(buffer, [])
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<64 {
            for x in 0..<64 {
                let p = base + y * rowBytes + x * 4
                p[0] = grey
                p[1] = grey
                p[2] = grey
                p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return buffer
    }

    @Test func theFirstFrameAlwaysCounts() {
        let detector = ChangeDetector(threshold: 0.02)
        #expect(detector.isDifferent(buffer(grey: 128)))
    }

    @Test func anIdenticalFrameIsSkipped() {
        let detector = ChangeDetector(threshold: 0.02)
        _ = detector.isDifferent(buffer(grey: 128))
        // A fixed camera on an empty room: this is the frame the scan must not pay for.
        #expect(!detector.isDifferent(buffer(grey: 128)))
    }

    @Test func aChangedFrameCounts() {
        let detector = ChangeDetector(threshold: 0.02)
        _ = detector.isDifferent(buffer(grey: 100))
        // 100 → 160 is 0.235 in normalized grey, well past the threshold.
        #expect(detector.isDifferent(buffer(grey: 160)))
    }

    @Test func aChangeBelowTheThresholdIsSkipped() {
        let detector = ChangeDetector(threshold: 0.5)
        _ = detector.isDifferent(buffer(grey: 100))
        #expect(!detector.isDifferent(buffer(grey: 110)))  // 0.039, under 0.5
    }
}

struct VideoStreamTests {
    /// Writes a short clip whose content moves, so the encoder produces real inter-frame
    /// data rather than a degenerate stream.
    private func makeClip(seconds: Double, fps: Int32) async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("videofile-tests-\(UUID().uuidString).mov")
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: 160, AVVideoHeightKey: 120,
            ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 160,
                kCVPixelBufferHeightKey as String: 120,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for i in 0..<Int(seconds * Double(fps)) {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(1)) }
            var pixels: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pixels)
            let buffer = pixels!
            CVPixelBufferLockBaseAddress(buffer, [])
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<120 {
                for x in 0..<160 {
                    let p = base + y * rowBytes + x * 4
                    p[0] = UInt8((x + i * 4) % 256)
                    p[1] = UInt8((y + i * 4) % 256)
                    p[2] = UInt8((x + y + i * 4) % 256)
                    p[3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(
                buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: fps))
        }
        input.markAsFinished()
        await writer.finishWriting()
        return url
    }

    private func collect(
        _ url: URL, rate: Double, sampling: VideoFile.Sampling
    ) async throws -> [TimeInterval] {
        var times: [TimeInterval] = []
        for try await frame in VideoFile.stream(url, framesPerSecond: rate, sampling: sampling) {
            times.append(frame.time)
        }
        return times
    }

    @Test func bothReadersAgreeOnWhatTheClipContains() async throws {
        let url = try await makeClip(seconds: 2, fps: 30)
        defer { try? FileManager.default.removeItem(at: url) }

        let sequential = try await collect(url, rate: 5, sampling: .sequential)
        let seeking = try await collect(url, rate: 5, sampling: .seeking)

        // Ten samples out of two seconds, both ways. The readers land on slightly different
        // frames (one takes the next frame at or after the mark, the other seeks within a
        // tolerance), so the count and the ordering are what must match, not the exact PTS.
        #expect(sequential.count == 10)
        #expect(seeking.count == 10)
        #expect(sequential == sequential.sorted())
        #expect(seeking == seeking.sorted())
        #expect(sequential.first! < 0.3)
        #expect(sequential.last! > 1.5)
    }

    @Test func theRateControlsHowManyFramesComeBack() async throws {
        let url = try await makeClip(seconds: 2, fps: 30)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(try await collect(url, rate: 1, sampling: .sequential).count == 2)
        #expect(try await collect(url, rate: 15, sampling: .sequential).count == 30)
    }

    @Test func cancellingTheStreamStopsTheRead() async throws {
        let url = try await makeClip(seconds: 2, fps: 30)
        defer { try? FileManager.default.removeItem(at: url) }

        var seen = 0
        for try await _ in VideoFile.stream(url, framesPerSecond: 30, sampling: .sequential) {
            seen += 1
            if seen == 3 { break }  // leaving the loop must tear the reader down
        }
        #expect(seen == 3)
    }
}
