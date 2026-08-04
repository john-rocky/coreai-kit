// VideoFile.swift — video-file glue: frames out of any AVFoundation-readable clip, either
// as an evenly-spaced handful (`frames`) or as a timestamped stream at a chosen rate
// (`stream`). The video-side sibling of `ImageFile.load`.
//
// There are two ways to get frames out of a clip and they are not interchangeable.
// **Seeking** (`AVAssetImageGenerator`) jumps to each sample and costs per sample, so it
// wins when samples are sparse — it skips whole groups of pictures it never has to decode.
// **Sequential decode** (`AVAssetReader`) reads the track once and costs per *clip*, so it
// wins as soon as samples are dense enough that seeking repeats work.
//
// Measured on an M4 Max over a 60 s 640×480 H.264 clip at 30 fps:
//
// | samples | seeking | sequential | |
// |---|---|---|---|
// | 6 (0.1/s) | 0.11 s | 0.47 s | seeking 4.2× |
// | 60 (1/s) | 0.35 s | 0.47 s | seeking 1.4× |
// | 300 (5/s) | 2.59 s | 0.47 s | **sequential 5.5×** |
// | 900 (15/s) | 8.20 s | 0.48 s | **sequential 17.1×** |
//
// Sequential is flat, seeking is linear — the crossover sits near one sample per second of
// 30 fps source, which is where `.automatic` switches. Getting this wrong by picking one
// implementation and standing by it costs 17× at one end of the range and 4× at the other,
// which is why the choice is made from the numbers rather than from taste.

import AVFoundation
import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// One frame of a clip: capture-shaped pixels plus where it sits in the timeline.
///
/// Pixels rather than a `CGImage` because that is what the model layer's real-time paths
/// take (`KitDetector.prepare`, `DepthEstimator.prepare`) — a scan that produced images
/// would make every consumer pay for a render it does not need.
public struct VideoFrame: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    /// Presentation time from the start of the clip.
    public let time: TimeInterval

    public init(pixelBuffer: CVPixelBuffer, time: TimeInterval) {
        self.pixelBuffer = pixelBuffer
        self.time = time
    }

    /// The frame as a `CGImage`, for the samples that earn one — a still to hand a
    /// vision-language model, a thumbnail for a timeline. Costs a render, so a scan should
    /// call it after it has decided the frame matters, not before.
    public func cgImage() -> CGImage? {
        CameraFrame(pixelBuffer: pixelBuffer).cgImage()
    }
}

public enum VideoFile {
    /// `count` frames sampled at even timestamps across the clip (each at the middle of
    /// its segment, so a 1-frame request grabs the clip's midpoint).
    public static func frames(_ count: Int, from url: URL) async throws -> [CGImage] {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else {
            throw VisionError.bundleLayout("could not read video duration of \(url.path)")
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        let tolerance = CMTime(seconds: duration / Double(max(count, 1)) / 2, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance

        var frames: [CGImage] = []
        frames.reserveCapacity(count)
        for i in 0..<count {
            let seconds = duration * (Double(i) + 0.5) / Double(count)
            let time = CMTime(seconds: seconds, preferredTimescale: 600)
            frames.append(try await generator.image(at: time).image)
        }
        return frames
    }

    /// How the frames are fetched. `.automatic` is the right answer unless you are
    /// measuring something.
    public enum Sampling: Sendable {
        /// Picks from the requested rate against the clip's own frame rate, using the
        /// crossover measured in this file's header.
        case automatic
        /// One seek per sample. Flat memory, cost proportional to sample count.
        case seeking
        /// Decode the track once. Cost proportional to clip length, independent of rate.
        case sequential
    }

    /// Frames at `framesPerSecond`, in order, each carrying its timestamp.
    ///
    /// - `size` asks the decoder for smaller buffers — free downscaling, so a 4K clip does
    ///   not become 4K buffers a model immediately shrinks. Ignored by the seeking path,
    ///   which is not sparse enough for it to matter.
    /// - `minimumChange` skips frames too similar to the last one emitted, measured as mean
    ///   absolute difference on a 32×32 grey thumbnail (0…1). Static footage then costs
    ///   decode instead of inference, which is the whole difference on a fixed camera.
    ///   `nil` emits every sample.
    public static func stream(
        _ url: URL, framesPerSecond: Double = 1, sampling: Sampling = .automatic,
        size: CGSize? = nil, minimumChange: Float? = nil
    ) -> AsyncThrowingStream<VideoFrame, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    let asset = AVURLAsset(url: url)
                    let duration = try await asset.load(.duration).seconds
                    guard duration.isFinite, duration > 0 else {
                        throw VisionError.bundleLayout(
                            "could not read video duration of \(url.path)")
                    }
                    guard
                        let track = try await asset.loadTracks(withMediaType: .video).first
                    else {
                        throw VisionError.bundleLayout("no video track in \(url.path)")
                    }
                    let sourceRate = Double(try await track.load(.nominalFrameRate))
                    let rate = max(framesPerSecond, 0.001)
                    let detector = minimumChange.map { ChangeDetector(threshold: $0) }

                    switch resolve(sampling, rate: rate, sourceRate: sourceRate) {
                    case .sequential:
                        try await readSequentially(
                            asset: asset, track: track, rate: rate, size: size,
                            changes: detector, into: continuation)
                    default:
                        try await readBySeeking(
                            asset: asset, duration: duration, rate: rate,
                            changes: detector, into: continuation)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Sequential decode pays for the whole clip; seeking pays per sample. Below roughly one
    /// kept frame in 25, seeking is ahead — see the measurements in this file's header. The
    /// ratio is codec- and device-dependent, so this is a threshold, not a constant anyone
    /// should read a guarantee into.
    static func resolve(_ sampling: Sampling, rate: Double, sourceRate: Double) -> Sampling {
        guard case .automatic = sampling else { return sampling }
        guard sourceRate > 0 else { return .sequential }
        return rate * 25 >= sourceRate ? .sequential : .seeking
    }

    private static func readSequentially(
        asset: AVURLAsset, track: AVAssetTrack, rate: Double, size: CGSize?,
        changes: ChangeDetector?, into continuation: AsyncThrowingStream<VideoFrame, any Error>.Continuation
    ) async throws {
        let reader = try AVAssetReader(asset: asset)
        var settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if let size {
            settings[kCVPixelBufferWidthKey as String] = Int(size.width)
            settings[kCVPixelBufferHeightKey as String] = Int(size.height)
        }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        // OS 27's provider API: `next()` suspends instead of blocking a thread, and the
        // vended buffers are safe to hold without `alwaysCopiesSampleData` (which is
        // deprecated on 27 for exactly that reason).
        let provider = reader.outputProvider(for: output)
        try reader.start()
        defer { reader.cancelReading() }

        let interval = 1 / rate
        var nextAt = 0.0
        while let sample = try await provider.next() {
            if Task.isCancelled { return }
            let time = sample.presentationTimeStamp.seconds
            // The frame is built inside the closure: `withUnsafeSampleBuffer` returns a
            // `sending` result, and a bare `CVImageBuffer` out of a task-isolated context
            // is not one. `VideoFrame` is, so the wrapper crosses and the buffer does not.
            guard time + 1e-9 >= nextAt,
                let frame = sample.withUnsafeSampleBuffer({ buffer in
                    buffer.imageBuffer.map { VideoFrame(pixelBuffer: $0, time: time) }
                })
            else { continue }
            nextAt = time + interval
            if let changes, !changes.isDifferent(frame.pixelBuffer) { continue }
            continuation.yield(frame)
            // A long scan is many back-to-back GPU submissions; giving the cooperative pool
            // a turn keeps a cancel responsive and the UI alive.
            await Task.yield()
        }
    }

    private static func readBySeeking(
        asset: AVURLAsset, duration: Double, rate: Double, changes: ChangeDetector?,
        into continuation: AsyncThrowingStream<VideoFrame, any Error>.Continuation
    ) async throws {
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        let interval = 1 / rate
        let tolerance = CMTime(seconds: interval / 2, preferredTimescale: 600)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance

        var time = interval / 2
        while time < duration {
            if Task.isCancelled { return }
            let image = try await generator.image(
                at: CMTime(seconds: time, preferredTimescale: 600)
            ).image
            defer { time += interval }
            guard let pixels = pixelBuffer(from: image) else { continue }
            if let changes, !changes.isDifferent(pixels) { continue }
            continuation.yield(VideoFrame(pixelBuffer: pixels, time: time))
            await Task.yield()
        }
    }

    /// The seeking path produces `CGImage`, and the model layer's fast paths take pixel
    /// buffers, so one render per *sample* buys a uniform frame type. At ~9 ms a seek and a
    /// fraction of a millisecond a render, this is not the part that costs.
    private static func pixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attributes =
            [
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                kCVPixelBufferCGImageCompatibilityKey: true,
            ] as CFDictionary
        guard
            CVPixelBufferCreate(
                kCFAllocatorDefault, image.width, image.height, kCVPixelFormatType_32BGRA,
                attributes, &buffer) == kCVReturnSuccess, let pixels = buffer
        else { return nil }

        CVPixelBufferLockBaseAddress(pixels, [])
        defer { CVPixelBufferUnlockBaseAddress(pixels, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixels),
            let context = CGContext(
                data: base, width: image.width, height: image.height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixels),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        context.draw(
            image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return pixels
    }
}

/// Decides whether a frame differs enough from the last kept one to be worth a model call.
///
/// Deliberately crude: a 32×32 grey thumbnail and a mean absolute difference. A fixed camera
/// watching an empty room produces near-zero differences, and skipping those is the entire
/// point — anything more sophisticated would cost more than the inference it saves.
final class ChangeDetector: @unchecked Sendable {
    private let threshold: Float
    private let lock = NSLock()
    private var previous: [Float]?
    /// Scratch for the difference, reused under the same lock as `previous`.
    private var scratch = [Float](repeating: 0, count: side * side)

    private static let side = 32

    init(threshold: Float) {
        self.threshold = threshold
    }

    func isDifferent(_ pixelBuffer: CVPixelBuffer) -> Bool {
        guard let thumbnail = Self.thumbnail(pixelBuffer) else { return true }
        return lock.withLock {
            defer { previous = thumbnail }
            guard let previous else { return true }  // the first frame always counts
            var difference: Float = 0
            let n = vDSP_Length(thumbnail.count)
            vDSP_vsub(previous, 1, thumbnail, 1, &scratch, 1, n)
            vDSP_vabs(scratch, 1, &scratch, 1, n)
            vDSP_meanv(scratch, 1, &difference, n)
            return difference >= threshold
        }
    }

    private static func thumbnail(_ pixelBuffer: CVPixelBuffer) -> [Float]? {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA
        else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

        var source = vImage_Buffer(
            data: base, height: vImagePixelCount(CVPixelBufferGetHeight(pixelBuffer)),
            width: vImagePixelCount(CVPixelBufferGetWidth(pixelBuffer)),
            rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer))
        var scaled = [UInt8](repeating: 0, count: side * side * 4)
        let error = scaled.withUnsafeMutableBytes { destination -> vImage_Error in
            var buffer = vImage_Buffer(
                data: destination.baseAddress, height: vImagePixelCount(side),
                width: vImagePixelCount(side), rowBytes: side * 4)
            return vImageScale_ARGB8888(&source, &buffer, nil, vImage_Flags(kvImageNoFlags))
        }
        guard error == kvImageNoError else { return nil }

        // Green alone stands in for luminance: it carries most of it, and this is a
        // change detector, not a photometric measurement.
        var grey = [Float](repeating: 0, count: side * side)
        scaled.withUnsafeBufferPointer { raw in
            vDSP_vfltu8(raw.baseAddress! + 1, 4, &grey, 1, vDSP_Length(side * side))
        }
        var scale = Float(1.0 / 255.0)
        vDSP_vsmul(grey, 1, &scale, &grey, 1, vDSP_Length(side * side))
        return grey
    }
}
