// VideoFile.swift — video-file glue: evenly-spaced frames from any AVFoundation-readable
// clip, the video-side sibling of `ImageFile.load`. Consumers hand a URL to a typed
// pipeline (e.g. `ActionRecognizer.classify(videoAt:)`) without touching AVFoundation.

import AVFoundation
import CoreGraphics
import Foundation

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
}
