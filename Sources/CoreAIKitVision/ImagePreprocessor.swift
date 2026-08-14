// ImagePreprocessor.swift — resize + per-channel normalize a CGImage into the planar CHW
// Float32 layout vision encoders expect. Adapted from apple/coreai-models' CoreAIShared
// ImagePreprocessor (BSD-3-Clause, Copyright 2026 Apple Inc.) — see NOTICE.txt.

import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

public struct ImagePreprocessor: Sendable {
    public let size: Int
    public let mean: SIMD3<Float>
    public let std: SIMD3<Float>

    public init(size: Int, mean: SIMD3<Float>, std: SIMD3<Float>) {
        self.size = size
        self.mean = mean
        self.std = std
    }

    /// OpenAI CLIP normalization at 224×224 (ViT-B variants).
    public static let clip224 = ImagePreprocessor(
        size: 224,
        mean: SIMD3(0.48145466, 0.4578275, 0.40821073),
        std: SIMD3(0.26862954, 0.26130258, 0.27577711))

    /// ImageNet normalization at 224×224 (Depth Anything, DINOv2-family encoders).
    public static let imagenet224 = ImagePreprocessor(
        size: 224,
        mean: SIMD3(0.485, 0.456, 0.406),
        std: SIMD3(0.229, 0.224, 0.225))

    /// Draws into an sRGB RGBA8 context at size×size (high interpolation, close to PIL
    /// BICUBIC), then computes `(pixel/255 - mean) / std` per channel into planar
    /// `[3, size, size]` Float32.
    public func chw(from image: CGImage) throws -> [Float] {
        let w = size
        let h = size
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: colorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            throw VisionError.imageRenderFailed
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let pixelData = ctx.data else {
            throw VisionError.imageRenderFailed
        }
        let raw = pixelData.bindMemory(to: UInt8.self, capacity: w * h * 4)

        let pixelCount = w * h
        var chw = [Float](repeating: 0, count: 3 * pixelCount)
        var channel = [Float](repeating: 0, count: pixelCount)
        let n = vDSP_Length(pixelCount)
        let means = [mean.x, mean.y, mean.z]
        let stds = [std.x, std.y, std.z]
        chw.withUnsafeMutableBufferPointer { dst in
            for c in 0..<3 {
                // Fold `(x/255 - mean) / std` into one vDSP_vsmsa: y = x*a + b.
                var a = (1.0 / 255.0) / stds[c]
                var b = -means[c] / stds[c]
                vDSP_vfltu8(raw.advanced(by: c), 4, &channel, 1, n)
                vDSP_vsmsa(channel, 1, &a, &b, dst.baseAddress! + c * pixelCount, 1, n)
            }
        }
        return chw
    }
}

/// The same resize-and-normalize, straight from a 32BGRA capture buffer: vImage scales and
/// vDSP splits the channels, with no `CGImage` and no `CIContext` anywhere in the path, and
/// scratch reused across frames.
///
/// This is the difference between a live pipeline that pays for a full render per frame and
/// one that does not. Measured on an M4 Max, 640×480 → 224², release build: **0.13 ms here
/// against 0.78 ms** for `CIContext.createCGImage` followed by `ImagePreprocessor.chw(from:)`
/// — about 6×, of which 0.36 ms is the render alone, producing a bitmap nobody ever looks at.
/// Re-run it yourself: `swift run -c release live-cli --bench` in `Examples/LiveCamera`.
///
/// Geometry matches `ImagePreprocessor.chw(from:)`: a plain square resize, aspect distorted,
/// because that is what these graphs were exported against. What differs is interpolation —
/// vImage's default against Core Graphics' `.high`. Measured against each other on the same
/// buffer: **flat colour is bit-identical** (so there is no colour-space or channel-order
/// difference between the paths), a smooth gradient differs by a mean of 0.002 normalized
/// units, and a deliberately aliasing high-frequency pattern by up to 1.7 — the last being
/// what any two downsamplers disagree about, not something specific to these two.
public final class PixelBufferPreprocessor: @unchecked Sendable {
    /// Square side the model consumes.
    public let size: Int

    /// Per-channel `(x/255 - mean) / std` folded into `y = x*a + b`, in R, G, B order.
    private let scales: [Float]
    private let biases: [Float]

    private let lock = NSLock()
    private var scaledBGRA: [UInt8] = []
    private var channelScratch: [Float] = []

    public init(
        size: Int, mean: SIMD3<Float> = SIMD3(0, 0, 0), std: SIMD3<Float> = SIMD3(1, 1, 1)
    ) {
        self.size = size
        let means = [mean.x, mean.y, mean.z]
        let stds = [std.x, std.y, std.z]
        self.scales = (0..<3).map { (1.0 / 255.0) / stds[$0] }
        self.biases = (0..<3).map { -means[$0] / stds[$0] }
    }

    /// The pixel-buffer twin of an existing `ImagePreprocessor` — same size, same
    /// normalization, so a model can offer both paths without them drifting apart.
    public convenience init(_ preprocessor: ImagePreprocessor) {
        self.init(
            size: preprocessor.size, mean: preprocessor.mean, std: preprocessor.std)
    }

    /// Planar `[3, size, size]` Float32 from a 32BGRA buffer. Thread-safe: calls serialize
    /// on the shared scratch, so this is the stage that overlaps with the GPU rather than
    /// one that can be run concurrently with itself.
    public func chw(from pixelBuffer: CVPixelBuffer) throws -> [Float] {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA
        else {
            throw VisionError.bundleLayout(
                "the pixel-buffer fast path expects a 32BGRA buffer")
        }
        let pixelCount = size * size

        lock.lock()
        defer { lock.unlock() }
        if scaledBGRA.count != pixelCount * 4 {
            scaledBGRA = [UInt8](repeating: 0, count: pixelCount * 4)
            channelScratch = [Float](repeating: 0, count: pixelCount)
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw VisionError.imageRenderFailed
        }
        var src = vImage_Buffer(
            data: base,
            height: vImagePixelCount(CVPixelBufferGetHeight(pixelBuffer)),
            width: vImagePixelCount(CVPixelBufferGetWidth(pixelBuffer)),
            rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer))

        var chw = [Float](repeating: 0, count: 3 * pixelCount)
        let error = scaledBGRA.withUnsafeMutableBytes { dst -> vImage_Error in
            var destination = vImage_Buffer(
                data: dst.baseAddress, height: vImagePixelCount(size),
                width: vImagePixelCount(size), rowBytes: size * 4)
            return vImageScale_ARGB8888(&src, &destination, nil, vImage_Flags(kvImageNoFlags))
        }
        guard error == kvImageNoError else { throw VisionError.imageRenderFailed }

        // BGRA byte order: B=0, G=1, R=2 → planar R, G, B.
        let n = vDSP_Length(pixelCount)
        scaledBGRA.withUnsafeBufferPointer { raw in
            chw.withUnsafeMutableBufferPointer { out in
                for (plane, offset) in [(0, 2), (1, 1), (2, 0)] {
                    var a = scales[plane]
                    var b = biases[plane]
                    vDSP_vfltu8(raw.baseAddress! + offset, 4, &channelScratch, 1, n)
                    vDSP_vsmsa(
                        channelScratch, 1, &a, &b,
                        out.baseAddress! + plane * pixelCount, 1, n)
                }
            }
        }
        return chw
    }
}
