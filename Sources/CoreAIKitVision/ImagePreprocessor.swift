// ImagePreprocessor.swift — resize + per-channel normalize a CGImage into the planar CHW
// Float32 layout vision encoders expect. Adapted from apple/coreai-models' CoreAIShared
// ImagePreprocessor (BSD-3-Clause, Copyright 2026 Apple Inc.) — see NOTICE.txt.

import Accelerate
import CoreGraphics
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
