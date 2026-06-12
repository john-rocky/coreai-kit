// MaskRenderer.swift — composite per-instance masks into one premultiplied-RGBA
// CGImage, fully vectorized (vDSP) with reused scratch buffers: per instance,
// binarize logits (> 0) and linearly blend its color into float RGBA planes
// (later = higher score wins overlaps), then one strided float->u8 conversion.
// No per-pixel Swift loops, no transcendental math — the yolo-ios-app rendering
// recipe adapted to full-frame DETR masks.

import Accelerate
import CoreGraphics
import Foundation

public final class MaskRenderer {
    public struct RGBA: Sendable {
        public let r: Float
        public let g: Float
        public let b: Float
        public init(_ r: UInt8, _ g: UInt8, _ b: UInt8) {
            self.r = Float(r)
            self.g = Float(g)
            self.b = Float(b)
        }
    }

    /// Default palette (matches the DetectCamera box palette).
    public static let palette: [RGBA] = [
        RGBA(255, 59, 48), RGBA(52, 199, 89), RGBA(0, 122, 255), RGBA(255, 149, 0),
        RGBA(175, 82, 222), RGBA(50, 200, 250), RGBA(255, 204, 0), RGBA(0, 199, 190),
        RGBA(255, 45, 85),
    ]

    private let alpha: Float
    private var width = 0
    private var height = 0
    private var mask01: [Float] = []
    private var planes: [[Float]] = []  // premultiplied R, G, B, A accumulators
    private var scratch: [Float] = []
    private var rgba: [UInt8] = []

    public init(alpha: Float = 110 / 255) {
        self.alpha = alpha
    }

    /// Composites all detections that carry a mask. Call from one thread at a
    /// time (scratch buffers are reused across frames).
    public func composite(_ detections: [Detection]) -> CGImage? {
        guard let first = detections.first(where: { $0.mask != nil })?.mask else {
            return nil
        }
        let w = first.width
        let h = first.height
        let n = w * h
        if width != w || height != h {
            width = w
            height = h
            mask01 = [Float](repeating: 0, count: n)
            planes = (0..<4).map { _ in [Float](repeating: 0, count: n) }
            scratch = [Float](repeating: 0, count: n)
            rgba = [UInt8](repeating: 0, count: n * 4)
        }
        for i in 0..<4 {
            vDSP.fill(&planes[i], with: 0)
        }

        // ascending score: the strongest instance blends last and wins overlaps
        for det in detections.sorted(by: { $0.score < $1.score }) {
            guard let mask = det.mask, mask.width == w, mask.height == h else { continue }
            // logits > 0 -> {0, 1}: clip(sign-limit(x, ±0.5) + 0.5)
            vDSP_vlim(mask.logits, 1, [Float(0)], [Float(0.5)], &mask01, 1, vDSP_Length(n))
            vDSP.add(0.5, mask01, result: &mask01)

            let color = Self.palette[det.classID % Self.palette.count]
            let a = alpha * 255
            // premultiplied target components for this instance
            let target: [Float] = [color.r * alpha, color.g * alpha, color.b * alpha, a]
            for (c, value) in target.enumerated() {
                // plane = plane + mask01 * (value - plane)
                //       = plane * (1 - mask01) + value * mask01
                vDSP.multiply(-1, planes[c], result: &scratch)
                vDSP.add(value, scratch, result: &scratch)
                vDSP.multiply(mask01, scratch, result: &scratch)
                vDSP.add(planes[c], scratch, result: &planes[c])
            }
        }

        rgba.withUnsafeMutableBufferPointer { out in
            for c in 0..<4 {
                vDSP_vfixru8(planes[c], 1, out.baseAddress! + c, 4, vDSP_Length(n))
            }
        }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true,
            intent: .defaultIntent)
    }
}
