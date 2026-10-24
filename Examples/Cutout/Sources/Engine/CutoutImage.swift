// CutoutImage — the photo with the matte as its alpha channel, written as a PNG (straight, not premultiplied alpha,
// as PNG stores it). Several instances make one matte: their probabilities added and clipped to 1 (the cut-out keeps
// everything any of them covers; two that touch keep their shared edge).

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum CutoutImage {
    /// The per-pixel alpha of the chosen instances: min(1, Σ p) → 0 … 255.
    public static func combinedAlpha(_ probabilities: [[Float]]) -> [UInt8] {
        guard let first = probabilities.first else { return [] }
        var out = [UInt8](repeating: 0, count: first.count)
        for i in 0..<first.count {
            var s: Float = 0
            for p in probabilities { s += p[i] }
            out[i] = UInt8(min(255, max(0, (min(s, 1) * 255).rounded())))
        }
        return out
    }

    /// RGBA pixels: the photo's RGB and `alpha` (straight).
    public static func rgba(_ image: RGB8Image, alpha: [UInt8]) -> [UInt8] {
        precondition(alpha.count == image.width * image.height)
        var out = [UInt8](repeating: 0, count: alpha.count * 4)
        for i in 0..<alpha.count {
            out[i * 4] = image.pixels[i * 3]
            out[i * 4 + 1] = image.pixels[i * 3 + 1]
            out[i * 4 + 2] = image.pixels[i * 3 + 2]
            out[i * 4 + 3] = alpha[i]
        }
        return out
    }

    /// A CGImage of straight RGBA (for the screen; ImageIO writes it as is).
    public static func cgImage(rgba: [UInt8], width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    /// The PNG bytes of the photo with `alpha` as its alpha channel.
    public static func png(_ image: RGB8Image, alpha: [UInt8]) throws -> Data {
        guard let cg = cgImage(rgba: rgba(image, alpha: alpha), width: image.width, height: image.height) else {
            throw CutoutError.image("cannot make the cut-out image")
        }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data as CFMutableData, UTType.png.identifier as CFString, 1, nil) else {
            throw CutoutError.image("cannot make a PNG writer")
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw CutoutError.image("the PNG writer failed") }
        return data as Data
    }

    /// The tight bounding box of alpha > 0 (x, y, width, height in pixels), for cropping a cut-out to its subject.
    public static func bounds(alpha: [UInt8], width: Int, height: Int, threshold: UInt8 = 8) -> (x: Int, y: Int, w: Int, h: Int)? {
        var x0 = width, y0 = height, x1 = -1, y1 = -1
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] > threshold {
                if x < x0 { x0 = x }
                if x > x1 { x1 = x }
                if y < y0 { y0 = y }
                if y > y1 { y1 = y }
            }
        }
        return x1 < 0 ? nil : (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    }
}
