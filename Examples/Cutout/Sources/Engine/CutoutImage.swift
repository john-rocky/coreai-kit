// CutoutImage — the photo with the matte as its alpha channel, written as a PNG (straight, not premultiplied alpha,
// as PNG stores it). Several instances make one matte: their probabilities added and clipped to 1 (the cut-out keeps
// everything any of them covers; two that touch keep their shared edge). Lined up, each instance is cropped to its own
// matte and the pieces are set side by side on one sheet.

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
        try png(rgba: rgba(image, alpha: alpha), width: image.width, height: image.height)
    }

    /// The PNG bytes of straight RGBA pixels.
    public static func png(rgba: [UInt8], width: Int, height: Int) throws -> Data {
        guard let cg = cgImage(rgba: rgba, width: width, height: height) else {
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

    /// The pieces apart, on one sheet: each instance's pixels where its matte is, cropped to that matte's bounds, side by
    /// side at the photo's scale over transparency — bottoms on one line, in the order given, at most `columns` to a row
    /// (by default one row up to four pieces, then about as many rows as columns), each row centered, 4 % of the
    /// tallest piece between the pieces and around the sheet. Straight RGBA; nil when no matte covers a pixel.
    /// One matte at the photo's size is held at a time.
    public static func lineup(_ image: RGB8Image, logits: [[Float]], columns: Int? = nil) -> (rgba: [UInt8], width: Int, height: Int)? {
        var pieces: [(rgba: [UInt8], w: Int, h: Int)] = []
        for l in logits {
            let alpha = Alpha.alpha8(l, width: image.width, height: image.height)
            guard let b = bounds(alpha: alpha, width: image.width, height: image.height) else { continue }
            var px = [UInt8](repeating: 0, count: b.w * b.h * 4)
            image.pixels.withUnsafeBufferPointer { rgb in
                alpha.withUnsafeBufferPointer { a in
                    px.withUnsafeMutableBufferPointer { d in
                        for y in 0..<b.h {
                            for x in 0..<b.w {
                                let s = (b.y + y) * image.width + b.x + x, o = (y * b.w + x) * 4
                                d[o] = rgb[s * 3]
                                d[o + 1] = rgb[s * 3 + 1]
                                d[o + 2] = rgb[s * 3 + 2]
                                d[o + 3] = a[s]
                            }
                        }
                    }
                }
            }
            pieces.append((px, b.w, b.h))
        }
        guard !pieces.isEmpty else { return nil }
        let n = pieces.count
        let cols = max(1, min(n, columns ?? (n <= 4 ? n : Int(Double(n).squareRoot().rounded(.up)))))
        let gap = max(1, Int((Double(pieces.map(\.h).max() ?? 1) * 0.04).rounded()))
        let rows = stride(from: 0, to: n, by: cols).map { Array($0..<min($0 + cols, n)) }
        let rowWidths = rows.map { row in row.map { pieces[$0].w }.reduce(0, +) + gap * (row.count - 1) }
        let rowHeights = rows.map { row in row.map { pieces[$0].h }.max() ?? 0 }
        let inner = rowWidths.max() ?? 0
        let width = inner + 2 * gap, height = rowHeights.reduce(0, +) + gap * (rows.count + 1)
        var sheet = [UInt8](repeating: 0, count: width * height * 4)
        sheet.withUnsafeMutableBufferPointer { d in
            var top = gap
            for (r, row) in rows.enumerated() {
                var left = gap + (inner - rowWidths[r]) / 2
                for i in row {
                    let p = pieces[i], y0 = top + rowHeights[r] - p.h
                    p.rgba.withUnsafeBufferPointer { s in
                        for y in 0..<p.h {
                            let src = y * p.w * 4, dst = ((y0 + y) * width + left) * 4
                            for k in 0..<(p.w * 4) { d[dst + k] = s[src + k] }
                        }
                    }
                    left += p.w + gap
                }
                top += rowHeights[r] + gap
            }
        }
        return (sheet, width, height)
    }
}
