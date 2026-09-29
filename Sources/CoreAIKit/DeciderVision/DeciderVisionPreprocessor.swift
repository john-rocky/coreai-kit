// DeciderVisionPreprocessor.swift — image → the vision tower's patches, the host half of decider-2b-vision's
// image contract. Ported from the model zoo's `apps/DeciderVision/Sources/DeciderVision/ImagePreprocess.swift`
// (2307ecf), which copies the zoo's `conversion/decider_vision/host.py`:
//
//   RGB8 → bicubic resize to (32 · grid)² in Pillow's pass order → /255 → (x − 0.5) / 0.5
//        → merge-block-major patches [4 · grid², 1536] float32, vector (C, T = 2, 16, 16), the frame twice.
//
// The resize is `host.resize_bicubic` written out, and it is part of the numbers: Pillow's antialiased bicubic
// (a = −0.5, support 2 × the reduction factor), the HORIZONTAL pass first, the intermediate rounded
// (floor(x + 0.5)) and clipped to 0…255 before the vertical pass, every sum in float64 in NumPy's order (weights
// normalised by NumPy's pairwise sum, taps accumulated in window order with a fused multiply-add from zero). A
// CGContext resize is a different image, and a vertical-first resize lands up to 17 levels from Pillow's. The
// aspect ratio is not kept: a 256×240 frame is stretched to 256×256, as the author's fixed-grid arms were captured.
//
// Decoding keeps the image's 8-bit samples as stored, the way Pillow's `convert("RGB")` does: no colour matching,
// alpha dropped, EXIF orientation not applied (the author's code applies none either). Formats the direct path does
// not read (16-bit, CMYK, premultiplied alpha) are drawn into an sRGB context instead.

import CoreGraphics
import Foundation

enum DeciderVisionPreprocessor {
    static let patch = 16
    static let merge = 2
    static let temporal = 2
    static let channels = 3
    /// One patch vector: channels × temporal × 16 × 16.
    static let patchVector = channels * temporal * patch * patch  // 1536

    /// 8-bit RGB pixels, row-major [height][width][3].
    struct RGB8: Sendable, Equatable {
        let width: Int
        let height: Int
        var pixels: [UInt8]
        /// How the pixels were obtained: "direct <layout>" (the stored samples) or "drawn sRGB" (colour-matched).
        var decodePath: String
    }

    /// Everything the tower needs from one image at one grid, with each step's time.
    struct Prepared: Sendable {
        let width: Int
        let height: Int
        let decodePath: String
        /// [4 · grid², 1536], row-major.
        let patches: [Float]
        let decodeSeconds: Double
        let resizeSeconds: Double
        let patchSeconds: Double
    }

    /// Pixels per side of the square tile a merged `grid` × `grid` covers (8: 256, 14: 448).
    static func tileSide(grid: Int) -> Int { patch * merge * grid }

    static func prepare(_ image: CGImage, grid: Int) throws -> Prepared {
        let t0 = ContinuousClock.now
        let rgb = try rgb8(image)
        let t1 = ContinuousClock.now
        let side = tileSide(grid: grid)
        let resized = resizeBicubic(rgb, width: side, height: side)
        let t2 = ContinuousClock.now
        let p = try patches(resized)
        let t3 = ContinuousClock.now
        return Prepared(
            width: rgb.width, height: rgb.height, decodePath: rgb.decodePath, patches: p,
            decodeSeconds: seconds(t0, t1), resizeSeconds: seconds(t1, t2), patchSeconds: seconds(t2, t3))
    }

    // MARK: - Decode

    /// The image's 8-bit RGB samples, alpha dropped (Pillow `convert("RGB")`).
    static func rgb8(_ image: CGImage) throws -> RGB8 {
        if let direct = directRGB8(image) { return direct }
        return try drawnRGB8(image)
    }

    private static func directRGB8(_ image: CGImage) -> RGB8? {
        guard image.bitsPerComponent == 8, let space = image.colorSpace,
            let data = image.dataProvider?.data, let base = CFDataGetBytePtr(data)
        else { return nil }
        let w = image.width, h = image.height, rowBytes = image.bytesPerRow
        let bpp = image.bitsPerPixel / 8
        let alpha = image.alphaInfo
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let little = order == .byteOrder32Little || order == .byteOrder16Little
        var out = [UInt8](repeating: 0, count: w * h * 3)
        switch space.model {
        case .rgb:
            // Offsets of R, G, B inside one pixel as stored.
            var offsets: [Int]
            let layout: String
            switch (bpp, alpha) {
            case (3, .none):
                offsets = [0, 1, 2]
                layout = "RGB"
            case (4, .noneSkipLast), (4, .last):
                offsets = [0, 1, 2]
                layout = alpha == .last ? "RGBA" : "RGBX"
            case (4, .noneSkipFirst), (4, .first):
                offsets = [1, 2, 3]
                layout = alpha == .first ? "ARGB" : "XRGB"
            default:
                return nil
            }
            if little && bpp == 4 { offsets = offsets.map { 3 - $0 } }
            for y in 0..<h {
                let row = base + y * rowBytes
                for x in 0..<w {
                    let p = row + x * bpp, o = (y * w + x) * 3
                    out[o] = p[offsets[0]]
                    out[o + 1] = p[offsets[1]]
                    out[o + 2] = p[offsets[2]]
                }
            }
            return RGB8(
                width: w, height: h, pixels: out,
                decodePath: "direct \(layout)\(little && bpp == 4 ? " little-endian" : "")")
        case .monochrome:
            let lOffset: Int
            switch (bpp, alpha) {
            case (1, .none): lOffset = 0
            case (2, .last), (2, .noneSkipLast): lOffset = little ? 1 : 0
            case (2, .first), (2, .noneSkipFirst): lOffset = little ? 0 : 1
            default: return nil
            }
            for y in 0..<h {
                let row = base + y * rowBytes
                for x in 0..<w {
                    let v = row[x * bpp + lOffset], o = (y * w + x) * 3
                    out[o] = v
                    out[o + 1] = v
                    out[o + 2] = v
                }
            }
            return RGB8(width: w, height: h, pixels: out, decodePath: "direct gray")
        default:
            return nil
        }
    }

    private static func drawnRGB8(_ image: CGImage) throws -> RGB8 {
        let w = image.width, h = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw DeciderVisionError.image("cannot make an RGB context \(w)x\(h)") }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw DeciderVisionError.image("empty context") }
        let src = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = src[i * 4]
            out[i * 3 + 1] = src[i * 4 + 1]
            out[i * 3 + 2] = src[i * 4 + 2]
        }
        return RGB8(width: w, height: h, pixels: out, decodePath: "drawn sRGB")
    }

    // MARK: - Resize (host.resize_bicubic)

    /// Pillow's bicubic kernel at distance `t` (already divided by the filter scale), a = −0.5, in NumPy's
    /// operation order.
    static func bicubic(_ t: Double) -> Double {
        let x = abs(t)
        let a = -0.5
        if x < 1.0 {
            var w = (a + 2.0) * x
            w = w - (a + 3.0)
            w = w * x
            w = w * x
            return w + 1.0
        }
        if x < 2.0 {
            var w = x - 5.0
            w = w * x
            w = w + 8.0
            w = w * x
            w = w - 4.0
            return w * a
        }
        return 0.0
    }

    /// Per output sample, the window's first input index and its normalised weights (`_filter_weights`).
    static func filterWeights(inSize: Int, outSize: Int) -> (starts: [Int], weights: [[Double]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(1.0, scale)
        let support = 2.0 * filterscale
        var starts = [Int](repeating: 0, count: outSize)
        var weights = [[Double]](repeating: [], count: outSize)
        for i in 0..<outSize {
            let center = (Double(i) + 0.5) * scale
            let xmin = Int(max(0.0, (center - support + 0.5).rounded(.down)))
            let xmax = Int(min(Double(inSize), (center + support + 0.5).rounded(.up)))
            var w = [Double]()
            w.reserveCapacity(max(0, xmax - xmin))
            for x in xmin..<max(xmin, xmax) {
                w.append(bicubic(((Double(x) + 0.5) - center) / filterscale))
            }
            let total = DeciderVisionNumerics.numpySum(w)
            if total > 0 { w = w.map { $0 / total } }
            starts[i] = xmin
            weights[i] = w
        }
        return (starts, weights)
    }

    /// One pass along `axis` (1 = horizontal, 0 = vertical) of a float [H][W][3] plane, then floor(x + 0.5) and clip.
    private static func pass(_ src: [Double], h: Int, w: Int, axis: Int, out: Int) -> [Double] {
        let inSize = axis == 1 ? w : h
        let (starts, weights) = filterWeights(inSize: inSize, outSize: out)
        let oh = axis == 1 ? h : out, ow = axis == 1 ? out : w
        var dst = [Double](repeating: 0, count: oh * ow * 3)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                for y in 0..<oh {
                    for x in 0..<ow {
                        let i = axis == 1 ? x : y
                        let ws = weights[i], s0 = starts[i]
                        for c in 0..<3 {
                            var acc = 0.0
                            for k in 0..<ws.count {
                                let sy = axis == 1 ? y : s0 + k, sx = axis == 1 ? s0 + k : x
                                acc = acc.addingProduct(ws[k], s[(sy * w + sx) * 3 + c])
                            }
                            d[(y * ow + x) * 3 + c] = min(255.0, max(0.0, (acc + 0.5).rounded(.down)))
                        }
                    }
                }
            }
        }
        return dst
    }

    /// `host.resize_bicubic`: the horizontal pass first (when the width changes), the uint8 intermediate, then the
    /// vertical pass.
    static func resizeBicubic(_ image: RGB8, width: Int, height: Int) -> RGB8 {
        var x = image.pixels.map(Double.init)
        var h = image.height, w = image.width
        if w != width {
            x = pass(x, h: h, w: w, axis: 1, out: width)
            w = width
        }
        if h != height {
            x = pass(x, h: h, w: w, axis: 0, out: height)
            h = height
        }
        return RGB8(width: w, height: h, pixels: x.map { UInt8($0) }, decodePath: image.decodePath)
    }

    // MARK: - Patches (host.patchify)

    /// A resized tile → [4 · grid², 1536] float32: (x / 255 − 0.5) / 0.5 in float64, then float32; patches in
    /// (block row, block column, row in block, column in block) order, each vector (C, T, 16, 16) with the frame at
    /// both T.
    static func patches(_ tile: RGB8) throws -> [Float] {
        let side = tile.width
        guard tile.height == side, side % (patch * merge) == 0 else {
            throw DeciderVisionError.image(
                "tile \(tile.width)x\(tile.height) is not a square multiple of \(patch * merge)")
        }
        let g = side / patch  // patches per side (2 · grid)
        let blocks = g / merge
        var lut = [Float](repeating: 0, count: 256)
        for v in 0..<256 { lut[v] = Float((Double(v) / 255.0 - 0.5) / 0.5) }
        var out = [Float](repeating: 0, count: g * g * patchVector)
        tile.pixels.withUnsafeBufferPointer { px in
            out.withUnsafeMutableBufferPointer { o in
                for bh in 0..<blocks {
                    for bw in 0..<blocks {
                        for mh in 0..<merge {
                            for mw in 0..<merge {
                                let n = ((bh * blocks + bw) * merge + mh) * merge + mw
                                let row0 = (bh * merge + mh) * patch, col0 = (bw * merge + mw) * patch
                                for c in 0..<channels {
                                    for t in 0..<temporal {
                                        let base = n * patchVector + (c * temporal + t) * patch * patch
                                        for py in 0..<patch {
                                            for pxx in 0..<patch {
                                                let v = px[((row0 + py) * side + col0 + pxx) * 3 + c]
                                                o[base + py * patch + pxx] = lut[Int(v)]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    private static func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
        let d = b - a
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}

/// The two NumPy reductions the reference numbers go through, so a weight or a probability here is the same double
/// the zoo's Python gates compute: float64 pairwise summation, and a softmax at T in float64.
enum DeciderVisionNumerics {
    /// NumPy's float64 `add.reduce` over a contiguous 1-D array (pairwise summation, 8 accumulators per block of at
    /// most 128): the order `w.sum()` and `p.sum()` add in.
    static func numpySum(_ a: [Double]) -> Double {
        guard !a.isEmpty else { return 0 }
        return a.withUnsafeBufferPointer { pairwise($0.baseAddress!, a.count) }
    }

    private static func pairwise(_ a: UnsafePointer<Double>, _ n: Int) -> Double {
        if n < 8 {
            var res = 0.0
            for i in 0..<n { res += a[i] }
            return res
        } else if n <= 128 {
            var r = (0..<8).map { a[$0] }
            var i = 8
            while i < n - (n % 8) {
                for j in 0..<8 { r[j] += a[i + j] }
                i += 8
            }
            var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n {
                res += a[i]
                i += 1
            }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwise(a, n2) + pairwise(a + n2, n - n2)
        }
    }

    /// Softmax of `logits / temperature` in float64, as the gates compute it: exp(x − max) / sum.
    static func softmax(_ logits: [Double], temperature: Double = 1) -> [Double] {
        let scaled = temperature == 1 ? logits : logits.map { $0 / temperature }
        guard let m = scaled.max() else { return [] }
        let e = scaled.map { Foundation.exp($0 - m) }
        let s = numpySum(e)
        return e.map { $0 / s }
    }

    /// The first index of the largest value (numpy.argmax).
    static func argmax<T: Comparable>(_ xs: [T]) -> Int {
        guard !xs.isEmpty else { return 0 }
        var best = 0
        for i in 1..<xs.count where xs[i] > xs[best] { best = i }
        return best
    }
}
