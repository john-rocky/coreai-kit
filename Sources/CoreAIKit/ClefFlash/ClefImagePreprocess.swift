// From the model zoo's apps/ClefFlash/Sources/ClefFlash/ImagePreprocess.swift (5ef2247, sha256 6f18f0592fb6, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// ClefImagePreprocess — image -> tower patches, the host half of the vision contract (conversion/clef_flash/host.py
// `preprocess(image, tile)`, the path every image gate fed):
//
//   RGB8 -> Pillow's BICUBIC resize to tile x tile (aspect not kept) -> /255 -> (x - 0.5) / 0.5
//        -> merge-block-major patches [4 * grid^2, 1536] float32, vector (C, T = 2, 16, 16), the frame twice.
//
// The resize is Pillow's own 8-bit resampler (libImaging/Resample.c, `ImagingResample` for an RGB image), not a float
// copy of it: per output sample a window of the input around center = (i + 0.5) * scale, bicubic weights (a = -0.5,
// support 2 x max(1, scale)) normalized by their sum and turned into 22-bit fixed point (round half away from zero),
// then integer sums started at 1 << 21, shifted right by 22 and clipped to 0...255; the horizontal pass first (into a
// uint8 image), then the vertical one, each only when that side changes. `host.resize_bicubic` (the float64 copy
// DeciderVision uses) lands 1–2 levels off Pillow on 12 of the 28 fixture tiles (results/test_host.json), and the
// oracle's pixel_values are Pillow's, so this file reproduces the integer arithmetic.
//
// Decoding keeps the file's 8-bit samples as they are, the way Pillow's `convert("RGB")` does: no color matching, alpha
// dropped. Formats the direct path does not read (16-bit, CMYK, premultiplied alpha) are drawn into an sRGB context
// instead and say so in `ClefRGB8Image.decodePath`.

import CoreGraphics
import Foundation
import ImageIO

/// 8-bit RGB pixels, row-major [height][width][3].
@available(macOS 27, iOS 27, *)
struct ClefRGB8Image: Sendable, Equatable {
    let width: Int
    let height: Int
    var pixels: [UInt8]
    /// How the pixels were obtained: "direct <layout>" (the decoded samples) or "drawn sRGB" (color-matched).
    var decodePath: String

    init(width: Int, height: Int, pixels: [UInt8], decodePath: String = "raw") {
        precondition(pixels.count == width * height * 3, "ClefRGB8Image: \(pixels.count) bytes for \(width)x\(height)")
        self.width = width
        self.height = height
        self.pixels = pixels
        self.decodePath = decodePath
    }
}

@available(macOS 27, iOS 27, *)
enum ClefImagePreprocess {
    static let patch = 16
    static let merge = 2
    static let temporal = 2
    static let channels = 3
    static let patchVector = channels * temporal * patch * patch   // 1536

    /// Pixels per side of the square tile a merged `grid` x `grid` covers (g256: 256, g448: 448).
    static func tileSide(grid: Int) -> Int { patch * merge * grid }

    // MARK: Decode

    static func loadCGImage(url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ClefFlashError.image("cannot decode \(url.path)") }
        return image
    }

    /// The image's 8-bit RGB samples, alpha dropped (Pillow `convert("RGB")`).
    static func rgb8(_ image: CGImage) throws -> ClefRGB8Image {
        if let direct = directRGB8(image) { return direct }
        return try drawnRGB8(image)
    }

    private static func directRGB8(_ image: CGImage) -> ClefRGB8Image? {
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
            // offsets of R, G, B inside one pixel as stored
            var offsets: [Int]
            let layout: String
            switch (bpp, alpha) {
            case (3, .none):
                offsets = [0, 1, 2]; layout = "RGB"
            case (4, .noneSkipLast), (4, .last):
                offsets = [0, 1, 2]; layout = alpha == .last ? "RGBA" : "RGBX"
            case (4, .noneSkipFirst), (4, .first):
                offsets = [1, 2, 3]; layout = alpha == .first ? "ARGB" : "XRGB"
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
            return ClefRGB8Image(width: w, height: h, pixels: out,
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
                    out[o] = v; out[o + 1] = v; out[o + 2] = v
                }
            }
            return ClefRGB8Image(width: w, height: h, pixels: out, decodePath: "direct gray")
        default:
            return nil
        }
    }

    private static func drawnRGB8(_ image: CGImage) throws -> ClefRGB8Image {
        let w = image.width, h = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw ClefFlashError.image("cannot make an RGB context \(w)x\(h)") }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw ClefFlashError.image("empty context") }
        let src = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = src[i * 4]; out[i * 3 + 1] = src[i * 4 + 1]; out[i * 3 + 2] = src[i * 4 + 2]
        }
        return ClefRGB8Image(width: w, height: h, pixels: out, decodePath: "drawn sRGB")
    }

    // MARK: Resize (Pillow's Resample.c, 8 bits per channel)

    static let precisionBits = 32 - 8 - 2      // PRECISION_BITS

    /// `bicubic_filter` (a = -0.5) in Resample.c's operation order.
    static func bicubic(_ t: Double) -> Double {
        let x = t < 0 ? -t : t
        let a = -0.5
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// `precompute_coeffs` + `normalize_coeffs_8bpc` for in0 = 0, in1 = inSize: per output sample the first input
    /// index, the window length and the 22-bit integer weights (ksize per sample, zeros past the window).
    static func coefficients(inSize: Int, outSize: Int) -> (ksize: Int, bounds: [(min: Int, count: Int)], kk: [Int]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let ksize = Int(support.rounded(.up)) * 2 + 1
        var bounds = [(min: Int, count: Int)](repeating: (0, 0), count: outSize)
        var kk = [Int](repeating: 0, count: outSize * ksize)
        let one = Double(1 << precisionBits)
        for xx in 0..<outSize {
            let center = 0.0 + (Double(xx) + 0.5) * scale
            var ww = 0.0
            let ss = 1.0 / filterscale
            var xmin = Int(center - support + 0.5)        // C (int): truncation toward zero
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var k = [Double](repeating: 0, count: ksize)
            for x in 0..<xmax {
                let w = bicubic((Double(x + xmin) - center + 0.5) * ss)
                k[x] = w
                ww += w
            }
            for x in 0..<xmax where ww != 0.0 { k[x] /= ww }
            for x in 0..<ksize {
                kk[xx * ksize + x] = k[x] < 0 ? Int(-0.5 + k[x] * one) : Int(0.5 + k[x] * one)
            }
            bounds[xx] = (xmin, xmax)
        }
        return (ksize, bounds, kk)
    }

    @inline(__always)
    static func clip8(_ v: Int) -> UInt8 {
        if v >= ((1 << precisionBits) << 8) { return 255 }
        if v <= 0 { return 0 }
        return UInt8(v >> precisionBits)
    }

    /// `ImagingResampleHorizontal_8bpc` (axis 1) / `ImagingResampleVertical_8bpc` (axis 0) on an RGB plane.
    private static func pass(_ src: [UInt8], h: Int, w: Int, horizontal: Bool, out: Int) -> [UInt8] {
        let (ksize, bounds, kk) = coefficients(inSize: horizontal ? w : h, outSize: out)
        let oh = horizontal ? h : out, ow = horizontal ? out : w
        var dst = [UInt8](repeating: 0, count: oh * ow * 3)
        let half = 1 << (precisionBits - 1)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                kk.withUnsafeBufferPointer { k in
                    for y in 0..<oh {
                        for x in 0..<ow {
                            let i = horizontal ? x : y
                            let (lo, n) = bounds[i]
                            let kb = i * ksize
                            var s0 = half, s1 = half, s2 = half
                            for t in 0..<n {
                                let sy = horizontal ? y : lo + t, sx = horizontal ? lo + t : x
                                let p = (sy * w + sx) * 3
                                let c = k[kb + t]
                                s0 += Int(s[p]) * c
                                s1 += Int(s[p + 1]) * c
                                s2 += Int(s[p + 2]) * c
                            }
                            let o = (y * ow + x) * 3
                            d[o] = clip8(s0)
                            d[o + 1] = clip8(s1)
                            d[o + 2] = clip8(s2)
                        }
                    }
                }
            }
        }
        return dst
    }

    /// Pillow's `Image.resize((width, height), BICUBIC)` of an RGB image: the horizontal pass when the width changes,
    /// then the vertical pass when the height changes (each on uint8).
    static func resizeBicubic(_ image: ClefRGB8Image, width: Int, height: Int) -> ClefRGB8Image {
        var x = image.pixels
        var h = image.height, w = image.width
        if w != width {
            x = pass(x, h: h, w: w, horizontal: true, out: width)
            w = width
        }
        if h != height {
            x = pass(x, h: h, w: w, horizontal: false, out: height)
            h = height
        }
        return ClefRGB8Image(width: w, height: h, pixels: x, decodePath: image.decodePath)
    }

    // MARK: Patches (host.patchify)

    /// Resized tile -> [4 * grid^2, 1536] float32: (x / 255 - 0.5) / 0.5 in float64, then float32; patches in
    /// (block row, block col, row in block, col in block) order, each vector (C, T, 16, 16) with the frame at both T.
    static func patches(_ tile: ClefRGB8Image) throws -> [Float] {
        let side = tile.width
        guard tile.height == side, side % (patch * merge) == 0 else {
            throw ClefFlashError.image("tile \(tile.width)x\(tile.height) is not a square multiple of \(patch * merge)")
        }
        let g = side / patch             // patches per side (2 * grid)
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

    /// Everything the tower needs from one image at one grid, with each step's time.
    struct Prepared: Sendable {
        let decoded: ClefRGB8Image
        let resized: ClefRGB8Image
        let patches: [Float]
        let decodeSeconds: Double
        let resizeSeconds: Double
        let patchSeconds: Double
    }

    static func prepare(_ image: CGImage, grid: Int) throws -> Prepared {
        let t0 = ContinuousClock.now
        let rgb = try rgb8(image)
        let t1 = ContinuousClock.now
        let side = tileSide(grid: grid)
        let resized = resizeBicubic(rgb, width: side, height: side)
        let t2 = ContinuousClock.now
        let p = try patches(resized)
        let t3 = ContinuousClock.now
        func s(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
            let d = b - a
            return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
        }
        return Prepared(decoded: rgb, resized: resized, patches: p, decodeSeconds: s(t0, t1),
                        resizeSeconds: s(t1, t2), patchSeconds: s(t2, t3))
    }
}
