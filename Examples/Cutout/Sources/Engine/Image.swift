// Image — a photo file → upright 8-bit RGB → the two inputs the model reads from it:
//
//   the vision tower   Pillow's BICUBIC to 448 × 448 (width and height each, the aspect ratio not kept), then the
//                      processor's patches: (x / 255 − 0.5) / 0.5, the frame twice (temporal 2), merge-block-major
//                      [784, 1536] fp16
//   the mask decoder   Pillow's BICUBIC to 1024 × 1024, / 255, [3, 1024, 1024]
//
// The resize is Pillow's `ImagingResample` for 8-bit images (libImaging/Resample.c) written out, as the NumPy copy in
// the zoo's conversion/wevisone/mask/mask_host.py does: the a = −0.5 cubic widened by the downscale factor, the weights
// normalized in double and rounded to 22-bit fixed point, a horizontal pass to uint8, then a vertical pass to uint8.
// Integer sums are exact, so the copy can equal Pillow byte for byte; the NumPy copy did on 51 of 51 fixture resizes
// (`mask_host.py check`). (DeciderVision's ImagePreprocess resizes in float64 instead, a different rounding: not used
// here.)
//
// Decoding keeps the file's 8-bit samples (Pillow's `convert("RGB")`: no color matching, alpha dropped), then turns
// the picture upright by its EXIF orientation with exact pixel moves. The package hands Pillow's image over as stored
// (no EXIF transpose); a phone photo stored sideways is turned upright here before the model sees it, so the boxes and
// the mattes are the upright picture's. A long side over 4096 px is subsampled while it decodes (ImageIO), as Look's
// photos are.

import CoreGraphics
import Foundation
import ImageIO

/// 8-bit RGB pixels, row-major [height][width][3].
public struct RGB8Image: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]
    /// How the pixels were obtained: "direct <layout>" (the decoded samples) or "drawn sRGB" (color-matched).
    public var decodePath: String

    public init(width: Int, height: Int, pixels: [UInt8], decodePath: String = "raw") {
        precondition(pixels.count == width * height * 3, "RGB8Image: \(pixels.count) bytes for \(width)x\(height)")
        self.width = width
        self.height = height
        self.pixels = pixels
        self.decodePath = decodePath
    }
}

public enum ImageHost {
    public static let towerSide = 448
    public static let canvasSide = 1024
    public static let patch = 16
    public static let merge = 2
    public static let temporal = 2
    public static let patchVector = 3 * 2 * 16 * 16   // 1536
    public static let maxDecodeSide = 4096

    // MARK: Decode

    /// The first image of a file and its EXIF orientation, subsampled to a long side of at most 4096 px.
    public static func load(url: URL) throws -> (image: CGImage, orientation: CGImagePropertyOrientation) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw CutoutError.image("cannot read \(url.lastPathComponent)")
        }
        return try load(source: source, name: url.lastPathComponent)
    }

    public static func load(data: Data) throws -> (image: CGImage, orientation: CGImagePropertyOrientation) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw CutoutError.image("cannot read the photo's data")
        }
        return try load(source: source, name: "the photo")
    }

    static func load(source: CGImageSource, name: String) throws -> (image: CGImage, orientation: CGImagePropertyOrientation) {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let raw = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        let side = max(properties?[kCGImagePropertyPixelWidth] as? Int ?? 0, properties?[kCGImagePropertyPixelHeight] as? Int ?? 0)
        let image = side > maxDecodeSide
            ? CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: maxDecodeSide,
                kCGImageSourceCreateThumbnailWithTransform: false,
            ] as CFDictionary)
            : CGImageSourceCreateImageAtIndex(source, 0, nil)
        guard let image else { throw CutoutError.image("cannot decode \(name)") }
        return (image, CGImagePropertyOrientation(rawValue: raw) ?? .up)
    }

    /// The image's 8-bit RGB samples, alpha dropped (Pillow `convert("RGB")`), turned upright.
    public static func uprightRGB8(_ image: CGImage, orientation: CGImagePropertyOrientation) throws -> RGB8Image {
        let rgb = try directRGB8(image) ?? drawnRGB8(image)
        return upright(rgb, orientation)
    }

    private static func directRGB8(_ image: CGImage) -> RGB8Image? {
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
            return RGB8Image(width: w, height: h, pixels: out, decodePath: "direct \(layout)\(little && bpp == 4 ? " little-endian" : "")")
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
            return RGB8Image(width: w, height: h, pixels: out, decodePath: "direct gray")
        default:
            return nil
        }
    }

    private static func drawnRGB8(_ image: CGImage) throws -> RGB8Image {
        let w = image.width, h = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw CutoutError.image("cannot make an RGB context \(w)x\(h)") }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw CutoutError.image("empty context") }
        let src = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = src[i * 4]; out[i * 3 + 1] = src[i * 4 + 1]; out[i * 3 + 2] = src[i * 4 + 2]
        }
        return RGB8Image(width: w, height: h, pixels: out, decodePath: "drawn sRGB")
    }

    /// The stored pixels moved so the picture stands upright (EXIF orientation 1–8; exact, no resampling).
    static func upright(_ img: RGB8Image, _ o: CGImagePropertyOrientation) -> RGB8Image {
        if o == .up { return img }
        let w = img.width, h = img.height
        let swap = [.left, .right, .leftMirrored, .rightMirrored].contains(o)
        let ow = swap ? h : w, oh = swap ? w : h
        var out = [UInt8](repeating: 0, count: img.pixels.count)
        img.pixels.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<oh {
                    for x in 0..<ow {
                        // the stored pixel shown at upright (x, y)
                        let sx: Int, sy: Int
                        switch o {
                        case .upMirrored: sx = w - 1 - x; sy = y
                        case .down: sx = w - 1 - x; sy = h - 1 - y
                        case .downMirrored: sx = x; sy = h - 1 - y
                        case .leftMirrored: sx = y; sy = x
                        case .right: sx = y; sy = h - 1 - x
                        case .rightMirrored: sx = w - 1 - y; sy = h - 1 - x
                        case .left: sx = w - 1 - y; sy = x
                        default: sx = x; sy = y
                        }
                        let si = (sy * w + sx) * 3, di = (y * ow + x) * 3
                        d[di] = s[si]; d[di + 1] = s[si + 1]; d[di + 2] = s[si + 2]
                    }
                }
            }
        }
        return RGB8Image(width: ow, height: oh, pixels: out, decodePath: img.decodePath + ", upright from EXIF \(o.rawValue)")
    }

    // MARK: Pillow's BICUBIC (libImaging/Resample.c, 8 bits per channel)

    static let precisionBits = 32 - 8 - 2

    static func bicubic(_ t: Double) -> Double {
        let a = -0.5
        let x = abs(t)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// precompute_coeffs + normalize_coeffs_8bpc for the box (0, inSize): per output sample its window start, length and
    /// fixed-point weights.
    static func coefficients(inSize: Int, outSize: Int) -> (xmin: [Int], xlen: [Int], kk: [[Int]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let ss = 1.0 / filterscale
        var xmins = [Int](repeating: 0, count: outSize)
        var xlens = [Int](repeating: 0, count: outSize)
        var kks = [[Int]](repeating: [], count: outSize)
        let one = Double(1 << precisionBits)
        for xx in 0..<outSize {
            let center = 0.0 + (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var w = [Double](repeating: 0, count: max(0, xmax))
            var ww = 0.0
            for x in 0..<max(0, xmax) {
                let v = bicubic((Double(x + xmin) - center + 0.5) * ss)
                w[x] = v
                ww += v
            }
            var k = [Int](repeating: 0, count: w.count)
            for x in 0..<w.count {
                let v = ww != 0.0 ? w[x] / ww : w[x]
                k[x] = v < 0 ? Int(-0.5 + v * one) : Int(0.5 + v * one)
            }
            xmins[xx] = xmin
            xlens[xx] = max(0, xmax)
            kks[xx] = k
        }
        return (xmins, xlens, kks)
    }

    @inline(__always) static func clip8(_ v: Int) -> UInt8 {
        if v >= ((1 << precisionBits) << 8) { return 255 }
        if v <= 0 { return 0 }
        return UInt8(v >> precisionBits)
    }

    /// `Image.fromarray(rgb).resize((outWidth, outHeight), BICUBIC)`: horizontal pass first (when the width changes),
    /// the uint8 intermediate, then the vertical pass.
    public static func pillowBicubic(_ image: RGB8Image, width outW: Int, height outH: Int) -> RGB8Image {
        var src = image.pixels
        var w = image.width
        let h = image.height
        let half = 1 << (precisionBits - 1)
        if outW != w {
            let c = coefficients(inSize: w, outSize: outW)
            var dst = [UInt8](repeating: 0, count: outW * h * 3)
            src.withUnsafeBufferPointer { s in
                dst.withUnsafeMutableBufferPointer { d in
                    for y in 0..<h {
                        let row = y * w * 3
                        for xx in 0..<outW {
                            let k = c.kk[xx], x0 = c.xmin[xx]
                            var s0 = half, s1 = half, s2 = half
                            for t in 0..<c.xlen[xx] {
                                let p = row + (x0 + t) * 3
                                let kt = k[t]
                                s0 += Int(s[p]) * kt
                                s1 += Int(s[p + 1]) * kt
                                s2 += Int(s[p + 2]) * kt
                            }
                            let o = (y * outW + xx) * 3
                            d[o] = clip8(s0); d[o + 1] = clip8(s1); d[o + 2] = clip8(s2)
                        }
                    }
                }
            }
            src = dst
            w = outW
        }
        if outH != h {
            let c = coefficients(inSize: h, outSize: outH)
            var dst = [UInt8](repeating: 0, count: w * outH * 3)
            src.withUnsafeBufferPointer { s in
                dst.withUnsafeMutableBufferPointer { d in
                    for yy in 0..<outH {
                        let k = c.kk[yy], y0 = c.xmin[yy], n = c.xlen[yy]
                        for x in 0..<w {
                            var s0 = half, s1 = half, s2 = half
                            for t in 0..<n {
                                let p = ((y0 + t) * w + x) * 3
                                let kt = k[t]
                                s0 += Int(s[p]) * kt
                                s1 += Int(s[p + 1]) * kt
                                s2 += Int(s[p + 2]) * kt
                            }
                            let o = (yy * w + x) * 3
                            d[o] = clip8(s0); d[o + 1] = clip8(s1); d[o + 2] = clip8(s2)
                        }
                    }
                }
            }
            return RGB8Image(width: w, height: outH, pixels: dst, decodePath: image.decodePath)
        }
        return RGB8Image(width: w, height: h, pixels: src, decodePath: image.decodePath)
    }

    // MARK: The model's inputs

    /// The processor's pixel_values for a tile already at 448 × 448 (`mask_host.patches`): (x / 255 − 0.5) / 0.5 in
    /// float32, the frame at both temporal slots, patch n = ((block row · blocks + block col) · 2 + row in block) · 2 +
    /// col in block, each vector (C, T, 16, 16); then float16 (the tower's input type).
    public static func patches(_ tile: RGB8Image) throws -> [Float16] {
        let side = tile.width
        guard tile.height == side, side % (patch * merge) == 0 else {
            throw CutoutError.image("tile \(tile.width)x\(tile.height) is not a square multiple of \(patch * merge)")
        }
        let g = side / patch
        let blocks = g / merge
        var lut = [Float16](repeating: 0, count: 256)
        for v in 0..<256 {
            let x = Float(v) / Float(255.0)
            lut[v] = Float16((x - Float(0.5)) / Float(0.5))
        }
        var out = [Float16](repeating: 0, count: g * g * patchVector)
        tile.pixels.withUnsafeBufferPointer { px in
            out.withUnsafeMutableBufferPointer { o in
                for bh in 0..<blocks {
                    for bw in 0..<blocks {
                        for mh in 0..<merge {
                            for mw in 0..<merge {
                                let n = ((bh * blocks + bw) * merge + mh) * merge + mw
                                let row0 = (bh * merge + mh) * patch, col0 = (bw * merge + mw) * patch
                                for c in 0..<3 {
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

    /// The mask decoder's canvas [3, 1024, 1024] in [0, 1] (`_pil_to_float_rgb` of the 1024 × 1024 resize): channel-major.
    public static func canvas(_ resized: RGB8Image) -> [Float] {
        let n = resized.width * resized.height
        var out = [Float](repeating: 0, count: 3 * n)
        var lut = [Float](repeating: 0, count: 256)
        for v in 0..<256 { lut[v] = Float(v) / Float(255.0) }
        resized.pixels.withUnsafeBufferPointer { px in
            out.withUnsafeMutableBufferPointer { o in
                for i in 0..<n {
                    o[i] = lut[Int(px[i * 3])]
                    o[n + i] = lut[Int(px[i * 3 + 1])]
                    o[2 * n + i] = lut[Int(px[i * 3 + 2])]
                }
            }
        }
        return out
    }

    /// The picture as a CGImage (for the screen and for composing the cutout).
    public static func cgImage(_ img: RGB8Image) -> CGImage? {
        var rgba = [UInt8](repeating: 255, count: img.width * img.height * 4)
        for i in 0..<(img.width * img.height) {
            rgba[i * 4] = img.pixels[i * 3]
            rgba[i * 4 + 1] = img.pixels[i * 3 + 1]
            rgba[i * 4 + 2] = img.pixels[i * 3 + 2]
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return CGImage(width: img.width, height: img.height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: img.width * 4, space: space,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
}
