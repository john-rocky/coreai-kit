// VLImagePreprocessor.swift — turns a CGImage into the ViT's `patches` tensor.
//
// Mirrors Qwen2VLImageProcessor for the fixed square grid (the numerics the gated python
// pipeline and the device-verified Qwen3-VL backend use): resize to `imageSide`², scale to
// [-1, 1] with x/127.5 - 1, then block-major patchify into per-patch [C, T(dup), P, P] rows.
// This is the numerics-critical step — keep it byte-for-byte with the reference. Orientation
// is normalized upstream (`CGImage.upright(_:)`); this routine assumes an upright image.

import CoreGraphics
import Foundation
import ImageIO

enum VLImagePreprocessor {
    // CLIP normalization (Qwen2-VL / MinerU): (x/255 − mean)/std.
    private static let clipMean: (Float, Float, Float) = (0.48145466, 0.4578275, 0.40821073)
    private static let clipStd: (Float, Float, Float) = (0.26862954, 0.26130258, 0.27577711)

    /// `[patches · patchDim]` f16, row-major per the ViT's expected layout. Supports square or
    /// non-square (`arch.imageWidth × arch.imageSide`) grids, `.stretch`/`.aspectFitPad` fit,
    /// and `.symmetric`/`.clip` normalization — the square/stretch/symmetric default is the
    /// device-verified Qwen3-VL path, numerically unchanged.
    static func patches(from cgImage: CGImage, arch: VLArchitecture) -> [Float16] {
        let width = arch.imageWidth
        let height = arch.imageHeight
        let patchSide = arch.patchSize
        let merge = arch.temporalPatchSize  // also the spatial merge factor on this grid
        let blocksWide = (width / patchSide) / merge
        let blocksTall = (height / patchSide) / merge
        var out = [Float16](repeating: 0, count: arch.patches * arch.patchDim)

        // Own the RGBA backing explicitly: the CGContext retains the pointer for the draw,
        // which outlives a `withUnsafeMutableBytes` closure (a use-after-scope footgun).
        let byteCount = width * height * 4
        let rgba = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
        defer { rgba.deallocate() }
        // White background for letterbox padding (documents have white margins); black is
        // fine under `.stretch` since the draw fills the whole canvas.
        let fill: UInt8 = arch.resize == .aspectFitPad ? 255 : 0
        rgba.initializeMemory(as: UInt8.self, repeating: fill, count: byteCount)

        guard
            let context = CGContext(
                data: rgba, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return out }
        context.interpolationQuality = .high
        switch arch.resize {
        case .stretch:
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        case .aspectFitPad:
            let s = min(Double(width) / Double(cgImage.width),
                        Double(height) / Double(cgImage.height))
            let dw = Double(cgImage.width) * s, dh = Double(cgImage.height) * s
            context.draw(cgImage, in: CGRect(
                x: (Double(width) - dw) / 2, y: (Double(height) - dh) / 2, width: dw, height: dh))
        }

        let pixels = rgba.assumingMemoryBound(to: UInt8.self)
        // Work in 0–255 space: (x/255 − mean)/std == (x − mean·255)/(std·255).
        let mean = arch.normalization == .clip
            ? (clipMean.0 * 255, clipMean.1 * 255, clipMean.2 * 255) : (127.5, 127.5, 127.5)
        let std = arch.normalization == .clip
            ? (clipStd.0 * 255, clipStd.1 * 255, clipStd.2 * 255) : (127.5, 127.5, 127.5)
        func normDivisor(_ c: Int) -> (Float, Float) {
            switch c { case 0: return (mean.0, std.0); case 1: return (mean.1, std.1)
            default: return (mean.2, std.2) }
        }
        func pixel(_ x: Int, _ y: Int, _ channel: Int) -> Float16 {
            let (m, d) = normDivisor(channel)
            return Float16((Float(pixels[(y * width + x) * 4 + channel]) - m) / d)
        }

        var patchIndex = 0
        // Block-major: iterate merge×merge sub-patches inside each merged block, so the ViT's
        // spatial-merge sees contiguous rows (Qwen2VLImageProcessor patch order).
        for blockRow in 0..<blocksTall {
            for blockCol in 0..<blocksWide {
                for innerRow in 0..<merge {
                    for innerCol in 0..<merge {
                        let pr = blockRow * merge + innerRow
                        let pc = blockCol * merge + innerCol
                        let y0 = pr * patchSide
                        let x0 = pc * patchSide
                        let base = patchIndex * arch.patchDim
                        for channel in 0..<3 {
                            for t in 0..<arch.temporalPatchSize {  // still frame duplicated
                                let channelBase = base
                                    + (channel * arch.temporalPatchSize + t) * patchSide * patchSide
                                for py in 0..<patchSide {
                                    for px in 0..<patchSide {
                                        out[channelBase + py * patchSide + px] =
                                            pixel(x0 + px, y0 + py, channel)
                                    }
                                }
                            }
                        }
                        patchIndex += 1
                    }
                }
            }
        }
        return out
    }

    /// `[1 · 3 · imageSide · imageSide]` f16, plain CHW — for towers that patchify in-graph
    /// (MiniCPM-V's SigLIP). Same resize + x/127.5 − 1 numerics as the patch path above,
    /// mirroring the gated reference preprocess (mean/std 0.5 overrides ImageNet).
    static func pixelValues(from cgImage: CGImage, arch: VLArchitecture) -> [Float16] {
        let side = arch.imageSide
        var out = [Float16](repeating: 0, count: 3 * side * side)

        let byteCount = side * side * 4
        let rgba = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 16)
        defer { rgba.deallocate() }
        rgba.initializeMemory(as: UInt8.self, repeating: 0, count: byteCount)

        guard
            let context = CGContext(
                data: rgba, width: side, height: side, bitsPerComponent: 8,
                bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return out }
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: side, height: side))

        let pixels = rgba.assumingMemoryBound(to: UInt8.self)
        for channel in 0..<3 {
            let channelBase = channel * side * side
            for y in 0..<side {
                for x in 0..<side {
                    out[channelBase + y * side + x] =
                        Float16(Float(pixels[(y * side + x) * 4 + channel]) / 127.5 - 1.0)
                }
            }
        }
        return out
    }
}

extension CGImage {
    /// Returns an upright copy for a non-`.up` EXIF orientation (camera photos), or `self`
    /// when already upright. Resolves orientation before the numerics-critical resize so the
    /// ViT sees the image the way a human does — also handy for an upright UI preview.
    public func upright(_ orientation: CGImagePropertyOrientation) -> CGImage {
        if orientation == .up { return self }
        let w = width, h = height
        // The 90°/270° cases swap the output's width and height.
        let swaps: Set<CGImagePropertyOrientation> = [.left, .right, .leftMirrored, .rightMirrored]
        let outW = swaps.contains(orientation) ? h : w
        let outH = swaps.contains(orientation) ? w : h
        guard
            let space = colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: outW, height: outH, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return self }

        // Transforms in the output (upright) space, origin bottom-left (CoreGraphics default).
        var transform = CGAffineTransform.identity
        switch orientation {
        case .down, .downMirrored:
            transform = transform.translatedBy(x: CGFloat(outW), y: CGFloat(outH)).rotated(by: .pi)
        case .left, .leftMirrored:
            transform = transform.translatedBy(x: CGFloat(outW), y: 0).rotated(by: .pi / 2)
        case .right, .rightMirrored:
            transform = transform.translatedBy(x: 0, y: CGFloat(outH)).rotated(by: -.pi / 2)
        default:
            break
        }
        switch orientation {
        case .upMirrored, .downMirrored:
            transform = transform.translatedBy(x: CGFloat(w), y: 0).scaledBy(x: -1, y: 1)
        case .leftMirrored, .rightMirrored:
            transform = transform.translatedBy(x: CGFloat(h), y: 0).scaledBy(x: -1, y: 1)
        default:
            break
        }
        context.concatenate(transform)
        context.draw(self, in: CGRect(x: 0, y: 0, width: w, height: h))
        return context.makeImage() ?? self
    }
}
