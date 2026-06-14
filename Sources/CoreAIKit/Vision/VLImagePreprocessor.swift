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
    /// `[patches · patchDim]` f16, row-major per the ViT's expected layout.
    static func patches(from cgImage: CGImage, arch: VLArchitecture) -> [Float16] {
        let side = arch.imageSide
        let patchSide = arch.patchSize
        let merge = arch.temporalPatchSize  // also the spatial merge factor on this grid
        let patchesPerSide = side / patchSide          // 28
        let blocksPerSide = patchesPerSide / merge      // 14
        var out = [Float16](repeating: 0, count: arch.patches * arch.patchDim)

        // Own the RGBA backing explicitly: the CGContext retains the pointer for the draw,
        // which outlives a `withUnsafeMutableBytes` closure (a use-after-scope footgun).
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
        func pixel(_ x: Int, _ y: Int, _ channel: Int) -> Float16 {
            Float16(Float(pixels[(y * side + x) * 4 + channel]) / 127.5 - 1.0)
        }

        var patchIndex = 0
        // Block-major: iterate merge×merge sub-patches inside each merged block, so the ViT's
        // spatial-merge sees contiguous rows.
        for blockRow in 0..<blocksPerSide {
            for blockCol in 0..<blocksPerSide {
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
