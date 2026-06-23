// SuperResolver.swift — single-step diffusion-GAN super-resolution (AdcSR, ×4) over GraphModel.
//
// The exported graph is ONE tile, fp32 I/O (fp16 weights convert transparently):
//   input  : lr [1,3,T,T]  in [-1,1]   (a low-resolution tile)
//   output : sr [1,3,T*4,T*4] in [-1,1]
// AdcSR is image→image with no noise / prompt / timestep. Larger images are split into
// overlapping T-px LR windows, each upscaled ×4, and feather-blended into the result.
//
// NOTE: tile seams / orientation / speed are best confirmed on-device (build is device-side).

import CoreGraphics
import Foundation

public final class SuperResolver: @unchecked Sendable {
    public let scale: Int        // output_side / input_side (4 for AdcSR)
    /// Caps the input's long side before upscaling (output ≤ maxInputSide·scale). Full-resolution
    /// phone photos would otherwise produce a gigapixel result and exhaust memory; SR is for
    /// enlarging small/low-res images, so large inputs are downscaled to fit first.
    public var maxInputSide = 512
    private let lrTile: Int      // model input side, read from the graph (e.g. 128)
    private let srTile: Int      // model output side (= lrTile * scale)
    private let overlap: Int     // LR-space overlap between tiles, for feather blending

    private let graph: GraphModel
    private let lrInput: String
    private let srOutput: String

    /// Loads a bundle directory holding one `*.aimodel` (or the `.aimodel` itself).
    public init(bundleAt url: URL, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        let modelURL: URL
        if url.pathExtension == "aimodel" {
            modelURL = url
        } else {
            guard
                let found = try FileManager.default.contentsOfDirectory(
                    at: url, includingPropertiesForKeys: nil
                ).first(where: { $0.pathExtension == "aimodel" })
            else {
                throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
            }
            modelURL = found
        }
        let graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.graph = graph

        guard let lrInput = graph.inputNames.first(where: { (graph.shape(ofInput: $0)?.count ?? 0) == 4 }),
            let srOutput = graph.outputNames.first,
            let inShape = graph.shape(ofInput: lrInput), let inSide = inShape.last, inSide > 0,
            let outShape = graph.shape(ofOutput: srOutput), let outSide = outShape.last, outSide > 0
        else {
            throw VisionError.bundleLayout(
                "expected one rank-4 image input + output, got \(graph.inputNames) -> \(graph.outputNames)")
        }
        self.lrInput = lrInput
        self.srOutput = srOutput
        self.lrTile = inSide
        self.srTile = outSide
        self.scale = max(1, outSide / inSide)
        self.overlap = max(4, inSide / 8)
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .adcsrX4,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    /// Upscales `image` by ×4. Deterministic (AdcSR has no stochastic input).
    public func upscale(_ image: CGImage) async throws -> CGImage {
        // Downscale the input so its long side ≤ maxInputSide (renderRGB scales while drawing),
        // bounding memory and tile count for large photos.
        let longSide = max(image.width, image.height)
        let s = longSide > maxInputSide ? Double(maxInputSide) / Double(longSide) : 1.0
        let lrW = max(1, Int((Double(image.width) * s).rounded()))
        let lrH = max(1, Int((Double(image.height) * s).rounded()))
        let lr = try renderRGB(image, width: lrW, height: lrH)  // [lrH*lrW*3] in [0,1], top-down

        let outW = lrW * scale
        let outH = lrH * scale
        var acc = [Float](repeating: 0, count: outW * outH * 3)  // weighted SR sum
        var wsum = [Float](repeating: 0, count: outW * outH)     // blend weights
        let feather = featherWeights(srTile)                     // [srTile*srTile] in (0,1]

        for (x0, y0) in tilePositions(width: lrW, height: lrH) {
            let lrCrop = cropNormalized(lr, srcW: lrW, srcH: lrH, x0: x0, y0: y0)  // [3,lrTile,lrTile] [-1,1]
            let out = try await graph.run([lrInput: .float32(lrCrop, shape: [1, 3, lrTile, lrTile])])
            guard let sr = out[srOutput]?.floats() else { throw VisionError.missingOutput(srOutput) }
            blend(sr: sr, into: &acc, weights: &wsum, feather: feather,
                  dx0: x0 * scale, dy0: y0 * scale, dstW: outW, dstH: outH)
        }

        // Normalize the weighted sum and convert [-1,1] -> [0,1] -> RGBA8 image.
        // Stitch: weighted average → raw SR in [-1,1].
        var sr = [Float](repeating: 0, count: outW * outH * 3)
        for p in 0..<(outW * outH) {
            let w = max(wsum[p], 1e-6)
            for c in 0..<3 { sr[p * 3 + c] = acc[p * 3 + c] / w }
        }
        // AdcSR's per-image color-match, applied ONCE on the whole image (not per-tile, which
        // blows up uniform tiles → pure-white squares): match each channel's mean/std to the LR's.
        colorMatch(&sr, count: outW * outH, toLR: lr, lrCount: lrW * lrH)
        // [-1,1] → [0,1] → RGBA8.
        var rgba = [UInt8](repeating: 255, count: outW * outH * 4)
        for p in 0..<(outW * outH) {
            for c in 0..<3 {
                let v = sr[p * 3 + c] * 0.5 + 0.5
                rgba[p * 4 + c] = UInt8(max(0, min(255, (v * 255).rounded())))
            }
        }
        return try makeImage(rgba: rgba, width: outW, height: outH)
    }

    /// Per-channel global color-match: rescales `sr` (interleaved RGB in [-1,1]) so each channel's
    /// mean/std matches the LR's (`lr` is interleaved RGB in [0,1], converted to [-1,1]).
    private func colorMatch(_ sr: inout [Float], count: Int, toLR lr: [Float], lrCount: Int) {
        for c in 0..<3 {
            var sumLr: Float = 0
            for p in 0..<lrCount { sumLr += lr[p * 3 + c] * 2 - 1 }
            let mLr = sumLr / Float(lrCount)
            var varLr: Float = 0
            for p in 0..<lrCount { let x = (lr[p * 3 + c] * 2 - 1) - mLr; varLr += x * x }
            let stdLr = (varLr / Float(lrCount)).squareRoot()

            var sumSr: Float = 0
            for p in 0..<count { sumSr += sr[p * 3 + c] }
            let mSr = sumSr / Float(count)
            var varSr: Float = 0
            for p in 0..<count { let x = sr[p * 3 + c] - mSr; varSr += x * x }
            let stdSr = max((varSr / Float(count)).squareRoot(), 1e-6)

            let gain = stdLr / stdSr
            for p in 0..<count { sr[p * 3 + c] = (sr[p * 3 + c] - mSr) * gain + mLr }
        }
    }

    // MARK: - Tiling (in LR space)

    /// Top-left LR positions of `lrTile`-sized windows covering the LR canvas with `overlap`,
    /// clamped so the last row/column ends exactly at the edge (no out-of-bounds reads).
    private func tilePositions(width: Int, height: Int) -> [(Int, Int)] {
        let step = lrTile - overlap
        func starts(_ extent: Int) -> [Int] {
            if extent <= lrTile { return [0] }
            var s = stride(from: 0, through: extent - lrTile, by: step).map { $0 }
            if s.last != extent - lrTile { s.append(extent - lrTile) }
            return s
        }
        let xs = starts(width), ys = starts(height)
        return ys.flatMap { y in xs.map { x in (x, y) } }
    }

    /// Extracts an `lrTile`×`lrTile` window from the LR RGB buffer, normalized to [-1,1], planar CHW.
    private func cropNormalized(_ src: [Float], srcW: Int, srcH: Int, x0: Int, y0: Int) -> [Float] {
        let n = lrTile
        var chw = [Float](repeating: 0, count: 3 * n * n)
        for ty in 0..<n {
            let sy = min(y0 + ty, srcH - 1)
            for tx in 0..<n {
                let sx = min(x0 + tx, srcW - 1)
                let s = (sy * srcW + sx) * 3
                let d = ty * n + tx
                chw[d] = src[s] * 2 - 1
                chw[n * n + d] = src[s + 1] * 2 - 1
                chw[2 * n * n + d] = src[s + 2] * 2 - 1
            }
        }
        return chw
    }

    /// Adds a tile's SR output (planar CHW, [-1,1]) into the accumulator at output-space (dx0,dy0).
    private func blend(
        sr: [Float], into acc: inout [Float], weights: inout [Float],
        feather: [Float], dx0: Int, dy0: Int, dstW: Int, dstH: Int
    ) {
        let n = srTile
        let plane = n * n
        for ty in 0..<n {
            let dy = dy0 + ty
            if dy >= dstH { break }
            for tx in 0..<n {
                let dx = dx0 + tx
                if dx >= dstW { break }
                let w = feather[ty * n + tx]
                let s = ty * n + tx
                let p = dy * dstW + dx
                acc[p * 3] += sr[s] * w
                acc[p * 3 + 1] += sr[plane + s] * w
                acc[p * 3 + 2] += sr[2 * plane + s] * w
                weights[p] += w
            }
        }
    }

    /// A separable triangular ramp: weight falls off toward tile edges so overlaps cross-fade.
    private func featherWeights(_ side: Int) -> [Float] {
        let edge = overlap * scale
        var ramp = [Float](repeating: 1, count: side)
        for i in 0..<max(1, edge) {
            let w = Float(i + 1) / Float(edge + 1)
            ramp[i] = w
            ramp[side - 1 - i] = w
        }
        var w2 = [Float](repeating: 0, count: side * side)
        for y in 0..<side {
            for x in 0..<side { w2[y * side + x] = ramp[y] * ramp[x] }
        }
        return w2
    }

    // MARK: - CoreGraphics helpers (top-down RGB, consistent orientation)

    /// Draws `image` into a top-down sRGB RGBA8 context and returns interleaved RGB in [0,1].
    private func renderRGB(_ image: CGImage, width: Int, height: Int) throws -> [Float] {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: 0, space: space,  // 0 → CG picks an aligned stride (may exceed width*4)
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw VisionError.imageRenderFailed }
        // No y-flip: a standard top-down CGBitmapContext already maps the image's top to row 0
        // (the extra flip was inverting the result vertically).
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = ctx.data else { throw VisionError.imageRenderFailed }
        // Use the context's ACTUAL row stride — for non-16-aligned widths it is padded past
        // width*4, and assuming width*4 shears the image (the tiling bug on odd-width photos).
        let rowBytes = ctx.bytesPerRow
        let raw = data.bindMemory(to: UInt8.self, capacity: rowBytes * height)
        var rgb = [Float](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            let row = y * rowBytes
            for x in 0..<width {
                let s = row + x * 4
                let d = (y * width + x) * 3
                rgb[d] = Float(raw[s]) / 255
                rgb[d + 1] = Float(raw[s + 1]) / 255
                rgb[d + 2] = Float(raw[s + 2]) / 255
            }
        }
        return rgb
    }

    private func makeImage(rgba: [UInt8], width: Int, height: Int) throws -> CGImage {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let provider = CGDataProvider(data: Data(rgba) as CFData),
            let img = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw VisionError.imageRenderFailed }
        return img
    }
}
