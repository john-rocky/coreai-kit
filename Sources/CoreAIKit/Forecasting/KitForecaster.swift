// KitForecaster.swift — TimesFM 2.5 200M time-series forecasting.
//
// One stateless Core AI transformer graph + a host DSP wrapper. Feed any univariate series;
// get a 128-step point + 10-quantile forecast. The graph is the transformer over patch tokens;
// all normalization / flip-invariance / quantile-head logic is deterministic host arithmetic,
// ported verbatim from the validated Python reference (conversion/timesfm/host_forecast.py) — so
// Mac and device produce identical numbers.
//
// Graph I/O (fixed context 2048 → N=64 patches of 32):
//   tok_in[1,64,64], cos[1,64,80], sin[1,64,80], attn_bias[1,1,64,64]
//     → proj_point[1,64,1280]  (128 horizon × 10 quantiles, per patch)
//       proj_q[1,64,10240]     (1024 × 10, per patch)
// We forecast from the LAST patch. Shorter series are front-padded + masked host-side.

import Foundation
import CoreAIKitVision

public struct Forecast: Sendable {
    /// 128-step point forecast (the model's mean / decode channel), already denormalized.
    public let mean: [Float]
    /// 128 × 10 quantile forecast. Channel 0 = mean; channels 1…9 = deciles `quantileLevels`.
    public let quantiles: [[Float]]
    /// The quantile levels for channels 1…9.
    public let quantileLevels: [Float]
}

public enum KitForecasterError: Error, Sendable {
    case bundleNotFound(URL)
    case outputMissing(String)
    case emptySeries
}

public final class KitForecaster {
    // Model constants (TimesFM 2.5 200M).
    static let patch = 32
    static let ctx = 2048
    static let nPatches = 64          // ctx / patch
    static let horizon = 128
    static let quantileCount = 10     // mean + 9 deciles
    static let outputQuantileLen = 1024
    static let headDim = 80
    static let decodeIndex = 5        // median channel
    static let ropeTheta: Float = 10000
    static let tolerance: Float = 1e-6
    static let quantileLevels: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    private let graph: GraphModel

    // MARK: - Loading

    /// Loads TimesFM by its catalog id: `KitForecaster(catalog: "timesfm-2.5-200m")`.
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        guard id == "timesfm-2.5-200m" else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        let model = ModelID(
            "mlboydaisuke/TimesFM-2.5-200M-CoreAI",
            path: "timesfm_2p5_200m_ctx2048_fp16.aimodel")
        let root = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: root, computeUnits: computeUnits)
    }

    /// Loads a local bundle (the `.aimodel` directory, or a folder containing it).
    public init(bundleAt root: URL, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        let bundle = try Self.resolve(in: root)
        self.graph = try await GraphModel(contentsOf: bundle, computeUnits: computeUnits)
    }

    private static func resolve(in root: URL) throws -> URL {
        if root.pathExtension == "aimodel" { return root }
        let fm = FileManager.default
        if let items = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
            if let b = items.first(where: { $0.pathExtension == "aimodel" }) { return b }
        }
        throw KitForecasterError.bundleNotFound(root)
    }

    // MARK: - Forecast

    /// Forecast 128 steps ahead from `series` (any length ≥ 1; ≤ 2048 used, shorter is front-padded).
    public func forecast(_ series: [Float]) async throws -> Forecast {
        guard !series.isEmpty else { throw KitForecasterError.emptySeries }

        // Use the last `ctx` points; front-pad (mask=1) if shorter.
        let tail = Array(series.suffix(Self.ctx))
        let inputMin = tail.min() ?? 0
        var input = [Float](repeating: 0, count: Self.ctx)
        var padMask = [Bool](repeating: false, count: Self.ctx)   // true = padded/invalid
        let pad = Self.ctx - tail.count
        for i in 0..<pad { padMask[i] = true }
        for (i, v) in tail.enumerated() { input[pad + i] = v }

        // Global RevIN over the (padded) context. std is unbiased (ddof=1), matching the reference.
        let muG = mean(input)
        let sigmaG = std(input, mean: muG)
        let safeSigmaG = sigmaG < Self.tolerance ? 1 : sigmaG
        let normalized = input.map { ($0 - muG) / safeSigmaG }
        let negNormalized = normalized.map { -$0 }

        // Flip-invariance: run the graph on +normalized and −normalized, combine.
        let pfPos = try await runGraph(normalized, padMask)     // point [H][Q]
        let pfNeg = try await runGraph(negNormalized, padMask)

        var pf = pfPos.point
        var qs = pfPos.quant
        flipCombine(&pf, pfNeg.point)
        flipCombine(&qs, pfNeg.quant)

        // Continuous-quantile head: rebase each non-median quantile onto the point median.
        var full = pf                                            // [H][Q]
        let mqh = min(Self.horizon, qs.count)
        for h in 0..<mqh {
            for idx in 1..<Self.quantileCount where idx != Self.decodeIndex {
                full[h][idx] = qs[h][idx] - qs[h][Self.decodeIndex] + full[h][Self.decodeIndex]
            }
        }

        // Global denorm + positivity clamp.
        let clamp = inputMin >= 0
        var meanOut = [Float](repeating: 0, count: Self.horizon)
        var quantOut = [[Float]](repeating: [Float](repeating: 0, count: Self.quantileCount),
                                 count: Self.horizon)
        for h in 0..<Self.horizon {
            for q in 0..<Self.quantileCount {
                var v = full[h][q] * sigmaG + muG
                if clamp { v = Swift.max(v, 0) }
                quantOut[h][q] = v
            }
            meanOut[h] = quantOut[h][Self.decodeIndex]
        }
        return Forecast(mean: meanOut, quantiles: quantOut, quantileLevels: Self.quantileLevels)
    }

    // MARK: - Graph call (patching + per-patch RevIN + mask + RoPE, then the transformer)

    private struct Projections { let point: [[Float]]; let quant: [[Float]] }

    private func runGraph(_ normalized: [Float], _ padMask: [Bool]) async throws -> Projections {
        let P = Self.patch, N = Self.nPatches, HD = Self.headDim, Q = Self.quantileCount

        // Per-patch causal Welford stats over valid (non-masked) points.
        var count: Float = 0, mean: Float = 0, std: Float = 0
        var ctxMu = [Float](repeating: 0, count: N)
        var ctxSigma = [Float](repeating: 0, count: N)
        for n in 0..<N {
            var incCount: Float = 0, sum: Float = 0
            for k in 0..<P where !padMask[n * P + k] { incCount += 1; sum += normalized[n * P + k] }
            let incMean = incCount == 0 ? 0 : sum / incCount
            var incVarSum: Float = 0
            for k in 0..<P where !padMask[n * P + k] {
                let d = normalized[n * P + k] - incMean; incVarSum += d * d
            }
            let incVar = incCount == 0 ? 0 : incVarSum / incCount
            let incStd = (incVar > 0 ? incVar : 0).squareRoot()
            let newCount = count + incCount
            let newCountSafe = newCount == 0 ? 1 : newCount
            let newMean = newCount == 0 ? 0 : (count * mean + incMean * incCount) / newCountSafe
            let t1 = count * std * std, t2 = incCount * incStd * incStd
            let t3 = count * (mean - newMean) * (mean - newMean)
            let t4 = incCount * (incMean - newMean) * (incMean - newMean)
            let newVar = newCount == 0 ? 0 : (t1 + t2 + t3 + t4) / newCountSafe
            count = newCount; mean = newMean; std = (newVar > 0 ? newVar : 0).squareRoot()
            ctxMu[n] = mean; ctxSigma[n] = std
        }

        // tok_in[n] = concat(normed_patch(32), mask_float(32)); patch_padding = mask of last element.
        var tokIn = [Float](repeating: 0, count: N * 2 * P)
        var patchPadding = [Bool](repeating: false, count: N)
        for n in 0..<N {
            let mu = ctxMu[n]
            let safe = ctxSigma[n] < Self.tolerance ? 1 : ctxSigma[n]
            for k in 0..<P {
                let masked = padMask[n * P + k]
                let normed = masked ? 0 : (normalized[n * P + k] - mu) / safe
                tokIn[n * 2 * P + k] = normed
                tokIn[n * 2 * P + P + k] = masked ? 1 : 0
            }
            patchPadding[n] = padMask[n * P + (P - 1)]
        }

        // Positions = arange(N) − numMaskedPatches; RoPE cos/sin over head_dim.
        var numMasked = 0
        for n in 0..<N where patchPadding[n] { numMasked += 1 }
        var cosArr = [Float](repeating: 0, count: N * HD)
        var sinArr = [Float](repeating: 0, count: N * HD)
        for n in 0..<N {
            let pos = Float(n - numMasked)
            for d in 0..<(HD / 2) {
                let inv = powf(Self.ropeTheta, -(Float(2 * d) / Float(HD)))
                let ang = pos * inv
                let c = cosf(ang), s = sinf(ang)
                cosArr[n * HD + d] = c; cosArr[n * HD + d + HD / 2] = c
                sinArr[n * HD + d] = s; sinArr[n * HD + d + HD / 2] = s
            }
        }

        // Single additive mask (fp16-safe): allowed = causal AND key-not-padded, else −1e4.
        let NEG: Float = -1e4
        var attnBias = [Float](repeating: 0, count: N * N)
        for i in 0..<N {
            for j in 0..<N {
                let allowed = (j <= i) && !patchPadding[j]
                attnBias[i * N + j] = allowed ? 0 : NEG
            }
        }

        let out = try await graph.run([
            "tok_in": .float32(tokIn, shape: [1, N, 2 * P]),
            "cos": .float32(cosArr, shape: [1, N, HD]),
            "sin": .float32(sinArr, shape: [1, N, HD]),
            "attn_bias": .float32(attnBias, shape: [1, 1, N, N]),
        ])
        let projPoint = try output(out, "proj_point")   // [N*1280]
        let projQ = try output(out, "proj_q")            // [N*10240]

        // Last patch only; un-RevIN with that patch's stats, reshape to [·][Q].
        let last = N - 1
        let sigmaLast = ctxSigma[last], muLast = ctxMu[last]
        var point = [[Float]](repeating: [Float](repeating: 0, count: Q), count: Self.horizon)
        let ppBase = last * Self.horizon * Q
        for h in 0..<Self.horizon {
            for q in 0..<Q {
                point[h][q] = projPoint[ppBase + h * Q + q] * sigmaLast + muLast
            }
        }
        var quant = [[Float]](repeating: [Float](repeating: 0, count: Q), count: Self.outputQuantileLen)
        let pqBase = last * Self.outputQuantileLen * Q
        for l in 0..<Self.outputQuantileLen {
            for q in 0..<Q {
                quant[l][q] = projQ[pqBase + l * Q + q] * sigmaLast + muLast
            }
        }
        return Projections(point: point, quant: quant)
    }

    /// pf = (pf − flipq(neg)) / 2, where flipq keeps channel 0 and reverses channels 1…9.
    private func flipCombine(_ pf: inout [[Float]], _ neg: [[Float]]) {
        let Q = Self.quantileCount
        for r in 0..<pf.count {
            for q in 0..<Q {
                let f = (q == 0) ? neg[r][0] : neg[r][Q - q]   // reverse of 1…9
                pf[r][q] = (pf[r][q] - f) / 2
            }
        }
    }

    private func output(_ d: [String: TensorValue], _ name: String) throws -> [Float] {
        guard let v = d[name] else { throw KitForecasterError.outputMissing(name) }
        return v.floats()
    }

    private func mean(_ a: [Float]) -> Float { a.reduce(0, +) / Float(a.count) }
    private func std(_ a: [Float], mean m: Float) -> Float {
        guard a.count > 1 else { return 0 }
        var s: Float = 0
        for v in a { let d = v - m; s += d * d }
        return (s / Float(a.count - 1)).squareRoot()   // unbiased (ddof=1)
    }
}
