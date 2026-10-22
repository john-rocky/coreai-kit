// TimestralHost.swift — Timestral-209M's host arithmetic: everything outside the graph.
//
// A line-by-line port of the model zoo's conversion/timestral/host_forecast.py, which mirrors
// `TimestralForecaster.forecast()` (github.com/mistralai/timestral @ a534bdb) for one univariate series without
// covariates and a horizon of at most one decode block (1,024 steps). float64 where the released code uses float64
// (the causal Welford statistics, the advancing refine frame), float32 everywhere else, in the released order of
// operations — and in NumPy's own order where NumPy's differs from a plain loop (its pairwise sums, its float32
// cosine and sine) — so the kit's forecast equals the Python host's on the same graph bundle. The port is gated
// window by window against the Python host on the zoo's fixture (max |Δq| / max |q| ≤ 1e-5).
//
// The graph (one bucket of L context patches and h horizon patches, R rows per call):
//   patches [R, L, 64] (values 32 + missing 32), cos [L+h, 64], sin [L+h, 64], attn_mask [1, 1, L+h, L+h]
//   (1 = attend) → quantiles [R, h·32, 21] in the normalized space.

import Foundation

enum TimestralHost {
    static let patch = 32
    /// The head's 21 quantile levels, low to high (config `quantile_knots`).
    static let levels: [Float] = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    static let quantileCount = 21
    static let medianIndex = 10
    /// config `minimum_scale`, a float64 in the released statistics.
    static let minimumScale = 1e-6
    /// `quantile_real_cap_k`: forecasts are clamped to [min − k·s, max + k·s] of the observed context.
    static let capK: Float = 1e4
    /// `max_context`, the trained context length in steps.
    static let maximumContext = 32768
    /// `per_shot_horizon_patches`: one decode block, 32 patches = 1,024 steps.
    static let blockPatches = 32
    static let headDim = 64
    static let ropeDim = 32
    static let ropeTheta: Float = 10000
    static let eps32 = Float.ulpOfOne  // np.finfo(np.float32).eps

    /// Test hook: one host part broken on purpose, for the parity gate's red arm.
    nonisolated(unsafe) static var redArm: RedArm = .off

    enum RedArm: String {
        case off = "none"
        /// The mirror row keeps the series' sign (symmetric averaging then averages a forecast with itself, reflected).
        case mirrorSign = "mirror-sign"
    }

    /// The trigonometry the RoPE tables are built with. `numpy` is NumPy's float32 kernel (what host_forecast.py
    /// gets from np.cos / np.sin); `libm` the system's cosf / sinf; `double` cos / sin in float64, rounded.
    nonisolated(unsafe) static var trig: Trig = .numpy

    enum Trig: String { case numpy, libm, double }

    // MARK: - NumPy semantics

    /// np.maximum: NaN when either is NaN (the first operand when both), else the larger.
    @inline(__always) static func npMax(_ a: Double, _ b: Double) -> Double { (a >= b || a.isNaN) ? a : b }
    @inline(__always) static func npMax(_ a: Float, _ b: Float) -> Float { (a >= b || a.isNaN) ? a : b }
    /// np.minimum: NaN when either is NaN, else the smaller.
    @inline(__always) static func npMin(_ a: Double, _ b: Double) -> Double { (a <= b || a.isNaN) ? a : b }
    @inline(__always) static func npMin(_ a: Float, _ b: Float) -> Float { (a <= b || a.isNaN) ? a : b }
    /// np.clip(x, lo, hi) = minimum(maximum(x, lo), hi).
    @inline(__always) static func npClip(_ x: Double, _ lo: Double, _ hi: Double) -> Double { npMin(npMax(x, lo), hi) }
    @inline(__always) static func npClip(_ x: Float, _ lo: Float, _ hi: Float) -> Float { npMin(npMax(x, lo), hi) }

    /// np.add.reduce over a contiguous float64 run (what `.sum(-1)` does on a row): NumPy's pairwise summation —
    /// fewer than 8 values in order, up to 128 with 8 interleaved accumulators, longer halves recursively.
    static func pairwiseSum(_ a: UnsafeBufferPointer<Double>) -> Double {
        let n = a.count
        if n < 8 {
            var res = 0.0
            for v in a { res += v }
            return res
        }
        if n <= 128 {
            var r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7]
            var i = 8
            while i < n - (n % 8) {
                r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
                r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n {
                res += a[i]
                i += 1
            }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return pairwiseSum(UnsafeBufferPointer(rebasing: a[0..<n2]))
            + pairwiseSum(UnsafeBufferPointer(rebasing: a[n2..<n]))
    }

    /// `.sum()` of a row as NumPy reduces it: the identity (0) plus the pairwise sum.
    static func npSum(_ a: [Double]) -> Double { a.withUnsafeBufferPointer { 0.0 + pairwiseSum($0) } }

    /// NumPy's float32 cosine / sine (numpy/_core/src/umath/loops_trigonometric.dispatch.cpp: Cody-Waite reduction
    /// by π/2 and Myklebust's polynomials with fused multiply-adds for |x| within Cody-Waite's range, libm beyond).
    static func numpyTrig(_ x: Float, cosine: Bool) -> Float {
        if x.isNaN { return .nan }
        if abs(x) > (cosine ? Float(71476.0625) : Float(117435.992)) { return cosine ? cosf(x) : sinf(x) }
        let twoOverPi: Float = 0x1.45f306p-1
        let pio2High: Float = -0x1.921fb0p0
        let pio2Med: Float = -0x1.5110b4p-22
        let pio2Low: Float = -0x1.846988p-48
        let rintMagic: Float = 0x1.800000p23
        var quadrant = x * twoOverPi
        quadrant = quadrant + rintMagic
        quadrant = quadrant - rintMagic
        var r = x.addingProduct(quadrant, pio2High)
        r = r.addingProduct(quadrant, pio2Med)
        r = r.addingProduct(quadrant, pio2Low)
        let r2 = r * r
        var c = Float(-0x1.6c06dcp-10).addingProduct(Float(0x1.98e616p-16), r2)
        c = Float(0x1.55553cp-5).addingProduct(c, r2)
        c = Float(-0x1.000000p-1).addingProduct(c, r2)
        c = Float(0x1.000000p0).addingProduct(c, r2)
        var s = Float(-0x1.a06bbap-13).addingProduct(Float(0x1.7d3bbcp-19), r2)
        s = Float(0x1.11119ap-7).addingProduct(s, r2)
        s = Float(-0x1.555556p-3).addingProduct(s, r2)
        s = Float(0).addingProduct(s, r2)
        s = r.addingProduct(s, r)
        var q = Int32(quadrant)
        if cosine { q &+= 1 }
        var v = (q & 1) == 0 ? s : c
        if (q & 2) == 2 { v = 0 - v }
        return v
    }

    @inline(__always) static func cosine(_ x: Float) -> Float {
        switch trig {
        case .numpy: return numpyTrig(x, cosine: true)
        case .libm: return cosf(x)
        case .double: return Float(cos(Double(x)))
        }
    }

    @inline(__always) static func sine(_ x: Float) -> Float {
        switch trig {
        case .numpy: return numpyTrig(x, cosine: false)
        case .libm: return sinf(x)
        case .double: return Float(sin(Double(x)))
        }
    }

    /// (32,) float32: rotary frequencies for the first ropeDim / 2 pairs, 0 (identity) for the rest.
    static let inverseFrequencies: [Float] = {
        var out = [Float](repeating: 0, count: headDim / 2)
        for i in 0..<(ropeDim / 2) {
            out[i] = Float(1) / powf(ropeTheta, Float(2 * i) / Float(ropeDim))
        }
        return out
    }()

    // MARK: - The released scaler, float64 statistics

    /// compute_causal_statistics: the causal running mean and std (Bessel) in float64, returned as float32.
    static func causalStatistics(_ data: [Float], _ mask: [Bool]) -> (means: [Float], scales: [Float]) {
        let n = data.count
        var means = [Float](repeating: 0, count: n)
        var scales = [Float](repeating: 0, count: n)
        var cumW = 0.0, cumX = 0.0, m2 = 0.0, previous = 0.0
        for t in 0..<n {
            let x = Double(data[t])
            let w: Double = mask[t] ? 1 : 0
            // np.cumsum starts from the first element itself (not 0 + it: the sign of a zero survives)
            cumW = t == 0 ? w : cumW + w
            cumX = t == 0 ? w * x : cumX + w * x
            let den = npMax(cumW, 1.0)
            let mean = cumX / den
            let term = (x - previous) * (x - mean) * w
            m2 = t == 0 ? term : m2 + term
            let variance = m2 / npMax(den - 1.0, 1.0)
            means[t] = Float(mean)
            scales[t] = Float(npMax(variance.squareRoot(), minimumScale))
            previous = mean
        }
        return (means, scales)
    }

    /// causal_patch_loc_scale: the statistics at each patch's last step, one value per patch.
    static func patchLocScale(_ data: [Float], _ mask: [Bool]) -> (loc: [Float], scale: [Float]) {
        let (means, scales) = causalStatistics(data, mask)
        let patches = data.count / patch
        var loc = [Float](repeating: 0, count: patches)
        var scale = [Float](repeating: 0, count: patches)
        for j in 0..<patches {
            loc[j] = means[j * patch + patch - 1]
            scale[j] = scales[j * patch + patch - 1]
        }
        return (loc, scale)
    }

    // MARK: - The graph's inputs

    /// A series as the graph's host reads it: float32 values with the unobserved steps at 0, the observation mask,
    /// left-padded to a multiple of 32 (forecast()'s _as_context + _assemble_rows + pad_to_patch).
    struct Context {
        var x: [Float]
        var mask: [Bool]
        /// Every observed value of the whole series is ≥ 0 (and there is one): the forecast is clipped at 0.
        var nonnegative: Bool
        /// Points of the series the model reads (the most recent ones, at most `capacity`).
        var used: Int
        /// Points left out from the start because the series is longer than the bucket holds.
        var dropped: Int
        /// Observed values among the points read.
        var observed: Int
    }

    /// NaN and ±inf are missing (value 0, mask false). The released code also forward-fills missing steps; every
    /// consumer weights them by the mask, so the fill changes no number and is skipped. A series longer than
    /// `capacity` keeps its most recent points (the released `max_context` truncation at the bucket's size); the
    /// non-negative test looks at the whole series, as the released clip does.
    static func prepareContext(_ series: [Float], capacity: Int) -> Context {
        var anyObserved = false
        var allNonnegative = true
        for v in series where v.isFinite {
            anyObserved = true
            if !(v >= 0) { allNonnegative = false }
        }
        let cap = min(capacity, maximumContext)
        let tail = series.count > cap ? Array(series.suffix(cap)) : series
        let pad = (patch - tail.count % patch) % patch
        var x = [Float](repeating: 0, count: pad + tail.count)
        var mask = [Bool](repeating: false, count: pad + tail.count)
        var observed = 0
        for (i, v) in tail.enumerated() where v.isFinite {
            x[pad + i] = v
            mask[pad + i] = true
            observed += 1
        }
        return Context(x: x, mask: mask, nonnegative: anyObserved && allNonnegative, used: tail.count,
                       dropped: series.count - tail.count, observed: observed)
    }

    struct GraphInputs {
        /// (L, 64): the series row; values 32 + missing 32.
        var patches: [Float]
        /// (L+h, 64)
        var cos: [Float]
        var sin: [Float]
        /// (L+h, L+h), 1 = attend
        var mask: [Float]
    }

    /// bucket_inputs: the graph's inputs for a prepared context left-padded to `nCtx` patches. The statistics of
    /// the padded row equal the unpadded row's at every real step (the padding has zero weight); the padding is
    /// never a valid key, and the positions count from the first observed patch.
    static func graphInputs(_ c: Context, nCtx: Int, nHor: Int) -> GraphInputs {
        let ll = c.x.count / patch
        precondition(ll <= nCtx, "context of \(ll) patches does not fit a \(nCtx)-patch bucket")
        let extra = (nCtx - ll) * patch
        let xb = [Float](repeating: 0, count: extra) + c.x
        let mb = [Bool](repeating: false, count: extra) + c.mask
        let (loc, scale) = patchLocScale(xb, mb)
        var patches = [Float](repeating: 0, count: nCtx * 64)
        var obs = [Bool](repeating: false, count: nCtx)
        for j in 0..<nCtx {
            let denominator = npMax(scale[j], eps32)
            for k in 0..<patch {
                let t = j * patch + k
                if mb[t] {
                    patches[j * 64 + k] = asinhf((xb[t] - loc[j]) / denominator)
                    obs[j] = true
                } else {
                    patches[j * 64 + patch + k] = 1
                }
            }
        }
        var keyValid = [Bool](repeating: false, count: nCtx)
        var seen = false
        for j in 0..<nCtx {
            seen = seen || obs[j]
            keyValid[j] = seen
        }
        let first = obs.firstIndex(of: true) ?? 0
        let s = nCtx + nHor
        let invFreq = inverseFrequencies
        var cos = [Float](repeating: 0, count: s * headDim)
        var sin = [Float](repeating: 0, count: s * headDim)
        for i in 0..<s {
            let position = Float(i - first)
            for k in 0..<(headDim / 2) {
                let angle = position * invFreq[k]
                let cv = cosine(angle), sv = sine(angle)
                cos[i * headDim + k] = cv
                cos[i * headDim + headDim / 2 + k] = cv
                sin[i * headDim + k] = sv
                sin[i * headDim + headDim / 2 + k] = sv
            }
        }
        var mask = [Float](repeating: 0, count: s * s)
        for i in 0..<nCtx {
            for j in 0..<nCtx where (j <= i && keyValid[j]) || i == j { mask[i * s + j] = 1 }
        }
        for i in nCtx..<s {
            for j in 0..<nCtx where keyValid[j] { mask[i * s + j] = 1 }
            for j in nCtx...i { mask[i * s + j] = 1 }
        }
        return GraphInputs(patches: patches, cos: cos, sin: sin, mask: mask)
    }

    /// The mirror row: the statistics of −x are the same scale and the negated location, so the normalized values
    /// negate and the missing channel stays.
    static func mirrored(_ patches: [Float]) -> [Float] {
        var m = patches
        if redArm == .mirrorSign { return m }
        for j in 0..<(patches.count / 64) {
            for k in 0..<patch { m[j * 64 + k] = -m[j * 64 + k] }
        }
        return m
    }

    // MARK: - After the graph

    /// +/-inf → the nearest finite value of the row (the 21 quantiles of one step); a row with no finite value → 0;
    /// NaN kept.
    static func clampNonfinite(_ q: inout [Float], rows: Int) {
        let n = quantileCount
        for t in 0..<rows {
            var hi = -Float.infinity, lo = Float.infinity
            for k in 0..<n {
                let v = q[t * n + k]
                if v.isFinite {
                    hi = npMax(hi, v)
                    lo = npMin(lo, v)
                }
            }
            for k in 0..<n {
                let v = q[t * n + k]
                var c = v == .infinity ? hi : (v == -.infinity ? lo : v)
                if c.isInfinite { c = 0 }
                q[t * n + k] = c
            }
        }
    }

    static func welfordCombine(
        _ wA: Double, _ meanA: Double, _ m2A: Double, _ wB: Double, _ meanB: Double, _ m2B: Double
    ) -> (Double, Double, Double) {
        let w = wA + wB
        let wSafe = npMax(w, Double.leastNormalMagnitude)  // np.finfo(np.float64).tiny
        let delta = meanB - meanA
        let mean = meanA + delta * wB / wSafe
        let m2 = m2A + m2B + delta * delta * wA * wB / wSafe
        return w == 0 ? (w, 0, 0) : (w, mean, m2)
    }

    /// Inclusive prefix scan of (count, mean, M2) along the patches (Hillis-Steele), as released.
    static func cumulativeWelford(_ w: inout [Double], _ mean: inout [Double], _ m2: inout [Double]) {
        let n = w.count
        var d = 1
        while d < n {
            var nw = w, nm = mean, n2 = m2
            for i in 0..<n {
                let (a0, a1, a2) = i >= d ? (w[i - d], mean[i - d], m2[i - d]) : (0.0, 0.0, 0.0)
                (nw[i], nm[i], n2[i]) = welfordCombine(a0, a1, a2, w[i], mean[i], m2[i])
            }
            w = nw
            mean = nm
            m2 = n2
            d *= 2
        }
    }

    /// advancing_block_frames for one row: the float64 location and scale of each horizon patch in the advancing
    /// refine frame. `sinhMedian` is (nPred, 32).
    static func advancingBlockFrames(_ context: [Float], _ mask: [Bool], _ sinhMedian: [Double], nPred: Int)
        -> (loc: [Double], scale: [Double])
    {
        let ps = patch
        let p = context.count / ps
        let total = p + nPred
        var cnt = [Double](repeating: 0, count: total)
        var mean = [Double](repeating: 0, count: total)
        var m2 = [Double](repeating: 0, count: total)
        var values = [Double](repeating: 0, count: ps)
        var weights = [Double](repeating: 0, count: ps)
        var scratch = [Double](repeating: 0, count: ps)
        for j in 0..<total {
            for k in 0..<ps {
                if j < p {
                    values[k] = Double(context[j * ps + k])
                    weights[k] = mask[j * ps + k] ? 1 : 0
                } else {
                    values[k] = 0
                    weights[k] = 0
                }
            }
            let c = npSum(weights)
            for k in 0..<ps { scratch[k] = values[k] * weights[k] }
            let mu = npSum(scratch) / npMax(c, 1.0)
            for k in 0..<ps {
                let dv = values[k] - mu
                scratch[k] = weights[k] * (dv * dv)
            }
            let s2 = npSum(scratch)
            cnt[j] = c
            mean[j] = c == 0 ? 0 : mu
            m2[j] = c == 0 ? 0 : s2
        }
        cumulativeWelford(&cnt, &mean, &m2)
        var count = cnt[total - nPred - 1]
        var mu = mean[total - nPred - 1]
        var sq = m2[total - nPred - 1]
        let offset = mu
        let clip = pow(Double(Float.greatestFiniteMagnitude), 0.5) / 1e3
        var locs = [Double](repeating: 0, count: nPred)
        var scales = [Double](repeating: 0, count: nPred)
        var folded = [Double](repeating: 0, count: ps)
        for h in 0..<nPred {
            let variance = npMax(sq / npMax(count - 1.0, 1.0), 0.0)
            let scale = npMax(variance.squareRoot(), minimumScale)
            locs[h] = mu
            scales[h] = scale
            for k in 0..<ps {
                folded[k] = npClip(sinhMedian[h * ps + k] * scale + mu - offset, -clip, clip) + offset
            }
            let bMean = npSum(folded) / Double(ps)
            for k in 0..<ps {
                let dv = folded[k] - bMean
                scratch[k] = dv * dv
            }
            let bM2 = npSum(scratch)
            (count, mu, sq) = welfordCombine(count, mu, sq, Double(ps), bMean, bM2)
        }
        return (locs, scales)
    }

    /// _refine_block for one row: horizon patch j ≥ 1 re-denormalized in the advancing refine frame. `q` is
    /// (nPred·32, 21) raw-space quantiles.
    static func refineBlock(_ q: [Float], _ x: [Float], _ mask: [Bool]) -> [Float] {
        let n = quantileCount
        let hl = q.count / n
        let nPred = hl / patch
        if nPred <= 1 { return q }
        let (loc, scale) = patchLocScale(x, mask)
        let loc0 = loc[loc.count - 1], scale0 = scale[scale.count - 1]
        var sinhMedian = [Double](repeating: 0, count: nPred * patch)
        for t in 0..<(nPred * patch) {
            sinhMedian[t] = (Double(q[t * n + medianIndex]) - Double(loc0)) / Double(scale0)
        }
        let (locJ, scaleJ) = advancingBlockFrames(x, mask, sinhMedian, nPred: nPred)
        var out = [Float](repeating: 0, count: q.count)
        for t in 0..<hl {
            let j = t / patch
            let lj = j == 0 ? loc0 : Float(locJ[j])
            let sj = j == 0 ? scale0 : Float(scaleJ[j])
            for k in 0..<n {
                let sinhNorm = (q[t * n + k] - loc0) / scale0
                out[t * n + k] = sinhNorm * sj + lj
            }
        }
        clampNonfinite(&out, rows: hl)
        return out
    }

    /// _cap_quantiles with k = 1e4: clamp to [min − k·s, max + k·s] of the observed context, s = the scale at the
    /// context's last step.
    static func capQuantiles(_ q: inout [Float], _ x: [Float], _ mask: [Bool]) {
        var ctxMax: Float = 0, ctxMin: Float = 0
        var any = false
        for (v, m) in zip(x, mask) where m {
            if !any {
                ctxMax = v
                ctxMin = v
                any = true
            } else {
                ctxMax = npMax(ctxMax, v)
                ctxMin = npMin(ctxMin, v)
            }
        }
        let (_, scales) = causalStatistics(x, mask)
        let a = scales[scales.count - 1] * capK
        let lo = ctxMin - a, hi = ctxMax + a
        for i in q.indices { q[i] = npClip(q[i], lo, hi) }
    }

    /// One _forecast_raw pass after the graph: (h·32, 21) normalized → (H, 21) raw, capped. Throws the released
    /// ValueError's case on NaN.
    static func finishPass(_ qNorm: ArraySlice<Float>, _ x: [Float], _ mask: [Bool], horizon: Int) throws -> [Float] {
        let n = quantileCount
        let nPred = (horizon + patch - 1) / patch
        let (loc, scale) = patchLocScale(x, mask)
        let loc0 = loc[loc.count - 1], scale0 = scale[scale.count - 1]
        var raw = [Float](repeating: 0, count: nPred * patch * n)
        var i = qNorm.startIndex
        for t in 0..<raw.count {
            raw[t] = sinhf(qNorm[i]) * scale0 + loc0
            i += 1
        }
        clampNonfinite(&raw, rows: nPred * patch)
        var refined = refineBlock(raw, x, mask)
        refined.removeLast(refined.count - horizon * n)
        capQuantiles(&refined, x, mask)
        let nan = refined.reduce(0) { $0 + ($1.isNaN ? 1 : 0) }
        if nan > 0 { throw KitTimestralForecasterError.degenerateInput(nanCount: nan) }
        return refined
    }

    /// Symmetric averaging and the non-negative clip: 0.5 · (q + flip(−q_mirror)), clamped at 0 when every observed
    /// value is ≥ 0.
    static func combine(_ a: [Float], _ b: [Float], horizon: Int, nonnegative: Bool) -> [Float] {
        let n = quantileCount
        var out = [Float](repeating: 0, count: horizon * n)
        for t in 0..<horizon {
            for k in 0..<n {
                var v = Float(0.5) * (a[t * n + k] + (-b[t * n + (n - 1 - k)]))
                if nonnegative { v = npMax(v, 0) }
                out[t * n + k] = v
            }
        }
        return out
    }
}
