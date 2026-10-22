// TimestralForecasterTests.swift — Timestral's host arithmetic (no bundle needed), and the parity runner that holds
// the Swift host against the Python host's reference forecasts on the zoo's fixture (needs local bundles; skipped
// unless TIMESTRAL_PARITY is set).
//
// Parity (the zoo's conversion/timestral/sample_ref.py writes <ref>; sample_score.py scores what this writes):
//   TIMESTRAL_PARITY=<ref dir> TIMESTRAL_PARITY_SETS='<tag>=<bundle>@<auto|gpu|ane|cpu>;…' \
//     [TIMESTRAL_OUT=swift] [TIMESTRAL_RED=mirror-sign] [TIMESTRAL_TRIG=numpy|libm|double] \
//     swift test --filter TimestralParity
//   → <ref>/<tag>/<out>/{forecast,inputs,raw}.f32.bin + manifest.json, in sample_ref.py's layouts. `auto` loads the
//   bundle (a graph or a folder) with no compute unit, as the catalog does; a load that fails is written to the
//   manifest as `load_error`. `catalog:<id>` in place of a bundle loads the catalog entry (downloading it on first
//   use), as `KitTimestralForecaster(catalog:)`, the CLI's `--model timestral-209m` and the app do.
// Numerics (NumPy's float32 trig / asinh / sinh and its float64 sums against the Swift port):
//   TIMESTRAL_NPCHECK=<ref>/npcheck swift test --filter TimestralNumerics → <npcheck>/swift_numerics.json

import CoreAIKitVision
import Foundation
import Testing

@testable import CoreAIKit

struct TimestralHostTests {
    @Test func causalStatisticsAreTheRunningMeanAndBesselStd() {
        let (m, s) = TimestralHost.causalStatistics([1, 2, 3, 4], [true, true, true, true])
        #expect(m == [1, 1.5, 2, 2.5])
        #expect(s[0] == Float(1e-6))  // one value: the minimum scale
        #expect(abs(s[1] - Float(0.5.squareRoot())) < 1e-7)
        #expect(abs(s[2] - 1) < 1e-7)
        #expect(abs(s[3] - Float((5.0 / 3.0).squareRoot())) < 1e-6)
    }

    @Test func missingStepsWeighNothing() {
        let (m, s) = TimestralHost.causalStatistics([1, 99, 3], [true, false, true])
        #expect(m == [1, 1, 2])
        #expect(abs(s[2] - Float(2.0.squareRoot())) < 1e-6)
    }

    @Test func contextIsMaskedLeftPaddedAndCapped() {
        var series = [Float](repeating: 1, count: 33)
        series[5] = .nan
        let c = TimestralHost.prepareContext(series, capacity: 2048)
        #expect(c.x.count == 64)
        #expect(c.observed == 32)
        #expect(!c.mask[31 + 5] && c.x[31 + 5] == 0)
        #expect(c.mask[31] && !c.mask[30])
        #expect(c.nonnegative)

        let long = TimestralHost.prepareContext([Float](repeating: 2, count: 5000), capacity: 2048)
        #expect(long.used == 2048 && long.dropped == 2952 && long.x.count == 2048)

        let missing = TimestralHost.prepareContext([.nan, .infinity, -.infinity], capacity: 2048)
        #expect(missing.observed == 0 && !missing.nonnegative)

        // the non-negative test reads the whole series, also the part a short bucket drops
        let early = TimestralHost.prepareContext([-1] + [Float](repeating: 3, count: 100), capacity: 64)
        #expect(!early.nonnegative && early.observed == 64)
    }

    @Test func sumsTakeNumpysPairwiseOrder() {
        let v: [Double] = (0..<32).map { (i: Int) -> Double in
            let d = Double(i)
            return d * 1.1 + 1e-9 * (d * d) - 7.3
        }
        var r: [Double] = Array(v[0..<8])
        for i in stride(from: 8, to: 32, by: 8) {
            for k in 0..<8 { r[k] += v[i + k] }
        }
        let low: Double = (r[0] + r[1]) + (r[2] + r[3])
        let high: Double = (r[4] + r[5]) + (r[6] + r[7])
        let expected: Double = low + high
        #expect(TimestralHost.npSum(v) == 0.0 + expected)
        #expect(TimestralHost.npSum([1, 2, 3]) == 6)
    }

    @Test func numpyTrigStaysWithinTwoUlpOfFloat64() {
        let inv = TimestralHost.inverseFrequencies
        var worst: Float = 0
        for p in -255...287 {
            for k in 0..<(TimestralHost.ropeDim / 2) {
                let a = Float(p) * inv[k]
                let c = TimestralHost.numpyTrig(a, cosine: true), s = TimestralHost.numpyTrig(a, cosine: false)
                let ce = Float(cos(Double(a))), se = Float(sin(Double(a)))
                worst = max(worst, abs(c - ce) / max(ce.ulp, Float.leastNormalMagnitude))
                worst = max(worst, abs(s - se) / max(se.ulp, Float.leastNormalMagnitude))
            }
        }
        #expect(worst <= 2)
    }

    @Test func maskReadsValidKeysCausallyAndPositionsStartAtTheFirstObservation() {
        // 3 context patches, the first one unobserved, 2 horizon patches
        let x = [Float](repeating: 0, count: 32) + (0..<64).map { Float($0) }
        let mask = [Bool](repeating: false, count: 32) + [Bool](repeating: true, count: 64)
        let c = TimestralHost.Context(x: x, mask: mask, nonnegative: true, used: 96, dropped: 0, observed: 64)
        let g = TimestralHost.graphInputs(c, nCtx: 3, nHor: 2)
        let s = 5
        let allowed = (0..<s).map { i in (0..<s).map { j in g.mask[i * s + j] == 1 } }
        #expect(allowed[0] == [true, false, false, false, false])  // a padded key: only itself
        #expect(allowed[2] == [false, true, true, false, false])  // causal over the valid keys
        #expect(allowed[3] == [false, true, true, true, false])  // a decoder token: valid context + itself
        #expect(allowed[4] == [false, true, true, true, true])
        #expect(g.patches[0 * 64 + 32] == 1 && g.patches[1 * 64 + 32] == 0)  // the missing channel
        #expect(g.cos[1 * 64] == 1 && g.sin[1 * 64] == 0)  // position 0 = the first observed patch
        let inv0 = TimestralHost.inverseFrequencies[0]
        #expect(g.cos[0] == TimestralHost.cosine(-1 * inv0))  // the padded patch sits at position −1
    }
}

// MARK: - Where a graph runs (no bundle needed)

struct TimestralPlacementTests {
    static let neuralEngineForm = "timestral_209m_L64_h32_r1_fp16p7-qkv4-tanh-conv"

    @available(macOS 27, iOS 27, *)
    @Test func theFormInTheNameGivesTheUnit() {
        func units(_ name: String) -> GraphModel.ComputeUnits {
            KitTimestralForecaster.computeUnits(forGraph: URL(fileURLWithPath: "/b/\(name)"))
        }
        #expect(units("\(Self.neuralEngineForm).aimodel") == .neuralEngine)
        #expect(units("timestral_209m_L64_h4_r2_fp16p7-qkv4-tanh-conv.h19p.aimodelc") == .neuralEngine)
        #expect(units("timestral_209m_L256_h32_r1_fp16p7-qkv4-tanh-conv.aimodelc") == .neuralEngine)
        #expect(units("timestral_209m_L256_h32_r2_fp16-attn32.aimodel") == .gpu)
        #expect(units("timestral_209m_L64_h32_r2_fp32.h16c.aimodelc") == .gpu)
        #expect(units("timestral_209m_L64_h32_r2_fp16w32.aimodel") == .gpu)
        #expect(units("my-conv.aimodel") == .gpu)  // outside the zoo's naming: the GPU, and the output check stands
        #expect(KitTimestralForecaster.form(of: URL(fileURLWithPath: "/b/\(Self.neuralEngineForm).aimodel"))
            == "fp16p7-qkv4-tanh-conv")
    }

    @available(macOS 27, iOS 27, *)
    @Test func aUnitTheNeuralEngineFormCannotRunOnIsRefusedBeforeLoading() async {
        // the path does not exist: the refusal comes before any load
        let graph = URL(fileURLWithPath: "/nonexistent/\(Self.neuralEngineForm).aimodel")
        await #expect(throws: KitTimestralForecasterError.wrongComputeUnits(
            graph: "\(Self.neuralEngineForm).aimodel", requested: "the GPU")) {
            _ = try await KitTimestralForecaster(bundleAt: graph, computeUnits: .gpu)
        }
        #expect(throws: KitTimestralForecasterError.self) { try KitTimestralForecaster.check(.cpuOnly, for: graph) }
        #expect(throws: Never.self) { try KitTimestralForecaster.check(.neuralEngine, for: graph) }
        let gpuForm = URL(fileURLWithPath: "/nonexistent/timestral_209m_L256_h32_r2_fp16-attn32.aimodel")
        #expect(throws: Never.self) { try KitTimestralForecaster.check(.neuralEngine, for: gpuForm) }
        #expect(throws: Never.self) { try KitTimestralForecaster.check(.gpu, for: gpuForm) }
    }

    @available(macOS 27, iOS 27, *)
    @Test func anOutputAtFp16sLimitCounts() {
        let values: [Float] = [0, 25.33, -7.1, 65503.99, 65504, -65504, .infinity, -.infinity, .nan]
        #expect(TimestralGraph.saturatedCount(values) == 4)
        #expect(TimestralGraph.saturatedCount([Float](repeating: 25.3, count: 43008)) == 0)
    }

    @available(macOS 27, iOS 27, *)
    @Test func formsOfOneGraphAcrossSubfoldersAreOneGraph() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(path: "timestral-forms-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let arch = GraphBundle.deviceArchitecture
        for path in ["ios/x.aimodel", "ios-\(arch)/x.\(arch).aimodelc", "ios/y.aimodel", "ios-other/y.h1z.aimodelc"] {
            try fm.createDirectory(at: root.appending(path: path), withIntermediateDirectories: true)
        }
        let urls = try KitTimestralForecaster.graphURLs(in: root).map { $0.pathComponents.suffix(2).joined(separator: "/") }
        #expect(urls.sorted() == ["ios-\(arch)/x.\(arch).aimodelc", "ios/y.aimodel"])
    }
}

// MARK: - Parity with the Python host (local bundles)

struct TimestralParity {
    struct Window: Decodable {
        let id: String
        let group: String
        let window: Int
        let horizon: Int
        let context_length: Int
        let context_offset: Int
    }

    struct Windows: Decodable {
        let windows: [Window]
        let contexts_file: String
    }

    struct SetEntry: Decodable {
        let id: String
        let horizon: Int
    }

    struct SetManifest: Decodable {
        let windows: [SetEntry]
    }

    static var enabled: Bool { ProcessInfo.processInfo.environment["TIMESTRAL_PARITY"] != nil }

    static func append(_ values: [Float], to handle: FileHandle) {
        values.withUnsafeBufferPointer { handle.write(Data(buffer: $0)) }
    }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: TimestralParity.enabled)) func everyWindowOfEverySet() async throws {
        let env = ProcessInfo.processInfo.environment
        let ref = URL(fileURLWithPath: env["TIMESTRAL_PARITY"]!)
        let outName = env["TIMESTRAL_OUT"] ?? "swift"
        if let red = env["TIMESTRAL_RED"] { TimestralHost.redArm = TimestralHost.RedArm(rawValue: red) ?? .off }
        if let trig = env["TIMESTRAL_TRIG"] { TimestralHost.trig = TimestralHost.Trig(rawValue: trig) ?? .numpy }
        defer {
            TimestralHost.redArm = .off
            TimestralHost.trig = .numpy
        }
        let windows = try JSONDecoder().decode(Windows.self, from: Data(contentsOf: ref.appending(path: "windows.json")))
        let byID = Dictionary(uniqueKeysWithValues: windows.windows.map { ($0.id, $0) })
        let contexts = try readFloats(ref.appending(path: windows.contexts_file))
        var loadedBundles: [String: (KitTimestralForecaster, Double)] = [:]  // one load per bundle@compute
        for spec in (env["TIMESTRAL_PARITY_SETS"] ?? "").split(separator: ";") {
            let parts = spec.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2, let at = parts[1].lastIndex(of: "@") else { continue }
            let tag = parts[0]
            let source = String(parts[1][..<at])
            let catalogID = source.hasPrefix("catalog:") ? String(source.dropFirst("catalog:".count)) : nil
            let bundle = URL(fileURLWithPath: source)
            let origin = catalogID == nil ? bundle.path : source  // what the manifest names
            let compute = String(parts[1][parts[1].index(after: at)...])
            let units: GraphModel.ComputeUnits?
            switch compute {
            case "auto": units = nil
            case "ane": units = .neuralEngine
            case "cpu": units = .cpuOnly
            default: units = .gpu
            }
            let set = try JSONDecoder().decode(
                SetManifest.self, from: Data(contentsOf: ref.appending(path: "\(tag)/manifest.json")))
            let out = ref.appending(path: "\(tag)/\(outName)")
            try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
            let forecaster: KitTimestralForecaster
            let loadS: Double
            if let hit = loadedBundles[parts[1]] {
                forecaster = hit.0
                loadS = hit.1
            } else {
                let t0 = Date()
                do {
                    if let catalogID {
                        forecaster = try await KitTimestralForecaster(catalog: catalogID, computeUnits: units)
                    } else if units == nil {
                        forecaster = try await KitTimestralForecaster(bundleAt: bundle)  // no unit, as the catalog's
                    } else {
                        forecaster = try await KitTimestralForecaster(bundleAt: bundle, computeUnits: units)
                    }
                } catch {
                    let manifest: [String: Any] = [
                        "bundle": origin, "compute": compute, "load_error": error.localizedDescription,
                        "load_error_case": "\(error)", "windows": [Any](),
                    ]
                    try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                        .write(to: out.appending(path: "manifest.json"))
                    print("PARITY_SET \(tag) \(outName): load error: \(error.localizedDescription)")
                    continue
                }
                loadS = Date().timeIntervalSince(t0)
                loadedBundles[parts[1]] = (forecaster, loadS)
            }
            for name in ["forecast", "inputs", "raw"] {
                FileManager.default.createFile(atPath: out.appending(path: "\(name).f32.bin").path, contents: nil)
            }
            let fc = try FileHandle(forWritingTo: out.appending(path: "forecast.f32.bin"))
            let inp = try FileHandle(forWritingTo: out.appending(path: "inputs.f32.bin"))
            let raw = try FileHandle(forWritingTo: out.appending(path: "raw.f32.bin"))
            var entries: [[String: Any]] = []
            var oFc = 0, oIn = 0, oRaw = 0
            for e in set.windows {
                guard let w = byID[e.id] else { continue }
                let series = Array(contexts[w.context_offset..<(w.context_offset + w.context_length)])
                do {
                    let t = try await forecaster.forecastTrace(series, horizon: w.horizon)
                    let flat = t.forecast.quantiles.flatMap { $0 }
                    Self.append(flat, to: fc)
                    Self.append(t.inputs, to: inp)
                    Self.append(t.raw, to: raw)
                    let b = t.forecast.details.bucket
                    entries.append([
                        "id": e.id, "horizon": w.horizon, "forecast_offset": oFc, "forecast_floats": flat.count,
                        "inputs_offset": oIn, "inputs_floats": t.inputs.count, "raw_offset": oRaw,
                        "raw_floats": t.raw.count, "ms": t.forecast.details.totalMilliseconds,
                        "graph_ms": t.forecast.details.graphMilliseconds, "bucket": [b.contextPatches, b.horizonPatches],
                        "rows": b.rows, "compute": "\(t.forecast.details.computeUnits)",
                    ])
                    oFc += flat.count
                    oIn += t.inputs.count
                    oRaw += t.raw.count
                } catch {
                    entries.append(["id": e.id, "error": "\(error)", "message": error.localizedDescription])
                }
            }
            try fc.close()
            try inp.close()
            try raw.close()
            let b = forecaster.buckets
            let manifest: [String: Any] = [
                "bundle": origin, "compute": compute, "load_s": loadS,
                "bucket": [b[0].contextPatches, b[0].horizonPatches],
                "buckets": b.map { [$0.contextPatches, $0.horizonPatches] },
                "rows": b.map(\.rows), "input_type": b.map(\.inputType), "red_arm": TimestralHost.redArm.rawValue,
                "trig": TimestralHost.trig.rawValue, "windows": entries,
            ]
            try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
                .write(to: out.appending(path: "manifest.json"))
            let errors = entries.filter { $0["error"] != nil }.count
            print("PARITY_SET \(tag) \(outName): \(entries.count - errors) windows, \(errors) errors, load \(loadS) s, buckets \(b)")
        }
    }
}

// MARK: - NumPy's numerics against the Swift port

struct TimestralNumerics {
    static var enabled: Bool { ProcessInfo.processInfo.environment["TIMESTRAL_NPCHECK"] != nil }

    @Test(.enabled(if: TimestralNumerics.enabled)) func trigAsinhSinhAgainstNumpy() throws {
        let dir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["TIMESTRAL_NPCHECK"]!)
        let trig = try readFloats(dir.appending(path: "trig.f32.bin"))
        let n = trig.count / 3
        let angles = trig[0..<n], npCos = trig[n..<(2 * n)], npSin = trig[(2 * n)..<(3 * n)]
        var result: [String: Any] = ["angles": n]
        for variant in [TimestralHost.Trig.numpy, .libm, .double] {
            TimestralHost.trig = variant
            var cosDiff = 0, sinDiff = 0
            var worst: Float = 0
            for i in 0..<n {
                let a = angles[angles.startIndex + i]
                let c = TimestralHost.cosine(a), s = TimestralHost.sine(a)
                let nc = npCos[npCos.startIndex + i], ns = npSin[npSin.startIndex + i]
                if c.bitPattern != nc.bitPattern { cosDiff += 1; worst = max(worst, abs(c - nc)) }
                if s.bitPattern != ns.bitPattern { sinDiff += 1; worst = max(worst, abs(s - ns)) }
            }
            result["\(variant.rawValue)_cos_differ"] = cosDiff
            result["\(variant.rawValue)_sin_differ"] = sinDiff
            result["\(variant.rawValue)_max_abs"] = Double(worst)
        }
        TimestralHost.trig = .numpy
        let ash = try readFloats(dir.appending(path: "asinh.f32.bin"))
        let m = ash.count / 3
        var asinhDiff = 0, sinhDiff = 0
        for i in 0..<m {
            let x = ash[i]
            if asinhf(x).bitPattern != ash[m + i].bitPattern { asinhDiff += 1 }
            if sinhf(x).bitPattern != ash[2 * m + i].bitPattern && !(sinhf(x).isNaN && ash[2 * m + i].isNaN) {
                sinhDiff += 1
            }
        }
        result["asinh_values"] = m
        result["asinhf_differ"] = asinhDiff
        result["sinhf_differ"] = sinhDiff
        result["inv_freq_bits"] = TimestralHost.inverseFrequencies.map { Int($0.bitPattern) }
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: dir.appending(path: "swift_numerics.json"))
        print("NUMERICS \(result.filter { $0.key != "inv_freq_bits" })")
    }
}

/// A little-endian float32 file.
private func readFloats(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
