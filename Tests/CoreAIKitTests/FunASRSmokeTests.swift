import Accelerate
import CoreAILanguageModels
import Foundation
import XCTest

@testable import CoreAIKit

/// Fun-ASR-Nano host parity against the Python port (zoo `conversion/funasr_nano`). The arithmetic
/// test always runs; the rest are opt-in:
///
///     KIT_FUNASR_DECODER=<decoder bundle dir> KIT_FUNASR_ENCODER=<encoder .aimodel> \
///     KIT_FUNASR_REF=<swift_ref dir> KIT_FUNASR_FIXTURES=<fixtures dir> \
///     swift test --filter FunASRSmoke
///
/// `swift_ref/` is what the zoo's `dump_features.py` wrote: `feats/<clip>.f32` (the NumPy front end,
/// float32 `[L, 560]`), `manifest.json`, `expected.json` (the fp32 oracle's ids and text, the Python
/// engine run of the same bundles, the oracle's top-2 margin per step) and `prompt_variants.json`
/// (funasr's prompt ids for hotwords / language / itn). Results go to `KIT_FUNASR_LOGS` (default
/// `<swift_ref>/../logs`) as `r3_swift_*.json`.
@available(macOS 27, iOS 27, *)
final class FunASRSmokeTests: XCTestCase {
    /// Oracle top-2 softmax gap below which a mismatch is a knife-edge, not a defect.
    static let marginFloor = 0.1
    /// The default prompt's text ids around the audio (the zoo's logs/r2_prompt_ids.json).
    static let defaultPrefix: [Int32] = [
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 105761,
        46670, 61443, 5122,
    ]
    static let defaultSuffix: [Int32] = [151645, 198, 151644, 77091, 198]

    // MARK: - Reference files

    struct Manifest: Decodable {
        struct Clip: Decodable {
            let name: String
            let wav: String
            let num_samples: Int
            let L: Int
            let N: Int
        }
        let clips: [Clip]
    }

    struct Expected: Decodable {
        struct Clip: Decodable {
            let name: String
            let N: Int
            let L: Int
            let oracle_text: String
            let oracle_gen_ids: [Int]
            let ship_arm: String
            let ship_text: String
            let ship_gen_ids: [Int]
            let margins_pos0: [Double]
            let runner_up_pos0: [Int]
        }
        let clips: [Clip]
    }

    struct PromptVariants: Decodable {
        struct Engine: Decodable {
            let arm: String
            let text: String
            let gen_ids: [Int]
        }
        struct Case: Decodable {
            let `case`: String
            let name: String
            let hotwords: [String]
            let language: String?
            let itn: Bool
            let source_ids: [Int]
            let fbank_beg: Int
            let N: Int
            let text: String
            let gen_ids: [Int]
            let python_engine: Engine?
        }
        let cases: [Case]
    }

    struct Paths {
        let ref: URL
        let fixtures: URL
        let logs: URL
    }

    static let environment = ProcessInfo.processInfo.environment

    private func paths() throws -> Paths {
        guard let ref = Self.environment["KIT_FUNASR_REF"],
            let fixtures = Self.environment["KIT_FUNASR_FIXTURES"]
        else {
            throw XCTSkip("Set KIT_FUNASR_REF (swift_ref/) and KIT_FUNASR_FIXTURES (fixtures/).")
        }
        let refURL = URL(fileURLWithPath: ref)
        let logs = Self.environment["KIT_FUNASR_LOGS"].map { URL(fileURLWithPath: $0) }
            ?? refURL.deletingLastPathComponent().appendingPathComponent("logs")
        return Paths(ref: refURL, fixtures: URL(fileURLWithPath: fixtures), logs: logs)
    }

    private func bundles() throws -> (decoder: URL, encoder: URL) {
        guard let decoder = Self.environment["KIT_FUNASR_DECODER"],
            let encoder = Self.environment["KIT_FUNASR_ENCODER"]
        else {
            throw XCTSkip("Set KIT_FUNASR_DECODER (decoder bundle dir) and KIT_FUNASR_ENCODER (.aimodel).")
        }
        return (URL(fileURLWithPath: decoder), URL(fileURLWithPath: encoder))
    }

    private func load<T: Decodable>(_ type: T.Type, _ url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }

    private func write<T: Encodable>(_ value: T, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url)
        print("PROBE wrote \(url.path)")
    }

    /// One model for the whole class: loading specializes a 1.2 GB pair of graphs.
    private actor ModelCache {
        static let shared = ModelCache()
        private var model: KitFunASRModel?

        func model(decoder: URL, encoder: URL) async throws -> KitFunASRModel {
            if let model { return model }
            let loaded = try await KitFunASRModel(decoderBundleAt: decoder, encoderModelAt: encoder)
            model = loaded
            return loaded
        }
    }

    private func model() async throws -> KitFunASRModel {
        let (decoder, encoder) = try bundles()
        return try await ModelCache.shared.model(decoder: decoder, encoder: encoder)
    }

    static func firstDivergence(_ a: [Int], _ b: [Int]) -> Int? {
        for (i, (x, y)) in zip(a, b).enumerated() where x != y { return i }
        return a.count == b.count ? nil : min(a.count, b.count)
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    // MARK: - 0. Host arithmetic (always runs)

    func test0HostArithmetic() {
        typealias P = FunASRFbankPreprocessor
        // 30 s = 480,000 samples -> 2998 fbank frames -> 500 LFR frames -> 63 audio slots.
        XCTAssertEqual(P.fbankFrames(samples: 480_000), 2998)
        XCTAssertEqual(P.lfrFrames(samples: 480_000), 500)
        XCTAssertEqual(P.audioTokenCount(lfrFrames: 500), 63)
        XCTAssertEqual(P.fbankFrames(samples: 399), 0)
        XCTAssertEqual(P.lfrFrames(samples: 400), 1)
        for l in 0...600 { _ = P.audioTokenCount(lfrFrames: l) }  // asserts ceil(L/8) == 3-stage form
        XCTAssertEqual(FunASRArchitecture.funASRNano2512.windowSamples, 480_000)

        // funasr's get_prompt strings.
        XCTAssertEqual(
            FunASRPromptRenderer.userText(hotwords: [], language: nil, itn: true), "语音转写：")
        XCTAssertEqual(
            FunASRPromptRenderer.userText(hotwords: [], language: "中文", itn: false),
            "语音转写成中文，不进行文本规整：")
        XCTAssertEqual(
            FunASRPromptRenderer.userText(hotwords: [], language: "ja", itn: true), "语音转写成日文：")
        XCTAssertEqual(
            FunASRPromptRenderer.userText(hotwords: ["a", "b"], language: nil, itn: true),
            "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n"
                + "**上下文信息：**\n\n\n热词列表：[a, b]\n语音转写：")

        // funasr's clean-up (Python whitespace set) plus a trim.
        XCTAssertEqual(FunASRRuntime.clean("a/silb"), "a b")
        XCTAssertEqual(FunASRRuntime.clean("  a \n\t b\u{3000}c "), "a b c")
        XCTAssertEqual(FunASRRuntime.clean("x\u{1F}y"), "x y")
        XCTAssertEqual(FunASRRuntime.clean("开饭时间。"), "开饭时间。")
    }

    // MARK: - 1. Front end vs the NumPy reference (and the oracle's own features)

    func test1FrontEndMatchesNumPyReference() throws {
        struct Row: Encodable {
            let name: String
            let L: Int
            let max_abs_delta: Float
            let mean_abs_delta: Float
            /// The same two measures against the fp32 oracle's own features, for Swift and for the
            /// NumPy reference (nil when swift_ref/oracle_feats/ is absent).
            let oracle_max_abs_delta_swift: Float?
            let oracle_max_abs_delta_numpy: Float?
            let samples: Int
        }
        struct Report: Encodable {
            let reference: String
            let threshold: Float
            let clips: Int
            let l_equal: Int
            let worst_clip: String
            let worst_max_abs_delta: Float
            let oracle_worst_clip_swift: String?
            let oracle_worst_max_abs_delta_swift: Float?
            let oracle_worst_max_abs_delta_numpy: Float?
            let rows: [Row]
        }
        func maxAbsDelta(_ a: [Float], _ b: [Float]) -> (max: Float, mean: Float) {
            var diff = [Float](repeating: 0, count: a.count)
            vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
            var maxAbs: Float = 0
            var meanAbs: Float = 0
            vDSP_maxmgv(diff, 1, &maxAbs, vDSP_Length(diff.count))
            vDSP_meamgv(diff, 1, &meanAbs, vDSP_Length(diff.count))
            return (maxAbs, meanAbs)
        }
        func floats(_ url: URL) throws -> [Float] {
            try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let p = try paths()
        let manifest = try load(Manifest.self, p.ref.appendingPathComponent("manifest.json"))
        let pre = FunASRFbankPreprocessor()
        let threshold: Float = 1e-2
        var rows: [Row] = []
        var lEqual = 0
        for clip in manifest.clips {
            let samples = try AudioFile.pcm16kMono(p.fixtures.appendingPathComponent(clip.wav))
            XCTAssertEqual(samples.count, clip.num_samples, clip.name)
            let (feats, l) = pre.features(samples)
            XCTAssertEqual(l, clip.L, clip.name)
            XCTAssertEqual(FunASRFbankPreprocessor.audioTokenCount(lfrFrames: l), clip.N, clip.name)
            let expected = try floats(p.ref.appendingPathComponent("feats/\(clip.name).f32"))
            guard l == clip.L, feats.count == expected.count else {
                XCTFail("\(clip.name): L \(l) vs \(clip.L), \(feats.count) vs \(expected.count) values")
                continue
            }
            lEqual += 1
            let (maxAbs, meanAbs) = maxAbsDelta(feats, expected)
            var vsOracle: (swift: Float, numpy: Float)?
            let oracleURL = p.ref.appendingPathComponent("oracle_feats/\(clip.name).f32")
            if FileManager.default.fileExists(atPath: oracleURL.path) {
                let oracle = try floats(oracleURL)
                XCTAssertEqual(oracle.count, feats.count, clip.name)
                if oracle.count == feats.count {
                    vsOracle = (maxAbsDelta(feats, oracle).max, maxAbsDelta(expected, oracle).max)
                    XCTAssertLessThan(vsOracle!.swift, threshold, "\(clip.name) vs oracle")
                }
            }
            print(
                "PROBE frontend \(clip.name): L=\(l) max|Δ|=\(maxAbs) mean|Δ|=\(meanAbs)"
                    + (vsOracle.map { " vs oracle: swift \($0.swift) numpy \($0.numpy)" } ?? ""))
            XCTAssertLessThan(maxAbs, threshold, clip.name)
            rows.append(
                Row(name: clip.name, L: l, max_abs_delta: maxAbs, mean_abs_delta: meanAbs,
                    oracle_max_abs_delta_swift: vsOracle?.swift,
                    oracle_max_abs_delta_numpy: vsOracle?.numpy, samples: samples.count))
        }
        let worst = rows.max { $0.max_abs_delta < $1.max_abs_delta }
        let oracleWorst = rows.filter { $0.oracle_max_abs_delta_swift != nil }
            .max { $0.oracle_max_abs_delta_swift! < $1.oracle_max_abs_delta_swift! }
        let numpyOracleWorst = rows.compactMap(\.oracle_max_abs_delta_numpy).max()
        print("PROBE frontend worst \(worst?.name ?? "-") max|Δ|=\(worst?.max_abs_delta ?? .nan) "
            + "over \(rows.count) clips, L equal \(lEqual)/\(manifest.clips.count); vs oracle: swift worst "
            + "\(oracleWorst?.name ?? "-") \(oracleWorst?.oracle_max_abs_delta_swift ?? .nan), numpy worst "
            + "\(numpyOracleWorst ?? .nan)")
        try write(
            Report(
                reference: "swift_ref/feats (conversion/funasr_nano/frontend.py, NumPy float64 -> float32); "
                    + "oracle = swift_ref/oracle_feats (oracle/<clip>.npz speech, torchaudio float32)",
                threshold: threshold, clips: manifest.clips.count, l_equal: lEqual,
                worst_clip: worst?.name ?? "", worst_max_abs_delta: worst?.max_abs_delta ?? .nan,
                oracle_worst_clip_swift: oracleWorst?.name,
                oracle_worst_max_abs_delta_swift: oracleWorst?.oracle_max_abs_delta_swift,
                oracle_worst_max_abs_delta_numpy: numpyOracleWorst,
                rows: rows),
            to: p.logs.appendingPathComponent("r3_swift_frontend.json"))
    }

    // MARK: - 2. Prompt ids vs funasr's

    func test2PromptIDsMatchFunASR() async throws {
        let p = try paths()
        let (decoder, _) = try bundles()
        let tokenizer = try await LanguageBundle(at: decoder).loadTokenizer()
        let arch = FunASRArchitecture.funASRNano2512

        // No BOS or other special token is added around either text segment.
        let (prefix, suffix) = FunASRPromptRenderer.segments(tokenizer: tokenizer)
        XCTAssertEqual(prefix.count, 18)
        XCTAssertEqual(suffix.count, 5)
        XCTAssertEqual(prefix, Self.defaultPrefix)
        XCTAssertEqual(suffix, Self.defaultSuffix)

        let variants = try load(PromptVariants.self, p.ref.appendingPathComponent("prompt_variants.json"))
        XCTAssertGreaterThanOrEqual(variants.cases.count, 4)
        for c in variants.cases {
            var expected = c.source_ids.map(Int32.init)
            XCTAssertEqual(Array(expected[c.fbank_beg..<(c.fbank_beg + c.N)]), [Int32](repeating: 0, count: c.N))
            for slot in 0..<c.N { expected[c.fbank_beg + slot] = arch.vocab + Int32(slot) }
            let ids = FunASRPromptRenderer.render(
                tokenizer: tokenizer, arch: arch, audioTokenCount: c.N, hotwords: c.hotwords,
                language: c.language, itn: c.itn)
            let equal = ids == expected
            print("PROBE prompt \(c.case): \(ids.count) ids (prefix \(c.fbank_beg)), equal=\(equal)")
            XCTAssertEqual(ids, expected, c.case)
        }
    }

    // MARK: - 3. End to end vs the oracle and the Python engine

    func test3EndToEndMatchesOracle() async throws {
        struct Row: Encodable {
            let name: String
            let N: Int
            let L: Int
            let windows: Int
            let prompt_length: Int
            let gen_ids: [Int]
            let text: String
            let exact_oracle: Bool
            let text_equal_oracle: Bool
            let exact_python_engine: Bool
            let text_equal_python_engine: Bool
            let first_divergence: Int?
            let oracle_margin_at_div: Double?
            let knife_edge: Bool?
            let ours_at_div: Int?
            let oracle_at_div: Int?
            let oracle_runner_up_at_div: Int?
            let hit_cap: Bool
            let frontend_ms: Double
            let encoder_ms: Double
            let prefill_ms: Double
            let decode_ms: Double
            /// Only where the Swift ids differ from the Python engine's: the same clip decoded from
            /// the NumPy front end's features, to separate the Swift front end from the decoder path.
            let numpy_features: Isolation?
        }
        struct Isolation: Encodable {
            let gen_ids: [Int]
            let equal_swift_features: Bool
            let equal_python_engine: Bool
            let first_divergence_oracle: Int?
        }
        struct Summary: Encodable {
            let clips: Int
            let exact_oracle: Int
            let exact_or_knife_edge: Int
            let knife_edge: Int
            let mismatch_above_floor: Int
            let text_equal_oracle: Int
            let exact_python_engine: Int
            let text_equal_python_engine: Int
            let python_engine_arm: String
            let hit_cap: Int
            let margin_floor: Double
            let public_transcribe_checked: Int
            let note: String
        }
        struct Report: Encodable {
            let summary: Summary
            let clips: [Row]
        }
        let p = try paths()
        let model = try await model()
        let expected = try load(Expected.self, p.ref.appendingPathComponent("expected.json"))
        let manifest = try load(Manifest.self, p.ref.appendingPathComponent("manifest.json"))
        let wavs = Dictionary(uniqueKeysWithValues: manifest.clips.map { ($0.name, $0.wav) })
        let examples: Set<String> = ["zh", "en", "ja", "ko", "yue"]

        var rows: [Row] = []
        var publicChecked = 0
        for clip in expected.clips {
            let samples = try AudioFile.pcm16kMono(p.fixtures.appendingPathComponent(wavs[clip.name]!))
            let windows = try await model.transcribeWindows(samples: samples)
            XCTAssertEqual(windows.count, 1, clip.name)
            guard let w = windows.first else { continue }
            let ids = w.tokenIDs.map(Int.init)
            let golden = clip.oracle_gen_ids
            let div = Self.firstDivergence(ids, golden)
            var margin: Double?
            var runnerUp: Int?
            if let div {
                let k = min(div, golden.count - 1)
                margin = clip.margins_pos0[k]
                runnerUp = clip.runner_up_pos0[k]
            }
            var isolation: Isolation?
            if ids != clip.ship_gen_ids {
                let feats: [Float] = try Data(
                    contentsOf: p.ref.appendingPathComponent("feats/\(clip.name).f32")
                ).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                try await model.runtime.attach(feats: feats, lfrFrames: clip.L)
                let rerun = try await model.runtime.decode(
                    hotwords: [], language: nil, itn: true, maxTokens: 512, onPartial: nil
                ).tokenIDs.map(Int.init)
                isolation = Isolation(
                    gen_ids: rerun, equal_swift_features: rerun == ids,
                    equal_python_engine: rerun == clip.ship_gen_ids,
                    first_divergence_oracle: Self.firstDivergence(rerun, golden))
                print(
                    "PROBE e2e isolate \(clip.name): NumPy features -> ids == Swift-features run "
                        + "\(rerun == ids), == python engine \(rerun == clip.ship_gen_ids)")
            }
            // The public entry point is the same window loop joined; check it on the examples.
            if examples.contains(clip.name) {
                let result = try await model.transcribe(samples: samples)
                XCTAssertEqual(result.text, w.text, clip.name)
                XCTAssertEqual(result.language, "", clip.name)
                publicChecked += 1
            }
            let row = Row(
                name: clip.name, N: clip.N, L: clip.L, windows: windows.count,
                prompt_length: w.promptLength, gen_ids: ids, text: w.text,
                exact_oracle: div == nil, text_equal_oracle: w.text == clip.oracle_text,
                exact_python_engine: ids == clip.ship_gen_ids,
                text_equal_python_engine: w.text == clip.ship_text,
                first_divergence: div, oracle_margin_at_div: margin,
                knife_edge: margin.map { $0 < Self.marginFloor },
                ours_at_div: div.flatMap { $0 < ids.count ? ids[$0] : nil },
                oracle_at_div: div.map { golden[min($0, golden.count - 1)] },
                oracle_runner_up_at_div: runnerUp, hit_cap: w.hitCap,
                frontend_ms: w.frontendMs, encoder_ms: w.encoderMs, prefill_ms: w.prefillMs,
                decode_ms: w.decodeMs, numpy_features: isolation)
            rows.append(row)
            let flag = row.exact_oracle ? "OK   " : (row.knife_edge == true ? "KNIFE" : "DIFF ")
            print(
                "PROBE e2e \(flag) \(clip.name): N=\(clip.N) gen \(ids.count)/\(golden.count) "
                    + "python-engine \(row.exact_python_engine ? "=" : "≠") "
                    + String(format: "enc %.0f ms prefill %.0f ms decode %.0f ms", w.encoderMs, w.prefillMs, w.decodeMs)
                    + (div.map { " div@\($0) margin \(margin ?? .nan)" } ?? ""))
        }

        let knife = rows.filter { !$0.exact_oracle && $0.knife_edge == true }
        let above = rows.filter { !$0.exact_oracle && $0.knife_edge != true }
        let summary = Summary(
            clips: rows.count, exact_oracle: rows.filter(\.exact_oracle).count,
            exact_or_knife_edge: rows.filter(\.exact_oracle).count + knife.count,
            knife_edge: knife.count, mismatch_above_floor: above.count,
            text_equal_oracle: rows.filter(\.text_equal_oracle).count,
            exact_python_engine: rows.filter(\.exact_python_engine).count,
            text_equal_python_engine: rows.filter(\.text_equal_python_engine).count,
            python_engine_arm: Set(expected.clips.map(\.ship_arm)).sorted().joined(separator: ","),
            hit_cap: rows.filter(\.hit_cap).count, margin_floor: Self.marginFloor,
            public_transcribe_checked: publicChecked,
            note: "Swift KitFunASRModel (decoder _s1 bundle via the pipelined engine, encoder GraphModel, "
                + "Mac GPU, contended); python engine = gate_e2e.py on the unified static twin of the same "
                + "quantized decoder. ms are per clip, debug build.")
        print(
            "PROBE e2e summary: exact \(summary.exact_oracle)/\(summary.clips), knife-edge \(summary.knife_edge), "
                + "above floor \(summary.mismatch_above_floor), text==python engine "
                + "\(summary.text_equal_python_engine)/\(summary.clips), ids==python engine "
                + "\(summary.exact_python_engine), cap \(summary.hit_cap)")
        for r in knife + above {
            print(
                "PROBE e2e non-exact \(r.name): step \(r.first_divergence ?? -1) margin "
                    + "\(r.oracle_margin_at_div ?? .nan) ours \(r.ours_at_div ?? -1) oracle "
                    + "\(r.oracle_at_div ?? -1) runner-up \(r.oracle_runner_up_at_div ?? -1)")
        }
        try write(
            Report(summary: summary, clips: rows),
            to: p.logs.appendingPathComponent("r3_swift_e2e.json"))
        XCTAssertEqual(rows.count, expected.clips.count)
        XCTAssertEqual(above.count, 0, "mismatches at an oracle margin >= \(Self.marginFloor)")
        XCTAssertEqual(summary.hit_cap, 0)
    }

    // MARK: - 4. Time (one clip, warm, contended)

    func test4Timing() async throws {
        struct Run: Encodable {
            let frontend_ms: Double
            let encoder_ms: Double
            let prefill_ms: Double
            let decode_ms_per_token: Double
            let tokens: Int
            let wall_ms: Double
            let rtf: Double
        }
        struct Report: Encodable {
            let clip: String
            let audio_s: Double
            let L: Int
            let N: Int
            let prompt_length: Int
            let runs: [Run]
            let median: Run
            let note: String
        }
        let p = try paths()
        let model = try await model()
        let name = "ja_jp_1719"
        let manifest = try load(Manifest.self, p.ref.appendingPathComponent("manifest.json"))
        guard let clip = manifest.clips.first(where: { $0.name == name }) else {
            return XCTFail("\(name) is not in the manifest")
        }
        let samples = try AudioFile.pcm16kMono(p.fixtures.appendingPathComponent(clip.wav))
        let audioSeconds = Double(samples.count) / 16000
        _ = try await model.transcribeWindows(samples: samples)  // warm-up
        var runs: [Run] = []
        var promptLength = 0
        let clock = ContinuousClock()
        for _ in 0..<5 {
            let start = clock.now
            let windows = try await model.transcribeWindows(samples: samples)
            let wall = FunASRRuntime.milliseconds(clock.now - start)
            let w = try XCTUnwrap(windows.first)
            promptLength = w.promptLength
            let steps = max(w.tokenIDs.count - 1, 1)
            runs.append(
                Run(frontend_ms: w.frontendMs, encoder_ms: w.encoderMs, prefill_ms: w.prefillMs,
                    decode_ms_per_token: w.decodeMs / Double(steps), tokens: w.tokenIDs.count,
                    wall_ms: wall, rtf: wall / 1000 / audioSeconds))
        }
        let median = Run(
            frontend_ms: Self.median(runs.map(\.frontend_ms)),
            encoder_ms: Self.median(runs.map(\.encoder_ms)),
            prefill_ms: Self.median(runs.map(\.prefill_ms)),
            decode_ms_per_token: Self.median(runs.map(\.decode_ms_per_token)),
            tokens: runs.first?.tokens ?? 0, wall_ms: Self.median(runs.map(\.wall_ms)),
            rtf: Self.median(runs.map(\.rtf)))
        print(
            "PROBE timing \(name) (L=\(clip.L), N=\(clip.N), Sp=\(promptLength)): "
                + String(
                    format: "%.2f s audio, frontend %.1f ms, encoder %.1f ms, prefill %.1f ms, "
                        + "decode %.2f ms/token (%d tokens), wall %.1f ms, RTF %.4f (median of 5, contended)",
                    audioSeconds, median.frontend_ms, median.encoder_ms, median.prefill_ms,
                    median.decode_ms_per_token, median.tokens, median.wall_ms, median.rtf))
        try write(
            Report(
                clip: name, audio_s: audioSeconds, L: clip.L, N: clip.N, prompt_length: promptLength,
                runs: runs, median: median,
                note: "Mac GPU shared with other sessions (contended); swift test debug build (host code "
                    + "unoptimized); one warm-up run before the 5 measured. prefill = generate call to "
                    + "first token (the prompt runs as S=1 pipelined steps); decode = first to last token."),
            to: p.logs.appendingPathComponent("r3_swift_timing.json"))
    }

    // MARK: - 5. Prompt options end to end (hotwords, language, itn)

    func test5PromptOptions() async throws {
        struct Row: Encodable {
            let `case`: String
            let name: String
            let hotwords: [String]
            let language: String?
            let itn: Bool
            let text: String
            let gen_ids: [Int]
            let oracle_text: String
            let exact_oracle: Bool
            let text_equal_oracle: Bool
            let python_engine_text: String?
            let exact_python_engine: Bool?
            let first_divergence_oracle: Int?
        }
        let p = try paths()
        let model = try await model()
        let variants = try load(PromptVariants.self, p.ref.appendingPathComponent("prompt_variants.json"))
        let manifest = try load(Manifest.self, p.ref.appendingPathComponent("manifest.json"))
        let wavs = Dictionary(uniqueKeysWithValues: manifest.clips.map { ($0.name, $0.wav) })
        var rows: [Row] = []
        for c in variants.cases where c.case != "default" {
            let samples = try AudioFile.pcm16kMono(p.fixtures.appendingPathComponent(wavs[c.name]!))
            let windows = try await model.transcribeWindows(
                samples: samples, hotwords: c.hotwords, language: c.language, itn: c.itn)
            let w = try XCTUnwrap(windows.first)
            let ids = w.tokenIDs.map(Int.init)
            let row = Row(
                case: c.case, name: c.name, hotwords: c.hotwords, language: c.language, itn: c.itn,
                text: w.text, gen_ids: ids, oracle_text: c.text, exact_oracle: ids == c.gen_ids,
                text_equal_oracle: w.text == c.text, python_engine_text: c.python_engine?.text,
                exact_python_engine: c.python_engine.map { ids == $0.gen_ids },
                first_divergence_oracle: Self.firstDivergence(ids, c.gen_ids))
            rows.append(row)
            print(
                "PROBE option \(c.case): swift \(w.text.debugDescription) | oracle \(c.text.debugDescription) "
                    + "(ids equal \(row.exact_oracle)) | python engine "
                    + "\(c.python_engine.map { $0.text.debugDescription } ?? "n/a") "
                    + "(ids equal \(row.exact_python_engine.map(String.init) ?? "n/a"))")
        }
        try write(rows, to: p.logs.appendingPathComponent("r3_swift_prompt_options.json"))
    }
}
