import Foundation
import XCTest

@testable import CoreAIKit

/// Audio8-TTS host parity against the Python port (zoo `conversion/audio8_tts`). The sampler test always runs;
/// the rest are opt-in:
///
///     KIT_AUDIO8_BUNDLES=<exports dir with the three .aimodel> KIT_AUDIO8_REF=<swift_ref dir> \
///     swift test --filter Audio8Smoke
///
/// `swift_ref/` is what the zoo's `dump_swift_ref.py` wrote: `manifest.json` (fixtures, packed prompts, the
/// Python engine run's codes), the oracle's recorded uniform draws per fixture, and `voices/` for the clone
/// fixtures. Test 2 asserts the prompt ids; test 3 replays the oracle's draws through this host and counts the
/// frames identical to the Python engine run on the same bundles (a divergence is a fp16 knife-edge flip, so the
/// report is the prefix length and the count, and the run must still end with eos); test 4 times a plain run.
/// Results go to `KIT_AUDIO8_LOGS` (default `<swift_ref>/../logs`) as `swift_smoke.json`.
@available(macOS 27, iOS 27, *)
final class Audio8SmokeTests: XCTestCase {
    struct Manifest: Decodable {
        struct Generation: Decodable { let max_new_tokens: Int; let temperature: Float; let top_p: Float; let top_k: Int }
        struct Fixture: Decodable {
            let name: String
            let lang: String
            let text: String
            let seed: Int
            let voice: String?
            let prompt_length: Int
            let prompt_rows: [Int32]
            let oracle_frames: Int
            let python_frames: Int
            let python_codes: String
            let noise_slow: String
            let noise_fast: String
            let noise_steps: Int
        }
        let tag: String
        let generation: Generation
        let fixtures: [Fixture]
    }

    /// The oracle's recorded draws, then (past the recorded steps) a seeded stream.
    struct RecordedNoise: Audio8NoiseSource {
        let slowDraws: [Float]      // [T, 2, 4097]
        let fastDraws: [Float]      // [T, 9, 4096]
        let steps: Int
        var t = 0
        var k = 0
        var fallback: Audio8SeededNoise

        mutating func slow() -> ([Float], [Float]) {
            defer { t += 1; k = 0 }
            guard t < steps else { return fallback.slow() }
            let a = Audio8Sampling.allowedCount
            let base = t * 2 * a
            return (Array(slowDraws[base ..< base + a]), Array(slowDraws[base + a ..< base + 2 * a]))
        }

        mutating func fast() -> [Float] {
            let tt = t - 1
            defer { k += 1 }
            guard tt < steps, k < 9 else { return fallback.fast() }
            let c = Audio8Sampling.codebookSize
            let base = (tt * 9 + k) * c
            return Array(fastDraws[base ..< base + c])
        }
    }

    static func env(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }

    static func f32(_ url: URL) throws -> [Float] {
        let d = try Data(contentsOf: url)
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    // MARK: - 1. host-side pieces (always runs)

    func testSamplerBasics() {
        // the seeded noise source: uniforms in (0, 1], deterministic per seed, the right widths
        var a = Audio8SeededNoise(seed: 7), b = Audio8SeededNoise(seed: 7)
        let (n1, h1) = a.slow(), (n2, _) = b.slow()
        XCTAssertEqual(n1.count, 4097); XCTAssertEqual(h1.count, 4097); XCTAssertEqual(n1, n2)
        XCTAssertTrue(n1.allSatisfy { $0 > 0 && $0 <= 1 })
        XCTAssertEqual(a.fast().count, 4096)
        // RAS window: nil before the first push, zeros after it, rolling afterwards
        var w = Audio8Sampling.RASWindow()
        XCTAssertNil(w.values)
        w.push(151700)
        XCTAssertEqual(w.values, [Int32](repeating: 0, count: 10))
        w.push(151701)
        XCTAssertEqual(w.values?.last, 151701)
        XCTAssertEqual(w.values?.count, 10)
    }

    // MARK: - helpers

    func loadAll() async throws -> (Audio8TTS, Manifest, URL)? {
        guard let bundles = Self.env("KIT_AUDIO8_BUNDLES"), let refDir = Self.env("KIT_AUDIO8_REF") else {
            print("Audio8Smoke: set KIT_AUDIO8_BUNDLES and KIT_AUDIO8_REF to run the parity tests")
            return nil
        }
        let ref = URL(fileURLWithPath: refDir)
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: ref.appendingPathComponent("manifest.json")))
        let root = URL(fileURLWithPath: bundles)
        let dualarName = Self.env("KIT_AUDIO8_DUALAR") ?? Audio8Paths.dualarName
        let paths = Audio8Paths.standard(root: root, dualar: dualarName)
        let tts = try await Audio8TTS(paths: paths)
        tts.maxNewTokens = manifest.generation.max_new_tokens
        print("Audio8Smoke: loaded in \(String(format: "%.2f", tts.loadSeconds)) s (\(dualarName))")
        return (tts, manifest, ref)
    }

    func voice(_ f: Manifest.Fixture, ref: URL) throws -> Audio8Voice? {
        guard let v = f.voice else { return nil }
        return try Audio8Voice(contentsOf: ref.appendingPathComponent(v))
    }

    // MARK: - 2. prompt ids == the publisher's processor (via the zoo's dump)

    func testPromptIDs() async throws {
        guard let (tts, manifest, ref) = try await loadAll() else { return }
        var ok = 0
        for f in manifest.fixtures {
            let (rows, P) = try tts.promptRows(text: f.text, voice: try voice(f, ref: ref))
            if P == f.prompt_length && rows == f.prompt_rows { ok += 1 } else {
                print("Audio8Smoke prompt MISMATCH \(f.name): P \(P) vs \(f.prompt_length); "
                      + "first diff at \(zip(rows, f.prompt_rows).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1)")
            }
        }
        print("Audio8Smoke prompts: \(ok)/\(manifest.fixtures.count) identical")
        XCTAssertEqual(ok, manifest.fixtures.count)
    }

    // MARK: - 3. replay the oracle's draws: codes vs the Python engine run

    func testReplayAgainstPythonEngine() async throws {
        guard let (tts, manifest, ref) = try await loadAll() else { return }
        let only = Self.env("KIT_AUDIO8_ONLY")?.split(separator: ",").map(String.init)
        var report: [[String: Any]] = []
        var identicalRuns = 0
        var eosRuns = 0
        var framesTotal = 0
        var prefixTotal = 0
        var wallTotal = 0.0
        var audioTotal = 0.0
        for f in manifest.fixtures where only == nil || only!.contains(f.name) {
            let py = try JSONDecoder().decode([[Int32]].self, from: Data(contentsOf: ref.appendingPathComponent(f.python_codes)))
            var noise = RecordedNoise(slowDraws: try Self.f32(ref.appendingPathComponent(f.noise_slow)),
                                      fastDraws: try Self.f32(ref.appendingPathComponent(f.noise_fast)),
                                      steps: f.noise_steps, fallback: Audio8SeededNoise(seed: UInt64(f.seed)))
            var audio: [Float] = []
            let stats = try await tts.generate(f.text, voice: try voice(f, ref: ref), noise: &noise, maxFrames: nil) {
                audio.append(contentsOf: $0)
            }
            let pyFrames = py.first?.count ?? 0
            let n = min(stats.frames, pyFrames)
            var prefix = n
            for t in 0..<n where (0..<10).contains(where: { stats.codes[$0][t] != py[$0][t] }) { prefix = t; break }
            let identical = prefix == n && stats.frames == pyFrames
            identicalRuns += identical ? 1 : 0
            eosRuns += stats.endedWithEOS ? 1 : 0
            framesTotal += stats.frames
            prefixTotal += prefix
            wallTotal += stats.wallSeconds
            audioTotal += stats.audioSeconds
            print("Audio8Smoke \(f.name): \(stats.frames) frames (python \(pyFrames), oracle \(f.oracle_frames)), identical prefix \(prefix)"
                  + (identical ? " (all)" : "") + ", eos \(stats.endedWithEOS), rtf \(String(format: "%.3f", stats.rtf)),"
                  + " prefill \(String(format: "%.0f", stats.prefillSeconds * 1000)) ms, frame \(String(format: "%.1f", stats.frameSeconds * 1000 / Double(max(stats.frames, 1)))) ms/f,"
                  + " codec \(String(format: "%.0f", stats.codecSeconds * 1000)) ms,"
                  + " first audio \(String(format: "%.2f", stats.firstAudioSeconds)) s, calls \(stats.engineCalls)")
            if let logs = Self.env("KIT_AUDIO8_LOGS") ?? Optional(ref.deletingLastPathComponent().appendingPathComponent("logs").path) {
                let dir = URL(fileURLWithPath: logs)
                try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                try? AudioFile.writeWAV(audio, sampleRate: Audio8TTS.sampleRate, to: dir.appendingPathComponent("\(f.name).swift.wav"))
            }
            report.append(["name": f.name, "frames": stats.frames, "python_frames": pyFrames, "oracle_frames": f.oracle_frames,
                           "identical_prefix": prefix, "identical": identical, "eos": stats.endedWithEOS,
                           "wall_s": stats.wallSeconds, "audio_s": stats.audioSeconds, "rtf": stats.rtf,
                           "prefill_s": stats.prefillSeconds, "frame_s": stats.frameSeconds,
                           "codec_s": stats.codecSeconds, "first_audio_s": stats.firstAudioSeconds,
                           "engine_calls": stats.engineCalls, "prompt_tokens": stats.promptTokens,
                           "codes": stats.codes.map { $0.map(Int.init) }])
            XCTAssertTrue(stats.endedWithEOS, "\(f.name) did not reach eos")
        }
        let summary: [String: Any] = ["tag": manifest.tag, "runs": report.count, "identical_runs": identicalRuns, "eos_runs": eosRuns,
                                      "frames": framesTotal, "identical_prefix_frames": prefixTotal,
                                      "rtf_overall": audioTotal > 0 ? wallTotal / audioTotal : -1, "load_s": tts.loadSeconds,
                                      "fixtures": report]
        print("Audio8Smoke summary: identical runs \(identicalRuns)/\(report.count), eos \(eosRuns)/\(report.count), "
              + "identical prefix frames \(prefixTotal)/\(framesTotal), rtf \(String(format: "%.3f", audioTotal > 0 ? wallTotal / audioTotal : -1))")
        if let logs = Self.env("KIT_AUDIO8_LOGS") ?? Optional(ref.deletingLastPathComponent().appendingPathComponent("logs").path) {
            let dir = URL(fileURLWithPath: logs)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: dir.appendingPathComponent("swift_smoke.json"))
        }
    }

    // MARK: - 4. a plain run with the host's own noise (what an app does)

    func testPlainRun() async throws {
        guard let (tts, _, _) = try await loadAll() else { return }
        let stats = try await tts.synthesizeStreaming("The meeting has been moved to Thursday afternoon.", seed: 7) { _ in }
        print("Audio8Smoke plain: \(stats.frames) frames, \(String(format: "%.2f", stats.audioSeconds)) s audio in "
              + "\(String(format: "%.2f", stats.wallSeconds)) s (rtf \(String(format: "%.3f", stats.rtf))), eos \(stats.endedWithEOS)")
        XCTAssertTrue(stats.endedWithEOS)
        XCTAssertGreaterThan(stats.frames, 20)
    }
}
