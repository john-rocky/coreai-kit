// diarize-gate — the diarizer alone against a reference, from the shell. Per-frame activity from
// `KitDiarizer.framePreds(fromSamples:)` is compared at 0.5 with a reference's probabilities
// ([frames, speakers] float32 LE — for Nemotron-3-Diarization the zoo's golden
// `<clip>_ll_probs.f32le`, transformers fp32), and the turns and wall times are printed. Public API
// only, so it measures what an app gets.
//
//   swift run diarize-gate --audio clip.wav --golden clip_ll_probs.f32le
//   swift run diarize-gate --audio clip.wav --golden clip_ll_probs.f32le \
//       --same-as clip_ll_logits.f32le          # another run's logits: are the probabilities bit-equal?
//
// --diarizer picks the catalog id (default nemotron-3-diarization), --diarizer-bundle a local
// bundle instead. Exit 0 = agreement at or above --bar (default 0.999), 3 = below it, 1 = an error.

import CoreAIKit
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func f32le(_ path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
    return data.withUnsafeBytes { raw in
        (0..<(data.count / 4)).map {
            Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
        }
    }
}

/// A 16 kHz mono PCM16 WAV as samples / 32768 — how the zoo's gates read their clips — or nil for
/// any other file.
func pcm16(_ url: URL) -> [Float]? {
    guard let data = try? Data(contentsOf: url), data.count > 44,
          String(decoding: data[0..<4], as: UTF8.self) == "RIFF" else { return nil }
    func u32(_ o: Int) -> Int { Int(data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) }) }
    func u16(_ o: Int) -> Int { Int(data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt16.self)) }) }
    var off = 12, pcm16Mono16k = false
    while off + 8 <= data.count {
        let id = String(decoding: data[off..<(off + 4)], as: UTF8.self), size = u32(off + 4), body = off + 8
        if id == "fmt " {
            pcm16Mono16k = u16(body) == 1 && u16(body + 2) == 1 && u32(body + 4) == 16_000 && u16(body + 14) == 16
        } else if id == "data" {
            guard pcm16Mono16k else { return nil }
            let n = min(size, data.count - body) / 2
            return data.withUnsafeBytes { raw in
                (0..<n).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: body + 2 * $0, as: Int16.self))) / 32768 }
            }
        }
        off = body + size + (size & 1)
    }
    return nil
}

func seconds(_ d: Duration) -> Double {
    Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

var diarizerID = "nemotron-3-diarization"
var bundlePath: String?
var audioPath: String?
var goldenPath: String?
var sameAsPath: String?
var bar = 0.999

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--diarizer": diarizerID = args.popFirst() ?? diarizerID
    case "--diarizer-bundle": bundlePath = args.popFirst()
    case "--audio": audioPath = args.popFirst()
    case "--golden": goldenPath = args.popFirst()
    case "--same-as": sameAsPath = args.popFirst()
    case "--bar": bar = args.popFirst().flatMap(Double.init) ?? bar
    default: fail("unknown argument: \(arg)")
    }
}
guard let audioPath, let goldenPath else {
    fail("usage: diarize-gate --audio <wav> --golden <probs.f32le> [--same-as <logits.f32le>] "
        + "[--diarizer <catalog-id> | --diarizer-bundle <path>] [--bar 0.999]")
}

do {
    let clock = ContinuousClock()
    let audioURL = URL(fileURLWithPath: (audioPath as NSString).expandingTildeInPath)
    let samples = try AudioFile.pcm16kMono(audioURL)
    if let raw = pcm16(audioURL) {
        let same = raw.count == samples.count
            && zip(raw, samples).allSatisfy { $0.bitPattern == $1.bitPattern }
        print("audio \(audioURL.lastPathComponent): \(String(format: "%.2f", Double(samples.count) / 16_000)) s, "
            + "AudioFile.pcm16kMono \(same ? "==" : "!=") PCM16 / 32768 (\(samples.count) vs \(raw.count) samples)")
    } else {
        print("audio \(audioURL.lastPathComponent): \(String(format: "%.2f", Double(samples.count) / 16_000)) s")
    }

    var t = clock.now
    let diarizer: KitDiarizer
    if let bundlePath {
        diarizer = try await KitDiarizer(
            bundleAt: URL(fileURLWithPath: (bundlePath as NSString).expandingTildeInPath))
    } else {
        diarizer = try await KitDiarizer(catalog: diarizerID)
    }
    let S = diarizer.nSpk
    print("diarizer \(bundlePath ?? diarizerID): \(S) speakers, \(Int(diarizer.frameSec * 1000)) ms frames, "
        + "loaded in \(String(format: "%.2f", seconds(clock.now - t))) s")

    t = clock.now
    let preds = try await diarizer.framePreds(fromSamples: samples)
    let predsWall = seconds(clock.now - t)
    t = clock.now
    let turns = try await diarizer.diarize(samples: samples)
    let turnsWall = seconds(clock.now - t)

    let ref = try f32le(goldenPath)
    let n = min(preds.count, ref.count / S)
    var differ = 0, maxDp = 0.0
    for f in 0..<n {
        for s in 0..<S {
            let a = preds[f][s], b = ref[f * S + s]
            if (a > 0.5) != (b > 0.5) { differ += 1 }
            maxDp = max(maxDp, abs(Double(a) - Double(b)))
        }
    }
    let elements = n * S
    let agreement = elements == 0 ? 0 : Double(elements - differ) / Double(elements)
    print("frames ours \(preds.count) / reference \(ref.count / S), compared \(n) x \(S) = \(elements)")
    print("agreement@0.5 \(String(format: "%.4f", agreement * 100)) % (\(differ) / \(elements) differ), "
        + "max|dp| \(String(format: "%.3e", maxDp))")
    let audioSec = Double(samples.count) / 16_000
    print("turns \(turns.count) (speakers \(Set(turns.map(\.speaker)).count)); framePreds wall "
        + "\(String(format: "%.3f", predsWall)) s (RTF \(String(format: "%.4f", predsWall / audioSec))), "
        + "diarize wall \(String(format: "%.3f", turnsWall)) s")

    if let sameAsPath {
        // The same float32 sigmoid the host applies: 1 / (1 + exp(-x)).
        let logits = try f32le(sameAsPath)
        var equal = 0, compared = 0
        for f in 0..<min(preds.count, logits.count / S) {
            for s in 0..<S {
                let p = Float(1) / (Float(1) + expf(-logits[f * S + s]))
                if p.bitPattern == preds[f][s].bitPattern { equal += 1 }
                compared += 1
            }
        }
        print("same-as \((sameAsPath as NSString).lastPathComponent): \(equal) / \(compared) probabilities bit-equal")
    }
    let pass = agreement >= bar
    print("\(pass ? "PASS" : "FAIL") (bar \(String(format: "%.1f", bar * 100)) %)")
    exit(pass ? 0 : 3)
} catch {
    fail("diarize-gate failed: \(error.localizedDescription)")
}
