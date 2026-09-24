// Nemotron-3-Diarization's host in the kit (Sources/CoreAIKit/Audio/NemotronDiarizer), gated without
// weights or audio against the zoo's Python host, bit for bit:
//
// - Fixtures/N3D/cache_unit* is one teacher-forced speaker-cache update from the zoo's golden
//   (export_golden.py @ 26241ae, cache/diarization_example_ll_s078: low latency, the second
//   compression, on a cache already compressed). The logits, the stored probabilities, the pooled
//   probabilities, the float64 compression scores and the probabilities after are the golden's
//   bytes. The 512-d rows only move through the update, so they ship as class ids — one per distinct
//   golden row, byte-identical rows sharing one — and the test builds a synthetic row per class.
// - Fixtures/N3D/mel_chunk0* is the first low-latency chunk (16,680 samples, centered) of a synthetic
//   clip, and the log-mel the zoo's NumPy mirror (mel_frontend.py @ 26241ae) computes from it.
//
// The whole golden — the 35 cache units and the two clips' mel chunks — runs when it is on this
// machine:
//
//     N3D_GOLDEN_DIR=<zoo>/conversion/nemotron3_diar/_work/golden \
//     N3D_HOST_DIR=<dir with the host constants> N3D_AUDIO_DIR=<dir with the two 16 kHz wavs> \
//     swift test --filter NemotronDiarizer

import Foundation
import Testing

@testable import CoreAIKit

struct NemotronDiarizerTests {
    @available(macOS 27, iOS 27, *)
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/N3D")

    @available(macOS 27, iOS 27, *)
    static func f32(_ name: String, in dir: URL = NemotronDiarizerTests.fixtures) throws -> [Float] {
        N3DAssets.readF32LE(try Data(contentsOf: dir.appendingPathComponent(name)))
    }

    @available(macOS 27, iOS 27, *)
    static func f64(_ name: String, in dir: URL = NemotronDiarizerTests.fixtures) throws -> [Double] {
        let data = try Data(contentsOf: dir.appendingPathComponent(name))
        return data.withUnsafeBytes { raw in
            (0..<(data.count / 8)).map {
                Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self)))
            }
        }
    }

    /// A distinct 512-d row per class id, exact in float32.
    @available(macOS 27, iOS 27, *)
    static func row(_ c: Int) -> [Float] {
        (0..<N3DSpeakerCache.hidden).map { Float(c * N3DSpeakerCache.hidden + $0) }
    }

    /// The kit's librosa-slaney filterbank: the same bytes as the repo's host/mel_filters_128x257.f32le.
    @available(macOS 27, iOS 27, *)
    static func bundledMelFilters() throws -> [Float] {
        let url = try #require(Bundle.module.url(forResource: "parakeet_mel_filters_128x257", withExtension: "f32"))
        return N3DAssets.readF32LE(try Data(contentsOf: url))
    }

    @available(macOS 27, iOS 27, *)
    @Test func speakerCacheUpdateIsBitExactWithTheZooHost() throws {
        let data = try Data(contentsOf: Self.fixtures.appendingPathComponent("cache_unit.json"))
        let meta = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let H = N3DSpeakerCache.hidden, S = N3DSpeakerCache.numSpeakers
        let L = try #require(meta["L"] as? Int), nChunk = try #require(meta["n_chunk"] as? Int)
        let nCache = try #require(meta["n_cache"] as? Int), nFifo = try #require(meta["n_fifo"] as? Int)
        let compressedBefore = try #require(meta["compressed_before"] as? Bool)
        let rows = try #require(meta["row_classes"] as? [Int]).flatMap(Self.row)
        let silence = Self.row(try #require(meta["silence_class"] as? Int))
        let logits = try Self.f32("cache_unit_logits.f32le")
        let probsBefore = try Self.f32("cache_unit_probs_before.f32le")
        let pooledRef = try Self.f32("cache_unit_pooled.f32le")
        #expect(rows.count == L * H && logits.count == L * S * S)

        let before = N3DSpeakerCache(
            fifoLength: try #require(meta["fifo_length"] as? Int),
            updatePeriod: try #require(meta["update_period"] as? Int),
            embeds: Array(rows[0..<(nCache * H)]), probs: probsBefore,
            fifo: Array(rows[(nCache * H)..<((nCache + nFifo) * H)]), isCompressed: compressedBefore)

        // sigmoid + the 8x pool, summed left to right
        let pooled = N3DSpeakerCache.poolProbs(logits: logits, frames: L)
        #expect(pooled.map(\.bitPattern) == pooledRef.map(\.bitPattern))

        // the float64 scores of what compress() ranks: the stored probabilities + the popped rows'
        let stored = compressedBefore ? probsBefore : Array(pooledRef[0..<(nCache * S)])
        let pop = before.numPopped(nFifo + nChunk)
        let scored = stored + pooledRef[(nCache * S)..<((nCache + nFifo + nChunk) * S)].prefix(pop * S)
        let scores = before.frameScores(probs: scored, frames: scored.count / S)
        let scoresRef = try Self.f64("cache_unit_scores.f64le")
        #expect(scores.map(\.bitPattern) == scoresRef.map(\.bitPattern))

        // the update: which rows stay, in which order, and the probabilities kept with them
        var cache = before
        cache.update(rows: rows, logits: logits, frames: L, silence: silence, chunkFrames: nChunk)
        let embedsAfter = try #require(meta["embeds_after_classes"] as? [Int]).flatMap(Self.row)
        let fifoAfter = try #require(meta["fifo_after_classes"] as? [Int]).flatMap(Self.row)
        #expect(cache.embeds == embedsAfter)
        #expect(cache.fifo == fifoAfter)
        let probsAfter = try Self.f32("cache_unit_probs_after.f32le")
        #expect(cache.probs.map(\.bitPattern) == probsAfter.map(\.bitPattern))
        #expect(cache.isCompressed == (meta["compressed_after"] as? Bool))
        #expect(cache.compressions == 1)

        // the fixture tells a compression from a cut: the zoo's no-compress control fails it
        var cut = before
        cut.poison = .noCompress
        cut.update(rows: rows, logits: logits, frames: L, silence: silence, chunkFrames: nChunk)
        #expect(cut.embeds != embedsAfter)
    }

    @available(macOS 27, iOS 27, *)
    @Test func melChunk0IsBitExactWithTheZooHost() throws {
        let pcm = try Data(contentsOf: Self.fixtures.appendingPathComponent("mel_chunk0_input.pcm16"))
        let samples: [Float] = pcm.withUnsafeBytes { raw in
            (0..<(pcm.count / 2)).map {
                Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: 2 * $0, as: Int16.self))) / 32768
            }
        }
        let mel = N3DMel(melFilters: try Self.bundledMelFilters(), hannWindow: try Self.f32("hann_window_400.f32le"))
        let chunk = try #require(N3DMel.streamChunks(samples: samples.count, chunkFrames: 9, lookaheadFrames: 4).first)
        #expect(chunk.isFirst && chunk.end == samples.count)
        let (m, frames) = mel.logMel(samples[chunk.start..<chunk.end], center: chunk.isFirst)
        let ref = try Self.f32("mel_chunk0.f32le")
        #expect(frames == ref.count / N3DMel.nMels)
        #expect(m.map(\.bitPattern) == ref.map(\.bitPattern))
    }

    /// The turn rule on 8-speaker rows at 10 ms: the strongest speaker above 0.5, same-speaker runs,
    /// gaps of up to 48 frames (0.48 s) bridged; a segment carries its frame length.
    @available(macOS 27, iOS 27, *)
    @Test func turnsFromEightSpeakerRows() {
        var rows = [[Float]](repeating: [Float](repeating: 0, count: 8), count: 300)
        for f in 0..<100 { rows[f][6] = 0.9 }
        for f in 148..<200 { rows[f][6] = 0.8 }            // 48-frame gap: bridged
        for f in 200..<260 { rows[f][7] = 0.7; rows[f][2] = 0.6 }   // overlap: the stronger voice
        for f in 260..<270 { rows[f][2] = 0.51 }
        let turns = KitDiarizer.segments(from: rows, bridgeFrames: 48, frameSec: 0.01)
        #expect(turns.map(\.speaker) == [6, 7, 2])
        #expect(turns.map(\.startFrame) == [0, 200, 260])
        #expect(turns.map(\.endFrame) == [200, 260, 270])
        #expect(abs(turns[1].startSec - 2.0) < 1e-12 && abs(turns[2].endSec - 2.7) < 1e-12)
        // one frame more and the gap stays
        #expect(KitDiarizer.segments(from: rows, bridgeFrames: 47, frameSec: 0.01).count == 4)
        // the 4-speaker default is unchanged: 80 ms frames
        let legacy = SpeakerSegment(speaker: 0, startFrame: 5, endFrame: 10)
        #expect(legacy.frameSec == 0.08 && abs(legacy.endSec - 0.8) < 1e-12)
    }

    /// A local bundle is Nemotron-3 when its host constants sit in `host/` beside the graph (the Hub
    /// layout) or next to it (an export directory); a Sortformer bundle has none.
    @available(macOS 27, iOS 27, *)
    @Test func localBundleLayouts() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("n3d-layout-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let hub = root.appendingPathComponent("hub"), flat = root.appendingPathComponent("flat")
        let sortformer = root.appendingPathComponent("sortformer")
        for dir in [hub.appendingPathComponent("host"), hub.appendingPathComponent("n3d_streaming_float16.aimodel"),
                    flat.appendingPathComponent("n3d_streaming_float16.aimodel"),
                    flat.appendingPathComponent("n3d_offline_float16.aimodel"),
                    sortformer.appendingPathComponent("sortformer_float16.aimodel")] {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        for dir in [hub.appendingPathComponent("host"), flat] {
            try Data().write(to: dir.appendingPathComponent("embedder_projection.f32le"))
        }
        let hubGraph = hub.appendingPathComponent("n3d_streaming_float16.aimodel")
        #expect(KitDiarizer.nemotronHost(near: hub)?.lastPathComponent == "host")
        #expect(KitDiarizer.nemotronHost(near: hubGraph)?.lastPathComponent == "host")
        #expect(try KitDiarizer.nemotronGraph(in: hub).lastPathComponent == "n3d_streaming_float16.aimodel")
        #expect(KitDiarizer.nemotronHost(near: flat)?.lastPathComponent == "flat")
        #expect(try KitDiarizer.nemotronGraph(in: flat).lastPathComponent == "n3d_streaming_float16.aimodel")
        #expect(KitDiarizer.nemotronHost(near: sortformer) == nil)
        #expect(KitDiarizer.nemotronHost(
            near: sortformer.appendingPathComponent("sortformer_float16.aimodel")) == nil)
    }
}

/// The zoo's whole golden against the kit's copy of the host, when it is on this machine (see the
/// file header): 35 teacher-forced cache updates, and chunks 0, 1 and the last of both clips'
/// low-latency mel plus their embeddings.
struct NemotronDiarizerGoldenTests {
    @available(macOS 27, iOS 27, *)
    static let environment = ProcessInfo.processInfo.environment
    @available(macOS 27, iOS 27, *)
    static let enabled = ["N3D_GOLDEN_DIR", "N3D_HOST_DIR", "N3D_AUDIO_DIR"].allSatisfy { environment[$0] != nil }
    @available(macOS 27, iOS 27, *)
    static func dir(_ key: String) -> URL { URL(fileURLWithPath: environment[key]!) }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: enabled, "set N3D_GOLDEN_DIR, N3D_HOST_DIR and N3D_AUDIO_DIR to run the zoo's golden"))
    @available(macOS 27, iOS 27, *)
    func everyCacheUnit() throws {
        let golden = Self.dir("N3D_GOLDEN_DIR")
        let assets = try N3DAssets(directory: Self.dir("N3D_HOST_DIR"))
        let data = try Data(contentsOf: golden.appendingPathComponent("cache/index.json"))
        let units = try #require(try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        let H = N3DSpeakerCache.hidden, S = N3DSpeakerCache.numSpeakers
        var compressions = 0
        for u in units {
            let prefix = try #require(u["prefix"] as? String)
            @available(macOS 27, iOS 27, *)
            func f(_ suffix: String) throws -> [Float] { try NemotronDiarizerTests.f32("\(prefix)_\(suffix).f32le", in: golden) }
            let L = try #require(u["L"] as? Int), nChunk = try #require(u["n_chunk"] as? Int)
            let nCache = try #require(u["n_cache"] as? Int), nFifo = try #require(u["n_fifo"] as? Int)
            let compressedBefore = try #require(u["compressed_before"] as? Bool)
            let rows = try f("rows"), logits = try f("logits"), probsBefore = try f("probs_before")
            let pooledRef = try f("pooled")
            var cache = N3DSpeakerCache(
                fifoLength: try #require(u["fifo_length"] as? Int), updatePeriod: try #require(u["update_period"] as? Int),
                embeds: Array(rows[0..<(nCache * H)]), probs: probsBefore,
                fifo: Array(rows[(nCache * H)..<((nCache + nFifo) * H)]), isCompressed: compressedBefore)
            #expect(N3DSpeakerCache.poolProbs(logits: logits, frames: L).map(\.bitPattern) == pooledRef.map(\.bitPattern),
                    "\(prefix): pooled")
            if u["compressed_now"] as? Bool == true {
                compressions += 1
                let stored = compressedBefore ? probsBefore : Array(pooledRef[0..<(nCache * S)])
                let pop = cache.numPopped(nFifo + nChunk)
                let scored = stored + pooledRef[(nCache * S)..<((nCache + nFifo + nChunk) * S)].prefix(pop * S)
                let ref = try NemotronDiarizerTests.f64("\(prefix)_scores.f64le", in: golden)
                #expect(cache.frameScores(probs: scored, frames: scored.count / S).map(\.bitPattern) == ref.map(\.bitPattern),
                        "\(prefix): scores")
            }
            cache.update(rows: rows, logits: logits, frames: L, silence: assets.silence, chunkFrames: nChunk)
            let (embedsAfter, probsAfter, fifoAfter) = (try f("embeds_after"), try f("probs_after"), try f("fifo_after"))
            #expect(cache.embeds.map(\.bitPattern) == embedsAfter.map(\.bitPattern), "\(prefix): cache rows")
            #expect(cache.probs.map(\.bitPattern) == probsAfter.map(\.bitPattern), "\(prefix): cache probs")
            #expect(cache.fifo.map(\.bitPattern) == fifoAfter.map(\.bitPattern), "\(prefix): fifo rows")
            #expect(cache.isCompressed == (u["compressed_after"] as? Bool), "\(prefix): compressed")
        }
        #expect(units.count == 35 && compressions == 17)
    }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: enabled, "set N3D_GOLDEN_DIR, N3D_HOST_DIR and N3D_AUDIO_DIR to run the zoo's golden"))
    @available(macOS 27, iOS 27, *)
    func melChunksOfBothClips() throws {
        let golden = Self.dir("N3D_GOLDEN_DIR")
        let assets = try N3DAssets(directory: Self.dir("N3D_HOST_DIR"))
        let mel = N3DMel(melFilters: assets.melFilters, hannWindow: assets.hannWindow)
        let embedder = N3DEmbedder(projection: assets.projection)
        let bundled = try NemotronDiarizerTests.bundledMelFilters()
        #expect(assets.melFilters.map(\.bitPattern) == bundled.map(\.bitPattern))
        for fixture in ["diarization_example", "test_multispk"] {
            let samples = try Self.pcm16Wav(Self.dir("N3D_AUDIO_DIR").appendingPathComponent("\(fixture)_16k.wav"))
            let chunks = N3DMel.streamChunks(samples: samples.count, chunkFrames: 9, lookaheadFrames: 4)
            for k in [0, 1, chunks.count - 1] {
                let c = chunks[k]
                let (m, frames) = mel.logMel(
                    samples[min(c.start, samples.count)..<min(c.end, samples.count)], center: c.isFirst)
                let ref = try NemotronDiarizerTests.f32("\(fixture)_ll_mel_chunk\(k).f32le", in: golden)
                #expect(frames == ref.count / N3DMel.nMels, "\(fixture) chunk \(k): frames")
                #expect(m.map(\.bitPattern) == ref.map(\.bitPattern), "\(fixture) chunk \(k): mel")
                #expect(embedder.embed(mel: m, frames: frames).embeds.map(\.bitPattern)
                        == embedder.embed(mel: ref, frames: frames).embeds.map(\.bitPattern), "\(fixture) chunk \(k): embeds")
            }
        }
    }

    /// 16 kHz mono PCM16 WAV -> samples / 32768 (how the zoo's gates read their fixtures).
    @available(macOS 27, iOS 27, *)
    static func pcm16Wav(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        @available(macOS 27, iOS 27, *)
        func u32(_ o: Int) -> Int { Int(data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) }) }
        @available(macOS 27, iOS 27, *)
        func u16(_ o: Int) -> Int { Int(data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt16.self)) }) }
        var off = 12, format = 0, channels = 0, rate = 0, bits = 0
        while off + 8 <= data.count {
            let id = String(decoding: data[off..<(off + 4)], as: UTF8.self), size = u32(off + 4), body = off + 8
            if id == "fmt " {
                (format, channels, rate, bits) = (u16(body), u16(body + 2), u32(body + 4), u16(body + 14))
            } else if id == "data" {
                try #require(format == 1 && channels == 1 && rate == 16_000 && bits == 16, "\(url.lastPathComponent): not 16 kHz mono PCM16")
                let n = min(size, data.count - body) / 2
                return data.withUnsafeBytes { raw in
                    (0..<n).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: body + 2 * $0, as: Int16.self))) / 32768 }
                }
            }
            off = body + size + (size & 1)
        }
        throw CocoaError(.fileReadCorruptFile)
    }
}
