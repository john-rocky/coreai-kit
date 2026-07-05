// KitNemotronModel.swift — NVIDIA Nemotron 3.5 ASR Streaming 0.6B on Core AI: the zoo's first
// STREAMING ASR (cache-aware FastConformer + pure-RNNT transducer, 40 locales, punctuation and
// capitalization built in, OpenMDW-1.1).
//
// ```swift
// let nemotron = try await KitNemotronModel(model: .nemotronASRStreaming)
// // live: feed mic packets as they arrive, read the running transcript back
// let session = nemotron.makeSession(language: "en-US")
// let partial = try await session.feed(samples: packet)
// let result = try await session.finish()
// // offline: any-length clip through the same streaming pipeline (no 30 s bucket)
// let result = try await nemotron.transcribe(samples: pcm) { partial in print(partial) }
// ```
//
// Pipeline per 320 ms chunk (lookahead 3): mel (25-then-32 frames) -> pre-encode graph (causal
// subsampling with explicit conv caches) -> conformer_a + conformer_b graphs (12 blocks each with
// an explicit 56-frame KV sliding window + depthwise-conv caches; b adds the language-one-hot
// prompt fusion + projector) -> 4 enc frames -> host greedy RNN-T (LSTM predictor + joint graphs).
// The conformer ships in TWO halves because the single 24-layer AOT bundle (2.4 GB resources.bin)
// fails to load on-device (instant POSIX-2); ~1.1 GB halves are device-proven.
// Validated token-exact vs the HF streaming reference
// (conversion/nemotron_asr/gate_e2e_streaming.py + gate_mel_swift_streaming.py).

import CoreAIKitVision
import Foundation
import Tokenizers

/// Failures specific to the Nemotron streaming transcription path.
public enum KitNemotronError: Error, LocalizedError {
    case unsupportedSampleRate(Int)
    case melFiltersMissing
    case graphMissing(String)
    case tokenizerMissing
    case outputMissing(String)
    case unknownLanguage(String)
    case sessionFinished

    public var errorDescription: String? {
        switch self {
        case .unsupportedSampleRate(let sr):
            return "Audio must be 16 kHz mono; got \(sr) Hz (resample before transcribing)."
        case .melFiltersMissing:
            return "Bundled mel filterbank resource is missing."
        case .graphMissing(let kind):
            return "Nemotron bundle is missing the \(kind) graph (.aimodel/.aimodelc)."
        case .tokenizerMissing:
            return "Nemotron bundle is missing tokenizer.json."
        case .outputMissing(let name):
            return "A Nemotron graph did not produce the expected output '\(name)'."
        case .unknownLanguage(let lang):
            return "Unknown language '\(lang)' — use a BCP-47 tag like \"en-US\", or \"auto\"."
        case .sessionFinished:
            return "This streaming session is finished — make a new one."
        }
    }
}

/// A Core AI Nemotron 3.5 ASR streaming bundle: six graphs (pre_first / pre / conformer_a /
/// conformer_b / predict / joint), the streaming mel frontend, and the tokenizer. One model
/// serves any number of consecutive sessions; run one session at a time.
public final class KitNemotronModel: @unchecked Sendable {
    // Model constants (config.json + conversion/nemotron_asr).
    static let blank = 13087
    static let vocab = 13088
    static let maxSymbolsPerStep = 10
    static let hidden = 640                       // predictor/joint/enc_proj width
    static let lstmLayers = 2
    static let layersPerHalf = 12, heads = 8, headDim = 128, encHidden = 1024
    static let kvWindow = 56                      // KV cache slots (sliding_window - 1)
    static let kvKeys = 60                        // cache + one 4-frame chunk
    static let encFramesPerChunk = 4              // lookahead 3 -> (3+1) enc frames / 320 ms
    static let convCacheLen = 8                   // depthwise conv kernel 9 -> 8 cached frames
    static let numPrompts = 128
    public static let sampleRate = 16000

    let preFirst: GraphModel
    let pre: GraphModel
    let conformerA: GraphModel
    let conformerB: GraphModel
    let predict: GraphModel
    let joint: GraphModel
    let tokenizer: any Tokenizer
    let melFilters: [Float]

    // MARK: - Init

    /// Loads Nemotron by its catalog id — the id shown on the model's card.
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        guard id == "nemotron-3.5-asr-streaming-0.6b" else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(
            model: .nemotronASRStreaming, store: store, computeUnits: computeUnits,
            downloadProgress: downloadProgress)
    }

    /// Downloads the Nemotron bundle from the Hub (if needed) and loads it.
    public convenience init(
        model: ModelID = .nemotronASRStreaming,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let root = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: root, computeUnits: computeUnits)
    }

    /// Loads a local Nemotron bundle directory (five graphs + `tokenizer.json`).
    public init(
        bundleAt root: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        let paths = try Self.resolve(in: root)
        self.preFirst = try await GraphModel(contentsOf: paths.preFirst, computeUnits: computeUnits)
        self.pre = try await GraphModel(contentsOf: paths.pre, computeUnits: computeUnits)
        self.conformerA = try await GraphModel(contentsOf: paths.conformerA, computeUnits: computeUnits)
        self.conformerB = try await GraphModel(contentsOf: paths.conformerB, computeUnits: computeUnits)
        self.predict = try await GraphModel(contentsOf: paths.predict, computeUnits: computeUnits)
        self.joint = try await GraphModel(contentsOf: paths.joint, computeUnits: computeUnits)
        guard let url = Bundle.module.url(
            forResource: "parakeet_mel_filters_128x257", withExtension: "f32")
        else { throw KitNemotronError.melFiltersMissing }
        self.melFilters = try Data(contentsOf: url).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        self.tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
    }

    // MARK: - Sessions

    /// Starts a live streaming session. `language` is a BCP-47 tag from
    /// `KitNemotronModel.languages` (e.g. `"en-US"`, `"ja-JP"`) or `"auto"` for language ID.
    public func makeSession(language: String = "en-US") throws -> NemotronStreamSession {
        guard let promptID = Self.promptDictionary[language] else {
            throw KitNemotronError.unknownLanguage(language)
        }
        return NemotronStreamSession(model: self, promptID: promptID)
    }

    /// Transcribe a whole 16 kHz mono clip through the streaming pipeline — any length, no
    /// 30 s bucket. `onPartial` streams the running transcript as chunks decode.
    public func transcribe(
        samples: [Float], sampleRate: Int = 16000, language: String = "en-US",
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        guard sampleRate == Self.sampleRate else {
            throw KitNemotronError.unsupportedSampleRate(sampleRate)
        }
        let session = try makeSession(language: language)
        // Feed in ~1 s slices so partials appear while long clips decode.
        var i = 0
        while i < samples.count {
            let end = min(i + Self.sampleRate, samples.count)
            let partial = try await session.feed(samples: Array(samples[i..<end]))
            if let onPartial, !partial.isEmpty { onPartial(partial) }
            i = end
        }
        return try await session.finish()
    }

    /// The supported language tags (sorted), plus `"auto"` for built-in language ID.
    public static var languages: [String] { promptDictionary.keys.sorted() }

    // MARK: - Bundle layout

    private struct Paths {
        let preFirst: URL, pre: URL, conformerA: URL, conformerB: URL
        let predict: URL, joint: URL, tokenizerDir: URL
    }

    /// Find the five graphs and the tokenizer anywhere under the bundle root — tolerant of both
    /// the flat dev layout and a per-graph subdirectory Hub layout.
    private static func resolve(in root: URL) throws -> Paths {
        let fm = FileManager.default
        var graphs: [URL] = []
        var tokenizerDir: URL?
        if let it = fm.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let url as URL in it {
                let ext = url.pathExtension
                if ext == "aimodel" || ext == "aimodelc" { graphs.append(url) }
                if url.lastPathComponent == "tokenizer.json" {
                    tokenizerDir = url.deletingLastPathComponent()
                }
            }
        }
        func graph(_ kind: String, exclude: String? = nil) throws -> URL {
            guard let u = graphs.first(where: {
                let name = $0.deletingPathExtension().lastPathComponent.lowercased()
                let dir = $0.deletingLastPathComponent().lastPathComponent.lowercased()
                let hit = name.contains(kind) || dir == kind
                let excluded = exclude.map { name.contains($0) || dir == $0 } ?? false
                return hit && !excluded
            }) else { throw KitNemotronError.graphMissing(kind) }
            return u
        }
        guard let tok = tokenizerDir else { throw KitNemotronError.tokenizerMissing }
        return Paths(
            preFirst: try graph("pre_first"),
            // "stream_pre" (not "pre") — a bare "pre" would also match "predict".
            pre: try graph("stream_pre", exclude: "pre_first"),
            conformerA: try graph("conformer_a"),
            conformerB: try graph("conformer_b"),
            predict: try graph("predict"),
            joint: try graph("joint"),
            tokenizerDir: tok)
    }

    /// The model's language-prompt indices (processor_config.json `prompt_dictionary`) — the
    /// one-hot input that conditions the single multilingual checkpoint at run time.
    static let promptDictionary: [String: Int] = [
        "af-ZA": 54, "am-ET": 49, "ar": 7, "auto": 101, "ay-BO": 81, "az-AZ": 66, "bg": 30,
        "bn-IN": 36, "cs": 22, "da": 25, "de": 9, "el": 21, "en-GB": 1, "en-US": 0, "es-ES": 2,
        "es-US": 3, "et": 60, "fa-IR": 38, "fi": 26, "fr-CA": 100, "fr-FR": 8, "gn-PY": 82,
        "gu-IN": 42, "ha-NG": 50, "haw-US": 97, "he-IL": 64, "hi-IN": 6, "hr": 29, "hu": 23,
        "hy-AM": 68, "id-ID": 34, "ig-NG": 53, "it": 15, "ja-JP": 10, "ka-GE": 67, "km-KH": 47,
        "kn-IN": 43, "ko-KR": 14, "ku-TR": 65, "ky-KG": 71, "ln-CD": 58, "lt": 31, "lv": 61,
        "mi-NZ": 96, "ml-IN": 44, "mr-IN": 41, "ms-MY": 35, "mt-MT": 102, "nah-MX": 83,
        "nb-NO": 103, "ne-NP": 46, "nl": 16, "nn-NO": 104, "no": 27, "ny-MW": 57, "or-KE": 59,
        "pl": 17, "pt-BR": 12, "pt-PT": 13, "qu-PE": 80, "ro": 20, "ru": 11, "rw-RW": 55,
        "si-LK": 45, "sk": 28, "sl": 62, "sm-WS": 98, "so-SO": 56, "sv": 24, "sw-KE": 48,
        "ta-IN": 39, "te-IN": 40, "tg-TJ": 70, "th-TH": 32, "to-TO": 99, "tr": 18, "uk": 19,
        "ur-PK": 37, "uz-UZ": 69, "vi-VN": 33, "yo-NG": 52, "zh-CN": 4, "zh-TW": 5, "zu-ZA": 51,
    ]
}

/// One live transcription stream: owns the mel frontend state, the encoder caches, and the
/// RNN-T decode state. Feed 16 kHz mono packets of any size; the transcript grows as complete
/// 320 ms chunks decode. Not concurrency-safe — feed from one task at a time.
public final class NemotronStreamSession: @unchecked Sendable {
    private let model: KitNemotronModel
    private let oneHot: TensorValue
    private let mel: NemotronMelStream

    // Encoder caches (round-tripped as the graphs' own fp16 tensors — no host conversion).
    private var kCacheA: TensorValue
    private var vCacheA: TensorValue
    private var convCacheA: TensorValue
    private var kCacheB: TensorValue
    private var vCacheB: TensorValue
    private var convCacheB: TensorValue
    private var cache0: TensorValue?
    private var cache1: TensorValue?
    private var cache2: TensorValue?
    private var chunkIndex = 0

    // RNN-T decode state.
    private var encFrames: [Float] = []           // growing [T, 640] buffer
    private var frame = 0
    private var symbolsOnFrame = 0
    private var h: [Float]
    private var c: [Float]
    private var dec: [Float] = []
    private var emitted: [Int] = []
    private var primed = false
    private var finished = false
    private var samplesSeen = 0

    /// Rolling per-chunk latency (encode + decode wall time), for RTF reporting.
    public private(set) var chunkCount = 0
    public private(set) var totalChunkSeconds = 0.0
    public private(set) var maxChunkSeconds = 0.0
    /// Per-stage wall-time totals (seconds) for profiling: pre-encode, conformer, joint, predict.
    public private(set) var stageSeconds: [String: Double] = [:]
    public private(set) var stageCalls: [String: Int] = [:]

    private func timed<T>(_ stage: String, _ body: () async throws -> T) async rethrows -> T {
        let t0 = ContinuousClock.now
        let result = try await body()
        let e = (ContinuousClock.now - t0).components
        stageSeconds[stage, default: 0] += Double(e.seconds) + Double(e.attoseconds) / 1e18
        stageCalls[stage, default: 0] += 1
        return result
    }

    /// The transcript so far (language tags stripped).
    public private(set) var transcript = ""
    /// The last language tag the model emitted (its built-in LID; useful with `language: "auto"`).
    public private(set) var detectedLanguage = ""

    init(model: KitNemotronModel, promptID: Int) {
        self.model = model
        var oh = [Float](repeating: 0, count: KitNemotronModel.numPrompts)
        oh[promptID] = 1
        self.oneHot = .float32(oh, shape: [1, KitNemotronModel.numPrompts])
        self.mel = NemotronMelStream(melFilters: model.melFilters)
        let L = KitNemotronModel.layersPerHalf
        let kvShape = [L, KitNemotronModel.heads, KitNemotronModel.kvWindow, KitNemotronModel.headDim]
        let convShape = [L, KitNemotronModel.encHidden, KitNemotronModel.convCacheLen]
        self.kCacheA = Self.zeros16(kvShape)
        self.vCacheA = Self.zeros16(kvShape)
        self.convCacheA = Self.zeros16(convShape)
        self.kCacheB = Self.zeros16(kvShape)
        self.vCacheB = Self.zeros16(kvShape)
        self.convCacheB = Self.zeros16(convShape)
        let stateLen = KitNemotronModel.lstmLayers * KitNemotronModel.hidden
        self.h = [Float](repeating: 0, count: stateLen)
        self.c = [Float](repeating: 0, count: stateLen)
    }

    /// Feed a packet of 16 kHz mono samples (any size); returns the transcript so far.
    @discardableResult
    public func feed(samples: [Float]) async throws -> String {
        guard !finished else { throw KitNemotronError.sessionFinished }
        samplesSeen += samples.count
        for chunk in mel.push(samples) {
            try await process(melChunk: chunk)
        }
        return transcript
    }

    /// Flush the tail (pads with silence until the chunk holding the last real-audio frame has
    /// been emitted; the leftover silence-only frames are discarded) and end the session.
    public func finish() async throws -> Transcription {
        guard !finished else { throw KitNemotronError.sessionFinished }
        // Last frame that carries any real audio: window [160t-200, 160t+200) ∋ final sample.
        let lastFrame = (samplesSeen + NemotronMelStream.winLength / 2 - 1) / NemotronMelStream.hop
        let silence = [Float](repeating: 0, count: NemotronMelStream.hop * 8)
        var guardIters = 0
        while framesInEmittedChunks() <= lastFrame && guardIters < 4096 {
            for chunk in mel.push(silence) { try await process(melChunk: chunk) }
            guardIters += 1
        }
        finished = true
        return Transcription(language: detectedLanguage, text: transcript)
    }

    /// Total mel frames already handed to the encoder in completed chunks.
    private func framesInEmittedChunks() -> Int {
        mel.chunksEmitted == 0
            ? 0
            : NemotronMelStream.firstChunkFrames
                + (mel.chunksEmitted - 1) * NemotronMelStream.chunkFrames
    }

    // MARK: - Per-chunk pipeline

    private func process(melChunk: [Float]) async throws {
        let t0 = ContinuousClock.now
        let frames = melChunk.count / NemotronMelStream.nMels

        // pre-encode (subsampling with conv caches) -> embeds [1,4,1024]
        let embeds: TensorValue
        if chunkIndex == 0 {
            let out = try await timed("pre") { try await model.preFirst.run([
                "mel": .float32(melChunk, shape: [1, frames, NemotronMelStream.nMels])
            ]) }
            embeds = try value(out, "embeds")
            cache0 = try value(out, "cache0")
            cache1 = try value(out, "cache1")
            cache2 = try value(out, "cache2")
        } else {
            let out = try await timed("pre") { try await model.pre.run([
                "mel": .float32(melChunk, shape: [1, frames, NemotronMelStream.nMels]),
                "cache0": cache0!, "cache1": cache1!, "cache2": cache2!,
            ]) }
            embeds = try value(out, "embeds")
            cache0 = try value(out, "cache0_out")
            cache1 = try value(out, "cache1_out")
            cache2 = try value(out, "cache2_out")
        }

        // conformer halves (KV + conv caches, additive mask over the not-yet-filled window).
        // I/O tensors are named "embeds"/"embeds_out", NOT "x" — the on-device AOT loader
        // rejects conformer_b with an input named "x" (instant POSIX-2; device-bisected).
        let neg = negMask()
        let outA = try await timed("conformer_a") { try await model.conformerA.run([
            "embeds": embeds,
            "neg_mask": neg,
            "k_cache": kCacheA, "v_cache": vCacheA, "conv_cache": convCacheA,
        ]) }
        kCacheA = try value(outA, "k_out")
        vCacheA = try value(outA, "v_out")
        convCacheA = try value(outA, "conv_out")
        let out = try await timed("conformer_b") { try await model.conformerB.run([
            "embeds": try value(outA, "embeds_out"),
            "one_hot": oneHot,
            "neg_mask": neg,
            "k_cache": kCacheB, "v_cache": vCacheB, "conv_cache": convCacheB,
        ]) }
        kCacheB = try value(out, "k_out")
        vCacheB = try value(out, "v_out")
        convCacheB = try value(out, "conv_out")
        let encChunk = try value(out, "enc_proj").floats()      // [4*640]
        encFrames.append(contentsOf: encChunk)
        chunkIndex += 1

        try await decodeAvailableFrames()

        let elapsed = (ContinuousClock.now - t0).components
        let dt = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        chunkCount += 1
        totalChunkSeconds += dt
        maxChunkSeconds = max(maxChunkSeconds, dt)
    }

    /// Greedy pure-RNNT over every not-yet-consumed encoder frame (gate_e2e_streaming.py,
    /// ported verbatim): blank advances a frame; a non-blank token is emitted and steps the
    /// predictor on the same frame; 10 symbols per frame force an advance.
    private func decodeAvailableFrames() async throws {
        let M = KitNemotronModel.self
        let H = M.hidden
        if !primed {
            dec = try await predictStep(token: M.blank)
            primed = true
        }
        let T = encFrames.count / H
        var newTokens = false
        while frame < T {
            let encFrame = Array(encFrames[frame * H ..< (frame + 1) * H])
            let jout = try await timed("joint") { try await model.joint.run([
                "dec_out": .float32(dec, shape: [1, H]),
                "enc_frame": .float32(encFrame, shape: [1, H]),
            ]) }
            let logits = try value(jout, "token_logits").floats()
            var best = 0
            var bestVal = -Float.greatestFiniteMagnitude
            for i in 0..<M.vocab where logits[i] > bestVal { bestVal = logits[i]; best = i }
            if best == M.blank {
                frame += 1
                symbolsOnFrame = 0
            } else {
                emitted.append(best)
                newTokens = true
                dec = try await predictStep(token: best)
                symbolsOnFrame += 1
                if symbolsOnFrame >= M.maxSymbolsPerStep {
                    frame += 1
                    symbolsOnFrame = 0
                }
            }
        }
        if newTokens { refreshTranscript() }
    }

    private func predictStep(token: Int) async throws -> [Float] {
        let M = KitNemotronModel.self
        let s = [M.lstmLayers, 1, M.hidden]
        let out = try await timed("predict") { try await model.predict.run([
            "token": .int32([Int32(token)], shape: [1, 1]),
            "h": .float32(h, shape: s),
            "c": .float32(c, shape: s),
        ]) }
        h = try value(out, "h_out").floats()
        c = try value(out, "c_out").floats()
        return try value(out, "dec_out").floats()
    }

    /// Decode + strip the trailing language-tag tokens the model emits at sentence ends
    /// (`<en-US>`, `<ja-JP>`, …) — the last one seen becomes `detectedLanguage`.
    private func refreshTranscript() {
        var text = model.tokenizer.decode(tokens: emitted)
        while let r = text.range(of: #"<[A-Za-z]{2,4}(-[A-Za-z]{2,4})?>"#, options: .regularExpression) {
            detectedLanguage = String(text[r].dropFirst().dropLast())
            text.removeSubrange(r)
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.unicodeScalars.contains("\u{FFFD}") { transcript = text }
    }

    // MARK: - Tensors

    /// Additive attention mask [1,1,4,60]: -inf over the cache slots not yet filled (the KV
    /// window fills over the first 14 chunks; afterwards a constant all-zeros mask).
    private func negMask() -> TensorValue {
        let M = KitNemotronModel.self
        let q = M.encFramesPerChunk, kv = M.kvKeys
        let valid = min(M.kvWindow, q * chunkIndex)
        var mask = [Float16](repeating: 0, count: q * kv)
        let maskedCols = M.kvWindow - valid
        if maskedCols > 0 {
            for row in 0..<q {
                for col in 0..<maskedCols { mask[row * kv + col] = -.infinity }
            }
        }
        return .float16(mask, shape: [1, 1, q, kv])
    }

    private static func zeros16(_ shape: [Int]) -> TensorValue {
        .float16([Float16](repeating: 0, count: shape.reduce(1, *)), shape: shape)
    }

    private func value(_ d: [String: TensorValue], _ name: String) throws -> TensorValue {
        guard let v = d[name] else { throw KitNemotronError.outputMissing(name) }
        return v
    }
}
