// KitDiarizer.swift — streaming speaker diarization (NVIDIA Streaming Sortformer 4-spk v2,
// CC-BY-4.0) on Core AI: "who spoke when", up to 4 speakers, fully on-device.
//
// ```swift
// let diarizer = try await KitDiarizer(catalog: "sortformer-diar-v2")
// let turns = try await diarizer.diarize(samples: pcm16kMono)
// // turns == [SpeakerSegment(speaker: 0, 0.3–4.1 s), SpeakerSegment(speaker: 1, 4.6–6.3 s), …]
// ```
//
// The exported graph is the fixed-buffer forward_for_export core (mask-from-`valid`); the host does
// everything around it — a NeMo 128-mel frontend, the streaming chunk loop, and the AOSC speaker-
// cache compression. This is the coreai-audio app's device-verified Swift port of
// conversion/sortformer_diar/{host_loop,gate_e2e_engine}.py, itself a faithful re-impl of NeMo
// sortformer_modules.py (inference path: permute_spk=false). All stages are byte-gated vs NeMo
// (speaker-activity agreement 100% on a 21.5 s and a 64.5 s clip, on Mac GPU and iPhone 17 Pro).
//
// Fixed-buffer graph contract (see the bundle's metadata.json):
//   chunk_mel[1,1520,128] (host zero-pads each mel chunk) · spkcache[1,188,512] · valid[1,378]
//     -> preds[1,378,4] (sigmoid activity) · chunk_pe[1,190,512] (pre-encode embeddings)

import Accelerate
import CoreAIKitVision
import Foundation

/// One speaker turn: `speaker` (0..<4) is active over `[startSec, endSec)` (frames are 80 ms).
public struct SpeakerSegment: Sendable, Hashable {
    public let speaker: Int
    public let startFrame: Int
    public let endFrame: Int          // exclusive
    public var startSec: Double { Double(startFrame) * KitDiarizer.frameSec }
    public var endSec: Double { Double(endFrame) * KitDiarizer.frameSec }

    public init(speaker: Int, startFrame: Int, endFrame: Int) {
        self.speaker = speaker
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
}

/// Failures specific to the diarization path.
public enum KitDiarizerError: Error, LocalizedError {
    case melFiltersMissing
    case graphMissing
    case outputMissing(String)

    public var errorDescription: String? {
        switch self {
        case .melFiltersMissing:
            return "Bundled 128-mel filterbank resource is missing."
        case .graphMissing:
            return "Diarizer bundle is missing the Sortformer graph (.aimodel/.aimodelc)."
        case .outputMissing(let name):
            return "The Sortformer graph did not produce the expected output '\(name)'."
        }
    }
}

// MARK: - NeMo 128-mel frontend (normalize=NA)

/// Log-mel exactly like NeMo AudioToMelSpectrogramPreprocessor (preemph 0.97 -> stft n_fft=512 /
/// win=400 Hann centered / hop=160 -> librosa-slaney mel[128,257] -> log(mel+2⁻²⁴)), but with
/// `normalize: NA` (NO per-channel normalization — the Sortformer config) and a VARIABLE length
/// padded to a multiple of 16 frames (NeMo pad_to). Mirrors ParakeetMelPreprocessor (the same
/// NeMo family, minus the normalization — the filterbank is the identical bundled resource).
/// Output is mel-major `[128, frames]`.
struct SortformerMel: Sendable {
    static let nFFT = 512, winLength = 400, hop = 160, nMels = 128
    static let nFreq = nFFT / 2 + 1        // 257
    static let preemph: Float = 0.97
    static let logGuard = Float(exactly: pow(2.0, -24))!
    static let padTo = 16

    private let window: [Float]            // [nFFT] 400-pt Hann centered in 512
    private let cosMat: [Float]            // [nFreq, nFFT]
    private let sinMat: [Float]            // [nFreq, nFFT]
    private let melFilters: [Float]        // [nMels, nFreq] librosa-slaney, mel-major

    init(melFilters: [Float]) {
        precondition(melFilters.count == Self.nMels * Self.nFreq, "mel_filters must be [128,257]")
        self.melFilters = melFilters
        var win = [Float](repeating: 0, count: Self.nFFT)
        let offset = (Self.nFFT - Self.winLength) / 2
        for n in 0..<Self.winLength {
            win[offset + n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(Self.winLength - 1))
        }
        window = win
        var c = [Float](repeating: 0, count: Self.nFreq * Self.nFFT)
        var s = [Float](repeating: 0, count: Self.nFreq * Self.nFFT)
        for k in 0..<Self.nFreq {
            for n in 0..<Self.nFFT {
                let a = 2 * Float.pi * Float(k) * Float(n) / Float(Self.nFFT)
                c[k * Self.nFFT + n] = cos(a)
                s[k * Self.nFFT + n] = sin(a)
            }
        }
        cosMat = c; sinMat = s
    }

    /// The extractor built from CoreAIKit's bundled librosa-slaney filterbank. No download.
    static func bundled() throws -> SortformerMel {
        guard let url = Bundle.module.url(
            forResource: "parakeet_mel_filters_128x257", withExtension: "f32")
        else { throw KitDiarizerError.melFiltersMissing }
        let filters = try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return SortformerMel(melFilters: filters)
    }

    /// 16 kHz mono waveform -> (mel-major `[128, frames]`, frames). Frames = ⌈(L/hop + 1)/16⌉·16.
    func logMel(_ samples: [Float]) -> (mel: [Float], frames: Int) {
        let nFFT = Self.nFFT, hop = Self.hop, nFreq = Self.nFreq, nMels = Self.nMels
        let L = samples.count
        let rawFrames = L / hop + 1
        let frames = ((rawFrames + Self.padTo - 1) / Self.padTo) * Self.padTo
        let pad = nFFT / 2

        // preemphasis 0.97 (front-to-back so x[t-1] is the original sample).
        var y = [Float](repeating: 0, count: L)
        if L > 0 {
            y[0] = samples[0]
            for t in 1..<L { y[t] = samples[t] - Self.preemph * samples[t - 1] }
        }
        // center pad by n_fft/2 with REFLECT (reflect_101, edge not repeated) — matches NeMo's
        // torch.stft(center=True, pad_mode='reflect'). Trailing frames from pad_to stay zero.
        let needed = max(pad + L + pad, (frames - 1) * hop + nFFT)
        var padded = [Float](repeating: 0, count: needed)
        for t in 0..<L { padded[pad + t] = y[t] }
        if L > 1 {
            for k in 1...pad where pad - k >= 0 && k < L { padded[pad - k] = y[k] }           // left
            for k in 1...pad where pad + L - 1 + k < needed && L - 1 - k >= 0 {
                padded[pad + L - 1 + k] = y[L - 1 - k]                                          // right
            }
        }

        // window each hop into [nFFT, frames] (column = frame).
        var win = [Float](repeating: 0, count: nFFT * frames)
        for t in 0..<frames {
            let base = t * hop
            for n in 0..<nFFT { win[n * frames + t] = padded[base + n] * window[n] }
        }
        // re/im = cos/sin DFT basis @ windowed frames -> [nFreq, frames]; power = re²+im².
        var re = [Float](repeating: 0, count: nFreq * frames)
        var im = [Float](repeating: 0, count: nFreq * frames)
        vDSP_mmul(cosMat, 1, win, 1, &re, 1, vDSP_Length(nFreq), vDSP_Length(frames), vDSP_Length(nFFT))
        vDSP_mmul(sinMat, 1, win, 1, &im, 1, vDSP_Length(nFreq), vDSP_Length(frames), vDSP_Length(nFFT))
        let count = vDSP_Length(nFreq * frames)
        vDSP_vsq(re, 1, &re, 1, count); vDSP_vsq(im, 1, &im, 1, count)
        var power = [Float](repeating: 0, count: nFreq * frames)
        vDSP_vadd(re, 1, im, 1, &power, 1, count)
        // mel = mel_filters @ power -> [nMels, frames]; log(mel + 2⁻²⁴). normalize=NA -> no norm.
        var mel = [Float](repeating: 0, count: nMels * frames)
        vDSP_mmul(melFilters, 1, power, 1, &mel, 1, vDSP_Length(nMels), vDSP_Length(frames), vDSP_Length(nFreq))
        var guardV = Self.logGuard
        vDSP_vsadd(mel, 1, &guardV, &mel, 1, vDSP_Length(mel.count))
        var n32 = Int32(mel.count)
        vvlogf(&mel, mel, &n32)
        return (mel, frames)
    }
}

// MARK: - Streaming diarizer

/// Any speaker-diarization model in the catalog (`ModelCatalog.builtin.available(.diarization)`)
/// behind one `diarize(samples:)` call. Serial use (one clip at a time) — the host streaming
/// loop owns the speaker cache.
public actor KitDiarizer {
    // streaming params (metadata.json / NeMo model_config.yaml)
    static let spkcacheLen = 188, fifoLen = 0, chunkLen = 188
    static let leftCtx = 1, rightCtx = 1, sub = 8, updatePeriod = 188
    static let silPerSpk = 3
    static let scoresBoostLatest: Float = 0.05
    static let silThreshold: Float = 0.2, predScoreThreshold: Float = 0.25
    static let strongRate = 0.75, weakRate = 1.5, minPosRate = 0.5
    static let maxIndex = 99_999
    static let emb = 512, mel = 128
    // fixed-buffer graph shapes
    static let tfMax = 1520, spk = 188, peMax = 190, tDim = 378

    /// The maximum number of concurrent speakers the model tracks.
    public static let nSpk = 4
    /// Seconds per activity frame (8 subsample × 10 ms hop).
    public static let frameSec = 0.08
    /// Expected input sample rate.
    public static let sampleRate = 16000

    private let model: GraphModel
    private let melFront: SortformerMel

    /// The catalog id this diarizer was loaded from ("" when loaded from a local bundle).
    public let id: String

    // MARK: - Init

    /// Loads a diarization model by its catalog id — the id shown on the model's card.
    /// Downloads on first use (progress via `downloadProgress`), then loads from the local cache.
    public init(
        catalog id: String = "sortformer-diar-v2",
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .diarization)
        guard entry.id == "sortformer-diar-v2", let variant = entry.variant else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        let root = try await store.download(
            entry.modelID(path: variant.path), progress: downloadProgress)
        try await self.init(bundleAt: root, computeUnits: computeUnits, id: entry.id)
    }

    /// Loads a local Sortformer bundle — either the `.aimodel`/`.aimodelc` directory itself or a
    /// directory containing one (a JIT `.aimodel` and an AOT `.aimodelc` side by side resolve to
    /// the platform-native form). The mel filterbank ships inside CoreAIKit, so the bundle needs
    /// no extra asset.
    public init(
        bundleAt root: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        id: String = ""
    ) async throws {
        model = try await GraphModel(contentsOf: Self.resolveGraph(in: root), computeUnits: computeUnits)
        melFront = try SortformerMel.bundled()
        self.id = id
    }

    /// The graph inside `root` (or `root` itself). With both forms present the platform-native
    /// one wins: AOT `.aimodelc` on iOS (the device JIT is avoided), JIT `.aimodel` on macOS.
    private static func resolveGraph(in root: URL) throws -> URL {
        let ext = root.pathExtension
        if ext == "aimodel" || ext == "aimodelc" { return root }
        var graphs: [URL] = []
        if let it = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let url as URL in it
            where url.pathExtension == "aimodel" || url.pathExtension == "aimodelc" {
                graphs.append(url)
                it.skipDescendants()
            }
        }
        #if os(iOS)
        let preferred = "aimodelc"
        #else
        let preferred = "aimodel"
        #endif
        guard let graph = graphs.first(where: { $0.pathExtension == preferred }) ?? graphs.first
        else { throw KitDiarizerError.graphMissing }
        return graph
    }

    /// dw_striding output length: 3 stages of (l-1)/2 + 1 (k=3, stride=2, pad=1).
    private static func preEncodeLen(_ tf: Int) -> Int {
        var l = tf
        for _ in 0..<3 { l = (l - 1) / 2 + 1 }
        return l
    }

    // MARK: - Diarize

    /// Diarize a 16 kHz mono clip into speaker turns (dominant speaker per frame, contiguous runs
    /// merged; gaps ≤ `bridgeFrames` within one speaker bridged so a brief pause doesn't split a
    /// turn). Speakers are indexed 0..<4 in no particular order — stable within a clip.
    public func diarize(samples: [Float], bridgeFrames: Int = 6) async throws -> [SpeakerSegment] {
        Self.segments(from: try await framePreds(fromSamples: samples), bridgeFrames: bridgeFrames)
    }

    /// Full pipeline: 16 kHz mono -> per-frame speaker activity `[nOut][4]` (sigmoid). frame =
    /// 80 ms. Use this for a raw activity timeline; `diarize(samples:)` for speaker turns.
    public func framePreds(fromSamples samples: [Float]) async throws -> [[Float]] {
        let (m, frames) = melFront.logMel(samples)
        return try await framePreds(mel: m, melFrames: frames)
    }

    /// Drive the streaming loop over a mel-major `[128, melFrames]` buffer. This is the gated core
    /// (the self-gate feeds the golden mel here). Returns per-output-frame activity `[nOut][4]`.
    public func framePreds(mel: [Float], melFrames: Int) async throws -> [[Float]] {
        let E = Self.emb, S = Self.nSpk
        var spkcache = [Float]()          // [spkFrames * 512]
        var spkFrames = 0
        var spkcachePreds: [Float]? = nil // [spkFrames * 4]
        var meanSil = [Float](repeating: 0, count: E)
        var nSil = 0
        var total = [Float]()             // [nOut * 4]

        var stt = 0
        while stt < melFrames {
            let left = min(Self.leftCtx * Self.sub, stt)
            let end = min(stt + Self.chunkLen * Self.sub, melFrames)
            let right = min(Self.rightCtx * Self.sub, melFrames - end)
            let cstart = stt - left, cend = end + right, tf = cend - cstart

            // chunk_feat [tf, 128] frame-major from mel-major [128, melFrames]
            var chunkFeat = [Float](repeating: 0, count: tf * Self.mel)
            for f in 0..<tf {
                let col = cstart + f
                for mi in 0..<Self.mel { chunkFeat[f * Self.mel + mi] = mel[mi * melFrames + col] }
            }

            let (predsConcat, chunkPe, peLen) =
                try await engineForward(chunkFeat: chunkFeat, tf: tf, spkcache: spkcache, spkLen: spkFrames)
            let Sc = spkFrames
            let lc = Int((Double(left) / Double(Self.sub)).rounded())
            let rc = Int((Double(right) / Double(Self.sub)).rounded(.up))
            let chunkFramesN = peLen - lc - rc

            // chunk_emb = chunk_pe[lc : lc+chunkFramesN]; chunk_preds = preds[Sc+lc : Sc+lc+chunkFramesN]
            var popEmbs = [Float](repeating: 0, count: chunkFramesN * E)
            for f in 0..<chunkFramesN {
                let src = (lc + f) * E
                for d in 0..<E { popEmbs[f * E + d] = chunkPe[src + d] }
            }
            var popPreds = [Float](repeating: 0, count: chunkFramesN * S)
            for f in 0..<chunkFramesN {
                let src = (Sc + lc + f) * S
                for s in 0..<S { popPreds[f * S + s] = predsConcat[src + s] }
            }

            // silence profile over the popped frames (FIFO_LEN=0 -> pop == this chunk)
            (meanSil, nSil) = Self.silenceProfile(meanSil, nSil, popEmbs, popPreds, frames: chunkFramesN)
            spkcache.append(contentsOf: popEmbs); spkFrames += chunkFramesN
            if spkcachePreds != nil { spkcachePreds!.append(contentsOf: popPreds) }

            if spkFrames > Self.spkcacheLen {
                if spkcachePreds == nil {
                    // spkcache_preds = cat(preds[:Sc], pop_preds) — Sc spkcache-region rows from this run
                    var sp = [Float](repeating: 0, count: Sc * S)
                    for f in 0..<Sc { for s in 0..<S { sp[f * S + s] = predsConcat[f * S + s] } }
                    sp.append(contentsOf: popPreds)
                    spkcachePreds = sp
                }
                let (ne, np) = Self.compressSpkcache(
                    embSeq: spkcache, preds: spkcachePreds!, frames: spkFrames, meanSil: meanSil)
                spkcache = ne; spkcachePreds = np; spkFrames = Self.spkcacheLen
            }
            total.append(contentsOf: popPreds)   // total_preds = accumulated chunk_preds
            stt = end
        }

        let nOut = total.count / S
        var out = [[Float]](repeating: [Float](repeating: 0, count: S), count: nOut)
        for f in 0..<nOut { for s in 0..<S { out[f][s] = total[f * S + s] } }
        return out
    }

    // MARK: - fixed-buffer graph driving (engine_forward)

    private func engineForward(chunkFeat: [Float], tf: Int, spkcache: [Float], spkLen: Int)
        async throws -> (predsConcat: [Float], chunkPe: [Float], peLen: Int)
    {
        let S = Self.nSpk, E = Self.emb
        let peLen = Self.preEncodeLen(tf)

        var chunkMel = [Float](repeating: 0, count: Self.tfMax * Self.mel)
        for i in 0..<min(chunkFeat.count, chunkMel.count) { chunkMel[i] = chunkFeat[i] }
        var spk188 = [Float](repeating: 0, count: Self.spk * E)
        for i in 0..<min(spkcache.count, spk188.count) { spk188[i] = spkcache[i] }
        var valid = [Float](repeating: 0, count: Self.tDim)
        for i in 0..<spkLen { valid[i] = 1 }
        for i in 0..<peLen { valid[Self.spk + i] = 1 }

        let out = try await model.run([
            "chunk_mel": .float32(chunkMel, shape: [1, Self.tfMax, Self.mel]),
            "spkcache": .float32(spk188, shape: [1, Self.spk, E]),
            "valid": .float32(valid, shape: [1, Self.tDim]),
        ])
        guard let predsOut = out["preds"] else { throw KitDiarizerError.outputMissing("preds") }
        guard let peOut = out["chunk_pe"] else { throw KitDiarizerError.outputMissing("chunk_pe") }
        let predsFixed = predsOut.floats()        // [378 * 4]
        let chunkPeFixed = peOut.floats()          // [190 * 512]

        // concat layout: [spkcache-region (spkLen) | chunk-region (peLen)]
        var predsConcat = [Float](repeating: 0, count: (spkLen + peLen) * S)
        for f in 0..<spkLen { for s in 0..<S { predsConcat[f * S + s] = predsFixed[f * S + s] } }
        for f in 0..<peLen {
            for s in 0..<S { predsConcat[(spkLen + f) * S + s] = predsFixed[(Self.spk + f) * S + s] }
        }
        let chunkPe = Array(chunkPeFixed[0..<peLen * E])
        return (predsConcat, chunkPe, peLen)
    }

    // MARK: - AOSC (NeMo sortformer_modules.py, inference path — 1:1)

    private static func silenceProfile(
        _ meanSil: [Float], _ nSil: Int, _ embSeq: [Float], _ preds: [Float], frames F: Int
    ) -> ([Float], Int) {
        let E = emb, S = nSpk
        var silCount = 0
        var silSum = [Float](repeating: 0, count: E)
        for f in 0..<F {
            var psum: Float = 0
            for s in 0..<S { psum += preds[f * S + s] }
            if psum < silThreshold {
                silCount += 1
                for d in 0..<E { silSum[d] += embSeq[f * E + d] }
            }
        }
        if silCount == 0 { return (meanSil, nSil) }
        let updN = nSil + silCount
        var out = [Float](repeating: 0, count: E)
        let denom = Float(max(updN, 1))
        for d in 0..<E { out[d] = (meanSil[d] * Float(nSil) + silSum[d]) / denom }
        return (out, updN)
    }

    private static func logPredScores(_ preds: [Float], frames F: Int) -> [Float] {
        let S = nSpk
        var scores = [Float](repeating: 0, count: F * S)
        let logHalf = Float(log(0.5))
        for f in 0..<F {
            var l1sum: Float = 0
            for s in 0..<S { l1sum += log(max(1 - preds[f * S + s], predScoreThreshold)) }
            for s in 0..<S {
                let lp = log(max(preds[f * S + s], predScoreThreshold))
                let l1 = log(max(1 - preds[f * S + s], predScoreThreshold))
                scores[f * S + s] = lp - l1 + l1sum - logHalf
            }
        }
        return scores
    }

    private static func disableLowScores(_ preds: [Float], _ scores: [Float], frames F: Int, minPos: Int) -> [Float] {
        let S = nSpk
        let ninf = -Float.infinity
        var out = scores
        // non-speech -> -inf
        for f in 0..<F { for s in 0..<S where !(preds[f * S + s] > 0.5) { out[f * S + s] = ninf } }
        // per-speaker count of positive scores
        var posCount = [Int](repeating: 0, count: S)
        for f in 0..<F { for s in 0..<S where out[f * S + s] > 0 { posCount[s] += 1 } }
        for f in 0..<F {
            for s in 0..<S {
                let isSpeech = preds[f * S + s] > 0.5
                let isPos = out[f * S + s] > 0
                if !isPos && isSpeech && posCount[s] >= minPos { out[f * S + s] = ninf }
            }
        }
        return out
    }

    /// Boost the `nBoost` highest scores per speaker by `-scale·log(0.5)` (in place, per column).
    private static func boostTopk(_ scores: inout [Float], frames F: Int, nBoost: Int, scale: Float) {
        let S = nSpk
        let add = -scale * Float(log(0.5))
        let k = min(nBoost, F)
        if k <= 0 { return }
        for s in 0..<S {
            // indices of the k largest scores in column s
            var idx = Array(0..<F)
            idx.sort { scores[$0 * S + s] > scores[$1 * S + s] }
            for i in 0..<k { scores[idx[i] * S + s] += add }
        }
    }

    /// NeMo _get_topk_indices: `scores` is the PADDED [(F+SIL)·S] buffer; n_frames := F+SIL,
    /// n_frames_no_sil := F. Returns spkcacheLen frame indices (into [0,F)) + a disabled mask.
    private static func topkIndices(_ scores: [Float], paddedFrames Fp: Int)
        -> (idx: [Int], disabled: [Bool])
    {
        let S = nSpk, K = spkcacheLen
        let nNoSil = Fp - silPerSpk
        // flat[s*Fp + f] = scores[f*S + s]   (permute(0,2,1).reshape)
        var flat = [Float](repeating: 0, count: S * Fp)
        for s in 0..<S { for f in 0..<Fp { flat[s * Fp + f] = scores[f * S + s] } }
        // K largest flat indices (values that matter; ties among -inf all map to maxIndex anyway)
        var order = Array(0..<(S * Fp))
        order.sort { flat[$0] > flat[$1] }
        var top = Array(order[0..<K])
        for i in 0..<K where flat[top[i]] == -Float.infinity { top[i] = maxIndex }
        top.sort()
        var idx = [Int](repeating: 0, count: K)
        var disabled = [Bool](repeating: false, count: K)
        for i in 0..<K {
            let wasMax = top[i] == maxIndex
            let rem = wasMax ? 0 : top[i] % Fp
            let dis = wasMax || rem >= nNoSil
            disabled[i] = dis
            idx[i] = dis ? 0 : rem
        }
        return (idx, disabled)
    }

    private static func compressSpkcache(embSeq: [Float], preds: [Float], frames F: Int, meanSil: [Float])
        -> (emb: [Float], preds: [Float])
    {
        let S = nSpk, E = emb, K = spkcacheLen
        let perSpk = K / S - silPerSpk
        let strong = Int((Double(perSpk) * strongRate).rounded(.down))
        let weak = Int((Double(perSpk) * weakRate).rounded(.down))
        let minPos = Int((Double(perSpk) * minPosRate).rounded(.down))

        var scores = logPredScores(preds, frames: F)
        scores = disableLowScores(preds, scores, frames: F, minPos: minPos)
        if scoresBoostLatest > 0 {
            for f in spkcacheLen..<F { for s in 0..<S { scores[f * S + s] += scoresBoostLatest } }
        }
        boostTopk(&scores, frames: F, nBoost: strong, scale: 2)
        boostTopk(&scores, frames: F, nBoost: weak, scale: 1)

        // append SIL_PER_SPK rows of +inf -> padded [(F+SIL) * S]
        let Fp = F + silPerSpk
        var padded = scores
        padded.append(contentsOf: [Float](repeating: .infinity, count: silPerSpk * S))
        let (idx, disabled) = topkIndices(padded, paddedFrames: Fp)

        // gather: disabled -> mean_sil emb / zero preds; else frame `idx[i]`
        var ne = [Float](repeating: 0, count: K * E)
        var np = [Float](repeating: 0, count: K * S)
        for i in 0..<K {
            if disabled[i] {
                for d in 0..<E { ne[i * E + d] = meanSil[d] }
            } else {
                let g = idx[i]
                for d in 0..<E { ne[i * E + d] = embSeq[g * E + d] }
                for s in 0..<S { np[i * S + s] = preds[g * S + s] }
            }
        }
        return (ne, np)
    }

    // MARK: - segmentation

    /// Per-frame -> speaker turns. A frame is assigned to the strongest active speaker (activity>0.5,
    /// else silence); contiguous same-speaker frames merge into a turn, and gaps ≤ `bridgeFrames`
    /// within one speaker are bridged so a brief pause doesn't split a turn.
    public static func segments(from frames: [[Float]], bridgeFrames: Int = 6) -> [SpeakerSegment] {
        var lab = [Int](repeating: -1, count: frames.count)      // -1 = silence
        for (f, act) in frames.enumerated() {
            var best = -1; var bestV: Float = 0.5
            for s in 0..<act.count where act[s] > bestV { bestV = act[s]; best = s }
            lab[f] = best
        }
        // build raw runs
        var segs = [SpeakerSegment]()
        var i = 0
        while i < lab.count {
            let s = lab[i]
            if s < 0 { i += 1; continue }
            var j = i + 1
            while j < lab.count && lab[j] == s { j += 1 }
            segs.append(SpeakerSegment(speaker: s, startFrame: i, endFrame: j))
            i = j
        }
        // bridge short gaps between same-speaker turns
        var merged = [SpeakerSegment]()
        for seg in segs {
            if let last = merged.last, last.speaker == seg.speaker,
               seg.startFrame - last.endFrame <= bridgeFrames {
                merged[merged.count - 1] = SpeakerSegment(
                    speaker: seg.speaker, startFrame: last.startFrame, endFrame: seg.endFrame)
            } else {
                merged.append(seg)
            }
        }
        return merged
    }
}
