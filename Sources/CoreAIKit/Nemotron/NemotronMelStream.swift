// NemotronMelStream.swift — incremental log-mel frontend for Nemotron 3.5 ASR streaming, in
// Accelerate, matching transformers' `NemotronAsrStreamingFeatureExtractor`. Push raw 16 kHz
// samples in ANY packet sizes; pop fixed mel chunks (25 frames first, then 32 each) sized for the
// cache-aware streaming encoder at lookahead 3. Validated token-exact end-to-end against the HF
// reference (conversion/nemotron_asr/gate_mel_swift_streaming.py).
//
// Recipe (differs from Parakeet: NO normalization):
//   1. preemphasis 0.97, continuous across the stream:  y[0]=x[0];  y[t]=x[t] - 0.97·x[t-1]
//   2. STFT n_fft=512, win=400 (Hann periodic=false, centered in 512), hop=160
//   3. power = |stft|²  ->  mel = librosa-slaney[128,257] @ power  ->  log(mel + 2⁻²⁴)
//
// Streaming form: mel frame t depends ONLY on samples [160t-200, 160t+200) — the 56-zero margins
// of the padded window absorb both the stream-start zero pad (HF center=True on the first chunk)
// and the per-chunk preemphasis boundary (HF center=False afterwards) — so frame t is emitted as
// soon as sample 160t+200 arrives, independent of packet size.

import Accelerate
import Foundation

/// Stateful streaming log-mel extractor for one audio stream. Not thread-safe — feed from one task.
final class NemotronMelStream {
    static let nFFT = 512
    static let winLength = 400
    static let hop = 160
    static let nMels = 128
    static let nFreq = nFFT / 2 + 1                 // 257
    static let preemphasis: Float = 0.97
    static let logGuard = Float(exactly: pow(2.0, -24))!

    /// Chunk protocol at lookahead 3 (`_required_stream_chunk_frames`): 25 mel first, then 32.
    static let firstChunkFrames = 25
    static let chunkFrames = 32

    private let window: [Float]                     // [nFFT], 400-pt Hann centered in 512
    private let cosMat: [Float]                     // [nFreq, nFFT]
    private let sinMat: [Float]                     // [nFreq, nFFT]
    private let melFilters: [Float]                 // [nMels, nFreq]

    private var buffer: [Float] = []                // preemphasized samples from `bufferBase` on
    private var bufferBase = 0                      // absolute index of buffer[0] in the stream
    private var prevSample: Float?                  // last raw sample (preemphasis continuity)
    private var nextFrame = 0                       // next mel frame index to compute
    private var pending: [[Float]] = []             // frames awaiting chunk assembly
    private(set) var chunksEmitted = 0
    /// Total mel frames handed out in completed chunks + pending.
    var framesComputed: Int { nextFrame }

    /// `melFilters` is the librosa-slaney filterbank, row-major `[128, 257]` (Parakeet's bundled
    /// table — same sample rate / n_fft / fmax, shared between the two frontends).
    init(melFilters: [Float]) {
        precondition(melFilters.count == Self.nMels * Self.nFreq, "mel_filters must be [128, 257]")
        self.melFilters = melFilters

        var win = [Float](repeating: 0, count: Self.nFFT)
        let offset = (Self.nFFT - Self.winLength) / 2
        for n in 0..<Self.winLength {
            win[offset + n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(Self.winLength - 1))
        }
        self.window = win

        var c = [Float](repeating: 0, count: Self.nFreq * Self.nFFT)
        var s = [Float](repeating: 0, count: Self.nFreq * Self.nFFT)
        for k in 0..<Self.nFreq {
            for n in 0..<Self.nFFT {
                let a = 2 * Float.pi * Float(k) * Float(n) / Float(Self.nFFT)
                c[k * Self.nFFT + n] = cos(a)
                s[k * Self.nFFT + n] = sin(a)
            }
        }
        self.cosMat = c
        self.sinMat = s
    }

    /// Push raw samples; returns zero or more completed mel chunks, each row-major
    /// `[frames, 128]` (frame-major — the encoder graphs take `mel [1, L, 128]`).
    func push(_ samples: [Float]) -> [[Float]] {
        guard !samples.isEmpty else { return [] }
        // 1. preemphasis (continuous: the first sample uses the previous packet's last sample).
        var y = [Float](repeating: 0, count: samples.count)
        y[0] = prevSample.map { samples[0] - Self.preemphasis * $0 } ?? samples[0]
        for t in 1..<samples.count { y[t] = samples[t] - Self.preemphasis * samples[t - 1] }
        prevSample = samples[samples.count - 1]
        buffer.append(contentsOf: y)

        // 2. compute every frame whose window support [160t-200, 160t+200) has fully arrived.
        let total = bufferBase + buffer.count
        while Self.hop * nextFrame + Self.winLength / 2 <= total {
            pending.append(computeFrame(nextFrame))
            nextFrame += 1
        }
        // drop samples no frame will need again (keep from 160·nextFrame - 256).
        let keepFrom = max(0, Self.hop * nextFrame - Self.nFFT / 2)
        if keepFrom > bufferBase {
            buffer.removeFirst(keepFrom - bufferBase)
            bufferBase = keepFrom
        }

        // 3. assemble fixed-size chunks (25 first, then 32).
        var out: [[Float]] = []
        while pending.count >= (chunksEmitted == 0 ? Self.firstChunkFrames : Self.chunkFrames) {
            let n = chunksEmitted == 0 ? Self.firstChunkFrames : Self.chunkFrames
            var chunk = [Float](repeating: 0, count: n * Self.nMels)
            for (i, f) in pending.prefix(n).enumerated() {
                chunk.replaceSubrange(i * Self.nMels ..< (i + 1) * Self.nMels, with: f)
            }
            pending.removeFirst(n)
            chunksEmitted += 1
            out.append(chunk)
        }
        return out
    }

    /// One log-mel frame [128] for absolute frame index `t` (window [160t-256, 160t+256), zeros
    /// before the stream start; callers guarantee the right edge has arrived).
    private func computeFrame(_ t: Int) -> [Float] {
        let nFFT = Self.nFFT, nFreq = Self.nFreq, nMels = Self.nMels
        let lo = Self.hop * t - nFFT / 2
        var seg = [Float](repeating: 0, count: nFFT)
        let srcLo = max(lo, 0)
        let start = srcLo - bufferBase
        let count = min(lo + nFFT, bufferBase + buffer.count) - srcLo
        if count > 0 {
            seg.replaceSubrange(srcLo - lo ..< srcLo - lo + count, with: buffer[start ..< start + count])
        }
        var w = [Float](repeating: 0, count: nFFT)
        vDSP_vmul(seg, 1, window, 1, &w, 1, vDSP_Length(nFFT))

        var re = [Float](repeating: 0, count: nFreq)
        var im = [Float](repeating: 0, count: nFreq)
        vDSP_mmul(cosMat, 1, w, 1, &re, 1, vDSP_Length(nFreq), 1, vDSP_Length(nFFT))
        vDSP_mmul(sinMat, 1, w, 1, &im, 1, vDSP_Length(nFreq), 1, vDSP_Length(nFFT))
        var power = [Float](repeating: 0, count: nFreq)
        vDSP_vsq(re, 1, &re, 1, vDSP_Length(nFreq))
        vDSP_vsq(im, 1, &im, 1, vDSP_Length(nFreq))
        vDSP_vadd(re, 1, im, 1, &power, 1, vDSP_Length(nFreq))

        var mel = [Float](repeating: 0, count: nMels)
        vDSP_mmul(melFilters, 1, power, 1, &mel, 1, vDSP_Length(nMels), 1, vDSP_Length(nFreq))
        var guardV = Self.logGuard
        vDSP_vsadd(mel, 1, &guardV, &mel, 1, vDSP_Length(nMels))
        var n32 = Int32(nMels)
        vvlogf(&mel, mel, &n32)
        return mel
    }
}
