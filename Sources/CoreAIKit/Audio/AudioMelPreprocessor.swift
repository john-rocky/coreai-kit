// AudioMelPreprocessor.swift — Whisper-large-v3 log-mel frontend in Accelerate, matching
// transformers' `WhisperFeatureExtractor` (the front end of Qwen2.5-Omni's audio tower) so a raw
// 16 kHz waveform becomes the encoder's `input_features`.
//
// Recipe (torch.stft center=True + [:-1], then Whisper log-normalize):
//   1. reflect-pad the waveform by n_fft/2 (200) on each side
//   2. per hop (160): Hann(400)-window 400 samples, real DFT -> 201 power bins
//   3. mel = mel_filters^T @ power            (mel_filters [201,128] from the HF extractor)
//   4. log10(max(mel,1e-10)); max(., globalMax-8); (.+4)/4
//
// n_fft=400 is not a power of two, so the per-frame DFT is a precomputed cos/sin MATMUL (exact,
// BLAS-fast) rather than vDSP's radix-2 FFT. The mel filterbank is supplied by the caller (dumped
// once from the HF extractor) so we never re-derive librosa's mel matrix.

import Accelerate
import Foundation

/// Stateless Whisper log-mel extractor. Build once (precomputes the DFT/window tables) and reuse.
public struct AudioMelPreprocessor: Sendable {
    public let nFFT: Int
    public let hop: Int
    public let nMels: Int
    public let nFreq: Int  // nFFT/2 + 1

    private let window: [Float]        // Hann(nFFT), periodic
    private let cosMat: [Float]        // [nFreq, nFFT]
    private let sinMat: [Float]        // [nFreq, nFFT]
    private let melFiltersT: [Float]   // [nMels, nFreq]  (transpose of the [nFreq,nMels] dump)

    /// `melFilters` is the HF extractor's `mel_filters`, row-major `[nFreq, nMels]` (n_fft/2+1 by
    /// feature_size — e.g. `[201, 128]`).
    public init(melFilters: [Float], nFFT: Int = 400, hop: Int = 160, nMels: Int = 128) {
        precondition(melFilters.count == (nFFT / 2 + 1) * nMels, "mel_filters shape mismatch")
        self.nFFT = nFFT
        self.hop = hop
        self.nMels = nMels
        let nFreq = nFFT / 2 + 1
        self.nFreq = nFreq

        // Hann window, periodic (torch.hann_window default): 0.5 - 0.5 cos(2*pi*n/N).
        var win = [Float](repeating: 0, count: nFFT)
        for n in 0..<nFFT { win[n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(nFFT)) }
        self.window = win

        // Real-DFT basis: re[k] = sum_n x[n] cos(2*pi*k*n/N), im[k] = -sum_n x[n] sin(...).
        var c = [Float](repeating: 0, count: nFreq * nFFT)
        var s = [Float](repeating: 0, count: nFreq * nFFT)
        for k in 0..<nFreq {
            for n in 0..<nFFT {
                let a = 2 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                c[k * nFFT + n] = cos(a)
                s[k * nFFT + n] = sin(a)
            }
        }
        self.cosMat = c
        self.sinMat = s

        // Transpose mel_filters [nFreq,nMels] -> [nMels,nFreq] for mel = filters^T @ power.
        var mt = [Float](repeating: 0, count: nMels * nFreq)
        for f in 0..<nFreq {
            for m in 0..<nMels { mt[m * nFreq + f] = melFilters[f * nMels + m] }
        }
        self.melFiltersT = mt
    }

    /// The Qwen2.5-Omni / Whisper-large-v3 extractor, built from CoreAIKit's bundled mel
    /// filterbank (n_fft 400, hop 160, 128 mel). No external file or download needed.
    public static func qwen2_5Omni() throws -> AudioMelPreprocessor {
        guard let url = Bundle.module.url(forResource: "mel_filters", withExtension: "f32") else {
            throw KitAudioError.melFiltersMissing
        }
        let filters = try Data(contentsOf: url).withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        return AudioMelPreprocessor(melFilters: filters)
    }

    /// The Whisper-large-v3 extractor used by `WhisperRuntime`. Same filterbank as `qwen2_5Omni()`
    /// — CoreAIKit's bundled `mel_filters.f32` is bit-exact with the HF `mel_filters_128.npy`.
    public static func whisperLargeV3() throws -> AudioMelPreprocessor { try qwen2_5Omni() }

    /// Log-mel for a 16 kHz mono waveform. Returns `mel` row-major `[nMels, frames]` and `frames`
    /// (= `samples.count / hop`, matching torch.stft center=True then dropping the trailing frame).
    public func logMel(_ samples: [Float]) -> (mel: [Float], frames: Int) {
        let frames = samples.count / hop
        guard frames > 0 else { return ([], 0) }

        // center reflect-pad by nFFT/2 (no edge repeat), then window each hop into [nFFT, frames].
        let pad = nFFT / 2
        let padded = reflectPad(samples, pad: pad)
        var win = [Float](repeating: 0, count: nFFT * frames)  // [nFFT, frames], column = frame
        for t in 0..<frames {
            let base = t * hop
            for n in 0..<nFFT { win[n * frames + t] = padded[base + n] * window[n] }
        }

        // re/im = cos/sin DFT basis @ windowed frames -> [nFreq, frames]; power = re^2 + im^2.
        var re = [Float](repeating: 0, count: nFreq * frames)
        var im = [Float](repeating: 0, count: nFreq * frames)
        matmul(cosMat, win, &re, m: nFreq, n: frames, k: nFFT)
        matmul(sinMat, win, &im, m: nFreq, n: frames, k: nFFT)
        let count = vDSP_Length(nFreq * frames)
        vDSP_vsq(re, 1, &re, 1, count)
        vDSP_vsq(im, 1, &im, 1, count)
        var power = [Float](repeating: 0, count: nFreq * frames)
        vDSP_vadd(re, 1, im, 1, &power, 1, count)

        // mel = mel_filters^T @ power -> [nMels, frames].
        var mel = [Float](repeating: 0, count: nMels * frames)
        matmul(melFiltersT, power, &mel, m: nMels, n: frames, k: nFreq)

        // Whisper log-normalize: log10(max(.,1e-10)); max(., globalMax-8); (.+4)/4.
        // vDSP_vthr(A,_,thr,C,_,N) = max(A, thr) elementwise.
        let melCount = vDSP_Length(mel.count)
        var floorV: Float = 1e-10
        vDSP_vthr(mel, 1, &floorV, &mel, 1, melCount)
        var n32 = Int32(mel.count)
        vvlog10f(&mel, mel, &n32)
        var globalMax: Float = -.greatestFiniteMagnitude
        vDSP_maxv(mel, 1, &globalMax, melCount)
        var clampLow = globalMax - 8.0
        vDSP_vthr(mel, 1, &clampLow, &mel, 1, melCount)
        var add: Float = 4.0
        var div: Float = 4.0
        vDSP_vsadd(mel, 1, &add, &mel, 1, melCount)
        vDSP_vsdiv(mel, 1, &div, &mel, 1, melCount)
        return (mel, frames)
    }

    // MARK: - helpers

    /// C[m,n] = A[m,k] · B[k,n], all row-major (Accelerate single-precision matmul).
    private func matmul(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int) {
        vDSP_mmul(
            a, 1, b, 1, &c, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    }

    /// Reflect padding without repeating the edge sample (numpy 'reflect' / torch.stft center).
    private func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad { out[i] = x[pad - i] }                 // x[pad..1]
        for i in 0..<n { out[pad + i] = x[i] }
        for i in 0..<pad { out[pad + n + i] = x[n - 2 - i] }     // x[n-2..n-1-pad]
        return out
    }
}
