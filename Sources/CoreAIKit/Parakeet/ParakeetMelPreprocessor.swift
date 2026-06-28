// ParakeetMelPreprocessor.swift — log-mel frontend for NVIDIA Parakeet TDT, in Accelerate, matching
// transformers' `ParakeetFeatureExtractor` so a raw 16 kHz waveform becomes the FastConformer
// encoder's `mel [1, 128, 2885]` input. Validated token-exact end-to-end on Core AI against the HF
// reference (conversion/parakeet/gate_mel_swift.py, recipe "S").
//
// Recipe (preemphasis -> torch.stft-equivalent -> librosa-slaney mel -> log -> per-utterance norm):
//   1. pad/trim the waveform to the fixed encoder bucket (2885 frames = 461600 samples @ 16 kHz)
//   2. preemphasis 0.97:  y[0]=x[0];  y[t]=x[t] - 0.97·x[t-1]
//   3. STFT n_fft=512, win=400 (Hann periodic=false, centered in 512), hop=160, center pad=zeros
//   4. power = |stft|²  -> [257, frames]
//   5. mel = mel_filters[128,257] @ power            (librosa-slaney; the bundled .f32)
//   6. log(mel + 2⁻²⁴)
//   7. per-mel-channel zero-mean unit-variance normalization (unbiased, over the bucket frames)
//
// Unlike the Whisper frontend this normalizes (ParakeetFeatureExtractor always does, despite the
// `do_normalize` arg), pads the AUDIO (not the mel) to the bucket — the trailing-silence frames are
// normalized in place and the decoder emits blanks over them — and uses n_fft 512 ≠ win 400, so the
// 400-sample Hann sits centered inside the 512-point DFT (56 zero samples each side).

import Accelerate
import Foundation

/// Stateless Parakeet log-mel extractor. Build once (precomputes the DFT/window/mel tables) and reuse.
public struct ParakeetMelPreprocessor: Sendable {
    public static let nFFT = 512
    public static let winLength = 400
    public static let hop = 160
    public static let nMels = 128
    public static let nFreq = nFFT / 2 + 1      // 257
    public static let preemphasis: Float = 0.97
    public static let logGuard = Float(exactly: pow(2.0, -24))!   // ≈ 5.96e-8
    public static let normEpsilon: Float = 1e-5

    /// Number of mel frames the encoder bucket expects (30 s clips bucket to ~28.84 s here).
    public let bucketFrames: Int
    private let bucketSamples: Int              // bucketFrames * hop

    private let window: [Float]                 // [nFFT], 400-pt Hann centered in 512
    private let cosMat: [Float]                 // [nFreq, nFFT]
    private let sinMat: [Float]                 // [nFreq, nFFT]
    private let melFilters: [Float]             // [nMels, nFreq]  (librosa-slaney, mel-major)

    /// `melFilters` is the librosa-slaney filterbank, row-major `[nMels, nFreq]` = `[128, 257]`.
    public init(melFilters: [Float], bucketFrames: Int = 2885) {
        precondition(
            melFilters.count == Self.nMels * Self.nFreq, "mel_filters must be [128, 257] row-major")
        self.bucketFrames = bucketFrames
        self.bucketSamples = bucketFrames * Self.hop
        self.melFilters = melFilters

        // Hann(win_length, periodic=false) = 0.5 - 0.5·cos(2πn/(N-1)), centered inside the n_fft
        // window with (n_fft - win_length)/2 zero samples on each side (torch.stft win padding).
        var win = [Float](repeating: 0, count: Self.nFFT)
        let offset = (Self.nFFT - Self.winLength) / 2
        for n in 0..<Self.winLength {
            win[offset + n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(Self.winLength - 1))
        }
        self.window = win

        // Real-DFT basis: re[k]=Σ x[n]·cos(2πkn/N), im[k]=Σ x[n]·sin(2πkn/N); power = re²+im².
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

    /// The default extractor built from CoreAIKit's bundled librosa-slaney filterbank. No download.
    public static func parakeet(bucketFrames: Int = 2885) throws -> ParakeetMelPreprocessor {
        guard let url = Bundle.module.url(
            forResource: "parakeet_mel_filters_128x257", withExtension: "f32")
        else { throw KitParakeetError.melFiltersMissing }
        let filters = try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return ParakeetMelPreprocessor(melFilters: filters, bucketFrames: bucketFrames)
    }

    /// Log-mel for a 16 kHz mono waveform, padded/trimmed to the encoder bucket. Returns the encoder
    /// input row-major `[nMels, bucketFrames]` (mel-major) — ready for `mel [1, 128, bucketFrames]`.
    public func logMel(_ samples: [Float]) -> [Float] {
        let frames = bucketFrames
        let nFFT = Self.nFFT, hop = Self.hop, nFreq = Self.nFreq, nMels = Self.nMels

        // 1. pad/trim the waveform to exactly bucketSamples (silence-pad short clips).
        var x = samples
        if x.count < bucketSamples {
            x.append(contentsOf: repeatElement(0, count: bucketSamples - x.count))
        } else if x.count > bucketSamples {
            x = Array(x[0..<bucketSamples])
        }

        // 2. preemphasis 0.97 (in place, front to back so x[t-1] is the original sample).
        var y = [Float](repeating: 0, count: bucketSamples)
        y[0] = x[0]
        for t in 1..<bucketSamples { y[t] = x[t] - Self.preemphasis * x[t - 1] }

        // 3. center zero-pad by nFFT/2, then window each hop into [nFFT, frames] (column = frame).
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: bucketSamples + 2 * pad)
        for i in 0..<bucketSamples { padded[pad + i] = y[i] }
        var win = [Float](repeating: 0, count: nFFT * frames)
        for t in 0..<frames {
            let base = t * hop
            for n in 0..<nFFT { win[n * frames + t] = padded[base + n] * window[n] }
        }

        // 4. re/im = cos/sin DFT basis @ windowed frames -> [nFreq, frames]; power = re²+im².
        var re = [Float](repeating: 0, count: nFreq * frames)
        var im = [Float](repeating: 0, count: nFreq * frames)
        matmul(cosMat, win, &re, m: nFreq, n: frames, k: nFFT)
        matmul(sinMat, win, &im, m: nFreq, n: frames, k: nFFT)
        let count = vDSP_Length(nFreq * frames)
        vDSP_vsq(re, 1, &re, 1, count)
        vDSP_vsq(im, 1, &im, 1, count)
        var power = [Float](repeating: 0, count: nFreq * frames)
        vDSP_vadd(re, 1, im, 1, &power, 1, count)

        // 5. mel = mel_filters @ power -> [nMels, frames].  6. log(mel + 2⁻²⁴).
        var mel = [Float](repeating: 0, count: nMels * frames)
        matmul(melFilters, power, &mel, m: nMels, n: frames, k: nFreq)
        var guardV = Self.logGuard
        vDSP_vsadd(mel, 1, &guardV, &mel, 1, vDSP_Length(mel.count))
        var n32 = Int32(mel.count)
        vvlogf(&mel, mel, &n32)

        // 7. per-mel-channel zero-mean unit-variance norm (unbiased, std + EPSILON), over the bucket.
        let denom = Float(frames - 1)
        mel.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for m in 0..<nMels {
                let ch = p + m * frames
                var mean: Float = 0
                vDSP_meanv(ch, 1, &mean, vDSP_Length(frames))
                var negMean = -mean
                vDSP_vsadd(ch, 1, &negMean, ch, 1, vDSP_Length(frames))   // ch -= mean
                var sumsq: Float = 0
                vDSP_svesq(ch, 1, &sumsq, vDSP_Length(frames))            // Σ(x-mean)²
                var invStd = 1 / ((sumsq / denom).squareRoot() + Self.normEpsilon)
                vDSP_vsmul(ch, 1, &invStd, ch, 1, vDSP_Length(frames))    // ch *= 1/std
            }
        }
        return mel
    }

    /// C[m,n] = A[m,k] · B[k,n], all row-major (Accelerate single-precision matmul).
    private func matmul(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int) {
        vDSP_mmul(a, 1, b, 1, &c, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
    }
}
