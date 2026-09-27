// FunASRFbankPreprocessor.swift — Fun-ASR-Nano host front end in Accelerate: a 16 kHz mono waveform
// becomes the LFR features `[L, 560]` the SAN-M audio encoder reads. A line-for-line port of the
// zoo's NumPy spec (conversion/funasr_nano/frontend.py), which reproduces funasr 1.4.16's
// `WavFrontend` for the `frontend_conf` Fun-ASR-Nano ships with: waveform × 32768 -> torchaudio
// `kaldi.fbank` (80 mel, 25/10 ms, hamming, dither 0, snip_edges, remove_dc_offset, preemphasis
// 0.97, 512-point FFT, 20–8000 Hz) -> `apply_lfr(m: 7, n: 6)`, no CMVN.
//
// Per frame (400 samples, hop 160, only the frames that fit completely):
//   1. x × 32768, minus the frame mean                         (remove_dc_offset)
//   2. pre-emphasis inside the frame: y[j] = x[j] − 0.97·x[j−1], y[0] = x[0] − 0.97·x[0]
//   3. hamming 0.54 − 0.46·cos(2πn/399)
//   4. power spectrum of the frame zero-padded to 512 (cos/sin DFT basis) -> 257 bins
//   5. 80 triangular kaldi mel filters, mel(f) = 1127·ln(1 + f/700), 20–8000 Hz (bin 256 has weight 0)
//   6. log(max(energy, FLT_EPSILON))
// LFR: output frame i stacks fbank frames 6i−3 … 6i+3, each index clamped to [0, T−1] (funasr
// left-pads three copies of frame 0 and repeats the last frame), so L = ceil(T / 6).
//
// The frame arithmetic (scale, DC, pre-emphasis, window) runs in Float, as torchaudio's does for the
// oracle; the DFT, mel projection and log run in Double. A 400-term Float dot product drifts from the
// oracle's float32 FFT in the weak bins (max 2.3e-3 over the 155 fixture clips, against the NumPy
// spec's own 1.4e-3); in Double the drift is 1.3e-3 and the decoded ids equal the Python engine's on
// all 155. Gated row by row against the NumPy reference and the oracle's features
// (Tests/CoreAIKitTests/FunASRSmokeTests.swift).

import Accelerate
import Foundation

/// Stateless Fun-ASR-Nano feature extractor. Build once (the window, DFT and mel tables) and reuse.
@available(macOS 27, iOS 27, *)
public struct FunASRFbankPreprocessor: Sendable {
    public static let sampleRate = 16000
    public static let frameLength = 400             // 25 ms
    public static let frameShift = 160              // 10 ms
    public static let nFFT = 512                    // next power of two >= frameLength
    public static let nFreq = nFFT / 2 + 1          // 257
    public static let nMels = 80
    public static let lfrM = 7
    public static let lfrN = 6
    public static let featureDim = nMels * lfrM     // 560
    public static let preemphasis: Float = 0.97
    /// torchaudio's log floor: FLT_EPSILON (1.1920929e-7).
    public static let logFloor: Float = .ulpOfOne
    static let lowFreq = 20.0
    static let highFreq = Double(sampleRate) / 2

    private let window: [Float]                     // [frameLength]
    private let dftCos: [Double]                    // [frameLength, nFreq]; the 112 padded samples are 0
    private let dftSin: [Double]                    // [frameLength, nFreq]
    private let melBanksT: [Double]                 // [nFreq, nMels]

    public init() {
        let fl = Self.frameLength, nFreq = Self.nFreq, nFFT = Self.nFFT, nMels = Self.nMels

        self.window = (0..<fl).map { n in
            Float(0.54 - 0.46 * cos(2 * Double.pi * Double(n) / Double(fl - 1)))
        }

        // re[k] = Σ x[n]·cos(2πkn/512), im[k] = Σ x[n]·sin(2πkn/512); the angle is reduced mod 512
        // before it becomes a Double so every entry is the correctly rounded value.
        var c = [Double](repeating: 0, count: fl * nFreq)
        var s = [Double](repeating: 0, count: fl * nFreq)
        for n in 0..<fl {
            for k in 0..<nFreq {
                let angle = 2 * Double.pi * Double((k * n) % nFFT) / Double(nFFT)
                c[n * nFreq + k] = cos(angle)
                s[n * nFreq + k] = sin(angle)
            }
        }
        self.dftCos = c
        self.dftSin = s

        // torchaudio.compliance.kaldi.get_mel_banks, plus the zero column fbank() appends for bin 256.
        func mel(_ f: Double) -> Double { 1127 * log(1 + f / 700) }
        let binWidth = Double(Self.sampleRate) / Double(nFFT)
        let melLow = mel(Self.lowFreq), melHigh = mel(Self.highFreq)
        let delta = (melHigh - melLow) / Double(nMels + 1)
        var banks = [Double](repeating: 0, count: nFreq * nMels)
        for b in 0..<nMels {
            let left = melLow + Double(b) * delta
            let center = melLow + Double(b + 1) * delta
            let right = melLow + Double(b + 2) * delta
            for i in 0..<(nFFT / 2) {
                let m = mel(binWidth * Double(i))
                let up = (m - left) / (center - left)
                let down = (right - m) / (right - center)
                banks[i * nMels + b] = max(0, min(up, down))
            }
        }
        self.melBanksT = banks
    }

    // MARK: - Frame counts

    /// fbank frames in `samples` samples (kaldi snip_edges: only the frames that fit completely).
    public static func fbankFrames(samples: Int) -> Int {
        samples < frameLength ? 0 : 1 + (samples - frameLength) / frameShift
    }

    /// LFR frames for `fbankFrames` fbank frames: ceil(T / 6).
    public static func lfrFrames(fbankFrames t: Int) -> Int { (t + lfrN - 1) / lfrN }

    /// LFR frames (encoder rows) in `samples` samples: 480,000 samples (30 s) -> 2998 -> 500.
    public static func lfrFrames(samples: Int) -> Int {
        lfrFrames(fbankFrames: fbankFrames(samples: samples))
    }

    /// Audio slots the decoder receives for `lfrFrames` encoder rows: N = ceil(L / 8) (500 -> 63).
    /// funasr writes it as three stride-2 stages (`data_load_speech`, `use_low_frame_rate`); both
    /// forms are computed and must agree.
    public static func audioTokenCount(lfrFrames l: Int) -> Int {
        func floorDiv(_ a: Int, _ b: Int) -> Int { a >= 0 ? a / b : -((b - 1 - a) / b) }  // Python //
        let n = (l + 7) / 8
        let o1 = 1 + floorDiv(l - 3 + 2, 2)
        let o2 = 1 + floorDiv(o1 - 3 + 2, 2)
        assert(n == floorDiv(o2 - 1, 2) + 1, "ceil(L/8) != funasr's three-stage count for L=\(l)")
        return n
    }

    // MARK: - Features

    /// Log-mel fbank for a 16 kHz mono waveform (floats in [-1, 1)): row-major `[T, 80]`.
    public func fbank(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let t = Self.fbankFrames(samples: samples.count)
        guard t > 0 else { return ([], 0) }
        let fl = Self.frameLength, nFreq = Self.nFreq, nMels = Self.nMels

        // 1–3. frame, scale, remove DC, pre-emphasize, window -> [T, 400].
        var frames = [Float](repeating: 0, count: t * fl)
        var scratch = [Float](repeating: 0, count: fl)
        var scale: Float = 32768
        var negCoefficient = -Self.preemphasis
        samples.withUnsafeBufferPointer { src in
            frames.withUnsafeMutableBufferPointer { dst in
                scratch.withUnsafeMutableBufferPointer { tmp in
                    window.withUnsafeBufferPointer { win in
                        guard let s = src.baseAddress, let d = dst.baseAddress, let x = tmp.baseAddress,
                            let w = win.baseAddress
                        else { return }
                        let length = vDSP_Length(fl)
                        for f in 0..<t {
                            vDSP_vsmul(s + f * Self.frameShift, 1, &scale, x, 1, length)
                            var mean: Float = 0
                            vDSP_meanv(x, 1, &mean, length)
                            var negMean = -mean
                            vDSP_vsadd(x, 1, &negMean, x, 1, length)
                            let row = d + f * fl
                            // row[j] = x[j] - 0.97·x[j-1] for j >= 1; row[0] = x[0] - 0.97·x[0].
                            vDSP_vsma(x, 1, &negCoefficient, x + 1, 1, row + 1, 1, length - 1)
                            row[0] = x[0] - Self.preemphasis * x[0]
                            vDSP_vmul(row, 1, w, 1, row, 1, length)
                        }
                    }
                }
            }
        }

        // 4. re/im = frames [T, 400] · basis [400, 257] in Double; power = re² + im².
        var framesD = [Double](repeating: 0, count: t * fl)
        vDSP_vspdp(frames, 1, &framesD, 1, vDSP_Length(t * fl))
        var re = [Double](repeating: 0, count: t * nFreq)
        var im = [Double](repeating: 0, count: t * nFreq)
        vDSP_mmulD(framesD, 1, dftCos, 1, &re, 1, vDSP_Length(t), vDSP_Length(nFreq), vDSP_Length(fl))
        vDSP_mmulD(framesD, 1, dftSin, 1, &im, 1, vDSP_Length(t), vDSP_Length(nFreq), vDSP_Length(fl))
        var power = [Double](repeating: 0, count: t * nFreq)
        vDSP_vmmaD(re, 1, re, 1, im, 1, im, 1, &power, 1, vDSP_Length(t * nFreq))

        // 5. mel energies [T, 80] = power [T, 257] · banks [257, 80].  6. log(max(e, FLT_EPSILON)).
        var melD = [Double](repeating: 0, count: t * nMels)
        vDSP_mmulD(power, 1, melBanksT, 1, &melD, 1, vDSP_Length(t), vDSP_Length(nMels), vDSP_Length(nFreq))
        var floorValue = Double(Self.logFloor)
        var count = Int32(melD.count)
        melD.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            vDSP_vthrD(p, 1, &floorValue, p, 1, vDSP_Length(buf.count))
            vvlog(p, p, &count)
        }
        var mel = [Float](repeating: 0, count: t * nMels)
        vDSP_vdpsp(melD, 1, &mel, 1, vDSP_Length(t * nMels))
        return (mel, t)
    }

    /// LFR features for a 16 kHz mono waveform: row-major `[L, 560]` (row i = fbank frames
    /// 6i−3 … 6i+3, clamped) and L. A clip shorter than one fbank frame (400 samples) gives L = 0.
    public func features(_ samples: [Float]) -> (feats: [Float], lfrFrames: Int) {
        let (fb, t) = fbank(samples)
        let l = Self.lfrFrames(fbankFrames: t)
        guard l > 0 else { return ([], 0) }
        let nMels = Self.nMels, dim = Self.featureDim, half = (Self.lfrM - 1) / 2
        var out = [Float](repeating: 0, count: l * dim)
        fb.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                guard let s = src.baseAddress, let d = dst.baseAddress else { return }
                for i in 0..<l {
                    for k in 0..<Self.lfrM {
                        let frame = min(max(i * Self.lfrN + k - half, 0), t - 1)
                        (d + i * dim + k * nMels).update(from: s + frame * nMels, count: nMels)
                    }
                }
            }
        }
        return (out, l)
    }
}
