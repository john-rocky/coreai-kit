// KitParakeetModel.swift — a Core AI NVIDIA Parakeet-TDT-0.6B transcription bundle. The zoo's first
// transducer / TDT (RNN-T family) ASR model.
//
// ```swift
// let parakeet = try await KitParakeetModel(model: .parakeetTDT)
// let result = try await parakeet.transcribe(samples: pcm16kMono) { partial in print(partial) }
// // result.text == "With her white paint and her scarlet smokestack, the Inverashiel, …"
// ```
//
// Pipeline: log-mel -> FastConformer encoder (+projector) -> host-driven greedy TDT loop (LSTM
// predictor + joint) -> token ids -> detokenize. Three stateless `.aimodel` graphs run on the GPU;
// the transducer loop is host code. Validated token-exact end-to-end vs the HF reference.

import CoreAIKitVision
import Foundation
import Tokenizers

/// Failures specific to the Parakeet / TDT transcription path.
public enum KitParakeetError: Error, LocalizedError {
    case unsupportedSampleRate(Int)
    case melFiltersMissing
    case graphMissing(String)
    case tokenizerMissing
    case outputMissing(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedSampleRate(let sr):
            return "Audio must be 16 kHz mono; got \(sr) Hz (resample before transcribing)."
        case .melFiltersMissing:
            return "Bundled Parakeet mel filterbank resource is missing."
        case .graphMissing(let kind):
            return "Parakeet bundle is missing the \(kind) graph (.aimodel/.aimodelc)."
        case .tokenizerMissing:
            return "Parakeet bundle is missing tokenizer.json."
        case .outputMissing(let name):
            return "A Parakeet graph did not produce the expected output '\(name)'."
        }
    }
}

/// A Core AI Parakeet-TDT bundle: encoder + predictor + joint graphs, the mel frontend, and the
/// tokenizer. Serial use (one transcription at a time) — the host TDT loop owns the LSTM state.
public final class KitParakeetModel: @unchecked Sendable {
    // TDT / model constants (config + gate_e2e.py).
    private static let blank = 8192
    private static let vocab = 8193                 // token logits width
    private static let durations: [Int] = [0, 1, 2, 3, 4]
    private static let hidden = 640                 // predictor/joint width
    private static let lstmLayers = 2               // predictor LSTM state is [2, 1, 640]
    public static let sampleRate = 16000

    private let encoder: GraphModel
    private let predict: GraphModel
    private let joint: GraphModel
    private let mel: ParakeetMelPreprocessor
    private let tokenizer: any Tokenizer

    // MARK: - Init

    /// Downloads the Parakeet bundle from the Hub (if needed) and loads it.
    public convenience init(
        model: ModelID = .parakeetTDT,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let root = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: root, computeUnits: computeUnits)
    }

    /// Loads a local Parakeet bundle directory (the three graphs + `tokenizer.json`). The mel
    /// filterbank ships inside CoreAIKit, so the bundle needs no extra asset.
    public init(
        bundleAt root: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        let paths = try Self.resolve(in: root)
        self.encoder = try await GraphModel(contentsOf: paths.encoder, computeUnits: computeUnits)
        self.predict = try await GraphModel(contentsOf: paths.predict, computeUnits: computeUnits)
        self.joint = try await GraphModel(contentsOf: paths.joint, computeUnits: computeUnits)
        // Bucket = the encoder's declared mel length (so the frontend always fills the graph).
        let bucket = encoder.shape(ofInput: "mel")?.last ?? 2885
        self.mel = try ParakeetMelPreprocessor.parakeet(bucketFrames: bucket)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: paths.tokenizerDir)
    }

    // MARK: - Transcribe

    /// Transcribe a raw 16 kHz mono waveform. The clip is padded/trimmed to the encoder bucket
    /// (~28.8 s). `onPartial` (optional) streams the running transcript as the loop emits tokens.
    public func transcribe(
        samples: [Float], sampleRate: Int = 16000,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        guard sampleRate == Self.sampleRate else {
            throw KitParakeetError.unsupportedSampleRate(sampleRate)
        }
        let H = Self.hidden

        // mel -> encoder(+projector) -> enc_proj [T, 640].
        let feats = mel.logMel(samples)             // [128, bucketFrames] row-major
        let encOut = try await encoder.run([
            "mel": .float32(feats, shape: [1, ParakeetMelPreprocessor.nMels, mel.bucketFrames])
        ])
        let encProj = try output(encOut, "enc_proj")           // [T*640]
        let T = encProj.count / H

        // Greedy TDT loop (gate_e2e.py, ported verbatim). LSTM state [2,1,640]; init with start=blank.
        var h = [Float](repeating: 0, count: Self.lstmLayers * H)
        var c = [Float](repeating: 0, count: Self.lstmLayers * H)
        var dec = try await predictStep(token: Self.blank, h: &h, c: &c)   // dec_out [640]

        var frame = 0
        var emitted: [Int] = []
        var lastEmit = ""
        let maxEmits = 12 * T
        while frame < T && emitted.count < maxEmits {
            let encFrame = Array(encProj[frame * H ..< (frame + 1) * H])
            let jout = try await joint.run([
                "dec_out": .float32(dec, shape: [1, H]),
                "enc_frame": .float32(encFrame, shape: [1, H]),
            ])
            let tokLogits = try output(jout, "token_logits")        // [8193]
            let durLogits = try output(jout, "dur_logits")          // [5]
            let token = argmax(tokLogits, 0, Self.vocab)
            var dur = Self.durations[argmax(durLogits, 0, Self.durations.count)]
            if token == Self.blank && dur == 0 { dur = 1 }          // forward-progress guard
            frame += dur
            if token != Self.blank {
                emitted.append(token)
                dec = try await predictStep(token: token, h: &h, c: &c)   // advance LSTM only on non-blank
                if let onPartial {
                    let text = clean(tokenizer.decode(tokens: emitted))
                    if text != lastEmit && !text.unicodeScalars.contains("\u{FFFD}") {
                        lastEmit = text
                        onPartial(text)
                    }
                }
            }
        }
        return Transcription(language: "", text: clean(tokenizer.decode(tokens: emitted)))
    }

    // MARK: - Internals

    /// One predictor step: embedding(token) -> 2-layer LSTM(h,c) -> projector. Updates `h`/`c` in
    /// place and returns `dec_out` [640].
    private func predictStep(token: Int, h: inout [Float], c: inout [Float]) async throws -> [Float] {
        let s = [Self.lstmLayers, 1, Self.hidden]
        let out = try await predict.run([
            "token": .int32([Int32(token)], shape: [1, 1]),
            "h": .float32(h, shape: s),
            "c": .float32(c, shape: s),
        ])
        h = try output(out, "h_out")
        c = try output(out, "c_out")
        return try output(out, "dec_out")
    }

    private func output(_ d: [String: TensorValue], _ name: String) throws -> [Float] {
        guard let v = d[name] else { throw KitParakeetError.outputMissing(name) }
        return v.floats()
    }

    /// Index of the max element in `a[lo..<hi]`, relative to `lo`.
    private func argmax(_ a: [Float], _ lo: Int, _ count: Int) -> Int {
        var best = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for i in 0..<count where a[lo + i] > bestVal { bestVal = a[lo + i]; best = i }
        return best
    }

    private func clean(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }

    // MARK: - Bundle layout

    private struct Paths { let encoder: URL; let predict: URL; let joint: URL; let tokenizerDir: URL }

    /// Find the three graphs (`encoder`/`predict`/`joint` in the filename, `.aimodel` or AOT
    /// `.aimodelc`) and the tokenizer directory anywhere under the bundle root — tolerant of both
    /// the flat dev layout and a `encoder/ predict/ joint/` Hub layout.
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
        func graph(_ kind: String) throws -> URL {
            guard let u = graphs.first(where: {
                $0.deletingPathExtension().lastPathComponent.lowercased().contains(kind)
                    || $0.deletingLastPathComponent().lastPathComponent.lowercased() == kind
            }) else { throw KitParakeetError.graphMissing(kind) }
            return u
        }
        guard let tok = tokenizerDir else { throw KitParakeetError.tokenizerMissing }
        return Paths(
            encoder: try graph("encoder"), predict: try graph("predict"),
            joint: try graph("joint"), tokenizerDir: tok)
    }
}
