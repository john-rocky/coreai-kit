// KitSeparator.swift — Mel-Band RoFormer (Kim Vocal, MIT) music source separation: a song in,
// a vocals stem and an instrumental stem out, entirely on device.
//
// The exported bundle folds STFT + the band-split RoFormer + iSTFT into ONE graph
// (`frames[1,2,801,2048] -> recon[1,2,801,2048]`), so the host never touches an FFT — it only
// reflect-pads, slices frames, and overlap-adds. Two overlap-add levels, both mirroring the
// validated Python reference (conversion/melband_roformer/{export_core2,pipeline_engine}.py):
//   (a) inside an 8 s chunk: STFT-frame overlap-add (stride hop) / Σ(win²) -> chunk audio
//   (b) across the song:     chunk overlap-add (stride C/numOverlap) with fade windows
// `instrumental = mix - vocals`.

import CoreAIKitVision
import Foundation

/// The two stems a separator produces, plus the rate they are sampled at.
/// Each stem is `[channel][sample]` — always 2 channels, same length as the input mix.
public struct Stems: Sendable {
    public let vocals: [[Float]]
    public let instrumental: [[Float]]
    public let sampleRate: Int

    public init(vocals: [[Float]], instrumental: [[Float]], sampleRate: Int) {
        self.vocals = vocals
        self.instrumental = instrumental
        self.sampleRate = sampleRate
    }
}

public enum KitSeparatorError: Error, Sendable {
    case bundleNotFound(URL)
    case outputMissing(String)
    case emptyInput
}

/// Music source separation by catalog id.
///
/// ```swift
/// let separator = try await KitSeparator(catalog: "melband-roformer-vocal")
/// let mix = try AudioFile.pcmStereo(songURL)
/// let stems = try await separator.separate(mix)      // stems.vocals / stems.instrumental
/// ```
public actor KitSeparator {
    /// Graph constants — fixed by the export, not tunable at run time.
    public static let sampleRate = 44_100
    /// Chunk the host feeds the graph: 8 s @ 44.1 kHz.
    public static let chunkSamples = 352_800
    static let nFFT = 2048, hop = 441, pad = 1024
    static let numOverlap = 2
    static var nFrames: Int { 1 + (chunkSamples + 2 * pad - nFFT) / hop }   // 801

    private let graph: GraphModel
    private let window: [Float]      // Hann periodic [nFFT]
    private let wsum: [Float]        // Σ(win²) over a padded chunk, for the frame overlap-add

    // MARK: - Loading

    /// Loads Mel-Band RoFormer by its catalog id: `KitSeparator(catalog: "melband-roformer-vocal")`.
    /// First use downloads the platform bundle (macOS `.aimodel` / iOS AOT `.aimodelc`), then caches.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .separation)
        guard let model = entry.modelID else { throw CoreAIKitError.modelNotInCatalog(id: id) }
        let root = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: root, computeUnits: computeUnits)
    }

    /// Loads a local bundle (the `.aimodel`/`.aimodelc` directory itself, or a folder holding one).
    public init(bundleAt root: URL, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        let bundle = try Self.resolveGraph(in: root)
        self.graph = try await GraphModel(contentsOf: bundle, computeUnits: computeUnits)
        var w = [Float](repeating: 0, count: Self.nFFT)
        for n in 0..<Self.nFFT { w[n] = 0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(Self.nFFT)) }
        self.window = w
        let total = Self.nFFT + Self.hop * (Self.nFrames - 1)
        var s = [Float](repeating: 0, count: total)
        for i in 0..<Self.nFrames { for n in 0..<Self.nFFT { s[i * Self.hop + n] += w[n] * w[n] } }
        for k in 0..<total { s[k] = max(s[k], 1e-8) }
        self.wsum = s
    }

    /// The graph inside `root` (or `root` itself): AOT `.aimodelc` on iOS, JIT `.aimodel` on macOS.
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
        else { throw KitSeparatorError.bundleNotFound(root) }
        return graph
    }

    // MARK: - Separation

    /// Separate a decoded stereo mix (`[[left], [right]]`, 44.1 kHz — see `AudioFile.pcmStereo`).
    /// `progress` reports 0…1 over the song's chunks.
    public func separate(
        _ mix: [[Float]], progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Stems {
        guard mix.count == 2, !mix[0].isEmpty, mix[1].count == mix[0].count else {
            throw KitSeparatorError.emptyInput
        }
        let C = Self.chunkSamples, step = C / Self.numOverlap, fade = C / 10
        let T = mix[0].count
        let border = C - step
        let mixp = [Self.reflectPad(mix[0], border), Self.reflectPad(mix[1], border)]
        let total = mixp[0].count

        // fade window (ones with fade-in/out at the ends); linspace(0,1,fade) = k/(fade-1)
        var win = [Float](repeating: 1, count: C)
        for k in 0..<fade { let v = Float(k) / Float(fade - 1); win[k] = v; win[C - 1 - k] = v }

        var result = [[Float]](repeating: [Float](repeating: 0, count: total), count: 2)
        var counter = [Float](repeating: 0, count: total)
        var i = 0
        let nChunks = max(1, (total + step - 1) / step)
        var done = 0
        while i < total {
            var part = [[Float]](repeating: [Float](repeating: 0, count: C), count: 2)
            let L = min(C, total - i)
            for c in 0..<2 { for k in 0..<L { part[c][k] = mixp[c][i + k] } }
            if L < C {   // reflect-pad the short tail to a full chunk (numpy 'reflect')
                for c in 0..<2 {
                    for k in L..<C {
                        let src = 2 * L - 2 - k
                        part[c][k] = src >= 0 ? part[c][src] : 0
                    }
                }
            }
            let voc = try await separateChunk(part)
            var w = win
            if i == 0 { for k in 0..<fade { w[k] = 1 } }
            else if i + C >= total { for k in 0..<fade { w[C - 1 - k] = 1 } }
            for c in 0..<2 { for k in 0..<L { result[c][i + k] += voc[c][k] * w[k] } }
            for k in 0..<L { counter[i + k] += w[k] }
            i += step
            done += 1
            progress?(min(1, Double(done) / Double(nChunks)))
        }

        var vocals = [[Float]](repeating: [Float](repeating: 0, count: T), count: 2)
        var inst = [[Float]](repeating: [Float](repeating: 0, count: T), count: 2)
        for c in 0..<2 {
            for k in 0..<T {
                let v = result[c][border + k] / max(counter[border + k], 1e-8)
                vocals[c][k] = v
                inst[c][k] = mix[c][k] - v
            }
        }
        return Stems(vocals: vocals, instrumental: inst, sampleRate: Self.sampleRate)
    }

    /// Separate an audio file directly (anything AVFoundation decodes).
    public func separate(
        contentsOf url: URL, progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> Stems {
        try await separate(AudioFile.pcmStereo(url, sampleRate: Double(Self.sampleRate)),
                           progress: progress)
    }

    /// One exactly-`chunkSamples` chunk → vocals `[2][chunkSamples]`. This is the graph's native
    /// unit; gates compare it against the published golden.
    public func separateChunk(_ chunk: [[Float]]) async throws -> [[Float]] {
        let N = Self.nFFT, H = Self.hop, F = Self.nFrames
        var flat = [Float](repeating: 0, count: 2 * F * N)
        for c in 0..<2 {
            let xp = Self.reflectPad(chunk[c], Self.pad)
            for i in 0..<F {
                let base = i * H
                let dst = (c * F + i) * N
                for n in 0..<N { flat[dst + n] = xp[base + n] }
            }
        }
        let out = try await graph.run(["frames": .float32(flat, shape: [1, 2, F, N])])
        guard let recon = out["recon"]?.floats() else {
            throw KitSeparatorError.outputMissing("recon")
        }
        let total = N + H * (F - 1)
        var acc = [[Float]](repeating: [Float](repeating: 0, count: total), count: 2)
        for c in 0..<2 {
            for i in 0..<F {
                let base = i * H, src = (c * F + i) * N
                for n in 0..<N { acc[c][base + n] += recon[src + n] }
            }
            for k in 0..<total { acc[c][k] /= wsum[k] }
        }
        return [Array(acc[0][Self.pad..<Self.pad + Self.chunkSamples]),
                Array(acc[1][Self.pad..<Self.pad + Self.chunkSamples])]
    }

    /// numpy 'reflect' padding (reflect_101: the edge sample is not repeated).
    private static func reflectPad(_ x: [Float], _ p: Int) -> [Float] {
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * p)
        for j in 0..<p { out[j] = x[p - j] }
        for k in 0..<n { out[p + k] = x[k] }
        for j in 0..<p { out[p + n + j] = x[n - 2 - j] }
        return out
    }
}
