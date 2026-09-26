// From john-rocky/coreai-model-zoo conversion/nemotron3_diar/swift/Sources/NemotronDiarizer/N3DAssets.swift @ 26241ae (same author); the kit's copy is internal and picks the graph by the kit's rule (`GraphBundle`).
// N3DAssets — the host-side files of a Nemotron-3-Diarization bundle directory (../export_n3d.py):
// four raw little-endian float32 constants (C order) and metadata.json, next to the graph bundles
// (`n3d_<profile>_float16.aimodel`, and any AOT `n3d_<profile>_float16.<arch>.aimodelc`).
// The sha256 values in metadata.json are informational; nothing here checks them.

import Foundation

@available(macOS 27, iOS 27, *)
enum N3DError: Error, CustomStringConvertible, Sendable {
    case missingFile(String)
    case badSize(file: String, expectedFloats: Int, gotBytes: Int)
    case functionNotFound(String)
    case contract(String)
    case iosBundleOnMac(String)
    case audioTooShort(samples: Int, minimum: Int)
    case graphOutput(String)

    var description: String {
        switch self {
        case .missingFile(let p): return "missing file \(p)"
        case .badSize(let f, let n, let b): return "\(f): expected \(n) float32 (\(n * 4) bytes), got \(b) bytes"
        case .functionNotFound(let n): return "function '\(n)' not in the bundle"
        case .contract(let s): return "graph contract: \(s)"
        case .iosBundleOnMac(let p): return "refusing a bundle compiled for another device on macOS: \(p)"
        case .audioTooShort(let n, let m):
            return "audio of \(n) samples is shorter than the first streaming chunk (\(m) samples)"
        case .graphOutput(let s): return "graph output: \(s)"
        }
    }
}

@available(macOS 27, iOS 27, *)
struct N3DAssets: Sendable {
    static let hidden = 512
    static let stackedWidth = N3DMel.nMels * N3DMel.stack      // 1024

    let directory: URL
    /// [512, 1024] `model.audio_tower.embedder.projection.weight` (Linear 1024 -> 512, no bias).
    let projection: [Float]
    /// [512] `silence_embeds`: the speaker cache's silence row.
    let silence: [Float]
    /// [128, 257] librosa slaney filterbank (bit-identical to the transformers feature extractor's).
    let melFilters: [Float]
    /// [400] `torch.hann_window(400, periodic=False)`.
    let hannWindow: [Float]
    /// metadata.json as stored (nil if absent).
    let metadata: Data?

    init(directory: URL) throws {
        self.directory = directory
        projection = try Self.readF32LE(directory.appendingPathComponent("embedder_projection.f32le"),
                                        count: Self.hidden * Self.stackedWidth)
        silence = try Self.readF32LE(directory.appendingPathComponent("silence_embeds.f32le"), count: Self.hidden)
        melFilters = try Self.readF32LE(directory.appendingPathComponent("mel_filters_128x257.f32le"),
                                        count: N3DMel.nMels * N3DMel.nFreq)
        hannWindow = try Self.readF32LE(directory.appendingPathComponent("hann_window_400.f32le"),
                                        count: N3DMel.winLength)
        metadata = try? Data(contentsOf: directory.appendingPathComponent("metadata.json"))
    }

    /// A raw little-endian float32 file of exactly `count` values.
    static func readF32LE(_ url: URL, count: Int) throws -> [Float] {
        guard let data = try? Data(contentsOf: url) else { throw N3DError.missingFile(url.path) }
        guard data.count == count * 4 else {
            throw N3DError.badSize(file: url.lastPathComponent, expectedFloats: count, gotBytes: data.count)
        }
        return readF32LE(data)
    }

    /// Every float32 of a little-endian buffer.
    static func readF32LE(_ data: Data) -> [Float] {
        let n = data.count / 4
        return data.withUnsafeBytes { raw in
            (0..<n).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    /// The graph for a profile kind in this directory: the AOT `.<arch>.aimodelc` when it was compiled
    /// for this device, else the `.aimodel` (`GraphBundle`; another device's AOT graph is never picked).
    func modelURL(for kind: N3DProfile.Kind) throws -> URL? {
        try GraphBundle.graph(named: "n3d_\(kind.rawValue)_float16", in: directory)
    }
}
