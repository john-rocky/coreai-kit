// KitDiarizer+Nemotron.swift — what KitDiarizer needs around the NemotronDiarizer host (the five
// N3D files beside this one): where a bundle keeps its host constants, which graph it runs, and the
// per-frame output of the low-latency streaming loop. The coreai-audio app's NemotronDiarizerBridge
// in the zoo is the model: the same mode, the same padding of a short clip, and turns by the rule
// of the 4-speaker path (`KitDiarizer.segments`) with 48 frames of 10 ms for its 6 of 80 ms.

import CoreAIKitVision
import Foundation

extension KitDiarizer {
    /// The repo subtree beside the graph holding the host constants: the 8-frame projection and
    /// the silence row (weights), the mel filterbank, the Hann window and metadata.json.
    static let nemotronHostPath = "host"
    /// Low latency: chunks of 9 + 4 encoder frames (0.72 s + 0.32 s look-ahead), graph T = 541.
    static let nemotronProfile = N3DProfile.streamingProfile(mode: .lowLatency)

    /// Where a local bundle keeps its Nemotron-3 host constants: `host/` beside the graph (the Hub
    /// layout) or the directory itself (an export directory). nil when there are none — a
    /// Sortformer bundle.
    static func nemotronHost(near root: URL) -> URL? {
        let dir = ["aimodel", "aimodelc"].contains(root.pathExtension)
            ? root.deletingLastPathComponent() : root
        return [dir.appendingPathComponent(nemotronHostPath), dir].first {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("embedder_projection.f32le").path)
        }
    }

    /// The streaming graph in `root`, or `root` itself when it is one. iOS prefers the AOT
    /// `.h18p.aimodelc`; macOS takes only the `.aimodel` (an iOS bundle is refused on a Mac).
    static func nemotronGraph(in root: URL) throws -> URL {
        if ["aimodel", "aimodelc"].contains(root.pathExtension) { return root }
        let base = "n3d_\(nemotronProfile.kind.rawValue)_float16"
        var names = ["\(base).aimodel"]
        #if os(iOS)
        names.insert("\(base).h18p.aimodelc", at: 0)
        #endif
        guard let url = names.map({ root.appendingPathComponent($0) })
            .first(where: { FileManager.default.fileExists(atPath: $0.path) })
        else { throw N3DError.missingFile("\(base) graph in \(root.path)") }
        return url
    }

    /// The streaming host over `graph`, with the constants in `host`.
    static func loadNemotron(
        graph: URL, host: URL, computeUnits: GraphModel.ComputeUnits
    ) async throws -> N3DDiarizer {
        try await N3DDiarizer(
            assets: N3DAssets(directory: host), computeUnits: N3DComputeUnits(computeUnits),
            profile: nemotronProfile, modelURL: graph)
    }

    /// 16 kHz mono -> activity `[frames][8]` at 10 ms. A clip shorter than the first chunk
    /// (16,680 samples, 1.04 s) is padded with silence, and its rows stop at the clip's last whole
    /// 10 ms frame.
    static func nemotronFramePreds(_ n3d: N3DDiarizer, samples: [Float]) async throws -> [[Float]] {
        let first = N3DMel.streamChunks(
            samples: 0, chunkFrames: nemotronProfile.chunkFrames,
            lookaheadFrames: nemotronProfile.lookaheadFrames)[0].end
        var input = samples
        if input.count < first {
            input.append(contentsOf: repeatElement(0, count: first - input.count))
        }
        let out = try await n3d.process(samples: input)
        let frames = input.count == samples.count
            ? out.frames : min(out.frames, samples.count / N3DMel.hop)
        let S = N3DSpeakerCache.numSpeakers
        return (0..<frames).map { Array(out.probs[($0 * S)..<(($0 + 1) * S)]) }
    }
}

extension N3DComputeUnits {
    /// The kit's compute-unit choice, lowered to the same `SpecializationOptions` as in
    /// `GraphModel`.
    init(_ units: GraphModel.ComputeUnits) {
        switch units {
        case .gpu: self = .gpu
        case .neuralEngine: self = .ane
        case .cpu: self = .cpu
        case .cpuOnly: self = .cpuOnly
        }
    }
}

extension N3DError: LocalizedError {
    var errorDescription: String? { "Nemotron-3-Diarization: \(description)" }
}
