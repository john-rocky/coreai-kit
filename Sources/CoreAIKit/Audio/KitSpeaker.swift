// KitSpeaker.swift — one text-to-speech entry point for any `tts` catalog id. The catalog
// entry names the platform bundle dir; the tokenizer and host-glue subtrees ride the same
// repo, so a card's id is all a caller holds (a picker or CLI flag away from the model card).

import Foundation

/// Synthesized speech: mono PCM in [-1, 1] plus its sample rate.
public struct SpokenAudio: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public var seconds: Double { Double(samples.count) / Double(sampleRate) }

    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

/// Any text-to-speech model in the catalog (`ModelCatalog.builtin.available(.tts)`) behind
/// one `synthesize(_:)`.
///
/// ```swift
/// let speaker = try await KitSpeaker(catalog: "voxcpm-0.5b")
/// let audio = try await speaker.synthesize("Hello from Core AI.")
/// ```
public struct KitSpeaker: Sendable {
    enum Engine: Sendable {
        case voxcpm(VoxCPMTTS)
        case voxcpm2(VoxCPM2TTS)
        case kokoro(KokoroTTS)
    }

    private let engine: Engine
    /// The catalog id this speaker was loaded from.
    public let catalogID: String

    /// Loads a text-to-speech model by its catalog id — the id shown on the model's card.
    /// First use downloads the platform bundles + tokenizer + glue tables (cached afterwards).
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .tts)
        guard let variant = entry.variant else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }

        // Kokoro is three stateless graph bundles + host glue at the repo root — no platform
        // dir, no tokenizer subtree (G2P rides the glue) — so it resolves its own subtrees.
        if entry.id == "kokoro-82m" {
            let predictor = try await store.download(
                ModelID(entry.repo, path: "kokoro_predictor.aimodel"), progress: downloadProgress)
            let prosody = try await store.download(
                ModelID(entry.repo, path: "kokoro_prosody.aimodel"), progress: downloadProgress)
            let vocoder = try await store.download(
                ModelID(entry.repo, path: "kokoro_vocoder.aimodel"), progress: downloadProgress)
            let glue = try await store.download(
                ModelID(entry.repo, path: "kokoro_host_glue"), progress: downloadProgress)
            self.engine = .kokoro(
                try await KokoroTTS(
                    predictorAt: predictor, prosodyAt: prosody, vocoderAt: vocoder,
                    glueDir: glue))
            self.catalogID = entry.id
            return
        }

        let glueName: String
        switch entry.id {
        case "voxcpm-0.5b": glueName = "voxcpm_host_glue"
        case "voxcpm2-2b": glueName = "voxcpm2_host_glue"
        default: throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        let platform = try await store.download(
            ModelID(entry.repo, path: variant.path), progress: downloadProgress)
        let tokenizer = try await store.download(
            ModelID(entry.repo, path: "tokenizer"), progress: downloadProgress)
        let glue = try await store.download(
            ModelID(entry.repo, path: glueName), progress: downloadProgress)
        switch entry.id {
        case "voxcpm-0.5b":
            #if os(iOS)
            var paths = VoxCPMPaths.aot(root: platform, arch: "h18p", tokenizerDir: tokenizer)
            #else
            var paths = VoxCPMPaths.standard(artifactsRoot: platform, tokenizerDir: tokenizer)
            #endif
            paths.glueDir = glue
            self.engine = .voxcpm(try await VoxCPMTTS(paths: paths))
        default:
            #if os(iOS)
            var paths = VoxCPM2Paths.aot(root: platform, arch: "h18p", tokenizerDir: tokenizer)
            #else
            var paths = VoxCPM2Paths.standard(artifactsRoot: platform, tokenizerDir: tokenizer)
            #endif
            paths.glueDir = glue
            self.engine = .voxcpm2(try await VoxCPM2TTS(paths: paths))
        }
        self.catalogID = entry.id
    }

    /// Synthesizes one utterance (16 kHz mono for VoxCPM, 48 kHz for VoxCPM2, 24 kHz for
    /// Kokoro).
    public func synthesize(_ text: String) async throws -> SpokenAudio {
        switch engine {
        case .voxcpm(let tts):
            return SpokenAudio(
                samples: try await tts.synthesize(text), sampleRate: VoxCPMTTS.sampleRate)
        case .voxcpm2(let tts):
            return SpokenAudio(
                samples: try await tts.synthesize(text), sampleRate: VoxCPM2TTS.sampleRate)
        case .kokoro(let tts):
            return SpokenAudio(
                samples: try await tts.synthesize(text), sampleRate: KokoroTTS.sampleRate)
        }
    }

    /// Streaming synthesis: `onChunk` receives ~0.5 s chunks as they decode, so playback can
    /// start before the whole clip exists. The concatenation equals `synthesize(_:)`.
    public func synthesizeStreaming(
        _ text: String, onChunk: @Sendable ([Float]) async -> Void
    ) async throws {
        switch engine {
        case .voxcpm(let tts): _ = try await tts.synthesizeStreaming(text, onChunk: onChunk)
        case .voxcpm2(let tts): _ = try await tts.synthesizeStreaming(text, onChunk: onChunk)
        case .kokoro(let tts): _ = try await tts.synthesizeStreaming(text, onChunk: onChunk)
        }
    }
}
