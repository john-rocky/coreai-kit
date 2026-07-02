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
    private let tts: VoxCPMTTS
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
        // VoxCPM is the only tts family today; a dispatch table (KitTranscriber-style)
        // lands with the second one.
        guard entry.id == "voxcpm-0.5b", let variant = entry.variant else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        let platform = try await store.download(
            ModelID(entry.repo, path: variant.path), progress: downloadProgress)
        let tokenizer = try await store.download(
            ModelID(entry.repo, path: "tokenizer"), progress: downloadProgress)
        let glue = try await store.download(
            ModelID(entry.repo, path: "voxcpm_host_glue"), progress: downloadProgress)
        #if os(iOS)
        var paths = VoxCPMPaths.aot(root: platform, arch: "h18p", tokenizerDir: tokenizer)
        #else
        var paths = VoxCPMPaths.standard(artifactsRoot: platform, tokenizerDir: tokenizer)
        #endif
        paths.glueDir = glue
        self.tts = try await VoxCPMTTS(paths: paths)
        self.catalogID = entry.id
    }

    /// Synthesizes one utterance (16 kHz mono for VoxCPM).
    public func synthesize(_ text: String) async throws -> SpokenAudio {
        SpokenAudio(samples: try await tts.synthesize(text), sampleRate: VoxCPMTTS.sampleRate)
    }

    /// Streaming synthesis: `onChunk` receives ~0.5 s chunks as they decode, so playback can
    /// start before the whole clip exists. The concatenation equals `synthesize(_:)`.
    public func synthesizeStreaming(
        _ text: String, onChunk: @Sendable ([Float]) async -> Void
    ) async throws {
        _ = try await tts.synthesizeStreaming(text, onChunk: onChunk)
    }
}
