// KitMusician.swift — one text-to-music entry point for any `music` catalog id. The catalog
// entry names the platform subtree (T5 conditioner + DiT + VAE + tokenizer in one dir); a
// card's id is all a caller holds.

import Foundation

/// Any text-to-music model in the catalog (`ModelCatalog.builtin.available(.music)`) behind
/// one `generate(_:seconds:)`.
///
/// ```swift
/// let musician = try await KitMusician(catalog: "stable-audio-open-small")
/// let audio = try await musician.generate("128 BPM tech house drum loop", seconds: 11)
/// ```
public struct KitMusician: Sendable {
    private let engine: StableAudioMusic
    /// The catalog id this musician was loaded from.
    public let catalogID: String

    /// Loads a text-to-music model by its catalog id — the id shown on the model's card.
    /// First use downloads the bundles (cached afterwards).
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .music)
        guard entry.id == "stable-audio-open-small", let variant = entry.variant else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        let root = try await store.download(
            entry.modelID(path: variant.path), progress: downloadProgress)
        let paths = StableAudioPaths(
            cond: root.appendingPathComponent("sa_cond_fp16b.aimodel"),
            dit: root.appendingPathComponent("sa_dit_fp16.aimodel"),
            vae: root.appendingPathComponent("sa_vae_fp16.aimodel"),
            tokenizerDir: root.appendingPathComponent("t5_tokenizer"))
        self.engine = try await StableAudioMusic(paths: paths)
        self.catalogID = entry.id
    }

    /// Generates audio for a prompt (44.1 kHz stereo, interleaved L/R).
    public func generate(_ prompt: String, seconds: Float = 11) async throws -> SpokenAudio {
        SpokenAudio(
            samples: try await engine.generate(prompt: prompt, seconds: seconds),
            sampleRate: StableAudioMusic.sampleRate)
    }
}
