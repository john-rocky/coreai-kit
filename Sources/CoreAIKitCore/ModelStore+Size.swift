// ModelStore+Size.swift — how big is it, before committing to finding out.
//
// The catalog carries a hand-written `sizeMB` per entry, and hand-written numbers drift: nine
// of them were wrong when this was added, four because an iOS figure had been estimated as the
// macOS one doubled. A number an adopter is shown before agreeing to a multi-gigabyte download
// should not be maintained by hand at all, so this asks the Hub.
//
// ```swift
// let bytes = try await ModelStore.default.remoteSize(of: entry.modelID!)   // the truth
// let onDisk = ModelStore.default.localSize(of: entry.modelID!)             // already paid
// ```
//
// The catalog number stays as the offline answer — `capability()` must work in airplane mode
// and must not block a picker on 53 network round trips — and this is what corrects it.

import Foundation

extension ModelStore {
    /// Bytes the Hub says this model's files add up to, without downloading any of them.
    ///
    /// This is the exact figure for the model's own subtree. It does **not** include sibling
    /// subtrees a loader resolves beside it (a vision tower, host glue, a tokenizer stored
    /// outside the variant path) — those are the loader's knowledge, not the store's, which is
    /// the same boundary `scripts/measure-catalog-sizes.py` refuses to cross.
    public func remoteSize(of model: ModelID) async throws -> Int64 {
        try await hubFiles(for: model).reduce(0) { $0 + $1.size }
    }

    /// Bytes this model currently occupies on disk, or nil if it is not downloaded.
    public nonisolated func localSize(of model: ModelID) -> Int64? {
        localURL(for: model).map { ModelStore.directorySize($0) }
    }

    /// Whether a complete copy is already on disk — the question `capability()` answers with,
    /// and the one that must never touch the network.
    ///
    /// Presence means complete: the store only ever moves a bundle to its final path once
    /// every file has landed, which is the invariant the whole cache layout is built on.
    public nonisolated func isCached(_ model: ModelID) -> Bool {
        localURL(for: model) != nil
    }

    /// Free space on the volume the store writes to, as the system would like it reported —
    /// `volumeAvailableCapacityForImportantUsage` is what iOS actually honours, and it is
    /// larger than the naive free-space number because it counts purgeable caches.
    public nonisolated static func availableBytes(at directory: URL) -> Int64? {
        let probe = FileManager.default.fileExists(atPath: directory.path)
            ? directory
            : directory.deletingLastPathComponent()
        return try? probe.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        ).volumeAvailableCapacityForImportantUsage
    }

    public nonisolated var availableBytes: Int64? {
        ModelStore.availableBytes(at: directory)
    }

    /// Every file a download of this model would fetch: where it comes from, where it lands,
    /// and how big it is.
    ///
    /// Exposed because the one download job this package cannot do for an adopter is the one
    /// Apple's `BackgroundAssets` framework owns — that needs an app extension, which a Swift
    /// package cannot ship. It can hand over the list to enqueue, which is this.
    public func downloadPlan(for model: ModelID) async throws -> [PlannedDownload] {
        let base = directory.appendingPathComponent(model.cacheSubpath, isDirectory: true)
        return try await hubFiles(for: model).map {
            PlannedDownload(
                url: $0.url, sizeBytes: $0.size,
                destination: base.appendingPathComponent($0.relativePath))
        }
    }
}

/// One file of a model download, for a caller doing the fetching itself.
public struct PlannedDownload: Sendable, Hashable {
    /// Where to fetch it from. Public and unauthenticated for the catalog's repos.
    public let url: URL
    public let sizeBytes: Int64
    /// Where `ModelStore` expects to find it afterwards.
    ///
    /// Writing files here directly is **not** supported: the store's contract is that a bundle
    /// appears at its final path only when every file is present, because a half-present bundle
    /// poisons the on-device compilation cache and later loads fail until it is wiped. Stage
    /// elsewhere and let the store do the final move, or simply call `download` — this is for
    /// planning and progress, not for bypassing.
    public let destination: URL
}
