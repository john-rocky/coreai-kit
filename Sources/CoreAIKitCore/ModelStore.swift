// ModelStore.swift — downloads and caches Core AI model bundles from the Hugging Face Hub.
//
// Every file streams into a hidden staging directory and the bundle is renamed into place
// only when ALL of its files are complete: the runtime must never see a partial bundle — a
// partially-present bundle poisons the content-keyed on-device compilation cache (later loads
// fail until the cache is wiped). Presence at the final location therefore means complete.
// Downloaded bundles are excluded from iCloud backup.
//
// Cache layout: <directory>/<org>/<name>/<revision>/<variant>/ — that directory is a complete
// bundle root (metadata.json + *.aimodel/ + tokenizer/) ready to hand to the runtime. <variant>
// is the subtree the bundle came from: on an iPhone that is `ios-<arch>/` when the repo has one
// for the device, else `ios/` (`ModelID.subtrees`).

import Foundation

public actor ModelStore {
    public static let `default` = ModelStore()

    public nonisolated let directory: URL

    private let hub: HubClient
    /// The architecture `ModelID.subtrees` matches an iPhone's `ios-<arch>/` against.
    private nonisolated let architecture: @Sendable () -> String?
    private var inflight: [ModelID: Task<URL, Error>] = [:]

    /// Store rooted at Application Support/CoreAIKit/Models.
    /// `hubBaseURL` selects an HF-compatible endpoint for both listing and file downloads.
    /// Credentials are not copied from Hugging Face to a custom endpoint.
    public init(hubBaseURL: URL = URL(string: "https://huggingface.co")!) {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
        self.directory = base.appendingPathComponent("CoreAIKit/Models", isDirectory: true)
        self.hub = HubClient(baseURL: hubBaseURL)
        self.architecture = { ModelStore.deviceArchitecture }
    }

    /// The endpoint does not change cache identity: repo, revision and variant still key it.
    public init(directory: URL, hubBaseURL: URL = URL(string: "https://huggingface.co")!) {
        self.directory = directory
        self.hub = HubClient(baseURL: hubBaseURL)
        self.architecture = { ModelStore.deviceArchitecture }
    }

    init(
        directory: URL, hub: HubClient,
        deviceArchitecture: @escaping @Sendable () -> String? = { ModelStore.deviceArchitecture }
    ) {
        self.directory = directory
        self.hub = hub
        self.architecture = deviceArchitecture
    }

    /// The subtrees this store looks for `model` in on this device, best first.
    nonisolated func subtrees(for model: ModelID) -> [String] {
        model.subtrees(deviceArchitecture: architecture())
    }

    /// Where `model`'s bundle from `subtree` sits in this store.
    nonisolated func bundleURL(_ model: ModelID, subtree: String) -> URL {
        directory.appendingPathComponent(model.cacheSubpath(subtree: subtree), isDirectory: true)
    }

    /// Local bundle root for a model, or nil if not downloaded. A model whose path is an iPhone's
    /// `ios` is looked for under `ios-<arch>/`, then `ios/`: a copy on disk is used whichever it
    /// is, and the Hub is not asked about the other.
    public nonisolated func localURL(for model: ModelID) -> URL? {
        subtrees(for: model).lazy
            .map { self.bundleURL(model, subtree: $0) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// Bundle roots present in this store (directories containing metadata.json).
    public nonisolated func downloadedModels() -> [(url: URL, sizeBytes: Int64)] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: directory, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        else { return [] }
        var found: [(URL, Int64)] = []
        for case let url as URL in enumerator {
            if fm.fileExists(atPath: url.appendingPathComponent("metadata.json").path) {
                found.append((url, Self.directorySize(url)))
                enumerator.skipDescendants()
            }
        }
        return found.map { (url: $0.0, sizeBytes: $0.1) }
    }

    /// Returns the local bundle root, downloading it first if needed. Concurrent calls for
    /// the same model join the in-flight download (only the first caller receives progress).
    ///
    /// On an iPhone, a model whose path is `ios` downloads `ios-<arch>/`, the graphs compiled for
    /// this device, when the repo has it, else `ios/` (`hubFiles(for:)`).
    ///
    /// Offline fallback: when the download fails at the transport level (airplane mode,
    /// no route to the Hub) and a complete copy of the same repo + variant is cached
    /// under a *different* revision — a catalog pin moved since it was downloaded — that
    /// copy is returned instead of surfacing the network error. A stale revision beats a
    /// dead Load button; the pinned revision downloads next time the network is back.
    @discardableResult
    public func download(
        _ model: ModelID,
        progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        if let url = localURL(for: model) { return url }
        if let task = inflight[model] { return try await task.value }
        let task = Task<URL, Error> {
            do {
                return try await self.performDownload(model, progress: progress)
            } catch let error as URLError {
                guard error.code != .cancelled,
                      let cached = self.siblingRevisionURL(for: model) else { throw error }
                return cached
            }
        }
        inflight[model] = task
        defer { inflight[model] = nil }
        return try await task.value
    }

    /// A complete cached copy of this model under another revision, newest first, or nil;
    /// within one revision, in the order `localURL` looks (`ios-<arch>/`, then `ios/`).
    /// Presence means complete — bundles only ever land at their final path by atomic
    /// rename (the staging directory is hidden), the same contract `localURL` relies on.
    nonisolated func siblingRevisionURL(for model: ModelID) -> URL? {
        let fm = FileManager.default
        let repoDir = directory.appendingPathComponent(model.repo, isDirectory: true)
        guard let revisions = try? fm.contentsOfDirectory(
            at: repoDir, includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles])
        else { return nil }
        let names = subtrees(for: model)
        let candidates = revisions
            .filter { $0.lastPathComponent != model.revision }
            .compactMap { rev in
                names.lazy
                    .map { $0.isEmpty ? rev : rev.appendingPathComponent($0) }
                    .first { fm.fileExists(atPath: $0.path) }
            }
        func mtime(_ url: URL) -> Date {
            (try? url.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
        }
        return candidates.max { mtime($0) < mtime($1) }
    }

    /// The subtree this device downloads the model from and the files in it, per the Hub. One
    /// place, so the download and the size probes can never disagree about what a model consists
    /// of.
    ///
    /// The first of `subtrees(for:)` the Hub lists files under. An iPhone's `ios-<arch>/` costs
    /// one listing, and `ios/` one more when that answers 404 or an empty tree. Any other failure
    /// of the first listing is thrown, not read as absence: a device that fell back to `ios/` on
    /// a Hub error would keep it, since a cached copy is used from then on.
    func hubFiles(
        for model: ModelID
    ) async throws -> (subtree: String, files: [HubClient.PlannedFile]) {
        let names = subtrees(for: model)
        for subtree in names.dropLast() {
            do {
                let files = try await hub.listFiles(
                    repo: model.repo, revision: model.revision, path: subtree)
                if !files.isEmpty { return (subtree, files) }
            } catch CoreAIKitError.variantNotFound {
                continue
            }
        }
        let subtree = names[names.count - 1]
        return (subtree, try await hub.listFiles(
            repo: model.repo, revision: model.revision, path: subtree))
    }

    /// Removes the model's cached bundle from every subtree `localURL` looks in, so an iPhone's
    /// `ios-<arch>/` and `ios/` of the revision both go. Throws when there is none.
    public func delete(_ model: ModelID) throws {
        let fm = FileManager.default
        let present = subtrees(for: model).map { bundleURL(model, subtree: $0) }
            .filter { fm.fileExists(atPath: $0.path) }
        guard !present.isEmpty else {
            // The file system's "no such file", as before.
            try fm.removeItem(at: bundleURL(model, subtree: model.resolvedPath))
            return
        }
        for url in present { try fm.removeItem(at: url) }
    }

    // MARK: - Download

    private func performDownload(
        _ model: ModelID,
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> URL {
        let (subtree, files) = try await hubFiles(for: model)
        guard !files.isEmpty else {
            throw CoreAIKitError.variantNotFound(
                repo: model.repo, path: subtree, revision: model.revision)
        }
        let totalBytes = files.reduce(0) { $0 + $1.size }

        let fm = FileManager.default
        let final = bundleURL(model, subtree: subtree)
        let parent = final.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)

        // Stage the whole bundle, then one atomic rename into place (same volume).
        let staging = parent.appendingPathComponent(".staging-\(final.lastPathComponent)")
        try? fm.removeItem(at: staging)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)

        defer { try? fm.removeItem(at: staging) }

        let gate = ProgressGate()
        var doneBytes: Int64 = 0
        for file in files {
            try Task.checkCancellation()
            let target = staging.appendingPathComponent(file.relativePath)
            try fm.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let base = doneBytes
            try await hub.download(file, repo: model.repo, revision: model.revision, to: target) {
                written in
                guard let progress, totalBytes > 0 else { return }
                let done = base + written
                let f = Double(done) / Double(totalBytes)
                if gate.pass(f) {
                    progress(DownloadProgress(
                        fraction: min(f, 1), completedBytes: done, totalBytes: totalBytes,
                        currentFile: file.relativePath))
                }
            }
            if file.size > 0 {
                let size = try target.resourceValues(forKeys: [.fileSizeKey]).fileSize
                guard size.map(Int64.init) == file.size else {
                    throw CocoaError(.fileReadCorruptFile, userInfo: [NSFilePathErrorKey: target.path])
                }
            }
            doneBytes += file.size
        }

        try? fm.removeItem(at: final)
        try fm.moveItem(at: staging, to: final)
        var noBackup = URLResourceValues()
        noBackup.isExcludedFromBackup = true
        var url = final
        try? url.setResourceValues(noBackup)

        progress?(DownloadProgress(
            fraction: 1, completedBytes: totalBytes, totalBytes: totalBytes, currentFile: ""))
        return final
    }

    nonisolated static func directorySize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }
}

// Rate-limits progress callbacks to visible changes (~0.002 of the total).
private final class ProgressGate: @unchecked Sendable {
    private let lock = NSLock()
    private var last = -1.0

    func pass(_ fraction: Double) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard fraction - last >= 0.002 || fraction >= 1 else { return false }
        last = fraction
        return true
    }
}
