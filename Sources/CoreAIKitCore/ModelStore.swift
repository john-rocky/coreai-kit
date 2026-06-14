// ModelStore.swift — downloads and caches Core AI model bundles from the Hugging Face Hub.
//
// Every file streams into a hidden staging directory and the bundle is renamed into place
// only when ALL of its files are complete: the runtime must never see a partial bundle — a
// partially-present bundle poisons the content-keyed on-device compilation cache (later loads
// fail until the cache is wiped). Presence at the final location therefore means complete.
// Downloaded bundles are excluded from iCloud backup.
//
// Cache layout: <directory>/<org>/<name>/<revision>/<variant>/ — that directory is a complete
// bundle root (metadata.json + *.aimodel/ + tokenizer/) ready to hand to the runtime.

import Foundation

public actor ModelStore {
    public static let `default` = ModelStore()

    public nonisolated let directory: URL

    private let hub = HubClient()
    private var inflight: [ModelID: Task<URL, Error>] = [:]

    /// Store rooted at Application Support/CoreAIKit/Models.
    public init() {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
        self.directory = base.appendingPathComponent("CoreAIKit/Models", isDirectory: true)
    }

    public init(directory: URL) {
        self.directory = directory
    }

    /// Local bundle root for a model, or nil if not downloaded.
    public nonisolated func localURL(for model: ModelID) -> URL? {
        let url = directory.appendingPathComponent(model.cacheSubpath, isDirectory: true)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
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
    @discardableResult
    public func download(
        _ model: ModelID,
        progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> URL {
        if let url = localURL(for: model) { return url }
        if let task = inflight[model] { return try await task.value }
        let task = Task<URL, Error> { try await self.performDownload(model, progress: progress) }
        inflight[model] = task
        defer { inflight[model] = nil }
        return try await task.value
    }

    public func delete(_ model: ModelID) throws {
        let url = directory.appendingPathComponent(model.cacheSubpath, isDirectory: true)
        try FileManager.default.removeItem(at: url)
    }

    // MARK: - Download

    private func performDownload(
        _ model: ModelID,
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> URL {
        let files = try await hub.listFiles(
            repo: model.repo, revision: model.revision, path: model.resolvedPath)
        let totalBytes = files.reduce(0) { $0 + $1.size }

        let fm = FileManager.default
        let final = directory.appendingPathComponent(model.cacheSubpath, isDirectory: true)
        let parent = final.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)

        // Stage the whole bundle, then one atomic rename into place (same volume).
        let staging = parent.appendingPathComponent(".staging-\(final.lastPathComponent)")
        try? fm.removeItem(at: staging)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)

        let delegate = DownloadDelegate()
        // Tolerate flaky networks and the app being briefly backgrounded: wait for connectivity and
        // allow a long total transfer time. Interrupted transfers resume from the partial (see
        // `downloadFile`) instead of restarting — critical for the multi-GB bundles.
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        config.timeoutIntervalForResource = 7 * 24 * 60 * 60
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let gate = ProgressGate()
        var doneBytes: Int64 = 0
        for file in files {
            try Task.checkCancellation()
            let target = staging.appendingPathComponent(file.relativePath)
            try fm.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let base = doneBytes
            let temp = try await Self.downloadFile(file.url, via: session, delegate: delegate) {
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
            try fm.moveItem(at: temp, to: target)
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

    // Single-file download; progress via delegate. The temp file is claimed synchronously
    // inside the delegate callback, then returned for the caller to move into staging.
    //
    // Retries with resume data so an interrupted transfer continues from the partial instead of
    // restarting the whole file. The common interruption is iOS cancelling the task when the app is
    // backgrounded: the cancellation is delivered when the app comes back to the foreground, and we
    // resume from the bytes already written (HF's CDN supports range requests). Without this, any
    // brief backgrounding restarted the multi-GB download from zero.
    private static func downloadFile(
        _ url: URL, via session: URLSession, delegate: DownloadDelegate,
        onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws -> URL {
        var resumeData: Data?
        var lastError: Error?
        for attempt in 0..<6 {
            try Task.checkCancellation()
            if attempt > 0 {
                try? await Task.sleep(nanoseconds: 1_500_000_000)
                try Task.checkCancellation()
            }
            do {
                return try await withCheckedThrowingContinuation { cont in
                    delegate.onBytes = onBytes
                    delegate.onFinish = { cont.resume(with: $0) }
                    let task = resumeData.map { session.downloadTask(withResumeData: $0) }
                        ?? session.downloadTask(with: url)
                    task.resume()
                }
            } catch {
                lastError = error
                // Keep the partial if the system handed back resume data; otherwise restart clean.
                resumeData = (error as NSError)
                    .userInfo[NSURLSessionDownloadTaskResumeData] as? Data
            }
        }
        throw lastError ?? CoreAIKitError.httpError(statusCode: -1, file: url.lastPathComponent)
    }

    private nonisolated static func directorySize(_ url: URL) -> Int64 {
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

// Delegate callbacks arrive serialized on the session's queue; `onFinish` is cleared after the
// first resume so the didComplete(error:) that follows a success cannot double-resume.
private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    var onBytes: (@Sendable (Int64) -> Void)?
    var onFinish: (@Sendable (Result<URL, Error>) -> Void)?

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64, totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onBytes?(totalBytesWritten)
    }

    func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let code = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? -1
        guard code == 200 else {
            finish(.failure(CoreAIKitError.httpError(
                statusCode: code,
                file: downloadTask.originalRequest?.url?.lastPathComponent ?? "?")))
            return
        }
        let keep = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        do {
            try FileManager.default.moveItem(at: location, to: keep)
            finish(.success(keep))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error { finish(.failure(error)) }
    }

    private func finish(_ r: Result<URL, Error>) {
        onFinish?(r)
        onFinish = nil
    }
}
