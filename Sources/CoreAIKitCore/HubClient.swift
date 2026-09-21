// Hub transport adapter. Bundle installation remains the responsibility of ModelStore.

import Foundation
import HuggingFace

struct HubClient: Sendable {
    let baseURL: URL
    private let client: HuggingFace.HubClient

    init(
        baseURL: URL = URL(string: "https://huggingface.co")!,
        session: URLSession? = nil,
        tokenProvider: TokenProvider = .environment,
        cache: HubCache? = .default
    ) {
        self.baseURL = baseURL
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        config.timeoutIntervalForResource = 7 * 24 * 60 * 60
        // Only the canonical HTTPS endpoint may use the user's HF credentials/cache.
        // Mirror content must not enter the shared, endpoint-independent Hub cache.
        let canonical = baseURL.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            == "https://huggingface.co"
        self.client = HuggingFace.HubClient(
            session: session ?? URLSession(configuration: config),
            host: baseURL,
            tokenProvider: canonical ? tokenProvider : .none,
            cache: canonical ? cache : nil)
    }

    struct PlannedFile: Sendable {
        let url: URL
        let repoPath: String
        let relativePath: String
        let size: Int64
    }

    private func endpoint(_ path: String) throws -> URL {
        guard let components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false),
              ["https", "http"].contains(components.scheme?.lowercased() ?? ""),
              let host = components.host, !host.isEmpty,
              components.user == nil, components.password == nil,
              components.query == nil, components.fragment == nil else {
            throw CoreAIKitError.invalidHubBaseURL
        }
        return baseURL.appendingPathComponent(path)
    }

    func downloadURL(repo: String, revision: String, path: String) throws -> URL {
        try endpoint("\(repo)/resolve").appending(component: revision).appending(path: path)
    }

    /// Accepts "https://huggingface.co/<org>/<name>[/...]" or a bare "<org>/<name>".
    static func repoId(from s: String) -> String? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let u = URL(string: t), let host = u.host, host.hasSuffix("huggingface.co") {
            let parts = u.path.split(separator: "/").map(String.init)
            return parts.count >= 2 ? "\(parts[0])/\(parts[1])" : nil
        }
        let parts = t.split(separator: "/").map(String.init)
        return parts.count == 2 ? t : nil
    }

    /// Enumerates the files under `path` in the repo at the given revision.
    func listFiles(repo: String, revision: String, path: String) async throws -> [PlannedFile] {
        _ = try endpoint("")
        var page: PaginatedResponse<Git.TreeEntry>?
        var files: [PlannedFile] = []
        let prefix = path.isEmpty ? "" : (path.hasSuffix("/") ? path : path + "/")
        // Fetch pages explicitly so a transient failure retries only the failed page.
        while let next = try await listingPage(after: page, repo: repo, revision: revision, path: path) {
            page = next
            for entry in next.items where entry.type == .file {
                let isSelectedFile = !path.isEmpty && entry.path == path
                guard isSelectedFile || entry.path.hasPrefix(prefix) else {
                    throw URLError(.badServerResponse)
                }
                let relative = isSelectedFile
                    ? (entry.path as NSString).lastPathComponent
                    : String(entry.path.dropFirst(prefix.count))
                let components = relative.split(separator: "/", omittingEmptySubsequences: false)
                guard !components.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }) else {
                    throw URLError(.badServerResponse)
                }
                files.append(PlannedFile(
                    url: try downloadURL(repo: repo, revision: revision, path: entry.path),
                    repoPath: entry.path, relativePath: relative,
                    size: Int64(entry.effectiveSize ?? 0)))
            }
        }
        return files
    }

    private func listingPage(
        after page: PaginatedResponse<Git.TreeEntry>?, repo: String, revision: String, path: String
    ) async throws -> PaginatedResponse<Git.TreeEntry>? {
        guard let id = Repo.ID(rawValue: repo) else {
            throw CoreAIKitError.notAHuggingFaceRepo(repo)
        }
        for attempt in 0..<5 {
            try Task.checkCancellation()
            if attempt > 0 {
                try await Task.sleep(for: .seconds(1 << attempt))
            }
            do {
                let result: PaginatedResponse<Git.TreeEntry>?
                if let page {
                    result = try await client.nextPage(after: page)
                } else {
                    result = try await client.listTree(
                        in: id, revision: revision, path: path, recursive: true)
                }
                try Task.checkCancellation()
                return result
            } catch HTTPClientError.responseError(let response, _) {
                let status = response.statusCode
                if status == 404 {
                    throw CoreAIKitError.variantNotFound(repo: repo, path: path, revision: revision)
                }
                guard attempt < 4, status == 429 || (500..<600).contains(status) else {
                    throw CoreAIKitError.httpError(statusCode: status, file: "\(repo)@\(revision)/\(path)")
                }
            }
        }
        throw URLError(.badServerResponse)
    }

    func download(
        _ file: PlannedFile, repo: String, revision: String, to destination: URL,
        onBytes: @escaping @Sendable (Int64) -> Void
    ) async throws {
        _ = try endpoint("")
        guard let id = Repo.ID(rawValue: repo) else {
            throw CoreAIKitError.notAHuggingFaceRepo(repo)
        }
        let progress = Progress(totalUnitCount: file.size)
        let observation = progress.observe(\.completedUnitCount, options: [.new]) { _, change in
            guard let completed = change.newValue else { return }
            onBytes(min(max(completed, 0), file.size))
        }
        defer { observation.invalidate() }
        try await downloadFile(file, from: id, revision: revision, to: destination, progress: progress)
    }

    private func downloadFile(
        _ file: PlannedFile, from id: Repo.ID, revision: String, to destination: URL, progress: Progress
    ) async throws {
        var resumeData: Data?
        for attempt in 0..<6 {
            try Task.checkCancellation()
            if attempt > 0 { try await Task.sleep(for: .milliseconds(1500)) }
            do {
                if let resumeData {
                    _ = try await client.resumeDownloadFile(
                        resumeData: resumeData, to: destination, progress: progress)
                } else {
                    _ = try await client.downloadFile(
                        at: file.repoPath, from: id, to: destination, revision: revision,
                        progress: progress)
                }
                return
            } catch HTTPClientError.responseError(let response, _) {
                resumeData = nil
                guard attempt < 5, response.statusCode == 429 || (500..<600).contains(response.statusCode) else {
                    throw CoreAIKitError.httpError(statusCode: response.statusCode, file: file.repoPath)
                }
            } catch let error as URLError {
                try Task.checkCancellation()
                guard attempt < 5 else { throw error }
                // Use the newest partial transfer, including after a resumed request fails.
                resumeData = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
            }
        }
    }
}
