// HubClient.swift — minimal Hugging Face Hub access: subtree listing + resolve URLs.
//
// The tree listing follows the API's pagination (`Link: <…>; rel="next"`). A JIT bundle
// subtree is a handful of files (~12); an AOT `.aimodelc` subtree is ~50 (one compiled
// region per graph function), and a chunked model multiplies that — the store must see every
// file or the bundle it assembles is incomplete.

import Foundation

struct HubClient: Sendable {
    let baseURL: URL

    init(baseURL: URL = URL(string: "https://huggingface.co")!) {
        self.baseURL = baseURL
    }

    struct PlannedFile: Sendable {
        let url: URL
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

    func listingURL(repo: String, revision: String, path: String) throws -> URL {
        // Empty path = repo root: no trailing slash, or the Hub API returns 404.
        let treePath = path.isEmpty ? "" : "/\(path)"
        let url = try endpoint("api/models/\(repo)/tree/\(revision)\(treePath)")
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        return components.url!
    }

    func downloadURL(repo: String, revision: String, path: String) throws -> URL {
        try endpoint("\(repo)/resolve/\(revision)/\(path)")
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

    private struct TreeEntry: Decodable {
        let type: String
        let path: String
        let size: Int64?
        let lfs: LFS?
        struct LFS: Decodable { let size: Int64? }
    }

    /// Enumerates the files under `path` in the repo at the given revision, every page.
    func listFiles(repo: String, revision: String, path: String) async throws -> [PlannedFile] {
        var page: URL? = try listingURL(repo: repo, revision: revision, path: path)
        var entries: [TreeEntry] = []
        while let url = page {
            let (data, response) = try await fetchPage(url, repo: repo, revision: revision, path: path)
            entries += try JSONDecoder().decode([TreeEntry].self, from: data)
            page = Self.nextPageURL(fromLinkHeader: response.value(forHTTPHeaderField: "Link"))
        }
        let prefix = path.isEmpty ? "" : (path.hasSuffix("/") ? path : path + "/")
        return try entries.filter { $0.type == "file" }.map { e in
            let rel = (!path.isEmpty && e.path == path)
                ? (e.path as NSString).lastPathComponent
                : String(e.path.dropFirst(prefix.count))
            let url = try downloadURL(repo: repo, revision: revision, path: e.path)
            return PlannedFile(url: url, relativePath: rel, size: e.lfs?.size ?? e.size ?? 0)
        }
    }

    /// One tree page. Hub tree requests can be rate-limited even when every bundle subtree
    /// exists: five attempts total, with cancellable 2/4/8/16-second waits between them.
    /// File transfers have their own resume/retry loop in ModelStore.
    private func fetchPage(
        _ api: URL, repo: String, revision: String, path: String
    ) async throws -> (Data, HTTPURLResponse) {
        for attempt in 0..<5 {
            try Task.checkCancellation()
            if attempt > 0 {
                try await Task.sleep(nanoseconds: (UInt64(1) << attempt) * 1_000_000_000)
            }
            let (data, response) = try await URLSession.shared.data(from: api)
            try Task.checkCancellation()
            let http = response as? HTTPURLResponse
            let status = http?.statusCode ?? -1
            if status == 200, let http { return (data, http) }
            if status == 404 {
                throw CoreAIKitError.variantNotFound(repo: repo, path: path, revision: revision)
            }
            guard attempt < 4, status == 429 || (500..<600).contains(status) else {
                // Authentication and other permanent errors are not missing variants.
                throw CoreAIKitError.httpError(statusCode: status, file: "\(repo)@\(revision)/\(path)")
            }
        }
        throw CoreAIKitError.httpError(statusCode: -1, file: "\(repo)@\(revision)/\(path)")
    }

    /// The `rel="next"` target of an RFC 8288 `Link` header, or nil when the listing is on
    /// its last page. Tolerates `rel=next` unquoted and several comma-separated links.
    static func nextPageURL(fromLinkHeader header: String?) -> URL? {
        guard let header else { return nil }
        for link in header.split(separator: ",") {
            let fields = link.split(separator: ";").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard let target = fields.first, target.hasPrefix("<"), target.hasSuffix(">") else {
                continue
            }
            let isNext = fields.dropFirst().contains {
                $0.replacingOccurrences(of: "\"", with: "").lowercased() == "rel=next"
            }
            if isNext { return URL(string: String(target.dropFirst().dropLast())) }
        }
        return nil
    }
}
