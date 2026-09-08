// HubClient.swift — minimal Hugging Face Hub access: subtree listing + resolve URLs.
//
// The tree API is not paginated here: bundle subtrees hold a handful of files (~12), well
// under the API's page size.

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

    /// Enumerates the files under `path` in the repo at the given revision.
    func listFiles(repo: String, revision: String, path: String) async throws -> [PlannedFile] {
        let api = try listingURL(repo: repo, revision: revision, path: path)
        let (data, resp) = try await URLSession.shared.data(from: api)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw CoreAIKitError.variantNotFound(repo: repo, path: path, revision: revision)
        }

        struct TreeEntry: Decodable {
            let type: String
            let path: String
            let size: Int64?
            let lfs: LFS?
            struct LFS: Decodable { let size: Int64? }
        }
        let entries = try JSONDecoder().decode([TreeEntry].self, from: data)
        let prefix = path.isEmpty ? "" : (path.hasSuffix("/") ? path : path + "/")
        return try entries.filter { $0.type == "file" }.map { e in
            let rel = (!path.isEmpty && e.path == path)
                ? (e.path as NSString).lastPathComponent
                : String(e.path.dropFirst(prefix.count))
            let url = try downloadURL(repo: repo, revision: revision, path: e.path)
            return PlannedFile(url: url, relativePath: rel, size: e.lfs?.size ?? e.size ?? 0)
        }
    }
}
