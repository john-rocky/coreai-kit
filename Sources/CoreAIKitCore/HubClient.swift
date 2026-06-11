// HubClient.swift — minimal Hugging Face Hub access: subtree listing + resolve URLs.
//
// The tree API is not paginated here: bundle subtrees hold a handful of files (~12), well
// under the API's page size.

import Foundation

struct HubClient: Sendable {
    struct PlannedFile: Sendable {
        let url: URL
        let relativePath: String
        let size: Int64
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
        guard let api = URL(string:
            "https://huggingface.co/api/models/\(repo)/tree/\(revision)/\(path)?recursive=true")
        else {
            throw CoreAIKitError.variantNotFound(repo: repo, path: path, revision: revision)
        }
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
        let prefix = path.hasSuffix("/") ? path : path + "/"
        return try entries.filter { $0.type == "file" }.map { e in
            let rel = e.path == path
                ? (e.path as NSString).lastPathComponent
                : String(e.path.dropFirst(prefix.count))
            guard let url = URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/\(e.path)")
            else {
                throw CoreAIKitError.httpError(statusCode: -1, file: e.path)
            }
            return PlannedFile(url: url, relativePath: rel, size: e.lfs?.size ?? e.size ?? 0)
        }
    }
}
