import Foundation

/// Byte-level progress of a model download, throttled to ~500 callbacks per download.
public struct DownloadProgress: Sendable, Equatable {
    public let fraction: Double
    public let completedBytes: Int64
    public let totalBytes: Int64
    public let currentFile: String
}

public enum CoreAIKitError: Error, LocalizedError, Sendable {
    case notAHuggingFaceRepo(String)
    case variantNotFound(repo: String, path: String, revision: String)
    case httpError(statusCode: Int, file: String)

    public var errorDescription: String? {
        switch self {
        case .notAHuggingFaceRepo(let s):
            return "Not a Hugging Face repo URL or id: \(s)"
        case .variantNotFound(let repo, let path, let revision):
            return "No '\(path)' variant in \(repo)@\(revision) — "
                + "this model may not be published for this platform."
        case .httpError(let code, let file):
            return "HTTP \(code) while downloading \(file)"
        }
    }
}
