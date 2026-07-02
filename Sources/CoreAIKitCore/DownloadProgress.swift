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
    case modelNotInCatalog(id: String)
    case modelNotAvailableOnPlatform(id: String)
    case catalogKindMismatch(id: String, expected: String, found: String)

    public var errorDescription: String? {
        switch self {
        case .notAHuggingFaceRepo(let s):
            return "Not a Hugging Face repo URL or id: \(s)"
        case .variantNotFound(let repo, let path, let revision):
            return "No '\(path)' variant in \(repo)@\(revision) — "
                + "this model may not be published for this platform."
        case .httpError(let code, let file):
            return "HTTP \(code) while downloading \(file)"
        case .modelNotInCatalog(let id):
            return "No model with catalog id '\(id)' — check the id on the model's card "
                + "(a newly published id may need a newer coreai-kit)."
        case .modelNotAvailableOnPlatform(let id):
            return "Model '\(id)' is not published for this platform."
        case .catalogKindMismatch(let id, let expected, let found):
            return "Model '\(id)' is a '\(found)' model, not '\(expected)' — "
                + "load it with the matching Kit model type."
        }
    }
}
