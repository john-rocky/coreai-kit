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
    /// No room for the download. The three cases below are the failures a shipped app
    /// actually hits, and none of them existed until now: every case above is the vocabulary
    /// of someone who knows how a model repository is laid out, and none of them fire on the
    /// developer's own machine — where the device is supported, the disk has room, and the
    /// model is already cached from the last run.
    case insufficientStorage(needsBytes: Int64, freeBytes: Int64)
    case unsupportedDevice(reason: String)
    /// The model is fine; this device has not been given the OS-side assets it needs (a
    /// locale pack for the system transcriber, say) and could not fetch them.
    case systemAssetsUnavailable(what: String)

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
        case .insufficientStorage(let needs, let free):
            return "Not enough room: this needs \(CoreAIKitError.gigabytes(needs)) and "
                + "\(CoreAIKitError.gigabytes(free)) is free. Ask the user to free space, or "
                + "call CoreAI.capability(_:) before offering the feature so this is a prompt "
                + "rather than a failure."
        case .unsupportedDevice(let reason):
            return "This device cannot run the model: \(reason)"
        case .systemAssetsUnavailable(let what):
            return "The system could not provide \(what). These assets belong to the OS and "
                + "are shared between apps; a network connection is needed the first time."
        }
    }

    /// Sizes in an error are read by a person deciding what to do, so they are rounded to
    /// something a person says out loud.
    static func gigabytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_000_000_000
        return gb >= 1
            ? String(format: "%.1f GB", gb)
            : String(format: "%.0f MB", Double(bytes) / 1_000_000)
    }
}
