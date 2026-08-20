// GraphBundle.swift — one rule for finding the graph inside a downloaded bundle directory,
// shared by every runtime that pairs a decoder with a second graph (VL tower, ASR/audio
// encoder). Each of those used to carry its own copy of this, and the copies drifted: the
// VL one looked for `.aimodel` alone, so an iOS bundle that ships only the AOT form was
// handed the path of a file that was never published ("Missing hash file").

import Foundation

/// A bundle directory that does not hold the graph it is supposed to hold.
public enum KitBundleError: Error, LocalizedError {
    case graphMissing(URL)

    public var errorDescription: String? {
        switch self {
        case .graphMissing(let url):
            return "Bundle \(url.lastPathComponent) holds no .aimodel or .aimodelc graph."
        }
    }
}

enum GraphBundle {
    /// The graph at `url`, or the one inside it when `url` is the bundle directory holding it.
    ///
    /// AOT (`.aimodelc`) wins over JIT (`.aimodel`) so a bundle carrying both skips the
    /// on-device specialization the AOT build exists to remove.
    static func resolve(in url: URL) throws -> URL {
        if url.pathExtension == "aimodel" || url.pathExtension == "aimodelc" { return url }
        let items = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil)
        if let aot = items.first(where: { $0.pathExtension == "aimodelc" }) { return aot }
        guard let jit = items.first(where: { $0.pathExtension == "aimodel" }) else {
            throw KitBundleError.graphMissing(url)
        }
        return jit
    }
}
