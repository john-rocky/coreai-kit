// PhotoFeaturePrintIndex.swift — a persisted on-device feature-print index built with YOUR
// CLIP model instead of Vision's `GenerateImageFeaturePrintRequest`.
//
// Apple's Visual Intelligence sample computes similarity with `VNGenerateImageFeaturePrintRequest`.
// Here every entry is a CLIP embedding produced by our converted Core AI bundle, so the
// system's "find visually similar" surface is backed by a model we trained/converted and can
// swap. Vectors are L2-normalized in-graph, so cosine similarity is a single matrix·vector.
//
// Each entry also caches a small JPEG thumbnail next to its embedding. This matters for the
// Visual Intelligence integration: the `IntentValueQuery` runs in a background launch of the
// app, and building a `DisplayRepresentation` image there must be cheap — we read a precomputed
// thumbnail from the app container rather than round-tripping through PhotoKit out-of-process.

import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Flat in-memory embedding matrix (rows × dimension) with binary persistence plus a JPEG
/// thumbnail cache. Thread-safe; shared between the in-app indexer and the background query.
final class PhotoFeaturePrintIndex: @unchecked Sendable {
    struct Match: Sendable {
        let id: String
        let score: Float
    }

    private let lock = NSLock()
    private var ids: [String] = []
    private var known: Set<String> = []
    private var matrix: [Float] = []

    let dimension: Int
    private let dir: URL
    private let matrixURL: URL
    private let idsURL: URL
    private let thumbsDir: URL

    init(dimension: Int) {
        self.dimension = dimension
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("VisualIntel", isDirectory: true)
        self.dir = base
        self.matrixURL = base.appendingPathComponent("embeddings-\(dimension).f32")
        self.idsURL = base.appendingPathComponent("ids-\(dimension).json")
        self.thumbsDir = base.appendingPathComponent("thumbs", isDirectory: true)
        try? FileManager.default.createDirectory(at: thumbsDir, withIntermediateDirectories: true)
        load()
    }

    var count: Int { lock.withLock { ids.count } }

    func contains(_ id: String) -> Bool { lock.withLock { known.contains(id) } }

    /// Adds one entry: its embedding plus a cached thumbnail (written to disk immediately).
    func add(id: String, vector: [Float], thumbnail: CGImage) {
        guard vector.count == dimension else { return }
        writeThumbnail(thumbnail, for: id)
        lock.withLock {
            guard !known.contains(id) else { return }
            ids.append(id)
            known.insert(id)
            matrix += vector
        }
    }

    /// Top-k ids by dot product (= cosine on normalized vectors).
    func search(_ query: [Float], top k: Int) -> [Match] {
        lock.withLock {
            let rows = ids.count
            guard rows > 0, query.count == dimension else { return [] }
            var scores = [Float](repeating: 0, count: rows)
            vDSP_mmul(
                matrix, 1, query, 1, &scores, 1,
                vDSP_Length(rows), 1, vDSP_Length(dimension))
            return scores.indices
                .sorted { scores[$0] > scores[$1] }
                .prefix(k)
                .map { Match(id: ids[$0], score: scores[$0]) }
        }
    }

    /// Cached thumbnail bytes for an indexed id, for a `DisplayRepresentation` image.
    func thumbnailData(for id: String) -> Data? {
        try? Data(contentsOf: thumbURL(for: id))
    }

    func save() {
        lock.withLock {
            let data = matrix.withUnsafeBufferPointer { Data(buffer: $0) }
            try? data.write(to: matrixURL, options: .atomic)
            try? JSONEncoder().encode(ids).write(to: idsURL, options: .atomic)
        }
    }

    // MARK: - Private

    private func load() {
        guard let idData = try? Data(contentsOf: idsURL),
            let loadedIds = try? JSONDecoder().decode([String].self, from: idData),
            let matrixData = try? Data(contentsOf: matrixURL),
            matrixData.count == loadedIds.count * dimension * MemoryLayout<Float>.size
        else { return }
        ids = loadedIds
        known = Set(loadedIds)
        matrix = matrixData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// A filesystem-safe filename derived from the (possibly path-like) photo identifier.
    private func thumbURL(for id: String) -> URL {
        let safe = Data(id.utf8).base64EncodedString()
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "=", with: "")
        return thumbsDir.appendingPathComponent("\(safe).jpg")
    }

    private func writeThumbnail(_ image: CGImage, for id: String) {
        let url = thumbURL(for: id)
        guard
            let dest = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { return }
        let options: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.7]
        CGImageDestinationAddImage(dest, image, options as CFDictionary)
        CGImageDestinationFinalize(dest)
    }
}
