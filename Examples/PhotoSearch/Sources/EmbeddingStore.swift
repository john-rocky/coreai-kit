import Accelerate
import Foundation

/// Flat in-memory embedding matrix (rows × dimension) with binary persistence.
/// Vectors are L2-normalized, so cosine similarity is one matrix·vector multiply.
final class EmbeddingStore: @unchecked Sendable {
    private let lock = NSLock()
    private var ids: [String] = []
    private var known: Set<String> = []
    private var matrix: [Float] = []

    let dimension: Int
    private let matrixURL: URL
    private let idsURL: URL

    init(dimension: Int) {
        self.dimension = dimension
        let dir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("PhotoSearch", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.matrixURL = dir.appendingPathComponent("embeddings-\(dimension).f32")
        self.idsURL = dir.appendingPathComponent("ids-\(dimension).json")
        load()
    }

    var count: Int {
        lock.withLock { ids.count }
    }

    func contains(_ id: String) -> Bool {
        lock.withLock { known.contains(id) }
    }

    func add(id: String, vector: [Float]) {
        lock.withLock {
            guard !known.contains(id), vector.count == dimension else { return }
            ids.append(id)
            known.insert(id)
            matrix += vector
        }
    }

    /// Top-k ids by dot product (= cosine on normalized vectors).
    func search(_ query: [Float], top k: Int) -> [(id: String, score: Float)] {
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
                .map { (ids[$0], scores[$0]) }
        }
    }

    func save() {
        lock.withLock {
            let data = matrix.withUnsafeBufferPointer { Data(buffer: $0) }
            try? data.write(to: matrixURL, options: .atomic)
            try? JSONEncoder().encode(ids).write(to: idsURL, options: .atomic)
        }
    }

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
}
