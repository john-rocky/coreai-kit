import CoreAIKitVision
import Foundation
import Observation
import Photos

@MainActor
@Observable
final class SearchModel {
    enum Phase: Equatable {
        case starting
        case downloading(Double)
        case needsPhotoAccess
        case indexing(done: Int, total: Int)
        case ready
        case error(String)

        var label: String {
            switch self {
            case .starting: return "Loading model…"
            case .downloading(let f): return "Downloading CLIP… \(Int(f * 100))%"
            case .needsPhotoAccess: return "Photo access denied — enable it in Settings"
            case .indexing(let done, let total): return "Indexing \(done)/\(total)"
            case .ready: return "Ready"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    var phase: Phase = .starting
    var results: [PHAsset] = []
    var indexedCount = 0
    var searchMilliseconds: Double?

    private var encoder: ImageTextEncoder?
    private var store: EmbeddingStore?

    func start() async {
        guard encoder == nil else { return }

        let access = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard access == .authorized || access == .limited else {
            phase = .needsPhotoAccess
            return
        }

        do {
            let encoder = try await ImageTextEncoder { progress in
                Task { @MainActor in self.phase = .downloading(progress.fraction) }
            }
            self.encoder = encoder
            let store = EmbeddingStore(dimension: encoder.embeddingDimension)
            self.store = store
            indexedCount = store.count
            phase = .ready

            // Index in the background; search works on whatever is indexed so far.
            let indexer = PhotoIndexer(encoder: encoder)
            Task {
                do {
                    try await indexer.indexLibrary(into: store) { progress in
                        Task { @MainActor in
                            self.indexedCount = progress.done
                            self.phase = progress.done < progress.total
                                ? .indexing(done: progress.done, total: progress.total)
                                : .ready
                        }
                    }
                } catch {
                    await MainActor.run { self.phase = .error(error.localizedDescription) }
                }
            }
        } catch {
            phase = .error(error.localizedDescription)
        }
    }

    func search(_ query: String) async {
        guard let encoder, let store else { return }
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            results = []
            searchMilliseconds = nil
            return
        }
        do {
            let start = SuspendingClock.now
            let vector = try await encoder.encode(text: trimmed)
            let hits = store.search(vector, top: 60)
            let elapsed = (SuspendingClock.now - start).components
            searchMilliseconds =
                Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15

            let fetched = PHAsset.fetchAssets(
                withLocalIdentifiers: hits.map(\.id), options: nil)
            var byId: [String: PHAsset] = [:]
            fetched.enumerateObjects { asset, _, _ in byId[asset.localIdentifier] = asset }
            results = hits.compactMap { byId[$0.id] }
        } catch {
            phase = .error(error.localizedDescription)
        }
    }
}
