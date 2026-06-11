import CoreAIKitVision
import CoreGraphics
import Photos

/// Walks the photo library and embeds every photo not yet in the store.
actor PhotoIndexer {
    struct Progress: Sendable {
        let done: Int
        let total: Int
    }

    private let encoder: ImageTextEncoder

    init(encoder: ImageTextEncoder) {
        self.encoder = encoder
    }

    func indexLibrary(
        into store: EmbeddingStore,
        progress: @escaping @Sendable (Progress) -> Void
    ) async throws {
        let fetch = PHAsset.fetchAssets(with: .image, options: nil)
        var assets: [PHAsset] = []
        assets.reserveCapacity(fetch.count)
        fetch.enumerateObjects { asset, _, _ in assets.append(asset) }

        let total = assets.count
        var done = 0
        let manager = PHImageManager.default()
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat  // exactly one callback per request
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = false

        for asset in assets {
            defer { done += 1 }
            if store.contains(asset.localIdentifier) { continue }
            guard let image = await thumbnail(of: asset, manager: manager, options: options)
            else { continue }
            let vector = try await encoder.encode(image: image)
            store.add(id: asset.localIdentifier, vector: vector)
            if done % 64 == 0 { store.save() }
            if done % 8 == 0 { progress(Progress(done: done, total: total)) }
        }
        store.save()
        progress(Progress(done: total, total: total))
    }

    private func thumbnail(
        of asset: PHAsset, manager: PHImageManager, options: PHImageRequestOptions
    ) async -> CGImage? {
        await withCheckedContinuation { continuation in
            manager.requestImage(
                for: asset,
                targetSize: CGSize(width: 224, height: 224),
                contentMode: .aspectFill,
                options: options
            ) { image, _ in
                continuation.resume(returning: image?.cgImage)
            }
        }
    }
}
