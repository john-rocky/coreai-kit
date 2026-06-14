// PhotoLibraryIndexer.swift — walks the photo library and embeds each image with the engine's
// CLIP model, populating the feature-print index that the Visual Intelligence "similar photos"
// surface searches. Mirrors the proven PhotoSearch example; the only difference is it stores a
// cached thumbnail alongside each embedding (so the out-of-process VI query needs no PhotoKit).

import CoreGraphics
import Foundation
import ImageIO
import Photos

struct PhotoLibraryIndexer {
    enum IndexerError: LocalizedError {
        case accessDenied
        var errorDescription: String? {
            "Photo library access was denied — enable it in Settings to index your photos."
        }
    }

    /// Indexes up to `limit` images. `progress` is called with (done, total).
    func index(
        into engine: VisualIntelEngine,
        limit: Int = 300,
        progress: @escaping @Sendable (Int, Int) -> Void
    ) async throws {
        let access = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard access == .authorized || access == .limited else { throw IndexerError.accessDenied }

        let fetch = PHAsset.fetchAssets(with: .image, options: nil)
        var assets: [PHAsset] = []
        assets.reserveCapacity(fetch.count)
        fetch.enumerateObjects { asset, _, _ in assets.append(asset) }
        if assets.count > limit { assets = Array(assets.prefix(limit)) }

        let total = assets.count
        let manager = PHImageManager.default()
        let options = PHImageRequestOptions()
        options.deliveryMode = .highQualityFormat  // exactly one callback per request
        options.isNetworkAccessAllowed = false

        var done = 0
        for asset in assets {
            defer { done += 1 }
            let id = asset.localIdentifier
            if engine.index.contains(id) {
                progress(done + 1, total)
                continue
            }
            guard let cgImage = await Self.image(for: asset, manager: manager, options: options)
            else {
                progress(done + 1, total)
                continue
            }
            try await engine.addPhoto(id: id, image: cgImage)
            if done % 16 == 0 { await engine.saveIndex() }
            progress(done + 1, total)
        }
        await engine.saveIndex()
    }

    /// Loads each asset as raw `Data` (works identically on iOS and macOS — `NSImage` has no
    /// `cgImage` property), then makes a downscaled `CGImage` thumbnail via ImageIO.
    private static func image(
        for asset: PHAsset, manager: PHImageManager, options: PHImageRequestOptions
    ) async -> CGImage? {
        await withCheckedContinuation { continuation in
            manager.requestImageDataAndOrientation(for: asset, options: options) {
                data, _, _, _ in
                guard let data, let source = CGImageSourceCreateWithData(data as CFData, nil)
                else {
                    continuation.resume(returning: nil)
                    return
                }
                let thumbOptions: [CFString: Any] = [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                    kCGImageSourceThumbnailMaxPixelSize: 512,
                ]
                let image = CGImageSourceCreateThumbnailAtIndex(
                    source, 0, thumbOptions as CFDictionary)
                continuation.resume(returning: image)
            }
        }
    }
}
