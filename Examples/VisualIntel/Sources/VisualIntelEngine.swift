// VisualIntelEngine.swift — the single on-device model engine shared by the in-app UI and the
// system Visual Intelligence query. THIS is "your model" in "your model behind Visual
// Intelligence": two converted Core AI bundles from the zoo, run through CoreAIKitVision —
//   • RF-DETR (object detection)  → "what is in this image?"
//   • CLIP ViT-B/32 (joint embed) → "find visually similar photos in my library"
// The system never sees the model: `IntentValueQuery.values(for:)` just hands us a pixel buffer
// and renders the `AppEntity`s we return. Everything inside is ours.
//
// Modeled as a `@unchecked Sendable` class (like the kit's `ImageTextEncoder`/`ObjectDetector`),
// not an actor: its methods take `CGImage` (which is not `Sendable`), so keeping them
// non-actor-isolated avoids forcing `CGImage` across an actor boundary. Lazy model loads are
// memoized as `Task`s under a lock (no `await` held while locked). Returned values are all
// `Sendable` (JPEG `Data` thumbnails, not `CGImage`) so they travel into the App Intents layer.

import CoreAIKit  // KitVisionModel — the VL executor that runs Qwen3-VL behind the FM stack.
import CoreAIKitVision
import CoreGraphics
import CoreVideo
import Foundation
import FoundationModels  // LanguageModelSession / Prompt / Attachment for the VLM turn.
import ImageIO
import UniformTypeIdentifiers
import VideoToolbox

/// One RF-DETR detection, ready to become a `DetectedObjectEntity`.
struct DetectionResult: Sendable {
    let classID: Int
    let label: String
    let score: Float
    /// Normalized box, origin top-left.
    let box: CGRect
    /// Small JPEG crop of the detected region, for a `DisplayRepresentation` image.
    let thumbnailJPEG: Data
}

/// One CLIP nearest-neighbor photo, ready to become a `PhotoMatchEntity`.
struct PhotoResult: Sendable {
    let id: String
    let score: Float
    let thumbnailJPEG: Data?
}

/// Both models' output for one captured frame.
struct AnalysisResult: Sendable {
    let detections: [DetectionResult]
    let photos: [PhotoResult]
}

/// The on-device VLM's answer about one captured frame (Qwen3-VL-2B via KitVisionModel), with its
/// own wall-clock latency. This is the cloud-free counterpart to Visual Intelligence's "ask" — no
/// network, no server model; the answer is computed on the device's GPU.
struct VLMAnswer: Sendable {
    let text: String
    let milliseconds: Double
}

/// The stages of `analyze`, surfaced to the in-app "Try it" mirror so it can draw a live trace
/// of the exact same work a Visual Intelligence query does. The system query passes no callback.
enum AnalysisPhase: Sendable { case detecting, matching, done }

final class VisualIntelEngine: @unchecked Sendable {
    /// Shared instance — registered as an App Intents dependency and used by the UI. In a
    /// background Visual Intelligence launch this is simply this process's instance.
    static let shared = VisualIntelEngine()

    /// CLIP ViT-B/32 embedding width. The index is created up front (cheap, just reads small
    /// files) so the query can tell whether photos are indexed before paying to load CLIP.
    private static let clipDimension = 512

    /// Thread-safe / `Sendable`, so the App Intents `EntityQuery` can read cached thumbnails
    /// directly without going through the engine.
    let index = PhotoFeaturePrintIndex(dimension: clipDimension)

    private let lock = NSLock()
    private var detectorTask: Task<ObjectDetector, Error>?
    private var encoderTask: Task<ImageTextEncoder, Error>?
    /// The VLM is heavy (~2 GB resident), so it is memoized and loaded ONLY when a caption is
    /// actually requested — never as a side effect of detection. Whether it loads inside the
    /// background Visual Intelligence query (vs only in the foreground) is the survival question
    /// this example measures.
    private var visionModelTask: Task<KitVisionModel, Error>?

    // MARK: - Lazy model loading (load only what a given query needs; memoized)

    private func detector() async throws -> ObjectDetector {
        // Scoped lock (returns before the `await`): raw lock()/unlock() are banned in async funcs.
        let task = lock.withLock { () -> Task<ObjectDetector, Error> in
            if let detectorTask { return detectorTask }
            // nano (384px, 103 MB) is the lightest variant — the right default for a query that
            // may run in a memory-constrained background launch. Swap to .rfdetrMedium for
            // accuracy if the device budget allows.
            let t = Task { try await ObjectDetector(model: .rfdetrNano, computeUnits: .gpu) }
            detectorTask = t
            return t
        }
        return try await task.value
    }

    private func encoder() async throws -> ImageTextEncoder {
        let task = lock.withLock { () -> Task<ImageTextEncoder, Error> in
            if let encoderTask { return encoderTask }
            let t = Task { () -> ImageTextEncoder in
                let e = try await ImageTextEncoder(
                    model: .clipViTB32, computeUnits: .neuralEngine)
                precondition(
                    e.embeddingDimension == Self.clipDimension,
                    "CLIP embedding width changed; update clipDimension and re-index")
                return e
            }
            encoderTask = t
            return t
        }
        return try await task.value
    }

    /// Loads Qwen3-VL-2B through the kit's VL executor (`KitVisionModel`), memoized. Uses the Hub
    /// preset, so first use downloads ~2.3 GB; a background Visual Intelligence launch has no
    /// bandwidth to download, so the model must already be cached (the in-app "Ask the VLM" button
    /// is the foreground step that downloads + caches it). HID is the only per-size difference;
    /// 2B is the iPhone-class default.
    private func visionModel() async throws -> KitVisionModel {
        let task = lock.withLock { () -> Task<KitVisionModel, Error> in
            if let visionModelTask { return visionModelTask }
            let t = Task { try await KitVisionModel(model: .qwen3VL2B) }
            visionModelTask = t
            return t
        }
        return try await task.value
    }

    // MARK: - On-device VLM (Qwen3-VL-2B) — the cloud-free "ask" surface

    /// One concise on-device answer about the captured frame. Apple's Visual Intelligence "ask"
    /// sends the image to ChatGPT (a server model); this runs a converted VLM on the device GPU
    /// instead — same idea, no cloud. `instrumented` logs `phys_footprint` + headroom around the
    /// load and inference so a background-launch jetsam is pinned by the last breadcrumb (G1).
    func caption(
        _ cgImage: CGImage,
        prompt: String = "Briefly describe the whole scene in this image in one sentence.",
        maximumResponseTokens: Int = 96,
        instrumented: Bool = false
    ) async throws -> VLMAnswer {
        if instrumented { MemoryProbe.mark("vlm-load-begin") }
        let model = try await visionModel()
        if instrumented { MemoryProbe.mark("vlm-load-done") }

        let session = LanguageModelSession(model: model)
        let start = SuspendingClock.now
        // Stream and keep the final snapshot — the proven VL path (matches the device-verified
        // backend); reasoning is stripped by the executor, so `content` is the clean answer.
        var text = ""
        let stream = session.streamResponse(
            to: Prompt {
                prompt
                Attachment(cgImage)
            },
            options: GenerationOptions(maximumResponseTokens: maximumResponseTokens))
        for try await snapshot in stream { text = snapshot.content }
        let ms = Self.milliseconds(since: start)
        if instrumented { MemoryProbe.mark("vlm-infer-done") }
        return VLMAnswer(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines), milliseconds: ms)
    }

    private static func milliseconds(since start: SuspendingClock.Instant) -> Double {
        let elapsed = (SuspendingClock.now - start).components
        return Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
    }

    // MARK: - Visual Intelligence query path

    /// Runs the detector over a captured image and returns the top detections by score (each
    /// object kept separately — multiple same-class objects, e.g. several cups, all surface).
    func detections(
        in cgImage: CGImage, scoreThreshold: Float = 0.35, maxResults: Int = 8
    ) async throws -> [DetectionResult] {
        let detector = try await detector()
        let raw = try await detector.detect(
            in: cgImage, scoreThreshold: scoreThreshold, maxDetections: 50)

        let imageSize = CGSize(width: cgImage.width, height: cgImage.height)
        return
            raw
            .sorted { $0.score > $1.score }
            .prefix(maxResults)
            .map { d in
                let crop = Self.crop(cgImage, normalized: d.box, imageSize: imageSize)
                let jpeg =
                    crop.flatMap { Self.jpeg(from: $0, maxSide: 160) }
                    ?? Self.jpeg(from: cgImage, maxSide: 160) ?? Data()
                return DetectionResult(
                    classID: d.classID, label: d.label, score: d.score, box: d.box,
                    thumbnailJPEG: jpeg)
            }
    }

    /// Embeds the captured image with CLIP and returns the nearest indexed photos. Returns
    /// empty (without loading CLIP) when nothing is indexed yet.
    func similarPhotos(
        to cgImage: CGImage, top k: Int = 6, minimumScore: Float = 0.20
    ) async throws -> [PhotoResult] {
        guard index.count > 0 else { return [] }
        let encoder = try await encoder()
        let vector = try await encoder.encode(image: cgImage)
        return index.search(vector, top: k)
            .filter { $0.score >= minimumScore }
            .map {
                PhotoResult(
                    id: $0.id, score: $0.score, thumbnailJPEG: index.thumbnailData(for: $0.id))
            }
    }

    /// Runs both models on one captured frame. The single entry point used by the Visual
    /// Intelligence query and the in-app "Try it" panel — the `CGImage` crosses into the engine
    /// exactly once. A failure in one model does not sink the other.
    func analyze(
        _ cgImage: CGImage, topPhotos: Int = 6, minimumPhotoScore: Float = 0.20,
        onPhase: (@Sendable (AnalysisPhase) -> Void)? = nil
    ) async throws -> AnalysisResult {
        onPhase?(.detecting)
        let detections = (try? await detections(in: cgImage)) ?? []
        onPhase?(.matching)
        let photos =
            (try? await similarPhotos(to: cgImage, top: topPhotos, minimumScore: minimumPhotoScore))
            ?? []
        onPhase?(.done)
        return AnalysisResult(detections: detections, photos: photos)
    }

    // MARK: - Indexing path (in-app, foreground)

    /// Embeds one library image with CLIP and stores it (vector + cached thumbnail) in the index.
    func addPhoto(id: String, image cgImage: CGImage) async throws {
        guard !index.contains(id) else { return }
        let encoder = try await encoder()
        let vector = try await encoder.encode(image: cgImage)
        let thumb = Self.downscale(cgImage, maxSide: 160) ?? cgImage
        index.add(id: id, vector: vector, thumbnail: thumb)
    }

    func saveIndex() { index.save() }

    // MARK: - Pixel buffer / image helpers

    /// A small JPEG of the captured frame, for an answer entity's `DisplayRepresentation` image.
    static func previewJPEG(_ cgImage: CGImage, maxSide: Int = 160) -> Data? {
        jpeg(from: cgImage, maxSide: maxSide)
    }

    /// Converts the Visual Intelligence `CVReadOnlyPixelBuffer` to a `CGImage` (format-agnostic).
    static func cgImage(from pixelBuffer: CVReadOnlyPixelBuffer) -> CGImage? {
        pixelBuffer.withUnsafeBuffer { buffer -> CGImage? in
            var out: CGImage?
            _ = VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &out)
            return out
        }
    }

    private static func crop(
        _ image: CGImage, normalized rect: CGRect, imageSize: CGSize
    ) -> CGImage? {
        let px =
            CGRect(
                x: rect.origin.x * imageSize.width,
                y: rect.origin.y * imageSize.height,
                width: rect.size.width * imageSize.width,
                height: rect.size.height * imageSize.height
            ).integral.intersection(CGRect(origin: .zero, size: imageSize))
        guard px.width >= 1, px.height >= 1 else { return nil }
        return image.cropping(to: px)
    }

    private static func downscale(_ image: CGImage, maxSide: Int) -> CGImage? {
        let w = image.width
        let h = image.height
        let longest = max(w, h)
        guard longest > maxSide else { return image }
        let scale = Double(maxSide) / Double(longest)
        let nw = max(1, Int(Double(w) * scale))
        let nh = max(1, Int(Double(h) * scale))
        guard
            let ctx = CGContext(
                data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        return ctx.makeImage()
    }

    private static func jpeg(from image: CGImage, maxSide: Int) -> Data? {
        let small = downscale(image, maxSide: maxSide) ?? image
        let data = NSMutableData()
        guard
            let dest = CGImageDestinationCreateWithData(
                data as CFMutableData, UTType.jpeg.identifier as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(
            dest, small, [kCGImageDestinationLossyCompressionQuality: 0.7] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
