// VLEngine.swift — the on-device VLM (Qwen3-VL-2B via the kit's VL executor, `KitVisionModel`)
// shared by the system Visual Intelligence query and the in-app "Try it" mirror.
//
// This is the cloud-free counterpart to Visual Intelligence's own "ask," which sends the photo to a
// server model (ChatGPT). Here a converted VLM runs on the device GPU and the answer is computed
// on-device, offline. The model is memoized; the heavy decode graph compiles once and then persists
// across processes, so a warm background VI launch answers in ~1.4 s.

import CoreAIKit
import CoreGraphics
import CoreVideo
import Foundation
import FoundationModels
import ImageIO
import UniformTypeIdentifiers
import VideoToolbox

/// The VLM's answer about one captured frame, with its wall-clock latency.
struct VLAnswer: Sendable {
    let text: String
    let milliseconds: Double
}

final class VLEngine: @unchecked Sendable {
    /// Shared instance — registered as an App Intents dependency and used by the UI. In a background
    /// Visual Intelligence launch this is simply this process's instance.
    static let shared = VLEngine()

    private let lock = NSLock()
    private var visionModelTask: Task<KitVisionModel, Error>?

    /// Loads Qwen3-VL-2B through the kit's VL executor, memoized. First use downloads ~2.3 GB; a
    /// background VI launch has no bandwidth, so the model must already be cached (the in-app
    /// "Ask the VLM" button is the foreground step that downloads + warms it).
    private func visionModel() async throws -> KitVisionModel {
        let task = lock.withLock { () -> Task<KitVisionModel, Error> in
            if let visionModelTask { return visionModelTask }
            let t = Task { try await KitVisionModel(model: .qwen3VL2B) }
            visionModelTask = t
            return t
        }
        return try await task.value
    }

    /// One concise on-device answer about the captured frame. `instrumented` logs footprint +
    /// headroom around the load and inference so a background-launch jetsam is pinned by the last
    /// breadcrumb in the unified log.
    func caption(
        _ cgImage: CGImage,
        prompt: String = "Briefly describe the whole scene in this image in one sentence.",
        maximumResponseTokens: Int = 96,
        instrumented: Bool = false
    ) async throws -> VLAnswer {
        if instrumented { MemoryProbe.mark("vlm-load-begin") }
        let model = try await visionModel()
        if instrumented { MemoryProbe.mark("vlm-load-done") }

        let session = LanguageModelSession(model: model)
        let start = SuspendingClock.now
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
        return VLAnswer(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines), milliseconds: ms)
    }

    private static func milliseconds(since start: SuspendingClock.Instant) -> Double {
        let elapsed = (SuspendingClock.now - start).components
        return Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
    }

    // MARK: - Pixel buffer / image helpers

    /// Converts the Visual Intelligence `CVReadOnlyPixelBuffer` to a `CGImage` (format-agnostic).
    static func cgImage(from pixelBuffer: CVReadOnlyPixelBuffer) -> CGImage? {
        pixelBuffer.withUnsafeBuffer { buffer -> CGImage? in
            var out: CGImage?
            _ = VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &out)
            return out
        }
    }

    /// A small JPEG of the captured frame, for an answer entity's `DisplayRepresentation` image.
    static func previewJPEG(_ cgImage: CGImage, maxSide: Int = 200) -> Data? {
        let small = downscale(cgImage, maxSide: maxSide) ?? cgImage
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

    /// Decodes `data` to an upright (EXIF-applied) CGImage capped at `maxPixel` on the long side.
    static func uprightCGImage(from data: Data, maxPixel: Int = 2048) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(src, 0, nil)
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
}
