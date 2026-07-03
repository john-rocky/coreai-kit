// ActionRecognizer.swift — video action recognition over GraphModel (V-JEPA 2 ViT-L export,
// Something-Something v2 head). One stateless graph: a 16-frame clip as
// `pixel_values_videos [1, 16, 3, 256, 256]` (aspect-fill center crop, ImageNet
// normalization on the host) → `logits [1, 174]`; `labels.json` in the same bundle names
// the classes. macOS ships the JIT `.aimodel`, iOS the AOT `.aimodelc` — both load through
// the same `GraphModel`.

import Accelerate
import CoreGraphics
import Foundation

public final class ActionRecognizer: @unchecked Sendable {
    /// One ranked class prediction for a clip.
    public struct Prediction: Sendable, Identifiable {
        /// Class index in the model's head.
        public let id: Int
        public let label: String
        public let probability: Float
    }

    /// Frames per clip the graph expects.
    public static let frameCount = 16

    private static let side = 256
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]

    private let graph: GraphModel
    private let labels: [String]

    /// Loads a bundle directory holding the model (`*.aimodel` on macOS, `*.aimodelc` on
    /// iOS) and its `labels.json`.
    public init(
        bundleAt root: URL, computeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        let entries = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        guard
            let modelURL = entries.first(where: {
                $0.pathExtension == "aimodel" || $0.pathExtension == "aimodelc"
            })
        else {
            throw VisionError.bundleLayout("no .aimodel/.aimodelc found under \(root.path)")
        }
        graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)

        let labelsData = try Data(contentsOf: root.appendingPathComponent("labels.json"))
        let map = try JSONDecoder().decode([String: String].self, from: labelsData)
        labels = (0..<map.count).map { map[String($0)] ?? "class \($0)" }
    }

    /// Loads an action-recognition model by its catalog id — the id shown on the model's
    /// card:
    ///
    /// ```swift
    /// let recognizer = try await ActionRecognizer(catalog: "vjepa2-vitl-ssv2")
    /// ```
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .video)
        let url = try await store.download(entry.modelID!, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    /// The class names of the model's head, by class index.
    public var classLabels: [String] { labels }

    /// Classifies a clip given as frames (any sizes; aspect-fill center crop + ImageNet
    /// normalization happen here). Other frame counts are uniformly resampled to 16.
    public func classify(frames: [CGImage], topK: Int = 3) async throws -> [Prediction] {
        guard !frames.isEmpty else {
            throw VisionError.bundleLayout("classify(frames:) needs at least one frame")
        }
        let clip: [CGImage]
        if frames.count == Self.frameCount {
            clip = frames
        } else {
            clip = (0..<Self.frameCount).map {
                frames[$0 * frames.count / Self.frameCount]
            }
        }
        let tensor = try Self.tensor(from: clip)
        let outputs = try await graph.run([
            "pixel_values_videos": .float32(
                tensor, shape: [1, Self.frameCount, 3, Self.side, Self.side])
        ])
        guard let logits = outputs["logits"]?.floats() else {
            throw VisionError.missingOutput("logits")
        }

        let maxLogit = logits.max() ?? 0
        let exps = logits.map { expf($0 - maxLogit) }
        let sum = exps.reduce(0, +)
        return exps.enumerated()
            .map {
                Prediction(
                    id: $0.offset,
                    label: labels.indices.contains($0.offset)
                        ? labels[$0.offset] : "class \($0.offset)",
                    probability: $0.element / sum)
            }
            .sorted { $0.probability > $1.probability }
            .prefix(topK).map { $0 }
    }

    /// Classifies a video file: 16 evenly-spaced frames → `classify(frames:)`.
    public func classify(videoAt url: URL, topK: Int = 3) async throws -> [Prediction] {
        let frames = try await VideoFile.frames(Self.frameCount, from: url)
        return try await classify(frames: frames, topK: topK)
    }

    /// Frames (any sizes) → normalized `[16, 3, 256, 256]` floats: sRGB draw with
    /// aspect-fill center crop, then `(x/255 − mean)/std` per channel via vDSP.
    private static func tensor(from frames: [CGImage]) throws -> [Float] {
        let s = side
        let plane = s * s
        var out = [Float](repeating: 0, count: frameCount * 3 * plane)
        var rgba = [UInt8](repeating: 0, count: plane * 4)
        var channel = [Float](repeating: 0, count: plane)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            throw VisionError.imageRenderFailed
        }
        for (t, image) in frames.prefix(frameCount).enumerated() {
            try rgba.withUnsafeMutableBytes { buffer in
                guard
                    let ctx = CGContext(
                        data: buffer.baseAddress, width: s, height: s, bitsPerComponent: 8,
                        bytesPerRow: s * 4, space: colorSpace,
                        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                else {
                    throw VisionError.imageRenderFailed
                }
                let w = CGFloat(image.width)
                let h = CGFloat(image.height)
                let scale = CGFloat(s) / min(w, h)
                let dw = w * scale
                let dh = h * scale
                ctx.interpolationQuality = .medium
                ctx.draw(
                    image,
                    in: CGRect(
                        x: (CGFloat(s) - dw) / 2, y: (CGFloat(s) - dh) / 2,
                        width: dw, height: dh))
            }
            let base = t * 3 * plane
            out.withUnsafeMutableBufferPointer { dst in
                rgba.withUnsafeBufferPointer { src in
                    for c in 0..<3 {
                        var a = (1.0 as Float / 255.0) / std[c]
                        var b = -mean[c] / std[c]
                        vDSP_vfltu8(
                            src.baseAddress! + c, 4, &channel, 1, vDSP_Length(plane))
                        vDSP_vsmsa(
                            channel, 1, &a, &b, dst.baseAddress! + base + c * plane, 1,
                            vDSP_Length(plane))
                    }
                }
            }
        }
        return out
    }
}
