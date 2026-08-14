// DepthEstimator.swift — typed monocular depth pipeline over GraphModel (Depth Anything
// 3 export). The shipped graph takes one square image [1, 3, R, R] as RAW [0, 1] RGB
// (ImageNet normalization is folded in-graph) and returns a relative depth map. It also
// accepts the legacy multi-view batch [1, N, 3, H, W], in which case a single photo is
// replicated across the view slots and view 0's depth is returned.

import CoreGraphics
import CoreVideo
import Foundation

public final class DepthEstimator: @unchecked Sendable {
    private let graph: GraphModel
    private let preprocessor: ImagePreprocessor
    private let pixelPreprocessor: PixelBufferPreprocessor
    private let imageInput: String
    private let imageShape: [Int]
    private let depthOutput: String

    /// Loads a bundle directory holding one `*.aimodel` (or the `.aimodel` itself).
    public init(
        bundleAt url: URL, computeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        let modelURL: URL
        if url.pathExtension == "aimodel" {
            modelURL = url
        } else {
            guard
                let found = try FileManager.default.contentsOfDirectory(
                    at: url, includingPropertiesForKeys: nil
                ).first(where: { $0.pathExtension == "aimodel" })
            else {
                throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
            }
            modelURL = found
        }
        let graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.graph = graph

        guard
            let imageInput = graph.inputNames.first(where: {
                let rank = graph.shape(ofInput: $0)?.count ?? 0
                return rank == 4 || rank == 5
            }),
            let imageShape = graph.shape(ofInput: imageInput)
        else {
            throw VisionError.bundleLayout(
                "could not identify the image input among \(graph.inputNames)")
        }
        self.imageInput = imageInput
        self.imageShape = imageShape

        guard
            let depthOutput = graph.outputNames.first(where: { $0.contains("depth") })
                ?? graph.outputNames.first(where: {
                    (graph.shape(ofOutput: $0)?.count ?? 0) >= 3
                })
        else {
            throw VisionError.bundleLayout(
                "could not identify the depth output among \(graph.outputNames)")
        }
        self.depthOutput = depthOutput

        // The shipped graph folds ImageNet normalization in-graph, so feed RAW [0, 1]
        // (identity mean/std). Double-normalizing here would corrupt the depth.
        let side = imageShape[imageShape.count - 1]
        self.preprocessor = ImagePreprocessor(
            size: side, mean: SIMD3(0, 0, 0), std: SIMD3(1, 1, 1))
        self.pixelPreprocessor = PixelBufferPreprocessor(self.preprocessor)
    }


    /// Loads a depth model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let estimator = try await DepthEstimator(catalog: "depth-anything-3-small")
    /// ```
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .depth)
        try await self.init(
            model: entry.modelID!, store: store, computeUnits: computeUnits,
            downloadProgress: downloadProgress)
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .depthAnything3Small,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    /// Relative depth for one image (preprocessing included). Higher = closer or
    /// farther depending on the model's convention; `DepthMap.cgImage` normalizes.
    public func estimateDepth(for image: CGImage) async throws -> DepthMap {
        try await estimateDepth(prepared(try preprocessor.chw(from: image)))
    }

    /// A frame preprocessed for this model, ready for `estimateDepth(_:)`. Lets a live
    /// pipeline run the CPU stage on frame N+1 while the GPU is still on frame N.
    public struct PreparedInput: Sendable {
        let pixels: [Float]
    }

    /// Real-time CPU half: vImage straight off a 32BGRA capture buffer, no `CGImage` and
    /// no `CIContext`. The live depth path used to render every frame into a bitmap before
    /// the model saw it; this is the same arithmetic without the render.
    public func prepare(_ pixelBuffer: CVPixelBuffer) throws -> PreparedInput {
        prepared(try pixelPreprocessor.chw(from: pixelBuffer))
    }

    /// Real-time both-halves, for a caller that is not pipelining.
    public func estimateDepth(for pixelBuffer: CVPixelBuffer) async throws -> DepthMap {
        try await estimateDepth(prepare(pixelBuffer))
    }

    /// The legacy multi-view graph takes N view slots; a single photo fills them all and
    /// view 0's depth comes back. Done once here so both input paths agree.
    private func prepared(_ chw: [Float]) -> PreparedInput {
        let views = max(imageShape.count == 5 ? imageShape[1] : 1, 1)
        guard views > 1 else { return PreparedInput(pixels: chw) }
        var pixels: [Float] = []
        pixels.reserveCapacity(chw.count * views)
        for _ in 0..<views { pixels += chw }
        return PreparedInput(pixels: pixels)
    }

    /// GPU half of the real-time path.
    public func estimateDepth(_ input: PreparedInput) async throws -> DepthMap {
        let outputs = try await graph.run(
            [imageInput: .float32(input.pixels, shape: imageShape)])
        guard let depth = outputs[depthOutput] else {
            throw VisionError.missingOutput(depthOutput)
        }
        let shape = depth.shape
        guard shape.count >= 2 else {
            throw VisionError.bundleLayout("unexpected depth shape \(shape)")
        }
        let height = shape[shape.count - 2]
        let width = shape[shape.count - 1]
        let flat = depth.floats()
        return DepthMap(
            width: width, height: height,
            values: Array(flat.prefix(width * height)))  // view 0
    }
}

/// A single-view relative depth map.
public struct DepthMap: Sendable {
    public let width: Int
    public let height: Int
    /// Row-major relative depth values.
    public let values: [Float]

    /// Min-max-normalized grayscale rendering (white = max value).
    public func cgImage() -> CGImage? {
        guard !values.isEmpty, width > 0, height > 0 else { return nil }
        var lo = Float.greatestFiniteMagnitude
        var hi = -Float.greatestFiniteMagnitude
        for v in values where v.isFinite {
            lo = min(lo, v)
            hi = max(hi, v)
        }
        let range = hi - lo
        var bytes = [UInt8](repeating: 0, count: width * height)
        if range > 0 {
            for i in 0..<bytes.count {
                bytes[i] = UInt8(max(0, min(255, (values[i] - lo) / range * 255)))
            }
        }
        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
    }
}
