// YOLOXDetector.swift — typed real-time object detection over a YOLOX (Megvii) GraphModel
// export. The graph is a single static function: image [1, 3, S, S] float32 in the
// YOLOX-native space (BGR, 0-255, letterboxed with pad value 114, top-left aligned —
// NO /255, NO mean/std) -> preds [1, A, 4+1+C] where A = (S/8)^2 + (S/16)^2 + (S/32)^2
// and the columns are [cx, cy, w, h, obj, cls_0 .. cls_(C-1)]. The box is already
// grid+stride DECODED to S-space pixels and obj/cls are already SIGMOID-ed in-graph.
//
// Unlike the DETR-family ObjectDetector (no NMS), YOLOX is a dense anchor-free
// detector: host decode is `score = obj * cls`, threshold, per-class greedy NMS, then
// un-letterbox the survivors back to normalized coordinates of the SOURCE frame.
//
// Shares `Detection`, `VisionError`, the COCO class table, and the GraphModel runner
// with ObjectDetector; only preprocessing (BGR letterbox vs RGB square resize) and
// decode (obj*cls + NMS vs sigmoid + top-k) differ.

import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

public final class YOLOXDetector: @unchecked Sendable {
    private let graph: GraphModel
    private let imageInput: String
    private let imageShape: [Int]
    private let predsOutput: String
    /// Model input side length (e.g. 640 for YOLOX-S).
    public let inputSize: Int
    /// Number of object classes the head predicts (80 for COCO).
    public let numClasses: Int

    /// YOLOX class index (contiguous 0..79) -> original COCO category id. The COCO_CLASSES
    /// order is ascending category id, so this is just the sorted id list; names resolve
    /// through `ObjectDetector.cocoClasses[id]`.
    private static let cocoIDs: [Int] = ObjectDetector.cocoClasses.keys.sorted()

    /// Loads a directory holding one `*.aimodel` (or the `.aimodel` itself).
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
                (graph.shape(ofInput: $0)?.count ?? 0) == 4
            }),
            let imageShape = graph.shape(ofInput: imageInput)
        else {
            throw VisionError.bundleLayout(
                "could not identify the image input among \(graph.inputNames)")
        }
        self.imageInput = imageInput
        self.imageShape = imageShape
        self.inputSize = imageShape[imageShape.count - 1]

        // The single 3-D output [1, A, 4+1+C]. Identify by rank, not last==4 (YOLOX's
        // last dim is 85, which is exactly how it differs from the DETR contract).
        guard
            let preds = graph.outputNames.first(where: {
                (graph.shape(ofOutput: $0)?.count ?? 0) == 3
            }),
            let predsShape = graph.shape(ofOutput: preds),
            let stride = predsShape.last, stride > 5
        else {
            throw VisionError.bundleLayout(
                "could not identify YOLOX preds [1,A,4+1+C] among \(graph.outputNames)")
        }
        self.predsOutput = preds
        self.numClasses = stride - 5
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .yoloxS,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    /// A preprocessed frame: letterboxed BGR pixels plus the geometry needed to map
    /// detections back (ratio + source dimensions). Carrying geometry per-input keeps
    /// the detector immutable and correct across differently-sized inputs.
    public struct PreparedInput: Sendable {
        let pixels: [Float]
        let ratio: Float
        let srcW: Int
        let srcH: Int
    }

    /// CPU half of the fast path (vImage letterbox + channel split). Thread-safe.
    public func prepare(_ pixelBuffer: CVPixelBuffer) throws -> PreparedInput {
        try letterbox(pixelBuffer)
    }

    /// Real-time fast path: detect objects in a 32BGRA capture buffer.
    public func detect(
        in pixelBuffer: CVPixelBuffer,
        scoreThreshold: Float = 0.3,
        nmsThreshold: Float = 0.45,
        maxDetections: Int = 50
    ) async throws -> [Detection] {
        try await detect(
            prepare(pixelBuffer), scoreThreshold: scoreThreshold,
            nmsThreshold: nmsThreshold, maxDetections: maxDetections)
    }

    /// GPU half of the fast path.
    public func detect(
        _ input: PreparedInput,
        scoreThreshold: Float = 0.3,
        nmsThreshold: Float = 0.45,
        maxDetections: Int = 50
    ) async throws -> [Detection] {
        let outputs = try await graph.run(
            [imageInput: .float32(input.pixels, shape: imageShape)])
        guard let preds = outputs[predsOutput] else {
            throw VisionError.missingOutput(predsOutput)
        }
        let p = preds.floats()
        let anchors = preds.shape[1]
        let stride = preds.shape[2]
        let classCount = stride - 5

        // 1) decode: score = obj * max-class, threshold (obj is an upper bound on the
        //    score since cls in [0,1], so reject on obj first — cheap).
        var boxes: [SIMD4<Float>] = []
        var scores: [Float] = []
        var classes: [Int] = []
        boxes.reserveCapacity(64)
        for i in 0..<anchors {
            let base = i * stride
            let obj = p[base + 4]
            if obj < scoreThreshold { continue }
            var bestClass = 0
            var bestCls = p[base + 5]
            for c in 1..<classCount where p[base + 5 + c] > bestCls {
                bestCls = p[base + 5 + c]
                bestClass = c
            }
            let score = obj * bestCls
            if score < scoreThreshold { continue }
            let cx = p[base + 0], cy = p[base + 1], w = p[base + 2], h = p[base + 3]
            boxes.append(SIMD4(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
            scores.append(score)
            classes.append(bestClass)
        }

        // 2) per-class greedy NMS in letterbox-pixel space.
        let keep = Self.nms(boxes: boxes, scores: scores, classes: classes, iouThreshold: nmsThreshold)

        // 3) un-letterbox (÷ratio) and normalize by the source frame.
        let r = input.ratio
        let invW = 1 / (r * Float(input.srcW))
        let invH = 1 / (r * Float(input.srcH))
        var found: [Detection] = []
        for k in keep {
            let b = boxes[k]
            let id = Self.cocoIDs[classes[k]]
            guard let label = ObjectDetector.cocoClasses[id] else { continue }
            let x0 = CGFloat(b.x * invW), y0 = CGFloat(b.y * invH)
            let x1 = CGFloat(b.z * invW), y1 = CGFloat(b.w * invH)
            found.append(
                Detection(
                    classID: id, label: label, score: scores[k],
                    box: CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)))
        }
        found.sort { $0.score > $1.score }
        if found.count > maxDetections { found.removeLast(found.count - maxDetections) }
        return found
    }

    /// Detects objects in one CGImage (used by the on-device gate).
    public func detect(
        in image: CGImage,
        scoreThreshold: Float = 0.3,
        nmsThreshold: Float = 0.45,
        maxDetections: Int = 50
    ) async throws -> [Detection] {
        try await detect(
            letterbox(image), scoreThreshold: scoreThreshold,
            nmsThreshold: nmsThreshold, maxDetections: maxDetections)
    }

    // MARK: - Per-class greedy NMS

    private static func nms(
        boxes: [SIMD4<Float>], scores: [Float], classes: [Int], iouThreshold: Float
    ) -> [Int] {
        let order = scores.indices.sorted { scores[$0] > scores[$1] }
        var suppressed = [Bool](repeating: false, count: scores.count)
        var keep: [Int] = []
        for ii in order.indices {
            let i = order[ii]
            if suppressed[i] { continue }
            keep.append(i)
            let a = boxes[i]
            let aArea = max(0, a.z - a.x) * max(0, a.w - a.y)
            for jj in (ii + 1)..<order.count {
                let j = order[jj]
                if suppressed[j] || classes[j] != classes[i] { continue }
                let b = boxes[j]
                let iw = max(0, min(a.z, b.z) - max(a.x, b.x))
                let ih = max(0, min(a.w, b.w) - max(a.y, b.y))
                let inter = iw * ih
                let union = aArea + max(0, b.z - b.x) * max(0, b.w - b.y) - inter
                if union > 0, inter / union > iouThreshold { suppressed[j] = true }
            }
        }
        return keep
    }

    // MARK: - Letterbox preprocessing (aspect-preserving, pad 114, BGR 0-255)

    private func letterbox(_ pixelBuffer: CVPixelBuffer) throws -> PreparedInput {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA
        else {
            throw VisionError.bundleLayout("YOLOXDetector expects a 32BGRA pixel buffer")
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw VisionError.imageRenderFailed
        }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        var src = vImage_Buffer(
            data: base, height: vImagePixelCount(srcH), width: vImagePixelCount(srcW),
            rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer))
        return try letterbox(src: &src, srcW: srcW, srcH: srcH)
    }

    private func letterbox(_ image: CGImage) throws -> PreparedInput {
        let w = image.width, h = image.height
        let rowBytes = w * 4
        var bgra = [UInt8](repeating: 0, count: rowBytes * h)
        guard
            let ctx = bgra.withUnsafeMutableBytes({ ptr -> CGContext? in
                CGContext(
                    data: ptr.baseAddress, width: w, height: h, bitsPerComponent: 8,
                    bytesPerRow: rowBytes, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                        | CGBitmapInfo.byteOrder32Little.rawValue)
            })
        else {
            throw VisionError.imageRenderFailed
        }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return try bgra.withUnsafeMutableBytes { ptr in
            var src = vImage_Buffer(
                data: ptr.baseAddress, height: vImagePixelCount(h),
                width: vImagePixelCount(w), rowBytes: rowBytes)
            return try letterbox(src: &src, srcW: w, srcH: h)
        }
    }

    /// Scales a BGRA source into the top-left of a 114-filled S×S canvas (aspect
    /// preserved), then splits to planar CHW BGR floats in [0, 255] (no normalization).
    private func letterbox(
        src: inout vImage_Buffer, srcW: Int, srcH: Int
    ) throws -> PreparedInput {
        let side = inputSize
        let ratio = min(Float(side) / Float(srcW), Float(side) / Float(srcH))
        let newW = Int((Float(srcW) * ratio).rounded(.down))
        let newH = Int((Float(srcH) * ratio).rounded(.down))

        // Pad value 114 in every byte; vImageScale overwrites only the newW×newH
        // top-left region (dst.rowBytes = side*4 leaves the right/bottom pad intact).
        var canvas = [UInt8](repeating: 114, count: side * side * 4)
        let err = canvas.withUnsafeMutableBytes { dst -> vImage_Error in
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress, height: vImagePixelCount(newH),
                width: vImagePixelCount(newW), rowBytes: side * 4)
            return vImageScale_ARGB8888(&src, &dstBuf, nil, vImage_Flags(kvImageNoFlags))
        }
        guard err == kvImageNoError else { throw VisionError.imageRenderFailed }

        let pixelCount = side * side
        let n = vDSP_Length(pixelCount)
        var chw = [Float](repeating: 0, count: 3 * pixelCount)
        // BGRA byte order: plane p reads byte offset p (B=0, G=1, R=2) -> planar BGR,
        // kept in [0, 255] (YOLOX's non-legacy checkpoints take raw pixel magnitudes).
        canvas.withUnsafeBufferPointer { raw in
            chw.withUnsafeMutableBufferPointer { out in
                for plane in 0..<3 {
                    vDSP_vfltu8(
                        raw.baseAddress! + plane, 4,
                        out.baseAddress! + plane * pixelCount, 1, n)
                }
            }
        }
        return PreparedInput(pixels: chw, ratio: ratio, srcW: srcW, srcH: srcH)
    }
}
