// ObjectDetector.swift — typed real-time object detection over GraphModel (RF-DETR
// exports). The graph is a single static function: image [1, 3, R, R] RGB in [0, 1]
// (ImageNet normalization folded in-graph) -> dets [1, Q, 4] cxcywh normalized +
// labels [1, Q, C] raw logits where the column index is the ORIGINAL COCO id.
// DETR-family models need no NMS: decode is sigmoid + threshold.
//
// Two input paths: CGImage (simple) and CVPixelBuffer (real-time fast path —
// vImage scale + vDSP channel split straight from the capture buffer, no
// CGImage/CIContext round-trip, reused scratch buffers).

import Accelerate
import CoreGraphics
import CoreVideo
import Foundation

/// A per-instance segmentation mask covering the FULL frame at the model's mask
/// stride (RF-DETR-Seg: input/4). Stored as raw logits — foreground is simply
/// `logit > 0` (≡ sigmoid > 0.5), so binarization needs no transcendental math.
/// Same normalized space as `Detection.box`.
public struct InstanceMask: Sendable {
    public let width: Int
    public let height: Int
    /// Raw mask logits, row-major.
    public let logits: [Float]

    public init(width: Int, height: Int, logits: [Float]) {
        self.width = width
        self.height = height
        self.logits = logits
    }

    /// Number of foreground pixels (logit > 0 ⇔ sigmoid > 0.5).
    public var foregroundCount: Int {
        logits.count(where: { $0 > 0 })
    }

    /// Sigmoid probabilities, computed vectorized on demand (display/IoU use
    /// the logits directly; only soft-probability consumers need this).
    public var probabilities: [Float] {
        var negated = [Float](repeating: 0, count: logits.count)
        vDSP.negative(logits, result: &negated)
        var exps = [Float](repeating: 0, count: logits.count)
        vvexpf(&exps, negated, [Int32(logits.count)])
        vDSP.add(1, exps, result: &exps)
        vDSP.divide(1, exps, result: &exps)
        return exps
    }
}

/// One detected object, in normalized image coordinates (origin top-left).
public struct Detection: Sendable, Identifiable {
    public let id = UUID()
    /// Original COCO category id (1 = person … 17 = cat …).
    public let classID: Int
    /// Human-readable category name.
    public let label: String
    /// Sigmoid confidence in [0, 1].
    public let score: Float
    /// Normalized bounding box, origin top-left, all components in [0, 1].
    public let box: CGRect
    /// Instance mask (segmentation models only).
    public let mask: InstanceMask?

    public init(
        classID: Int, label: String, score: Float, box: CGRect, mask: InstanceMask? = nil
    ) {
        self.classID = classID
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
    }
}

public final class ObjectDetector: @unchecked Sendable {
    private let graph: GraphModel
    /// Optional first stage (split deployment): image -> features, e.g. the ViT
    /// backbone on the Neural Engine while the deformable head stays on the GPU.
    private let backbone: GraphModel?
    private let backboneInput: String
    private let backboneOutput: String
    private let headInput: String
    private let preprocessor: ImagePreprocessor
    private let imageInput: String
    private let imageShape: [Int]
    private let detsOutput: String
    private let labelsOutput: String
    /// Model input side length (e.g. 384 for RF-DETR nano).
    public let inputSize: Int

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
        self.backbone = nil
        self.backboneInput = ""
        self.backboneOutput = ""
        self.headInput = ""

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

        guard
            let dets = graph.outputNames.first(where: {
                graph.shape(ofOutput: $0)?.last == 4
            }),
            let labels = graph.outputNames.first(where: {
                ($0 != dets) && (graph.shape(ofOutput: $0)?.count == 3)
            })
        else {
            throw VisionError.bundleLayout(
                "could not identify dets/labels among \(graph.outputNames)")
        }
        self.detsOutput = dets
        self.labelsOutput = labels

        // The export folds ImageNet mean/std into the graph: feed plain pixel/255.
        self.preprocessor = ImagePreprocessor(
            size: inputSize, mean: SIMD3(0, 0, 0), std: SIMD3(1, 1, 1))
    }

    /// Split deployment: a separate backbone graph (image -> features) chained into
    /// a head graph (features -> dets/labels), each with its own compute-unit
    /// preference — e.g. the ANE-friendly ViT backbone on .neuralEngine while the
    /// gather-heavy deformable head stays on .gpu.
    public init(
        backboneAt backboneURL: URL, headAt headURL: URL,
        backboneUnits: GraphModel.ComputeUnits = .neuralEngine,
        headUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        let bb = try await GraphModel(contentsOf: backboneURL, computeUnits: backboneUnits)
        let head = try await GraphModel(contentsOf: headURL, computeUnits: headUnits)
        guard
            let bbInput = bb.inputNames.first(where: { (bb.shape(ofInput: $0)?.count ?? 0) == 4 }),
            let bbShape = bb.shape(ofInput: bbInput),
            let bbOutput = bb.outputNames.first,
            let headIn = head.inputNames.first(where: { (head.shape(ofInput: $0)?.count ?? 0) == 4 })
        else {
            throw VisionError.bundleLayout("unexpected split-graph I/O")
        }
        guard
            let dets = head.outputNames.first(where: { head.shape(ofOutput: $0)?.last == 4 }),
            let labels = head.outputNames.first(where: {
                ($0 != dets) && (head.shape(ofOutput: $0)?.count == 3)
            })
        else {
            throw VisionError.bundleLayout("could not identify dets/labels among \(head.outputNames)")
        }
        self.graph = head
        self.backbone = bb
        self.backboneInput = bbInput
        self.backboneOutput = bbOutput
        self.headInput = headIn
        self.imageInput = bbInput
        self.imageShape = bbShape
        self.inputSize = bbShape[bbShape.count - 1]
        self.detsOutput = dets
        self.labelsOutput = labels
        self.preprocessor = ImagePreprocessor(
            size: bbShape[bbShape.count - 1], mean: SIMD3(0, 0, 0), std: SIMD3(1, 1, 1))
    }

    /// Downloads a split deployment (backbone + head) from the Hub, then loads it
    /// with per-stage compute-unit preferences.
    public convenience init(
        backboneModel: ModelID, headModel: ModelID,
        store: ModelStore = .default,
        backboneUnits: GraphModel.ComputeUnits = .neuralEngine,
        headUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let backboneURL = try await store.download(backboneModel, progress: downloadProgress)
        let headURL = try await store.download(headModel, progress: downloadProgress)
        try await self.init(
            backboneAt: backboneURL, headAt: headURL,
            backboneUnits: backboneUnits, headUnits: headUnits)
    }


    /// Loads a detection model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let detector = try await ObjectDetector(catalog: "rf-detr")
    /// ```
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .detection)
        try await self.init(
            model: entry.modelID!, store: store, computeUnits: computeUnits,
            downloadProgress: downloadProgress)
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .rfdetrMedium,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    /// Detects objects in one image (preprocessing included). Results are sorted by
    /// descending confidence; `box` is normalized with origin at the top-left.
    public func detect(
        in image: CGImage,
        scoreThreshold: Float = 0.5,
        maxDetections: Int = 50
    ) async throws -> [Detection] {
        let pixels = try preprocessor.chw(from: image)
        return try await detect(pixels: pixels, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
    }

    /// Real-time fast path: detects objects in a 32BGRA capture buffer. vImage scales
    /// the frame to the model input and vDSP splits channels into planar [0,1] RGB —
    /// no CGImage/CIContext round-trip, scratch buffers reused across calls.
    public func detect(
        in pixelBuffer: CVPixelBuffer,
        scoreThreshold: Float = 0.5,
        maxDetections: Int = 50
    ) async throws -> [Detection] {
        let pixels = try chwFloats(from: pixelBuffer)
        return try await detect(pixels: pixels, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
    }

    /// A preprocessed frame, ready for `detect(_:)`. Lets a pipeline overlap CPU
    /// preprocessing of frame N+1 with GPU inference of frame N.
    public struct PreparedInput: Sendable {
        let pixels: [Float]
    }

    /// CPU half of the fast path (vImage scale + channel split). Thread-safe.
    public func prepare(_ pixelBuffer: CVPixelBuffer) throws -> PreparedInput {
        PreparedInput(pixels: try chwFloats(from: pixelBuffer))
    }

    /// GPU half of the fast path.
    public func detect(
        _ input: PreparedInput,
        scoreThreshold: Float = 0.5,
        maxDetections: Int = 50
    ) async throws -> [Detection] {
        try await detect(pixels: input.pixels, scoreThreshold: scoreThreshold, maxDetections: maxDetections)
    }

    private func detect(
        pixels: [Float], scoreThreshold: Float, maxDetections: Int
    ) async throws -> [Detection] {
        var outputs: [String: TensorValue]
        if let backbone {
            let feats = try await backbone.run(
                [backboneInput: .float32(pixels, shape: imageShape)])
            guard let features = feats[backboneOutput] else {
                throw VisionError.missingOutput(backboneOutput)
            }
            outputs = try await graph.run([headInput: features])
        } else {
            outputs = try await graph.run(
                [imageInput: .float32(pixels, shape: imageShape)])
        }
        guard let dets = outputs[detsOutput], let labels = outputs[labelsOutput] else {
            throw VisionError.missingOutput("\(detsOutput)/\(labelsOutput)")
        }
        let boxes = dets.floats()
        let logits = labels.floats()
        let queries = dets.shape[1]
        let classes = labels.shape[2]
        // segmentation models add masks [1, Q, H/4, W/4]
        let maskTensor = outputs["masks"]
        let maskFloats = maskTensor?.floats()
        let maskH = maskTensor?.shape[2] ?? 0
        let maskW = maskTensor?.shape[3] ?? 0

        var found: [Detection] = []
        for q in 0..<queries {
            var bestClass = -1
            var bestLogit = -Float.greatestFiniteMagnitude
            // column 0 is unused in the COCO-id layout; start at 1
            for c in 1..<classes where logits[q * classes + c] > bestLogit {
                bestLogit = logits[q * classes + c]
                bestClass = c
            }
            let score = 1 / (1 + exp(-bestLogit))
            guard score >= scoreThreshold,
                let label = ObjectDetector.cocoClasses[bestClass]
            else { continue }
            let cx = CGFloat(boxes[q * 4 + 0])
            let cy = CGFloat(boxes[q * 4 + 1])
            let w = CGFloat(boxes[q * 4 + 2])
            let h = CGFloat(boxes[q * 4 + 3])
            var mask: InstanceMask?
            if let maskFloats, maskH > 0 {
                let plane = maskH * maskW
                mask = InstanceMask(
                    width: maskW, height: maskH,
                    logits: Array(maskFloats[q * plane..<(q + 1) * plane]))
            }
            found.append(
                Detection(
                    classID: bestClass, label: label, score: score,
                    box: CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h),
                    mask: mask))
        }
        found.sort { $0.score > $1.score }
        if found.count > maxDetections { found.removeLast(found.count - maxDetections) }
        return found
    }

    // MARK: - CVPixelBuffer preprocessing (reused scratch, no CGImage)

    private let prepLock = NSLock()
    private var pixelPreprocessor: PixelBufferPreprocessor?

    /// Built on first use rather than in the initializers: three of them set `inputSize` by
    /// different routes (single graph, split backbone/head, catalog), and one lazily-built
    /// instance behind the existing lock beats keeping four constructions in step.
    private func chwFloats(from pixelBuffer: CVPixelBuffer) throws -> [Float] {
        let preprocessor = prepLock.withLock { () -> PixelBufferPreprocessor in
            if let pixelPreprocessor { return pixelPreprocessor }
            // The RF-DETR graphs fold ImageNet normalization in, so this stage emits raw
            // [0, 1] — identity mean/std, matching `preprocessor`.
            let made = PixelBufferPreprocessor(size: inputSize)
            pixelPreprocessor = made
            return made
        }
        return try preprocessor.chw(from: pixelBuffer)
    }

    /// Original COCO category ids (with the official gaps) to names.
    public static let cocoClasses: [Int: String] = [
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
        11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
        22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
        28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
        35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
        39: "baseball bat", 40: "baseball glove", 41: "skateboard",
        42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
        47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
        52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
        57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
        62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
        67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
        75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
        79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
        85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
        89: "hair drier", 90: "toothbrush",
    ]
}
