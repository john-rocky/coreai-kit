// ObjectDetector.swift — typed real-time object detection over GraphModel (RF-DETR
// exports). The graph is a single static function: image [1, 3, R, R] RGB in [0, 1]
// (ImageNet normalization folded in-graph) -> dets [1, Q, 4] cxcywh normalized +
// labels [1, Q, C] raw logits where the column index is the ORIGINAL COCO id.
// DETR-family models need no NMS: decode is sigmoid + threshold.

import CoreGraphics
import Foundation

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
}

public final class ObjectDetector: @unchecked Sendable {
    private let graph: GraphModel
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
        let outputs = try await graph.run(
            [imageInput: .float32(pixels, shape: imageShape)])
        guard let dets = outputs[detsOutput], let labels = outputs[labelsOutput] else {
            throw VisionError.missingOutput("\(detsOutput)/\(labelsOutput)")
        }
        let boxes = dets.floats()
        let logits = labels.floats()
        let queries = dets.shape[1]
        let classes = labels.shape[2]

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
            found.append(
                Detection(
                    classID: bestClass, label: label, score: score,
                    box: CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)))
        }
        found.sort { $0.score > $1.score }
        if found.count > maxDetections { found.removeLast(found.count - maxDetections) }
        return found
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
