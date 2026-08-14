// CoreAI+Image.swift — anchored image operations over local catalog models.
//
// Same contract as the text ops in CoreAI.swift: each op fixes its task and output shape,
// and the catalog model behind it can change (catalog update, `options: .model(...)`)
// without breaking callers.
//
// ```swift
// let caption = try await CoreAI.caption(photo)              // image -> description
// let boxes   = try await CoreAI.detect(in: photo)           // image -> [Detection]
// let page    = try await CoreAI.read(documentAt: scanURL)   // document image -> markdown
// ```
//
// `caption` rides `KitVisionModel` behind a fresh `LanguageModelSession` per call (the op
// is stateless by contract); `detect` rides `KitDetector`; `read` resolves the catalog id
// to its OCR driver (GLM-OCR / MinerU / Unlimited-OCR). Ops are one-shot conveniences —
// a video-rate loop should hold its own `KitDetector` instead of paying the per-call
// actor hop.

import CoreAIKit
import CoreAIKitVision
import CoreGraphics
import CoreImage
import Foundation
import FoundationModels
import ImageIO

/// How `CoreAI.caption` shapes its output.
public enum CaptionStyle: String, Sendable, CaseIterable {
    /// One or two plain sentences.
    case concise
    /// A detailed paragraph: the scene, the subjects, and any visible text.
    case detailed

    var instruction: String {
        switch self {
        case .concise:
            "Describe this image in one or two plain sentences."
        case .detailed:
            "Describe this image in detail: the scene, the subjects, and any visible text."
        }
    }
}

extension CoreAI {
    /// Default vision-language model for `caption`. 2B keeps the download and the per-call
    /// prefill small; `options: .model("qwen3-vl-4b")` trades speed for fidelity.
    public static let defaultVisionModel = "qwen3-vl-2b"

    /// Image → description. First use downloads and loads the model (cached afterwards);
    /// captions on the same model serialize behind each other.
    public static func caption(
        _ image: CGImage, style: CaptionStyle = .concise, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await ImageOpModels.shared.caption(
            catalog: options.model ?? defaultVisionModel,
            image: image, orientation: .up, style: style)
    }

    /// Image file → description (any format the system decodes; EXIF orientation honored).
    public static func caption(
        imageAt url: URL, style: CaptionStyle = .concise, options: OpOptions = OpOptions()
    ) async throws -> String {
        let loaded = try ImageFile.load(url)
        return try await ImageOpModels.shared.caption(
            catalog: options.model ?? defaultVisionModel,
            image: loaded.cgImage, orientation: loaded.orientation, style: style)
    }

    /// Default detection model: RF-DETR nano — no NMS, small download.
    public static let defaultDetectionModel = "rf-detr"

    /// Image → labeled bounding boxes (preprocessing included). Results are sorted by
    /// descending confidence; `box` is normalized with origin at the top-left.
    public static func detect(
        in image: CGImage, scoreThreshold: Float = 0.5, options: OpOptions = OpOptions()
    ) async throws -> [Detection] {
        let id = options.model ?? defaultDetectionModel
        let detector = try await ImageOpModels.shared.detector(catalog: id)
        return try await withPinnedModel(ResidentKind.detector, id) {
            try await detector.detect(in: image, scoreThreshold: scoreThreshold)
        }
    }

    /// Image file → labeled bounding boxes. EXIF orientation is baked into the bitmap
    /// first, so boxes are normalized to the upright image.
    public static func detect(
        inImageAt url: URL, scoreThreshold: Float = 0.5, options: OpOptions = OpOptions()
    ) async throws -> [Detection] {
        let loaded = try ImageFile.load(url)
        return try await detect(
            in: upright(loaded.cgImage, loaded.orientation),
            scoreThreshold: scoreThreshold, options: options)
    }

    private static let uprightContext = CIContext()

    /// The bitmap with its EXIF orientation applied (detection has no orientation input).
    private static func upright(
        _ image: CGImage, _ orientation: CGImagePropertyOrientation
    ) -> CGImage {
        guard orientation != .up else { return image }
        let oriented = CIImage(cgImage: image).oriented(orientation)
        return uprightContext.createCGImage(oriented, from: oriented.extent) ?? image
    }

    /// Default document-OCR model: GLM-OCR (~4 s/page on device).
    public static let defaultOCRModel = "glm-ocr"

    /// Document image → markdown text — headings and tables survive as markup where
    /// the model emits them (`options: .model("mineru2.5-pro")` / `"unlimited-ocr"`).
    public static func read(
        _ image: CGImage, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await ImageOpModels.shared.read(
            catalog: options.model ?? defaultOCRModel, image: image)
    }

    /// Document image file → markdown text (any format the system decodes).
    public static func read(
        documentAt url: URL, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await read(ImageFile.load(url).cgImage, options: options)
    }

    /// Default super-resolution model: AdcSR — one-step diffusion, deterministic.
    public static let defaultUpscaleModel = "adcsr-x4"

    /// Image → ×4 upscaled image (tiled internally, so large photos are fine).
    public static func upscale(
        _ image: CGImage, options: OpOptions = OpOptions()
    ) async throws -> CGImage {
        let id = options.model ?? defaultUpscaleModel
        let resolver = try await ImageOpModels.shared.upscaler(catalog: id)
        return try await withPinnedModel(ResidentKind.upscaler, id) {
            try await resolver.upscale(image)
        }
    }

    /// Default depth model: Depth Anything 3 Small.
    public static let defaultDepthModel = "depth-anything-3-small"

    /// Image → relative depth map. `DepthMap.cgImage()` renders it as a normalized
    /// grayscale image.
    public static func estimateDepth(
        in image: CGImage, options: OpOptions = OpOptions()
    ) async throws -> DepthMap {
        let id = options.model ?? defaultDepthModel
        let estimator = try await ImageOpModels.shared.depthEstimator(catalog: id)
        return try await withPinnedModel(ResidentKind.depth, id) {
            try await estimator.estimateDepth(for: image)
        }
    }
}

/// Process-wide cache of loaded image-op models, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached.
/// Caption and read turns on one model serialize behind each other — a VL runtime holds
/// one attached image / KV state at a time.
actor ImageOpModels {
    static let shared = ImageOpModels()

    private let visionModels = ResidentCache<KitVisionModel>(kind: ResidentKind.vision)
    private let detectors = ResidentCache<KitDetector>(kind: ResidentKind.detector)
    private let readers = ResidentCache<DocReader>(kind: ResidentKind.reader)
    private let upscalers = ResidentCache<SuperResolver>(kind: ResidentKind.upscaler)
    private let depthEstimators = ResidentCache<DepthEstimator>(kind: ResidentKind.depth)
    private var visionTurns: [String: Task<Void, Never>] = [:]
    private var readerTurns: [String: Task<Void, Never>] = [:]

    /// The catalog's OCR drivers behind one `read(_:)`.
    enum DocReader {
        case glm(KitGlmOcrReader)
        case mineru(KitMineruReader)
        case unlimited(KitDocReader)

        func read(_ image: CGImage) async throws -> String {
            switch self {
            case .glm(let reader): try await reader.read(image)
            case .mineru(let reader): try await reader.read(image)
            case .unlimited(let reader): try await reader.read(image)
            }
        }
    }

    func caption(
        catalog id: String, image: CGImage, orientation: CGImagePropertyOrientation,
        style: CaptionStyle
    ) async throws -> String {
        let model = try await visionModel(catalog: id)
        let previous = visionTurns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.vision, id) {
                let session = LanguageModelSession(model: model)
                let reply = try await session.respond(
                    to: Prompt {
                        "\(style.instruction) Reply with only the description — no "
                            + "preambles or questions."
                        Attachment(image, orientation: orientation)
                    },
                    // Qwen3-VL thinks by default and the deliberation counts against this
                    // budget — a tight cap truncates the turn before the description starts.
                    options: GenerationOptions(maximumResponseTokens: 2048))
                return reply.content.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        visionTurns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func read(catalog id: String, image: CGImage) async throws -> String {
        let reader = try await reader(catalog: id)
        let previous = readerTurns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.reader, id) {
                try await reader.read(image)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        readerTurns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func detector(catalog id: String) async throws -> KitDetector {
        try await detectors.value(for: id) {
            try await KitDetector(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func upscaler(catalog id: String) async throws -> SuperResolver {
        try await upscalers.value(for: id) {
            try await SuperResolver(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func depthEstimator(catalog id: String) async throws -> DepthEstimator {
        try await depthEstimators.value(for: id) {
            try await DepthEstimator(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func visionModel(catalog id: String) async throws -> KitVisionModel {
        try await visionModels.value(for: id) {
            try await KitVisionModel(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func reader(catalog id: String) async throws -> DocReader {
        try await readers.value(for: id) {
            let entry = try await ModelCatalog.entry(forID: id, expecting: .ocr)
            switch entry.id {
            case "glm-ocr":
                return .glm(
                    try await KitGlmOcrReader(
                        catalog: entry.id, downloadProgress: OpDownloads.forward))
            case "mineru2.5-pro":
                return .mineru(
                    try await KitMineruReader(
                        catalog: entry.id, downloadProgress: OpDownloads.forward))
            case "unlimited-ocr":
                return .unlimited(
                    try await KitDocReader(
                        catalog: entry.id, downloadProgress: OpDownloads.forward))
            default: throw CoreAIKitError.modelNotInCatalog(id: id)
            }
        }
    }
}
