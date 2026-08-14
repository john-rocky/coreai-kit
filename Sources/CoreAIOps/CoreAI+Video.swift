// CoreAI+Video.swift — anchored video operation: action recognition over a clip.
//
// ```swift
// let actions = try await CoreAI.recognizeAction(videoAt: clipURL)
// print(actions.first?.label ?? "?")   // e.g. "pouring something into something"
// ```

import CoreAIKit
import CoreAIKitVision
import CoreGraphics
import Foundation

extension CoreAI {
    /// Default action-recognition model: V-JEPA 2 ViT-L (Something-Something v2 head).
    public static let defaultActionModel = "vjepa2-vitl-ssv2"

    /// Video clip → ranked action labels: 16 frames are sampled uniformly and scored
    /// against the model's action classes, best first.
    public static func recognizeAction(
        videoAt url: URL, topK: Int = 3, options: OpOptions = OpOptions()
    ) async throws -> [ActionRecognizer.Prediction] {
        let id = options.model ?? defaultActionModel
        let recognizer = try await VideoOpModels.shared.recognizer(catalog: id)
        return try await withPinnedModel(ResidentKind.recognizer, id) {
            try await recognizer.classify(videoAt: url, topK: topK)
        }
    }

    /// Frames → ranked action labels (any frame count; resampled to 16).
    public static func recognizeAction(
        frames: [CGImage], topK: Int = 3, options: OpOptions = OpOptions()
    ) async throws -> [ActionRecognizer.Prediction] {
        let id = options.model ?? defaultActionModel
        let recognizer = try await VideoOpModels.shared.recognizer(catalog: id)
        // `frames` is [CGImage] — not Sendable, so the pin wraps the load rather than the
        // call, and the classification runs in the caller's isolation as it did before.
        await ModelResidency.shared.pin(ResidentModel(kind: ResidentKind.recognizer, id: id))
        defer {
            ModelResidency.shared.unpinLater(
                ResidentModel(kind: ResidentKind.recognizer, id: id))
        }
        return try await recognizer.classify(frames: frames, topK: topK)
    }
}

/// Process-wide cache of loaded recognizers, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached.
actor VideoOpModels {
    static let shared = VideoOpModels()

    private let recognizers = ResidentCache<ActionRecognizer>(kind: ResidentKind.recognizer)

    func recognizer(catalog id: String) async throws -> ActionRecognizer {
        try await recognizers.value(for: id) {
            try await ActionRecognizer(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}
