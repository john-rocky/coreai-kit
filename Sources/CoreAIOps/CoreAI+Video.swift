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
        let recognizer = try await VideoOpModels.shared.recognizer(
            catalog: options.model ?? defaultActionModel)
        return try await recognizer.classify(videoAt: url, topK: topK)
    }

    /// Frames → ranked action labels (any frame count; resampled to 16).
    public static func recognizeAction(
        frames: [CGImage], topK: Int = 3, options: OpOptions = OpOptions()
    ) async throws -> [ActionRecognizer.Prediction] {
        let recognizer = try await VideoOpModels.shared.recognizer(
            catalog: options.model ?? defaultActionModel)
        return try await recognizer.classify(frames: frames, topK: topK)
    }
}

/// Process-wide cache of loaded recognizers, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached.
actor VideoOpModels {
    static let shared = VideoOpModels()

    private var recognizerLoads: [String: Task<ActionRecognizer, Error>] = [:]

    func recognizer(catalog id: String) async throws -> ActionRecognizer {
        if let load = recognizerLoads[id] { return try await load.value }
        let load = Task<ActionRecognizer, Error> {
            try await ActionRecognizer(catalog: id, downloadProgress: OpDownloads.forward)
        }
        recognizerLoads[id] = load
        do {
            return try await load.value
        } catch {
            recognizerLoads[id] = nil
            throw error
        }
    }
}
