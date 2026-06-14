// VLIntents.swift — the system-facing Visual Intelligence surface for the on-device VLM.
//
// Because Visual Intelligence shows ONE tab per app, this dedicated app surfaces as its own
// "Qwen3-VL" tab next to Google / RF-DETR. The query runs the VLM and returns a single answer
// entity; Visual Intelligence shows its own loading state while `values(for:)` runs (the "thinking"
// phase) and then renders the answer — the slow VLM no longer blocks a fast detection tab.

import AppIntents
import CoreVideo
import Foundation
import VisualIntelligence

// MARK: - Entity

@available(iOS 27.0, macOS 27.0, *)
struct VLAnswerEntity: AppEntity {
    var id: String
    var answer: String
    var thumbnail: Data?

    static let typeDisplayRepresentation: TypeDisplayRepresentation = "On-device Answer"

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(
            title: "\(answer)",
            subtitle: "Qwen3-VL · on-device, no cloud",
            image: thumbnail.map { .init(data: $0) })
    }

    static let defaultQuery = VLAnswerQuery()
}

@available(iOS 27.0, macOS 27.0, *)
struct VLAnswerQuery: EntityQuery {
    func entities(for identifiers: [String]) async throws -> [VLAnswerEntity] {
        // Transient — nothing to rehydrate from an id alone; the live entity carries the answer.
        identifiers.map { VLAnswerEntity(id: $0, answer: "", thumbnail: nil) }
    }
}

// MARK: - The query Visual Intelligence calls with the captured pixels

@available(iOS 27.0, macOS 27.0, *)
struct AskVLMValueQuery: IntentValueQuery {
    @Dependency var engine: VLEngine

    func values(for input: SemanticContentDescriptor) async throws -> [VLAnswerEntity] {
        guard let pixelBuffer = input.pixelBuffer,
            let cgImage = VLEngine.cgImage(from: pixelBuffer)
        else { return [] }

        MemoryProbe.mark("vi-query-enter")
        // Run the VLM on the whole captured frame. Visual Intelligence shows its loading state
        // while this runs; the model must already be cached (warmed in the foreground) or the
        // first cold compile (~30 s) can exceed the VI query timeout.
        guard let vlm = try? await engine.caption(cgImage, instrumented: true), !vlm.text.isEmpty
        else {
            MemoryProbe.log.log("vi-vlm NO-ANSWER (threw, killed, or not cached)")
            return []
        }
        MemoryProbe.log.log(
            "vi-vlm SURVIVED latency=\(vlm.milliseconds, format: .fixed(precision: 0))ms answer=\(vlm.text, privacy: .public)"
        )
        return [
            VLAnswerEntity(
                id: "vlm-answer", answer: vlm.text,
                thumbnail: VLEngine.previewJPEG(cgImage))
        ]
    }
}

// MARK: - OpenIntent — required, or the app does not surface in Visual Intelligence

@available(iOS 27.0, macOS 27.0, *)
struct OpenVLAnswerIntent: OpenIntent {
    static let title: LocalizedStringResource = "Open On-device Answer"

    @Parameter(title: "Answer")
    var target: VLAnswerEntity

    @Dependency var router: VLRouter

    func perform() async throws -> some IntentResult {
        await router.open(answer: target.answer)
        return .result()
    }
}
