// CoreAI+Redact.swift — anchored zero-shot entity extraction and PII redaction (GLiNER2).
//
// ```swift
// let safe  = try await CoreAI.redact(supportEmail)               // "[EMAIL]", "[PERSON]", …
// let found = try await CoreAI.extractEntities(from: supportEmail,
//                                              labels: ["person", "order number"])
// ```
//
// Unlike the other ops, the model here is not a catalog entry yet — GLiNER2 loads from
// its pinned `ModelID` preset, so these two take no `OpOptions`. The label set is
// zero-shot: pass any strings, not just the PII defaults.

import CoreAIKit
import CoreAIKitEmbeddings
import Foundation

extension CoreAI {
    /// Default label set for `redact`: the classic PII sweep.
    public static let piiLabels = [
        "person", "email", "phone number", "credit card number",
        "social security number", "address", "date of birth", "organization",
    ]

    /// Text → text with every detected entity replaced by its label — "Call Dana at
    /// 555-0123" becomes "Call [PERSON] at [PHONE NUMBER]". Labels are zero-shot; the
    /// default set sweeps common PII.
    public static func redact(
        _ text: String, labels: [String] = piiLabels, threshold: Float? = nil
    ) async throws -> String {
        try await RedactOpModels.shared.redact(text, labels: labels, threshold: threshold)
    }

    /// Text → entities for zero-shot `labels`: label → matched substrings,
    /// confidence-descending. Typed extraction into a `@Generable` value is `extract`.
    public static func extractEntities(
        from text: String, labels: [String], threshold: Float? = nil
    ) async throws -> [String: [String]] {
        try await RedactOpModels.shared.extract(text, labels: labels, threshold: threshold)
    }
}

/// Process-wide GLiNER2 instance. Turns serialize behind each other — two extractions
/// running concurrently on one instance corrupt the shared graph.
actor RedactOpModels {
    static let shared = RedactOpModels()

    /// GLiNER2-PII is a pinned preset rather than a catalog entry, so the cache holds one
    /// id — it still rides the shared residency budget, which is the point.
    static let presetID = "gliner2-pii"

    private let extractors = ResidentCache<InformationExtractor>(kind: ResidentKind.extractor)
    private var turns: Task<Void, Never>?

    func redact(_ text: String, labels: [String], threshold: Float?) async throws -> String {
        let extractor = try await self.extractor()
        return try await chained {
            try await extractor.redact(text, entities: labels, threshold: threshold)
        }
    }

    func extract(
        _ text: String, labels: [String], threshold: Float?
    ) async throws -> [String: [String]] {
        let extractor = try await self.extractor()
        return try await chained {
            try await extractor.extract(from: text, entities: labels, threshold: threshold)
        }
    }

    private func chained<Value: Sendable>(
        _ body: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        let previous = turns
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.extractor, Self.presetID) {
                try await body()
            }
        }
        turns = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func extractor() async throws -> InformationExtractor {
        try await extractors.value(for: Self.presetID) {
            try await InformationExtractor(
                model: .gliner2PII, downloadProgress: OpDownloads.forward)
        }
    }
}
