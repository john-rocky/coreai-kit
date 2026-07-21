// CoreAI.swift — anchored operations over local catalog models.
//
// Ops are the stable API; models are catalog entries resolved behind them. Anchored ops
// only — no free-generation API by design: each op fixes its own prompt contract and
// output shape, so the model behind it can change (catalog update, `options: .model(...)`)
// without breaking callers.
//
// ```swift
// let tldr = try await CoreAI.summarize(article, style: .bullets)
// let invoice = try await CoreAI.extract(email, as: Invoice.self)   // Invoice: @Generable
// ```
//
// v1 runs every op on the sequential engine: `extract`'s guided generation needs per-step
// logits, and sharing one engine keeps a single copy of the weights resident for both ops.

import CoreAIKit
import Foundation
import FoundationModels

/// How `CoreAI.summarize` shapes its output.
public enum SummaryStyle: String, Sendable, CaseIterable {
    /// 2–3 plain sentences.
    case concise
    /// Markdown bullet list of the key points.
    case bullets
    /// A single sentence.
    case oneLine

    var instruction: String {
        switch self {
        case .concise: "Reply with a summary of 2-3 plain sentences."
        case .bullets: "Reply with a Markdown bullet list of the key points."
        case .oneLine: "Reply with a single-sentence summary."
        }
    }
}

/// Per-call knobs for an op. The default resolves the op's default model from the catalog;
/// `.model("qwen3-4b")` runs the same op on another catalog chat model.
public struct OpOptions: Sendable {
    /// Catalog id of the chat model to run on (nil = the op's default).
    public var model: String?

    public init(model: String? = nil) { self.model = model }

    /// Run the op on this catalog chat model instead of the default.
    public static func model(_ id: String) -> OpOptions { OpOptions(model: id) }
}

/// Anchored text operations, each a thin prompt contract over a catalog model.
public enum CoreAI {
    /// Default model for text ops — small enough to load fast, strong enough for
    /// summarize/extract. Overridable per call via `options: .model(...)`.
    public static let defaultModel = "qwen3-0.6b"

    /// Summarizes `text` in the given style. First use downloads and loads the model
    /// (cached afterwards); later calls reuse the loaded engine.
    public static func summarize(
        _ text: String, style: SummaryStyle = .concise, options: OpOptions = OpOptions()
    ) async throws -> String {
        let model = try await OpModels.shared.model(catalog: options.model ?? defaultModel)
        let session = LanguageModelSession(
            model: model,
            instructions: """
                You summarize text. \(style.instruction) \
                Do not add preambles, commentary, or questions.
                """)
        let response = try await session.respond(
            to: "Summarize the following text:\n\n\(text)")
        return response.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Extracts a typed value from `text`: the framework derives the schema from the
    /// `@Generable` type and guided generation constrains decoding to it, so the result
    /// parses by construction.
    public static func extract<Value: Generable>(
        _ text: String, as type: Value.Type = Value.self, options: OpOptions = OpOptions()
    ) async throws -> Value {
        let model = try await OpModels.shared.model(catalog: options.model ?? defaultModel)
        let session = LanguageModelSession(
            model: model,
            instructions: """
                You extract structured data from text. Fill every field from the text; \
                use the closest supported value when a field is not stated verbatim.
                """)
        let response = try await session.respond(
            to: "Extract the requested data from the following text:\n\n\(text)",
            generating: Value.self)
        return response.content
    }
}

/// Process-wide cache of loaded op models, keyed by catalog id — ops are static calls, so
/// this is what keeps the second op call from reloading the weights. Concurrent first
/// calls share one load; a failed load is not cached (a later call retries).
actor OpModels {
    static let shared = OpModels()

    private var loads: [String: Task<KitLanguageModel, Error>] = [:]

    func model(catalog id: String) async throws -> KitLanguageModel {
        if let load = loads[id] { return try await load.value }
        let load = Task<KitLanguageModel, Error> {
            let entry = try await ModelCatalog.entry(forID: id, expecting: .chat)
            guard let modelID = entry.modelID else {
                throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
            }
            return try await KitLanguageModel(model: modelID, engineVariant: .sequential)
        }
        loads[id] = load
        do {
            return try await load.value
        } catch {
            loads[id] = nil
            throw error
        }
    }
}
