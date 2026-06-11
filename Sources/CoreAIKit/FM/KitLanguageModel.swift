// KitLanguageModel.swift — Core AI bundles behind FoundationModels' `LanguageModel`
// protocol, adding what Apple's `CoreAILanguageModel` adapter does not implement: tool
// calling, usage events, and append-only KV transcript reuse. Adapted from the
// coreai-model-zoo ZooFMProvider (see NOTICE.txt).

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

/// A Core AI language bundle as a `LanguageModelSession` provider.
///
/// ```swift
/// let model = try await KitLanguageModel(model: .qwen3_0_6B)
/// let session = LanguageModelSession(model: model, tools: [WeatherTool()])
/// let answer = try await session.respond(to: "Weather in Tokyo?")
/// ```
///
/// Tool calling uses the Hermes ChatML dialect and is therefore declared only for models
/// whose tokenizer speaks ChatML (qwen3 family); plain chat works on every bundle via its
/// own chat template. Guided generation is not declared: schema requests throw
/// `LanguageModelError.unsupportedCapability`.
///
/// One model instance assumes serial use (one `LanguageModelSession` at a time) — the
/// underlying engine traps on concurrent generate calls, same as Apple's adapter.
public struct KitLanguageModel: LanguageModel {
    public typealias Executor = KitExecutor

    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelID: String
    let profile: OutputProfile
    /// Whether the tokenizer speaks ChatML (`<|im_start|>` in vocab) — the precondition
    /// for the Hermes tool-calling dialect.
    let speaksChatML: Bool

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        if speaksChatML {
            capabilities.append(.toolCalling)
        }
        if profile.rules.contains(where: { $0.kind == .thinking }) {
            capabilities.append(.reasoning)
        }
        return LanguageModelCapabilities(capabilities: capabilities)
    }

    public var executorConfiguration: KitExecutor.Configuration {
        KitExecutor.Configuration(
            engine: engine,
            tokenizer: tokenizer,
            modelID: modelID,
            profile: profile,
            speaksChatML: speaksChatML)
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public init(
        model: ModelID,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url)
    }

    /// Loads a bundle directory (metadata.json + *.aimodel/ + tokenizer/) and auto-picks
    /// the engine, exactly like Apple's `CoreAILanguageModel(resourcesAt:)`.
    public init(bundleAt url: URL) async throws {
        let runtime = try await ModelRuntime(bundleAt: url)
        self.init(
            engine: runtime.engine,
            tokenizer: runtime.tokenizer,
            modelID: url.standardizedFileURL.path)
    }

    /// Wraps an already-created engine + tokenizer. `modelID` keys the session's executor
    /// cache — use a bundle-unique string (the bundle path); two models with the same ID
    /// share one executor and its engine state.
    public init(engine: any InferenceEngine, tokenizer: any Tokenizer, modelID: String) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.modelID = modelID
        self.profile = OutputProfile.detect(probing: tokenizer)
        self.speaksChatML = tokenizer.convertTokenToId("<|im_start|>") != nil
    }
}

/// Failures specific to this provider.
public enum KitFMError: Error, LocalizedError {
    /// The model emitted a `<tool_call>` block whose body is not a JSON object with a
    /// `name` string.
    case malformedToolCall(payload: String)

    public var errorDescription: String? {
        switch self {
        case .malformedToolCall(let payload):
            return "Model emitted an unparseable tool call: \(payload.prefix(200))"
        }
    }
}
