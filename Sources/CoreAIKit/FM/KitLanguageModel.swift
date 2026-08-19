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
/// Tool calling uses the Hermes ChatML dialect and is therefore declared only for ChatML
/// models that are NOT LFM-family: the precondition is `<|im_start|>` in the vocab AND the
/// absence of LFM's pythonic `<|tool_call_start|>` / `<|tool_call_end|>` tokens. LFM bundles
/// are ChatML-framed too but emit a pythonic call syntax this Hermes-only path can't parse,
/// so advertising `.toolCalling` for them would silently never fire — they're excluded.
/// (In practice this is the qwen3 family today.) Plain chat works on every bundle via its
/// own chat template. Guided generation (`session.respond(generating:)`) is declared when
/// the engine exposes per-step logits — load with `engineVariant: .sequential`; the
/// default pipelined engine samples on-GPU and rejects schema requests with
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
    /// Whether tool calling can use the Hermes ChatML dialect: `<|im_start|>` in the vocab
    /// (ChatML framing) AND no LFM pythonic tool tokens. LFM-family bundles are ChatML-framed
    /// but emit `<|tool_call_start|>[…]` calls this Hermes-only path can't parse, so they are
    /// excluded rather than advertising a `.toolCalling` capability that would never fire.
    let supportsHermesTools: Bool
    /// Vocabulary size from bundle metadata; sizes the grammar bitmask for guided
    /// generation (derived from the tokenizer when nil).
    let vocabSize: Int?

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        if supportsHermesTools {
            capabilities.append(.toolCalling)
        }
        if profile.rules.contains(where: { $0.kind == .thinking }) {
            capabilities.append(.reasoning)
        }
        if engine.supportsLogits {
            capabilities.append(.guidedGeneration)
        }
        return LanguageModelCapabilities(capabilities)
    }

    public var executorConfiguration: KitExecutor.Configuration {
        KitExecutor.Configuration(
            engine: engine,
            tokenizer: tokenizer,
            modelID: modelID,
            profile: profile,
            supportsHermesTools: supportsHermesTools,
            vocabSize: vocabSize)
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it. `modelID`
    /// overrides the executor-cache key (defaults to the bundle path) — pass a distinct
    /// value to load the SAME bundle twice as independent executors/engines (e.g. a
    /// sequential router alongside a pipelined answerer).
    public init(
        model: ModelID,
        store: ModelStore = .default,
        engineVariant: EngineVariant = .auto,
        modelID: String? = nil,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, engineVariant: engineVariant, modelID: modelID)
    }

    /// Loads a bundle directory (metadata.json + *.aimodel/ + tokenizer/) and auto-picks
    /// the engine, exactly like Apple's `CoreAILanguageModel(resourcesAt:)`. Pass
    /// `engineVariant: .sequential` to enable guided generation. `modelID` overrides the
    /// executor-cache key (defaults to the bundle path).
    public init(
        bundleAt url: URL, engineVariant: EngineVariant = .auto, modelID: String? = nil
    ) async throws {
        let runtime = try await ModelRuntime(bundleAt: url, engineVariant: engineVariant)
        self.init(
            engine: runtime.engine,
            tokenizer: runtime.tokenizer,
            modelID: modelID ?? url.standardizedFileURL.path,
            vocabSize: runtime.vocabSize)
    }

    /// Wraps an already-created engine + tokenizer. `modelID` keys the session's executor
    /// cache — use a bundle-unique string (the bundle path); two models with the same ID
    /// share one executor and its engine state.
    public init(
        engine: any InferenceEngine, tokenizer: any Tokenizer, modelID: String,
        vocabSize: Int? = nil
    ) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.modelID = modelID
        self.profile = OutputProfile.detect(probing: tokenizer)
        // Hermes tools need ChatML framing AND a non-LFM model: LFM bundles also carry
        // <|im_start|> but emit pythonic <|tool_call_start|>[…] calls this path can't parse.
        self.supportsHermesTools =
            tokenizer.convertTokenToId("<|im_start|>") != nil
            && tokenizer.convertTokenToId("<|tool_call_start|>") == nil
            && tokenizer.convertTokenToId("<|tool_call_end|>") == nil
        self.vocabSize = vocabSize
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
