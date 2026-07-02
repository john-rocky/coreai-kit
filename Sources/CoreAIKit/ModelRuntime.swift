// ModelRuntime.swift — loads a language bundle into ready-to-generate parts.

import CoreAILanguageModels
import Foundation
import Tokenizers

/// Which inference engine to load a language bundle with.
///
/// `.auto` (the default) lets the runtime pick the fastest engine for the model
/// structure — the GPU-pipelined engine for dynamic models. The pipelined engine
/// samples on-GPU and cannot expose per-step logits, so guided generation requires
/// `.sequential` (step-synchronous, logits-capable; slower decode).
public enum EngineVariant: String, Sendable {
    /// Auto-detect from the model structure (pipelined for dynamic models).
    case auto
    /// GPU-pipelined engine: fastest streaming chat, no per-step logits.
    case pipelined = "coreai-pipelined"
    /// Step-synchronous engine: per-step logits (required for guided generation).
    case sequential = "coreai-sequential"
    /// Neural Engine oriented engine for chunked static (AOT) bundles.
    case staticShape = "static-shape"

    /// The factory override string; nil = auto-detection.
    var factoryOverride: String? { self == .auto ? nil : rawValue }

    /// The engine a catalog entry's `engine` hint names. An unknown hint (a newer catalog
    /// than this build) degrades to `.auto`, mirroring `CatalogEntry.Kind.unknown`.
    init(catalogHint: String?) {
        switch catalogHint {
        case "sequential": self = .sequential
        case "pipelined": self = .pipelined
        case "static-shape": self = .staticShape
        default: self = .auto
        }
    }
}

struct ModelRuntime: Sendable {
    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelName: String
    let vocabSize: Int
    let loadSeconds: Double
    let outputProfile: OutputProfile

    /// Loads a bundle directory (metadata.json + *.aimodel/ + tokenizer/), creating the
    /// engine and tokenizer concurrently — the same path as Apple's CoreAIRunner.
    init(bundleAt url: URL, engineVariant: EngineVariant = .auto) async throws {
        // Zoo decode-only ports (catalog hint "pipelined") are S=1 graphs: any multi-token
        // prefill chunk is rejected by the runtime (shape substitution fatal), so chunking
        // must stay off — the same guard every other kit runtime for zoo ports carries
        // (Gemma/VL/Audio/ASR). Official dynamic bundles load via `.auto` and keep the
        // default threshold (single-pass prefill). Process-wide by necessity: the runtime
        // reads the env var per generation; a user-set value always wins.
        if engineVariant == .pipelined, getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let start = SuspendingClock.now
        let bundle = try LanguageBundle(at: url)
        let runner = CoreAIRunner(from: bundle, variant: engineVariant.factoryOverride)
        async let engineLoad = runner.makeInferenceEngine()
        async let tokenizerLoad = bundle.loadTokenizer()
        let (engine, tokenizer) = try await (engineLoad, tokenizerLoad)
        self.engine = engine
        self.tokenizer = tokenizer
        self.modelName = bundle.name
        self.vocabSize = bundle.vocabSize
        self.outputProfile = OutputProfile.detect(probing: tokenizer)
        self.loadSeconds = ProcessStats.seconds(from: start, to: .now)
    }
}
