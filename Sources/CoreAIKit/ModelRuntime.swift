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

/// How `ChatSession` turns conversation history into prompt tokens.
enum PromptRenderer: Sendable {
    /// The bundle tokenizer's embedded chat template (the standard path).
    case chatTemplate
    /// Gemma 4's turn/channel format, emitted explicitly — the published Gemma 4 E2B/E4B
    /// bundles ship the stock tokenizer with no embedded chat template.
    case gemma(GemmaArchitecture)
}

struct ModelRuntime: Sendable {
    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelName: String
    let vocabSize: Int
    let loadSeconds: Double
    let outputProfile: OutputProfile
    let promptRenderer: PromptRenderer
    /// Retains the Gemma runtime — and with it the static PLE table buffers the engine's
    /// graph reads — for this runtime's lifetime.
    private let gemma: GemmaRuntime?

    /// Wraps a loaded Gemma 4 runtime (decode bundle + bound PLE tables) in the same
    /// ready-to-generate shape as a plain bundle. The `…_tbl` graph fuses the PLE gather,
    /// head, and softcap, so the engine returns sampled tokens exactly like a text bundle
    /// and ChatSession's decode loop drives it unchanged.
    init(gemma runtime: GemmaRuntime, loadSeconds: Double) {
        self.engine = runtime.engine
        self.tokenizer = runtime.tokenizer
        self.modelName = runtime.modelName
        self.vocabSize = runtime.vocabSize
        self.loadSeconds = loadSeconds
        self.outputProfile = runtime.arch.outputProfile
        self.promptRenderer = .gemma(runtime.arch)
        self.gemma = runtime
    }

    /// Wraps the raw-Metal Gemma 4 E2B runtime (no .aimodel — hand-written kernels over
    /// the mmap'd mixed-bit pack). Greedy on-GPU argmax ⇒ no logits; prompt rendering is
    /// the same explicit Gemma 4 turn format as the stock Gemma bundles.
    init(rawMetal runtime: Gemma4MetalRuntime, loadSeconds: Double) {
        self.engine = runtime.engine
        self.tokenizer = runtime.tokenizer
        self.modelName = runtime.modelName
        self.vocabSize = runtime.vocabSize
        self.loadSeconds = loadSeconds
        self.outputProfile = GemmaArchitecture.gemma4.outputProfile
        self.promptRenderer = .gemma(.gemma4)
        self.gemma = nil
    }

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
        // NB (device-bisected 2026-07-24, nanbeige4.2-3b): the iOS on-device compiler
        // miscompiles this graph class when the bound KV state's seq dim is >=2048 —
        // output is corrupt from token 1 at full pipeline speed. Capacity <=1024 is
        // clean; Mac is clean at every shape; a full cache evict does not help, and
        // .fixedSize (4096 upfront) makes every generation ride the broken shape.
        // So: keep .auto (growing) and keep per-turn prompt+maxTokens inside 1024
        // at the app layer until the compiler-level fix lands.
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
        self.promptRenderer = .chatTemplate
        self.gemma = nil
    }
}
