// Gemma4MetalRuntime.swift — the raw-Metal Gemma-4-E2B chat runtime: the kit's first
// engine with NO Core AI .aimodel behind it. It mmaps the mixed-bit QAT pack
// (int2/int4/int8 + PLE tables, 2.18 GB), compiles the hand-written kernels at load
// (mathMode .safe — the losslessness proof depends on it), and decodes greedily with
// on-GPU argmax. iPhone 17 Pro fresh decode ~55 tok/s = LiteRT-LM parity, token-exact
// vs the fp16 oracle (S1 gate). Weights: Google's Gemma-4-E2B QAT release, repacked —
// see the zoo's conversion/gemma4_raw_metal/ and knowledge/gemma4-raw-metal-port.md.
//
// ChatSession dispatches to this runtime by catalog id (`gemma-4-e2b-metal`) — the same
// id→driving-class seam as GemmaRuntime. The engine face is a thin InferenceEngine
// conformance over Gemma4MetalEngine: greedy-only (on-GPU argmax ⇒ supportsLogits
// false, sampling configuration ignored), KV kept across calls with prefix reuse via
// reset(to:) + the core's own longest-common-prefix prefill.

import CoreAILanguageModels
import Foundation
import Synchronization
import Tokenizers

/// Loads and owns the raw-Metal Gemma 4 E2B pack: engine + tokenizer.
final class Gemma4MetalRuntime: Sendable {
    static let catalogID = "gemma-4-e2b-metal"

    let engine: Gemma4MetalInferenceEngine
    let tokenizer: any Tokenizer
    let modelName = "Gemma 4 E2B ⚡raw-Metal"
    var vocabSize: Int { engine.core.vocabSize }

    /// - Parameter packURL: downloaded pack dir (gemma4_pack.bin + gemma4_pack.json +
    ///   tokenizer/). Kernels ride the kit bundle (Gemma4Metal/g4msl resources).
    init(packAt packURL: URL) async throws {
        guard let mslDir = Bundle.module.url(forResource: "g4msl", withExtension: nil) else {
            throw InferenceRuntimeError.genericError("g4msl kernel resources missing from CoreAIKit bundle")
        }
        async let tokenizerLoad = AutoTokenizer.from(
            modelFolder: packURL.appendingPathComponent("tokenizer"), strict: false)
        // Pack mmap + kernel compilation take a few seconds — keep them off the caller.
        let core = try await Task.detached(priority: .userInitiated) {
            try Gemma4MetalEngine(packDir: packURL, mslDir: mslDir)
        }.value
        self.engine = Gemma4MetalInferenceEngine(core: core)
        self.tokenizer = try await tokenizerLoad
    }
}

/// Greedy, KV-preserving `InferenceEngine` face over the raw-Metal loop.
final class Gemma4MetalInferenceEngine: InferenceEngine, @unchecked Sendable {
    struct Config: Codable, InferenceConfiguration {
        let maxContextLength: Int
    }

    let core: Gemma4MetalEngine
    let config: Config
    /// Gemma-4 stop set: <eos>=1, <turn|>=106. The stream FINISHES at a stop token
    /// (without emitting it) — consumers treat end-of-stream as end-of-turn, the same
    /// contract as a maxTokens finish.
    private let stopIds: Set<Int> = [1, 106]

    init(core: Gemma4MetalEngine) {
        self.core = core
        self.config = Config(maxContextLength: core.maxContext)
    }

    func generate(
        with input: [Int32],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) throws -> GenerationSequence {
        guard inferenceOptions.forcedContinuation == nil else {
            throw InferenceRuntimeError.invalidArgument(
                "forcedContinuation is not supported by the raw-Metal engine")
        }
        guard !inferenceOptions.includeLogits else {
            throw InferenceRuntimeError.invalidArgument(
                "logits are not exposed (on-GPU argmax); load a sequential engine instead")
        }
        let ids = input.map(Int.init)
        let maxNew = inferenceOptions.maxTokens ?? max(1, core.maxContext - ids.count - 8)
        let (stream, cont) = AsyncThrowingStream<InferenceOutput, Error>.makeStream()
        let store = KitStopReasonStore()
        let core = self.core
        let stopIds = self.stopIds
        let task = Task.detached(priority: .userInitiated) {
            core.generate(promptIds: ids, maxNew: maxNew, stopIds: stopIds) { batch in
                for t in batch { cont.yield(InferenceOutput(tokenId: Int32(t))) }
            }
            // The core finishes at a stop id without emitting it; either way the
            // stream ends here — report the generic engine-driven reason.
            store.setIfUnset(.maxTokens)
            cont.finish()
        }
        cont.onTermination = { reason in
            if case .cancelled = reason {
                store.set(.cancelled)
                core.requestStop()
                task.cancel()
            }
        }
        return GenerationSequence(base: stream, store: store)
    }

    /// Tokens the core holds committed in its KV cache.
    var processedTokenCount: Int { core.committedCount }

    /// `reset(to: 0)` forgets the conversation; `reset(to: n)` retains the leading `n`
    /// committed tokens. The caller re-feeds the full sequence either way — the core's
    /// own longest-common-prefix match prefills only the un-cached suffix.
    func reset(to tokenIndex: Int) async throws {
        if tokenIndex <= 0 {
            core.reset()
        } else {
            core.trimCommitted(to: tokenIndex)
        }
    }
}

extension Gemma4MetalInferenceEngine {
    /// Minimal `InferenceOutputSequence` face over an `AsyncThrowingStream` (the
    /// upstream engines each carry their own; the store class is internal to
    /// CoreAILanguageModels, so the kit mirrors it here).
    struct GenerationSequence: InferenceOutputSequence {
        typealias Element = InferenceOutput
        typealias Failure = Error

        let base: AsyncThrowingStream<InferenceOutput, Error>
        let store: KitStopReasonStore

        var stopReason: StopReason? { store.stopReason }
        func setStopReason(_ reason: StopReason) { store.set(reason) }

        func makeAsyncIterator() -> AsyncThrowingStream<InferenceOutput, Error>.AsyncIterator {
            base.makeAsyncIterator()
        }
    }
}

/// Thread-safe `StopReason` slot shared between a `GenerationSequence` value, its
/// producer task, and the consumer (same contract as upstream's internal store).
final class KitStopReasonStore: Sendable {
    private let value = Mutex<StopReason?>(nil)

    var stopReason: StopReason? { value.withLock { $0 } }
    func set(_ reason: StopReason) { value.withLock { $0 = reason } }
    func setIfUnset(_ reason: StopReason) { value.withLock { if $0 == nil { $0 = reason } } }
}
