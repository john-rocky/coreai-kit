// TypedDecisions.swift — typed decisions over a Core AI chat bundle: one prompt scored at its
// last position, the answer read as probabilities over the listed options.
//
// ```swift
// let decider = try await TypedDecisions(catalog: "minicpm5-2b")
// let answer = try await decider.decide(
//     "Customer: my order arrived with the box crushed and the screen cracked.",
//     .noul("Does the customer want a replacement?"))
// answer.noul   // P(yes)
// ```
//
// ## One prefill, N decisions
//
// Every decision on the same state shares the prompt prefix up to the end of the state (the
// system line and the state itself); only the question and the options differ. The engine
// keeps its KV cache between calls, so a second question on the same state rewinds to that
// shared prefix and prefills just the tail. `prefill(_:)` runs the prefix on its own so the
// first question is as cheap as the rest; `Decision.Timing.reusedTokens` reports what was
// kept. Engines that cannot rewind mid-sequence (recurrent hybrids — Qwen3.5, LFM2.5,
// Granite 4) fall back to a full re-prefill on every decision, losslessly; the timing says so.
//
// ## Which engine
//
// The answer needs the logits at the answer slot, which the default GPU-pipelined engine does
// not expose (it samples on the GPU). A catalog model loads on the sequential engine, or on
// the static-shape engine for a chunked static (Neural Engine) bundle — the two engines that
// return logits. Decode speed does not matter here: a decision generates nothing.
//
// On iPhone keep a prompt under 1024 tokens: the on-device compiler miscompiles the growing
// KV cache of a dynamic bundle once it reaches 2048 positions (the same guard the pipelined
// engine enforces), and a decision has no reason to be longer than that.

import CoreAILanguageModels
import Foundation
import Tokenizers

/// A loaded decision model. One decision at a time; calls on the same instance serialize.
public actor TypedDecisions {
    public struct Configuration: Sendable {
        /// Softmax temperature over the answer-slot logits. 1 reads the model's own
        /// distribution; a model trained with a calibration temperature names it on its card.
        public var temperature: Double = 1
        /// Reuse the engine's KV cache across decisions on the same state (rewind to the shared
        /// prefix, prefill only the question). Off, every decision re-prefills its whole prompt
        /// — the `direct` column of a benchmark.
        public var sharePrefix: Bool = true
        /// Engine to load the bundle with. `.auto` picks the static-shape engine for a bundle
        /// the catalog marks `static-shape` and the sequential engine for everything else.
        public var engineVariant: EngineVariant = .auto
        /// Prefill one token per step. `nil` detects a decode-only (S=1) graph from the bundle
        /// name; set it explicitly for a local bundle the heuristic cannot see.
        public var singleTokenPrefill: Bool? = nil

        public init() {}
    }

    private let runtime: ModelRuntime
    private let configuration: Configuration
    /// The catalog id, or the bundle directory name for a local bundle.
    public let id: String
    /// The bundle's `max_context_length`; a prompt must leave one slot for the answer.
    public let maxContextLength: Int
    /// Exact token sequence the engine's KV cache holds (the last scored prompt).
    private var kvTokens: [Int32] = []
    /// Timing of the last decision or prefill.
    public private(set) var lastTiming: Decision.Timing?

    /// Display name from the bundle metadata.
    public var modelName: String { runtime.modelName }

    /// Loads a model by its catalog id (`kind: chat`); downloads on first use.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        configuration: Configuration = Configuration(),
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .chat)
        guard let model = entry.modelID else {
            throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
        }
        // The Gemma 4 pairs and the raw-Metal pack load through their own runtimes, which
        // sample on the GPU and expose no logits.
        if entry.id == Gemma4MetalRuntime.catalogID || GemmaModelID.byCatalogID[entry.id] != nil {
            throw DecisionError.unsupportedModel(
                id: id, reason: "its runtime samples on the GPU and exposes no logits")
        }
        let url = try await store.download(model, progress: downloadProgress)
        Self.configureSingleTokenPrefill(
            bundleName: model.resolvedPath, override: configuration.singleTokenPrefill)
        let bundle = try LanguageBundle(at: url)
        let runtime = try await ModelRuntime(
            bundleAt: url,
            engineVariant: Self.resolveEngine(configuration.engineVariant, hint: entry.engine))
        try self.init(
            runtime: runtime, configuration: configuration, id: id,
            maxContextLength: bundle.maxContextLength)
    }

    /// Loads a local bundle directory (metadata.json + *.aimodel/ + tokenizer/).
    public init(bundleAt url: URL, configuration: Configuration = Configuration()) async throws {
        Self.configureSingleTokenPrefill(
            bundleName: url.lastPathComponent, override: configuration.singleTokenPrefill)
        let bundle = try LanguageBundle(at: url)
        let runtime = try await ModelRuntime(
            bundleAt: url, engineVariant: Self.resolveEngine(configuration.engineVariant, hint: nil))
        try self.init(
            runtime: runtime, configuration: configuration, id: url.lastPathComponent,
            maxContextLength: bundle.maxContextLength)
    }

    private init(
        runtime: ModelRuntime, configuration: Configuration, id: String, maxContextLength: Int
    ) throws {
        guard runtime.engine.supportsLogits else {
            throw DecisionError.engineWithoutLogits(model: id)
        }
        // The first two letters cover every question shape; a tokenizer that cannot slot
        // them fails here, at load, not on the first decision.
        _ = try DecisionPrompt.slotTokens(count: 2, tokenizer: runtime.tokenizer)
        self.runtime = runtime
        self.configuration = configuration
        self.id = id
        self.maxContextLength = maxContextLength
    }

    /// `.auto` → the static-shape engine for a chunked static bundle, else the sequential
    /// engine. An explicit choice is kept, and refused at load if it cannot return logits.
    static func resolveEngine(_ requested: EngineVariant, hint: String?) -> EngineVariant {
        guard requested == .auto else { return requested }
        return EngineVariant(catalogHint: hint) == .staticShape ? .staticShape : .sequential
    }

    /// Zoo decode-only ports are S=1 graphs (their bundle names carry `_decode_`): every
    /// prefill chunk must be one token, on every engine. The runtime reads the threshold from
    /// the environment per generation, so this is process-wide, exactly as the kit's pipelined
    /// loads set it; a value already in the environment always wins.
    static func configureSingleTokenPrefill(bundleName: String, override: Bool?) {
        let single = override ?? bundleName.contains("_decode_")
        if single, getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
    }

    // MARK: - Decide

    /// One question on one state.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        let rendered = try DecisionPrompt.render(
            state: state, question: question, tokenizer: runtime.tokenizer)
        let (logits, timing) = try await score(rendered.tokens)
        let slotLogits = rendered.slots.map { Double(logits[Int($0)]) }
        let probabilities = DecisionPrompt.probabilities(
            logits: slotLogits, temperature: configuration.temperature)
        return DecisionPrompt.answer(for: question, probabilities: probabilities, timing: timing)
    }

    /// Several questions on one state, answered in order; the state is prefilled once and
    /// each question rewinds to it.
    public func decide(_ state: String, _ questions: [Decision.Question]) async throws -> [Decision.Answer] {
        var answers: [Decision.Answer] = []
        answers.reserveCapacity(questions.count)
        for question in questions {
            answers.append(try await decide(state, question))
        }
        return answers
    }

    /// Questions keyed by an id of the caller's choosing, answered in key order.
    public func decide(
        _ state: String, _ questions: [String: Decision.Question]
    ) async throws -> [String: Decision.Answer] {
        var answers: [String: Decision.Answer] = [:]
        for key in questions.keys.sorted() {
            answers[key] = try await decide(state, questions[key]!)
        }
        return answers
    }

    /// Runs the part of the prompt every question on `state` shares, so the first decision
    /// pays only for its question. Returns a handle whose `decide` calls score against it.
    public func prefill(_ state: String) async throws -> PrefilledState {
        let prefix = try DecisionPrompt.statePrefix(state: state, tokenizer: runtime.tokenizer)
        let (_, timing) = try await score(prefix, includeLogits: false)
        return PrefilledState(state: state, tokens: prefix.count, timing: timing, decider: self)
    }

    /// The exact token sequence a question on a state is scored as, and the answer-slot token
    /// per option — for checking the rendering against a reference token file without a
    /// decision.
    nonisolated public func promptTokens(
        _ state: String, _ question: Decision.Question
    ) throws -> (tokens: [Int32], slots: [Int32]) {
        let rendered = try DecisionPrompt.render(
            state: state, question: question, tokenizer: runtime.tokenizer)
        return (rendered.tokens, rendered.slots)
    }

    /// Drops the cached prompt: the next decision prefills from scratch.
    public func reset() async throws {
        kvTokens = []
        try await runtime.engine.reset()
    }

    // MARK: - Engine

    /// Feeds `tokens` and returns the logits at the last position. Rewinds to the longest
    /// prefix shared with the previous prompt first (unless sharing is off), so only the
    /// tail is processed.
    private func score(
        _ tokens: [Int32], includeLogits: Bool = true
    ) async throws -> ([LogitsScalarType], Decision.Timing) {
        guard tokens.count < maxContextLength else {
            throw DecisionError.promptTooLong(tokens: tokens.count, max: maxContextLength - 1)
        }
        let engine = runtime.engine
        let start = SuspendingClock.now
        let wanted = configuration.sharePrefix
            ? min(
                DecisionPrompt.commonPrefixLength(tokens, kvTokens),
                max(0, tokens.count - 1),
                engine.processedTokenCount)
            : 0
        // A mirror in an unknown state must not be trusted: clear it until the call lands.
        kvTokens = []
        let kept = try await engine.rewind(to: wanted)
        let stream = try await engine.generate(
            with: tokens,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1, includeLogits: includeLogits))
        var logits: [LogitsScalarType]? = nil
        var sawOutput = false
        for try await output in stream {
            sawOutput = true
            logits = output.logits
            break
        }
        guard sawOutput, let logits = includeLogits ? logits : (logits ?? []) else {
            throw DecisionError.noLogits
        }
        // The engine consumed exactly the prompt: the one sampled token was never fed back.
        kvTokens = tokens
        let timing = Decision.Timing(
            promptTokens: tokens.count, reusedTokens: kept,
            seconds: ProcessStats.seconds(from: start, to: .now))
        lastTiming = timing
        return (logits, timing)
    }
}

/// A state whose shared prompt prefix the engine already holds. Decisions made through it
/// pay for their question only.
public struct PrefilledState: Sendable {
    public let state: String
    /// Tokens of the shared prefix the engine holds.
    public let tokens: Int
    /// What the prefill cost.
    public let timing: Decision.Timing
    let decider: TypedDecisions

    public func decide(_ question: Decision.Question) async throws -> Decision.Answer {
        try await decider.decide(state, question)
    }

    public func decide(_ questions: [Decision.Question]) async throws -> [Decision.Answer] {
        try await decider.decide(state, questions)
    }

    public func decide(_ questions: [String: Decision.Question]) async throws -> [String: Decision.Answer] {
        try await decider.decide(state, questions)
    }
}
