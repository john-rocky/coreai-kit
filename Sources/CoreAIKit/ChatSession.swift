// ChatSession.swift — high-level streaming chat over a Core AI language bundle, on Apple's
// official runtime (CoreAILM). Adapted from the CoreAIChatMac sample's ChatEngine.

import CoreAILanguageModels
import Foundation
import Tokenizers

/// Streaming multi-turn chat with live performance stats.
///
/// ```swift
/// let chat = try await ChatSession(model: .qwen3_0_6B)
/// for try await event in chat.streamResponse(to: "Hello!") {
///     if case .response(let delta) = event { print(delta, terminator: "") }
/// }
/// ```
///
/// The session owns the conversation history and re-renders it through the bundle's chat
/// template every turn (full-history prefill — the proven official-runtime path). Assistant
/// turns feed back only the final answer; thinking output is surfaced as `.thinking` events
/// and on `Message.thinking` but never re-prompted. One generation at a time per session.
public actor ChatSession {
    public struct Configuration: Sendable {
        /// Sampling temperature; nil = greedy decoding.
        public var temperature: Double? = 0.7
        public var maxResponseTokens: Int = 2048
        public var systemPrompt: String? = nil
        /// Engine to load the bundle with. The default `.auto` picks the fastest engine
        /// (GPU-pipelined for dynamic models); guided generation needs `.sequential`.
        public var engineVariant: EngineVariant = .auto

        public init() {}
    }

    public enum Event: Sendable, Equatable {
        /// Answer text delta.
        case response(String)
        /// Chain-of-thought delta (only on models that emit reasoning markup).
        case thinking(String)
        /// Throttled live stats (TTFT, rolling tok/s, token counts).
        case stats(GenerationStats)
        /// The fully assembled assistant message, after it was appended to `history`.
        case complete(Message)
    }

    private let runtime: ModelRuntime
    private let configuration: Configuration
    public private(set) var history: [Message] = []
    public private(set) var stats = GenerationStats()
    private var generationTask: Task<Void, Never>?
    private var isGenerating = false

    /// Display name from the bundle metadata.
    public var modelName: String { runtime.modelName }

    /// Loads a model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let chat = try await ChatSession(catalog: "qwen3.5-2b")
    /// ```
    ///
    /// Resolves repo, platform variant, and engine hint from the live catalog (built-in
    /// snapshot offline), then downloads if needed — consumers never touch bundle paths or
    /// engine names. A catalog engine hint applies only when `configuration.engineVariant`
    /// is `.auto`; an explicit setting (e.g. `.sequential` for guided generation) wins.
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
        var configuration = configuration
        if configuration.engineVariant == .auto {
            configuration.engineVariant = EngineVariant(catalogHint: entry.engine)
        }
        try await self.init(
            model: model, store: store, configuration: configuration,
            downloadProgress: downloadProgress)
    }

    /// Downloads the model if needed (cached afterwards), then loads it.
    public init(
        model: ModelID,
        store: ModelStore = .default,
        configuration: Configuration = Configuration(),
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, configuration: configuration)
    }

    /// Loads a local bundle directory (metadata.json + *.aimodel/ + tokenizer/).
    public init(bundleAt url: URL, configuration: Configuration = Configuration()) async throws {
        self.runtime = try await ModelRuntime(
            bundleAt: url, engineVariant: configuration.engineVariant)
        self.configuration = configuration
        var stats = GenerationStats()
        stats.loadSeconds = runtime.loadSeconds
        stats.footprintBytes = ProcessStats.physFootprint()
        self.stats = stats
    }

    // MARK: - Generation

    public func streamResponse(to prompt: String) -> AsyncThrowingStream<Event, Error> {
        let (stream, continuation) = AsyncThrowingStream<Event, Error>.makeStream()
        guard !isGenerating else {
            continuation.finish(throwing: ChatSessionError.generationInProgress)
            return stream
        }
        isGenerating = true
        let task = Task { await self.generate(prompt: prompt, schema: nil, into: continuation) }
        generationTask = task
        continuation.onTermination = { reason in
            if case .cancelled = reason { task.cancel() }
        }
        return stream
    }

    /// Convenience: streams internally and returns the collected answer text.
    public func respond(to prompt: String) async throws -> String {
        var text = ""
        for try await event in streamResponse(to: prompt) {
            if case .response(let delta) = event { text += delta }
        }
        return text
    }

    // MARK: - Guided generation

    /// Streams a response constrained to a JSON schema (xgrammar bitmask enforcement):
    /// every decoded token is grammar-checked, so the collected text is guaranteed to
    /// parse as JSON conforming to `schema`.
    ///
    /// Requires a logits-capable engine — load the session with
    /// `configuration.engineVariant = .sequential`. The default pipelined engine samples
    /// on-GPU and cannot expose the per-step logits the grammar mask is applied to.
    /// Constrained turns produce no `.thinking` events (the grammar starts at the JSON).
    public func streamGuidedResponse(
        to prompt: String, schema: String
    ) -> AsyncThrowingStream<Event, Error> {
        let (stream, continuation) = AsyncThrowingStream<Event, Error>.makeStream()
        guard runtime.engine.supportsLogits else {
            continuation.finish(throwing: ChatSessionError.guidedGenerationUnsupported)
            return stream
        }
        guard !isGenerating else {
            continuation.finish(throwing: ChatSessionError.generationInProgress)
            return stream
        }
        isGenerating = true
        let task = Task {
            await self.generate(prompt: prompt, schema: schema, into: continuation)
        }
        generationTask = task
        continuation.onTermination = { reason in
            if case .cancelled = reason { task.cancel() }
        }
        return stream
    }

    /// Convenience: schema-constrained generation, returning the JSON text.
    public func respondJSON(to prompt: String, schema: String) async throws -> String {
        var text = ""
        for try await event in streamGuidedResponse(to: prompt, schema: schema) {
            if case .response(let delta) = event { text += delta }
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Convenience: schema-constrained generation decoded into a `Decodable` value.
    ///
    /// ```swift
    /// struct City: Codable { let name: String; let country: String }
    /// let city = try await chat.respond(
    ///     to: "Name one big city.", generating: City.self, schema: citySchema)
    /// ```
    public func respond<Value: Decodable>(
        to prompt: String, generating type: Value.Type, schema: String
    ) async throws -> Value {
        let json = try await respondJSON(to: prompt, schema: schema)
        guard let data = json.data(using: .utf8) else {
            throw ChatSessionError.malformedGuidedOutput(json)
        }
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw ChatSessionError.malformedGuidedOutput(json)
        }
    }

    /// Compiles the sampler graph and touches the weights (one 1-token generate +
    /// reset), so the first real turn doesn't pay the warm-up. Optional but recommended
    /// right after init, while the UI still shows a loading state.
    public func prewarm() async throws {
        guard !isGenerating else { return }
        let seed = runtime.tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
        let stream = try runtime.engine.generate(
            with: [seed],
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1))
        for try await _ in stream {}
        try await runtime.engine.reset()
    }

    /// Stops the running generation; the stream completes with the partial message.
    public func cancelGeneration() {
        generationTask?.cancel()
    }

    /// Starts a fresh conversation on the loaded model (clears history, keeps the engine).
    public func reset() {
        generationTask?.cancel()
        history = []
        stats.promptTokens = 0
        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = nil
    }

    private func generate(
        prompt: String, schema: String?,
        into continuation: AsyncThrowingStream<Event, Error>.Continuation
    ) async {
        defer { isGenerating = false }

        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            continuation.finish()
            return
        }

        let historyMark = history.count
        history.append(Message(role: .user, content: trimmed))

        var rendered: [[String: any Sendable]] = []
        if let system = configuration.systemPrompt {
            rendered.append(["role": "system", "content": system])
        }
        rendered += history.map { ["role": $0.role.rawValue, "content": $0.content] }

        stats.ttftSeconds = nil
        stats.generatedTokens = 0
        stats.tokensPerSecond = nil

        do {
            let promptTokens = try runtime.tokenizer.applyChatTemplate(messages: rendered)
            stats.promptTokens = promptTokens.count

            try await runtime.engine.reset()
            let sampling =
                configuration.temperature.map { SamplingConfiguration(temperature: $0) }
                ?? SamplingConfiguration.greedy
            // Constrained turns swap the decode loop: per-step logits are masked
            // through the schema's grammar before sampling, so output is valid JSON
            // by construction. Both loops share the same streaming result shape.
            let stream: AsyncThrowingStream<GenerationResult, Error>
            if let schema {
                stream = ConstrainedLoop.stream(
                    jsonSchema: schema,
                    promptTokens: promptTokens.map(Int32.init),
                    engine: runtime.engine,
                    tokenizer: runtime.tokenizer,
                    vocabSize: runtime.vocabSize,
                    sampling: sampling,
                    maxTokens: configuration.maxResponseTokens)
            } else {
                stream = DecodingStrategyFactory.create(type: .vanilla).decode(
                    from: .tokens(promptTokens),
                    tokenizer: runtime.tokenizer,
                    inferenceEngine: runtime.engine,
                    samplingConfiguration: sampling,
                    options: InferenceOptions(
                        maxTokens: configuration.maxResponseTokens, includeLogits: false),
                    // Chat templates end turns with a dedicated marker that tokenizer_config's
                    // eos doesn't always cover (gemma-3: eos = <eos>, turns end with
                    // <end_of_turn> — generation would run to maxTokens spewing the marker).
                    // Add every turn-end token the vocab actually has (round trip guards the
                    // UNK-vocab false positive); duplicates of eos dedupe inside StopSequences.
                    stopSequences: StopSequences(
                        for: runtime.tokenizer,
                        additionalSequences: ["<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<turn|>"]
                            .compactMap { marker in
                                guard let id = runtime.tokenizer.convertTokenToId(marker),
                                    runtime.tokenizer.convertIdToToken(id) == marker
                                else { return nil }
                                return [Int32(id)]
                            })
                )
            }

            let requestStart = SuspendingClock.now
            var parser = StreamingTagParser(profile: runtime.outputProfile)
            var content = ""
            var thinking = ""
            var window: [SuspendingClock.Instant] = []
            let windowSize = 32

            func handle(_ events: [StreamingTagParser.Event]) {
                for event in events {
                    switch event {
                    case .response(let delta):
                        content += delta
                        continuation.yield(.response(delta))
                    case .thinking(let delta):
                        thinking += delta
                        continuation.yield(.thinking(delta))
                    case .toolCallPayload:
                        // ChatSession does not execute tools; use KitLanguageModel with a
                        // LanguageModelSession for tool calling.
                        break
                    }
                }
            }

            do {
                for try await chunk in stream {
                    if Task.isCancelled { break }
                    let now = SuspendingClock.now
                    if stats.ttftSeconds == nil {
                        stats.ttftSeconds = ProcessStats.seconds(from: requestStart, to: now)
                    }
                    stats.generatedTokens += 1
                    window.append(now)
                    if window.count > windowSize { window.removeFirst() }
                    if window.count > 8 {
                        let span = ProcessStats.seconds(from: window[0], to: now)
                        if span > 0 {
                            stats.tokensPerSecond = Double(window.count - 1) / span
                        }
                    }
                    handle(parser.consume(chunk.text))
                    continuation.yield(.stats(stats))
                }
            } catch is CancellationError {
                // Treated like a stop button: keep the partial reply.
            }
            handle(parser.flush())

            stats.footprintBytes = ProcessStats.physFootprint()
            let reply = Message(role: .assistant, content: content, thinking: thinking)
            history.append(reply)
            continuation.yield(.stats(stats))
            continuation.yield(.complete(reply))
            continuation.finish()
        } catch {
            // Roll back the user turn so a retry re-sends cleanly.
            history.removeSubrange(historyMark...)
            continuation.finish(throwing: error)
        }
    }
}
