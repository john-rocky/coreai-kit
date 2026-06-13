// KitExecutor.swift — executor for `KitLanguageModel`: streaming generation with
// incremental tool-call/thinking detection, usage events, and an append-only KV fast
// path. Adapted from the coreai-model-zoo ZooFMProvider (see NOTICE.txt).

import CoreAILanguageModels
import Foundation
import FoundationModels
import Synchronization
import Tokenizers

/// ## Events
/// Text streams out as `.response` events the moment each token decodes cleanly;
/// reasoning streams as `.reasoning`; tool calls are emitted as one `.toolCalls` event
/// per call (consecutive events coalesce into a single transcript entry) once their JSON
/// body is complete. Metadata + usage are sent once per turn at the end, attached to the
/// kind of entry the turn produced — a usage event of a kind the turn never emits content
/// for would materialize an empty transcript entry (verified on the 27.0 beta).
///
/// ## KV reuse and the over-generation pump
/// The engine preserves KV between `generate()` calls and continues from its current
/// position, so when the new transcript's rendered tokens extend the tokens already in
/// the cache, only the suffix is prefilled (`Usage.Input.cachedTokenCount` reports the
/// reuse).
///
/// One engine reality shapes this design: breaking out of the token stream (at EOS) does
/// NOT stop the pipelined engine — it keeps generating to `maxTokens` in the background,
/// and those post-EOS tokens land in the KV cache. `respond` therefore pumps the engine
/// stream through a relay task that keeps draining (and recording) tokens after the
/// consumer stops, and the next `respond` settles that bookkeeping before deciding
/// whether the cache still matches the transcript. Post-EOS tokens usually diverge from
/// the next rendered prompt, in which case the executor falls back to reset + full
/// re-prefill — correctness first.
public struct KitExecutor: LanguageModelExecutor {
    public typealias Model = KitLanguageModel

    public struct Configuration: Hashable, Sendable {
        let engine: any InferenceEngine
        let tokenizer: any Tokenizer
        let modelID: String
        let profile: OutputProfile
        let supportsHermesTools: Bool
        let vocabSize: Int?

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.modelID == rhs.modelID
        }
        public func hash(into hasher: inout Hasher) {
            hasher.combine(modelID)
        }
    }

    private let engine: any InferenceEngine
    private let tokenizer: any Tokenizer
    private let modelID: String
    private let profile: OutputProfile
    private let supportsHermesTools: Bool
    private let vocabSize: Int?
    private let state = TurnState()

    public init(configuration: Configuration) throws {
        self.engine = configuration.engine
        self.tokenizer = configuration.tokenizer
        self.modelID = configuration.modelID
        self.profile = configuration.profile
        self.supportsHermesTools = configuration.supportsHermesTools
        self.vocabSize = configuration.vocabSize
    }

    // MARK: - Prewarm

    /// NOTE: the signature must be exactly `prewarm(model:transcript:)` — the protocol
    /// ships a default no-op, so a near-miss compiles and is silently never called.
    ///
    /// Warm = one real 1-token generate + reset (compiles the sampler graph and touches
    /// the weights). `engine.warmup()` is deliberately not used: its default query length
    /// builds a step shape some bundles reject.
    ///
    /// The warm work registers itself as the in-flight turn, so a respond that races
    /// prewarm awaits it in `settle()` instead of contending for the engine. It returns
    /// `[]`: the trailing reset leaves nothing in the cache, which is exactly what settle
    /// computes from an empty turn.
    public func prewarm(model: KitLanguageModel, transcript: Transcript) {
        let engine = self.engine
        let tokenizer = self.tokenizer
        let state = self.state
        let semaphore = DispatchSemaphore(value: 0)
        // Block via a GCD thread — semaphore.wait() on a cooperative thread while Task{}
        // needs one would starve (same pattern as Apple's adapter).
        DispatchQueue(label: "coreai.kit.fm.prewarm").async {
            // Snapshot the in-flight pump BEFORE creating the warm task — the warm task
            // must not settle() itself once registered (self-await).
            let previous = state.takePump()
            let warm = Task<[Int32], any Error> {
                defer { semaphore.signal() }
                if let previous {
                    _ = try? await previous.value  // engine free before generate
                }
                let seed = tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
                let stream = try engine.generate(
                    with: [seed],
                    samplingConfiguration: .greedy,
                    inferenceOptions: InferenceOptions(maxTokens: 1))
                for try await _ in stream {}
                try await engine.reset()
                return []
            }
            state.beginTurn(base: [], fed: [], pump: warm)
        }
        semaphore.wait()
    }

    // MARK: - respond

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: KitLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Constrained decoding needs per-step logits; the pipelined engine samples
        // on-GPU. Approximate-or-throw rule: there is no honest approximation of a
        // schema, so anything else throws.
        if let schema = request.schema {
            guard engine.supportsLogits else {
                throw LanguageModelError.unsupportedCapability(
                    .init(
                        capability: .guidedGeneration,
                        debugDescription:
                            "This engine cannot expose per-step logits. Load the model "
                            + "with engineVariant: .sequential for guided generation."))
            }
            try await respondConstrained(schema: schema, request: request, channel: channel)
            return
        }

        // 1) Settle the previous turn: await its background drain and learn exactly
        //    which tokens are in the KV cache (nil = unknown → reset).
        let kvTokens = await state.settle()

        // 2) Render the transcript. toolCallingMode (WWDC26 242): disallowed drops the
        //    tools block entirely; required adds a must-call instruction (prompt-level
        //    approximation — local models have no grammar enforcement).
        let mode = request.generationOptions.toolCallingMode?.kind ?? .allowed
        let tools = mode == .disallowed ? [] : request.enabledToolDefinitions
        let rendered = try TranscriptRenderer.render(
            transcript: request.transcript,
            tools: tools,
            requireToolCall: mode == .required,
            tokenizer: tokenizer,
            supportsHermesTools: supportsHermesTools)
        let promptTokens = rendered.tokens

        // 3) Append-only KV fast path: skip reset and feed only the suffix when the
        //    rendered prompt extends what's already in the cache.
        let fed: [Int32]
        let kvBase: [Int32]
        if let kv = kvTokens, kv.isEmpty {
            fed = promptTokens
            kvBase = []
        } else if let kv = kvTokens, promptTokens.count > kv.count,
            promptTokens.starts(with: kv)
        {
            fed = Array(promptTokens[kv.count...])
            kvBase = kv
            kitFMDebug("KV fast path: reusing \(kv.count) tokens, prefilling \(fed.count)")
        } else {
            if let kv = kvTokens {
                kitFMDebug(
                    "KV diverged (cache \(kv.count) tokens vs prompt \(promptTokens.count)) — reset")
            }
            try await engine.reset()
            fed = promptTokens
            kvBase = []
        }
        let cachedCount = kvBase.count

        // (WWDC 339 suggests metadata + usage upfront, but a usage-only .response event
        // materializes an EMPTY Response transcript entry when the turn ends in tool
        // calls — verified against the 27.0 beta framework. Until kind-agnostic usage
        // exists, both are sent once at end of turn, attached to the entry kind the turn
        // produced.)

        // 4) Generate through a relay: the pump owns the engine stream and keeps
        //    draining it (recording every token) after this function stops consuming at
        //    EOS — see the type comment.
        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        let sampling =
            request.generationOptions.temperature.map {
                SamplingConfiguration(temperature: $0)
            } ?? .greedy

        // Engine input contracts differ: the pipelined engine consumes its input as NEW
        // tokens (feed only the suffix), while logits-capable engines (sequential,
        // static-shape) take the cumulative token list and skip the already-processed
        // prefix internally. TurnState bookkeeping uses `fed` (the suffix) either way.
        let engineInput = engine.supportsLogits ? promptTokens : fed

        let (relay, relayContinuation) = AsyncThrowingStream<Int32, any Error>.makeStream()
        let engine = self.engine
        let pump = Task {
            var ids: [Int32] = []
            do {
                let stream = try engine.generate(
                    with: engineInput,
                    samplingConfiguration: sampling,
                    inferenceOptions: InferenceOptions(maxTokens: maxTokens))
                for try await output in stream {
                    ids.append(output.tokenId)
                    relayContinuation.yield(output.tokenId)
                }
                relayContinuation.finish()
            } catch {
                relayContinuation.finish(throwing: error)
                throw error
            }
            return ids
        }
        state.beginTurn(base: kvBase, fed: fed, pump: pump)

        // 5) Stream: incremental UTF-8-safe detok (U+FFFD hold + one retained context
        //    token, same contract as Apple's adapter) feeding the tag parser; text goes
        //    out the moment it decodes.
        var parser = StreamingTagParser(profile: profile)
        var pendingTokens: [Int] = []
        var previousDecodedText = ""
        var generatedCount = 0
        var reasoningEventCount = 0
        var sentResponseText = false
        var toolPayloads: [String] = []
        let eosTokenId = tokenizer.eosTokenId

        func dispatch(_ events: [StreamingTagParser.Event]) async {
            for event in events {
                switch event {
                case .response(let text):
                    sentResponseText = true
                    await channel.send(.response(action: .appendText(text, tokenCount: 1)))
                case .thinking(let text):
                    reasoningEventCount += 1
                    await channel.send(.reasoning(action: .appendText(text, tokenCount: 1)))
                case .toolCallPayload(let payload):
                    toolPayloads.append(payload)
                }
            }
        }

        do {
            for try await tokenId in relay {
                if let eos = eosTokenId, Int(tokenId) == eos { break }
                // Count AFTER the EOS check so the never-emitted EOS sentinel is
                // not folded into Usage.Output (matches Apple's reference adapter).
                generatedCount += 1

                pendingTokens.append(Int(tokenId))
                let decodedText = tokenizer.decode(tokens: pendingTokens)
                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                // Check U+FFFD on the full decode, not the delta: consecutive partial
                // tokens can decode to identical replacement strings, making the delta
                // empty and hiding the incomplete state.
                if decodedText.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                    previousDecodedText = decodedText
                    continue
                }
                await dispatch(parser.consume(delta))
                // Retain one token of context: SentencePiece-style tokenizers need a
                // predecessor to infer leading spaces, and one token bounds the
                // re-decode to O(1) per step.
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [last])
                }
            }
        } catch InferenceRuntimeError.contextLengthExceeded(let position, let maxContext) {
            throw LanguageModelError.contextSizeExceeded(
                .init(
                    contextSize: maxContext,
                    tokenCount: position,
                    debugDescription: "Transcript no longer fits the model context."))
        }
        // Flush content the parser held back for a possible marker match — without this,
        // text at the EOS boundary is lost.
        await dispatch(parser.flush())

        // 6) Tool calls: every complete payload becomes one toolCall with a minted id;
        //    consecutive .toolCalls events form one transcript entry, so a multi-call
        //    turn lands as a single ToolCalls entry.
        if !toolPayloads.isEmpty {
            let calls = try toolPayloads.flatMap(Self.parseToolCalls)
            for call in calls {
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: UUID().uuidString,
                            name: call.name,
                            action: .appendArguments(
                                call.argumentsJSON,
                                tokenCount: max(1, call.argumentsJSON.count / 4)))))
            }
        }

        // 7) Metadata + usage once per turn, attached to the kind of entry this turn
        //    actually produced (see the note above step 4).
        let metadata: [String: any Sendable & Codable & Equatable] = [
            "modelID": modelID,
            "requestID": request.id.uuidString,
        ]
        let usageInput = LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: promptTokens.count, cachedTokenCount: cachedCount)
        let usageOutput = LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: generatedCount, reasoningTokenCount: reasoningEventCount)
        if !toolPayloads.isEmpty {
            await channel.send(.toolCalls(action: .updateMetadata(metadata)))
            await channel.send(
                .toolCalls(action: .updateUsage(input: usageInput, output: usageOutput)))
        } else if sentResponseText || reasoningEventCount == 0 {
            await channel.send(.response(action: .updateMetadata(metadata)))
            await channel.send(
                .response(action: .updateUsage(input: usageInput, output: usageOutput)))
        } else {
            await channel.send(.reasoning(action: .updateMetadata(metadata)))
            await channel.send(
                .reasoning(action: .updateUsage(input: usageInput, output: usageOutput)))
        }

        await Task.yield()
    }

    // MARK: - Constrained respond (guided generation)

    /// Schema-constrained turn: per-step logits are masked through the schema's grammar
    /// (xgrammar bitmask) before sampling, so the streamed text is valid JSON for
    /// `request.schema` by construction. One engine step per token
    /// (`maxTokens: 1, includeLogits: true`) — there is no over-generation pump because
    /// each call stops after its single step.
    private func respondConstrained(
        schema: GenerationSchema,
        request: LanguageModelExecutorGenerationRequest,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // 1) Settle the previous turn, learning what the KV cache holds (nil = reset).
        let kvTokens = await state.settle()

        // 2) Render the transcript EXACTLY like a vanilla turn (same tools block), so
        //    guided and plain turns stay prefix-compatible for KV reuse.
        let mode = request.generationOptions.toolCallingMode?.kind ?? .allowed
        let tools = mode == .disallowed ? [] : request.enabledToolDefinitions
        let rendered = try TranscriptRenderer.render(
            transcript: request.transcript,
            tools: tools,
            requireToolCall: mode == .required,
            tokenizer: tokenizer,
            supportsHermesTools: supportsHermesTools)
        let promptTokens = rendered.tokens

        // 3) KV reuse. Logits-capable engines take the cumulative token list and skip
        //    the processed prefix internally, so reuse just means skipping the reset.
        let kvBase: [Int32]
        if let kv = kvTokens, promptTokens.count > kv.count, promptTokens.starts(with: kv) {
            kvBase = kv
            kitFMDebug("guided KV fast path: reusing \(kv.count) tokens")
        } else {
            try await engine.reset()
            kvBase = []
        }
        let fed = Array(promptTokens[kvBase.count...])

        // 4) A `GenerationSchema` JSON-encodes to the JSON schema the grammar compiler
        //    expects (same conversion as Apple's CoreAIExecutor).
        let schemaData = try JSONEncoder().encode(schema)
        guard let jsonSchema = String(data: schemaData, encoding: .utf8) else {
            preconditionFailure("GenerationSchema JSON encoding produced invalid UTF-8")
        }

        // 5) Step loop (shared ConstrainedLoop): logits -> grammar mask -> sample ->
        //    accept -> stream delta.
        let sampling =
            request.generationOptions.temperature.map {
                SamplingConfiguration(temperature: $0)
            } ?? .greedy
        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512

        var generated: [Int32] = []
        do {
            let stream = ConstrainedLoop.stream(
                jsonSchema: jsonSchema,
                promptTokens: promptTokens,
                engine: engine,
                tokenizer: tokenizer,
                vocabSize: try resolveVocabSize(),
                sampling: sampling,
                maxTokens: maxTokens)
            for try await result in stream {
                generated.append(result.tokenId)
                if !result.text.isEmpty {
                    await channel.send(
                        .response(action: .appendText(result.text, tokenCount: 1)))
                }
            }
        } catch {
            // The engine state is mid-step unknown; a throwing pump makes the next
            // respond settle to nil and reset.
            state.beginTurn(
                base: [], fed: [],
                pump: Task<[Int32], any Error> { throw error })
            if case InferenceRuntimeError.contextLengthExceeded(let position, let maxContext) =
                error
            {
                throw LanguageModelError.contextSizeExceeded(
                    .init(
                        contextSize: maxContext,
                        tokenCount: position,
                        debugDescription: "Transcript no longer fits the model context."))
            }
            throw error
        }

        // 6) Record the turn for KV bookkeeping. Every generated token except the last
        //    was fed back into the engine, so settle's `base + fed + ids.dropLast()`
        //    is exactly the cache contents — a pre-completed pump carries the ids.
        let ids = generated
        state.beginTurn(
            base: kvBase, fed: fed,
            pump: Task<[Int32], any Error> { ids })

        // 7) Metadata + usage, end of turn (see the vanilla path note). A guided turn
        //    always produces response text, so both attach to `.response`.
        let metadata: [String: any Sendable & Codable & Equatable] = [
            "modelID": modelID,
            "requestID": request.id.uuidString,
        ]
        await channel.send(.response(action: .updateMetadata(metadata)))
        await channel.send(
            .response(
                action: .updateUsage(
                    input: .init(
                        totalTokenCount: promptTokens.count, cachedTokenCount: kvBase.count),
                    output: .init(
                        totalTokenCount: generated.count, reasoningTokenCount: 0))))

        await Task.yield()
    }

    /// Bundle-metadata vocab size when available, else derived from the tokenizer by
    /// binary-searching the last valid token id (the grammar bitmask must cover the
    /// model's full output row, not the tokenizer's dense entries).
    private func resolveVocabSize() throws -> Int {
        if let vocabSize { return vocabSize }
        var low = 0
        var high = 524_288
        while low < high {
            let mid = (low + high) / 2
            if tokenizer.convertIdToToken(mid) != nil {
                low = mid + 1
            } else {
                high = mid
            }
        }
        guard low > 0 else {
            throw ConstrainedGenerationError.generationFailed(
                "Cannot determine vocabulary size from the tokenizer; "
                    + "construct KitLanguageModel with an explicit vocabSize.")
        }
        return low
    }

    // MARK: - Tool call parsing

    struct ParsedToolCall {
        let name: String
        let argumentsJSON: String
    }

    /// `{"name": "...", "arguments": {...}}` per block, or a top-level JSON ARRAY of such
    /// objects (some Hermes-tuned models emit a multi-call array in one block — matches
    /// `HermesDialect.parseToolCalls` in the zoo provider). Anything else throws.
    static func parseToolCalls(_ payload: String) throws -> [ParsedToolCall] {
        let trimmed = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            let data = trimmed.data(using: .utf8),
            let root = try? JSONSerialization.jsonObject(with: data)
        else {
            throw KitFMError.malformedToolCall(payload: trimmed)
        }
        let objects: [[String: Any]]
        if let object = root as? [String: Any] {
            objects = [object]
        } else if let array = root as? [[String: Any]], !array.isEmpty {
            objects = array
        } else {
            throw KitFMError.malformedToolCall(payload: trimmed)
        }
        return try objects.map { object in
            guard let name = object["name"] as? String, !name.isEmpty else {
                throw KitFMError.malformedToolCall(payload: trimmed)
            }
            let arguments = object["arguments"] ?? [String: Any]()
            guard
                let argumentsData = try? JSONSerialization.data(
                    withJSONObject: arguments, options: [.fragmentsAllowed, .sortedKeys]),
                let argumentsJSON = String(data: argumentsData, encoding: .utf8)
            else {
                throw KitFMError.malformedToolCall(payload: trimmed)
            }
            return ParsedToolCall(name: name, argumentsJSON: argumentsJSON)
        }
    }
}

// MARK: - Turn state

/// Mutex-protected bookkeeping shared by all copies of one executor: which token ids the
/// engine's KV cache holds, and the pump task still draining the previous turn's stream.
final class TurnState: Sendable {
    private struct Turn: Sendable {
        /// Tokens known to be consumed into the KV cache. nil = unknown — the initial
        /// state (a fresh executor can't know what an earlier session left in the
        /// engine) and the state after an engine error; either way the next respond
        /// resets first.
        var kvTokens: [Int32]? = nil
        var pendingBase: [Int32] = []
        var pendingFed: [Int32] = []
        var pump: Task<[Int32], any Error>?
    }

    private let turn = Mutex<Turn>(Turn())

    func beginTurn(base: [Int32], fed: [Int32], pump: Task<[Int32], any Error>) {
        turn.withLock {
            $0.pendingBase = base
            $0.pendingFed = fed
            $0.pump = pump
        }
    }

    /// Detaches the in-flight pump without folding its result (prewarm's reset discards
    /// the cache anyway).
    func takePump() -> Task<[Int32], any Error>? {
        turn.withLock {
            let pump = $0.pump
            $0.pump = nil
            return pump
        }
    }

    /// Awaits the previous turn's pump and folds its result into `kvTokens`. The last
    /// yielded token was sampled but never consumed by a subsequent step, so it is NOT
    /// in the cache — hence `dropLast()`.
    func settle() async -> [Int32]? {
        let (pump, base, fed) = turn.withLock {
            let values = ($0.pump, $0.pendingBase, $0.pendingFed)
            $0.pump = nil
            return values
        }
        guard let pump else {
            return turn.withLock { $0.kvTokens }
        }
        do {
            let ids = try await pump.value
            let kv = base + fed + ids.dropLast()
            turn.withLock { $0.kvTokens = kv }
            kitFMDebug("settled turn: \(ids.count) tokens streamed, cache now \(kv.count) tokens")
            return kv
        } catch {
            turn.withLock { $0.kvTokens = nil }
            kitFMDebug("settle: previous turn failed (\(error)) — cache unknown")
            return nil
        }
    }
}

// MARK: - Debug logging

let kitFMDebugEnabled = ProcessInfo.processInfo.environment["COREAI_KIT_DEBUG"] != nil

func kitFMDebug(_ message: @autoclosure () -> String) {
    guard kitFMDebugEnabled else { return }
    FileHandle.standardError.write(Data("[coreai-kit] \(message())\n".utf8))
}
