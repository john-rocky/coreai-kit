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
/// ## KV reuse and the relay pump
/// The engine preserves KV and its token history between `generate()` calls (implicit
/// prefix caching). Each turn the executor rewinds to the longest prefix its own mirror
/// shares with the newly rendered transcript (`reset(to:)`), then feeds the FULL token
/// list — the engine prefills only the tail (`Usage.Input.cachedTokenCount` reports the
/// reuse). The explicit rewind also trims the few post-EOS tokens the pipeline may have
/// consumed after the consumer broke, which would otherwise read as divergence inside
/// the engine and force a full re-prefill.
///
/// `respond` still pumps the engine stream through a relay task: the pump owns the
/// stream, records every token for the KV mirror, and (with the D1 EOS-stop in the
/// engine) finishes promptly once the consumer stops; the next `respond` settles that
/// bookkeeping before computing the shared prefix. A mirror in an unknown state (fresh
/// executor, engine error) settles to nil → `reset(to: 0)` — correctness first.
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
                let stream = try await engine.generate(
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

        // 3) KV fast path: rewind the engine to the longest shared prefix, then feed the
        //    FULL rendered prompt — the engine's implicit prefix caching prefills only the
        //    tail beyond its recorded history. The explicit reset(to:) trims tokens the
        //    pipeline consumed after the consumer broke (post-EOS drain) and turns a
        //    divergence (e.g. a re-rendered transcript) into a pure extension instead of
        //    the engine's divergence full-reset. nil mirror (fresh executor / prior error)
        //    → reset(to: 0). Engines that can't rewind (recurrent/SSM) degrade to a full
        //    reset in rewind(to:), which reports 0 kept — see EngineRewind.swift.
        let wanted: Int
        if let kv = kvTokens {
            wanted = min(
                Self.commonPrefixLength(kv, promptTokens),
                max(0, promptTokens.count - 1),
                engine.processedTokenCount)
        } else {
            wanted = 0
        }
        let common = try await engine.rewind(to: wanted)
        let kvBase = Array(promptTokens[..<common])
        let fed = Array(promptTokens[common...])
        if common > 0 {
            kitFMDebug("KV fast path: reusing \(common) tokens, prefilling \(fed.count)")
        } else if let kv = kvTokens, !kv.isEmpty {
            kitFMDebug(
                "KV diverged (cache \(kv.count) tokens vs prompt \(promptTokens.count)) — reset")
        }
        let cachedCount = common

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

        // Every engine now takes the full cumulative token list and slices off its cached
        // prefix internally (implicit prefix caching). TurnState bookkeeping still uses
        // `fed` (the suffix) to reconstruct the cache mirror.
        let (relay, relayContinuation) = AsyncThrowingStream<Int32, any Error>.makeStream()
        let engine = self.engine
        let eosForPump = tokenizer.eosTokenId
        let pump = Task {
            var ids: [Int32] = []
            do {
                let stream = try await engine.generate(
                    with: promptTokens,
                    samplingConfiguration: sampling,
                    inferenceOptions: InferenceOptions(maxTokens: maxTokens))
                for try await output in stream {
                    ids.append(output.tokenId)
                    relayContinuation.yield(output.tokenId)
                    // Stop at EOS instead of draining to maxTokens. With v0.1.1-zoo's D1, breaking
                    // the engine stream stops the engine, so the pump finishes promptly; without
                    // this break the engine over-generates and the NEXT turn's settle() waits on
                    // that long background drain — which is what made the 2nd consecutive generation
                    // time out ("something went wrong"). The few-tokens-past-EOS the pipeline may
                    // still hold only matter for cross-turn KV reuse; an independent next turn
                    // diverges and resets anyway.
                    if let eos = eosForPump, Int(output.tokenId) == eos { break }
                }
                relayContinuation.finish()
            } catch {
                relayContinuation.finish(throwing: error)
                throw error
            }
            return ids
        }
        state.beginTurn(base: kvBase, fed: fed, pump: pump)

        // 5) Stream: incremental UTF-8-safe detok (StreamingDetokenizer — holds only a
        //    trailing in-progress character, so a stray invalid byte cannot starve the
        //    turn) feeding the tag parser; text goes out the moment it decodes.
        var parser = StreamingTagParser(profile: profile)
        var detok = StreamingDetokenizer { [tokenizer = self.tokenizer] in
            tokenizer.decode(tokens: $0)
        }
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

                let delta = detok.consume(Int(tokenId))
                if !delta.isEmpty {
                    await dispatch(parser.consume(delta))
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
        let metadata: [String: any ConvertibleToGeneratedContent] = [
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

        // 3) KV reuse: same rewind-then-full-feed contract as the vanilla path (the
        //    sequential engine's implicit caching skips the processed prefix internally).
        let wanted: Int
        if let kv = kvTokens {
            wanted = min(
                Self.commonPrefixLength(kv, promptTokens),
                max(0, promptTokens.count - 1),
                engine.processedTokenCount)
        } else {
            wanted = 0
        }
        let common = try await engine.rewind(to: wanted)
        if common > 0 {
            kitFMDebug("guided KV fast path: reusing \(common) tokens")
        }
        let kvBase = Array(promptTokens[..<common])
        let fed = Array(promptTokens[common...])

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
        let metadata: [String: any ConvertibleToGeneratedContent] = [
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

    private static func commonPrefixLength(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

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
