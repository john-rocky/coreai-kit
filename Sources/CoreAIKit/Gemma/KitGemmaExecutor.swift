// KitGemmaExecutor.swift — executor for `KitGemmaModel`: renders the transcript in Gemma 4's
// turn format, resets the engine, then streams generation. Streaming/detok mirrors
// `KitExecutor` (UTF-8-safe incremental decode feeding the shared tag parser).
//
// v1 re-prefills the whole prompt every turn (no append-only KV fast path): correctness first,
// matching the device-verified Gemma backend, which resets before each generation. The two
// constant PLE table buffers are bound for the engine's lifetime, so a turn is just
// reset → feed prompt → stream. The turn ends on `<turn|>` (Gemma's end-of-turn) or the
// tokenizer's `<eos>`. KV reuse is a follow-up.

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

public struct KitGemmaExecutor: LanguageModelExecutor {
    public typealias Model = KitGemmaModel

    public struct Configuration: Hashable, Sendable {
        let runtime: GemmaRuntime
        let modelID: String
        let profile: OutputProfile

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.modelID == rhs.modelID
        }
        public func hash(into hasher: inout Hasher) { hasher.combine(modelID) }
    }

    private let runtime: GemmaRuntime
    private let modelID: String
    private let profile: OutputProfile

    public init(configuration: Configuration) throws {
        self.runtime = configuration.runtime
        self.modelID = configuration.modelID
        self.profile = configuration.profile
    }

    // MARK: - Prewarm

    /// Touches the weights + compiles the sampler graph (the signature must be exactly
    /// `prewarm(model:transcript:)` — see KitExecutor).
    public func prewarm(model: KitGemmaModel, transcript: Transcript) {
        let runtime = self.runtime
        let semaphore = DispatchSemaphore(value: 0)
        DispatchQueue(label: "coreai.kit.gemma.prewarm").async {
            Task {
                defer { semaphore.signal() }
                try? await runtime.warmup()
            }
        }
        semaphore.wait()
    }

    // MARK: - respond

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: KitGemmaModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Guided generation is not wired for the Gemma path: the schema grammar needs per-step
        // logits the pipelined engine cannot expose, and there is no honest approximation.
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "The Gemma model does not support guided generation."))
        }

        // 1) Render the transcript in Gemma 4's turn format (explicit bos + turn/channel tokens).
        let rendered = try GemmaPromptRenderer.render(
            transcript: request.transcript, tokenizer: runtime.tokenizer, arch: runtime.arch)

        // 2) Full re-prefill: reset, then feed the whole prompt (pipelined engine, no logits).
        let engine = runtime.engine
        try await engine.reset()

        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        let sampling =
            request.generationOptions.temperature.map { SamplingConfiguration(temperature: $0) }
            ?? .greedy

        // 3) Stream tokens with UTF-8-safe incremental detok feeding the tag parser.
        let tokenizer = runtime.tokenizer
        var parser = StreamingTagParser(profile: profile)
        var pendingTokens: [Int] = []
        var previousDecodedText = ""
        var generatedCount = 0
        var reasoningEventCount = 0
        var sentResponseText = false
        let eosTokenId = tokenizer.eosTokenId
        // Gemma ends every turn with <turn|>; stop on it as well as the rarer <eos>.
        let endOfTurnID = tokenizer.convertTokenToId(runtime.arch.endOfTurn)

        func dispatch(_ events: [StreamingTagParser.Event]) async {
            for event in events {
                switch event {
                case .response(let text):
                    sentResponseText = true
                    await channel.send(.response(action: .appendText(text, tokenCount: 1)))
                case .thinking(let text):
                    reasoningEventCount += 1
                    await channel.send(.reasoning(action: .appendText(text, tokenCount: 1)))
                case .toolCallPayload:
                    continue  // the Gemma path advertises no tools
                }
            }
        }

        do {
            let stream = try engine.generate(
                with: rendered.tokens,
                samplingConfiguration: sampling,
                inferenceOptions: InferenceOptions(maxTokens: maxTokens))
            for try await output in stream {
                let tokenId = output.tokenId
                if let eos = eosTokenId, Int(tokenId) == eos { break }
                if let eot = endOfTurnID, Int(tokenId) == eot { break }
                generatedCount += 1

                pendingTokens.append(Int(tokenId))
                let decodedText = tokenizer.decode(tokens: pendingTokens)
                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                if decodedText.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                    previousDecodedText = decodedText
                    continue
                }
                await dispatch(parser.consume(delta))
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [last])
                }
            }
        } catch InferenceRuntimeError.contextLengthExceeded(let position, let maxContext) {
            throw LanguageModelError.contextSizeExceeded(
                .init(
                    contextSize: maxContext, tokenCount: position,
                    debugDescription: "Transcript no longer fits the model context."))
        }
        await dispatch(parser.flush())

        // 4) Metadata + usage once, attached to the kind this turn produced.
        let metadata: [String: any Sendable & Codable & Equatable] = [
            "modelID": modelID,
            "requestID": request.id.uuidString,
        ]
        let usageInput = LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: rendered.tokens.count, cachedTokenCount: 0)
        let usageOutput = LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: generatedCount, reasoningTokenCount: reasoningEventCount)
        if sentResponseText || reasoningEventCount == 0 {
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
}
