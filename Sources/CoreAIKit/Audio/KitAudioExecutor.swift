// KitAudioExecutor.swift — executor for `KitAudioModel`: renders the transcript with the audio
// block (using the runtime's out-of-band-attached audio-token count), then streams generation.
// Streaming/detok mirrors `KitVisionExecutor` (UTF-8-safe incremental decode + thinking/response
// segmentation via the shared profile).
//
// The audio is attached to the runtime BEFORE this runs (the app calls `KitAudioModel.attach`),
// so respond does not touch the encoder — it re-prefills the whole prompt each turn (v1: no
// append-only KV fast path; the audio block makes cross-turn cache reuse fragile — correctness
// first). The static `audio_embeds` buffer stays bound across turns until detached.

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

public struct KitAudioExecutor: LanguageModelExecutor {
    public typealias Model = KitAudioModel

    public struct Configuration: Hashable, Sendable {
        let runtime: AudioRuntime
        let modelID: String
        let profile: OutputProfile

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.modelID == rhs.modelID
        }
        public func hash(into hasher: inout Hasher) { hasher.combine(modelID) }
    }

    private let runtime: AudioRuntime
    private let modelID: String
    private let profile: OutputProfile

    public init(configuration: Configuration) throws {
        self.runtime = configuration.runtime
        self.modelID = configuration.modelID
        self.profile = configuration.profile
    }

    // MARK: - Prewarm

    public func prewarm(model: KitAudioModel, transcript: Transcript) {
        let runtime = self.runtime
        let semaphore = DispatchSemaphore(value: 0)
        DispatchQueue(label: "coreai.kit.audio.prewarm").async {
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
        model: KitAudioModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "The audio model does not support guided generation."))
        }

        // 1) Render the transcript with the audio block (uses the out-of-band-attached count).
        let rendered = try AudioPromptRenderer.render(
            transcript: request.transcript, tokenizer: runtime.tokenizer, arch: runtime.arch,
            audioTokenCount: runtime.attachedAudioTokens)

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
        let imEndID = tokenizer.convertTokenToId(runtime.arch.imEnd)

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
                    continue  // the audio path advertises no tools
                }
            }
        }

        do {
            let stream = try await engine.generate(
                with: rendered.tokens,
                samplingConfiguration: sampling,
                inferenceOptions: InferenceOptions(maxTokens: maxTokens))
            for try await output in stream {
                let tokenId = output.tokenId
                if let eos = eosTokenId, Int(tokenId) == eos { break }
                if let imEnd = imEndID, Int(tokenId) == imEnd { break }
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
        let metadata: [String: any ConvertibleToGeneratedContent] = [
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
