// KitVisionExecutor.swift — executor for `KitVisionModel`: extracts the latest image from the
// transcript, vision-encodes it into the decoder's static buffers, renders the prompt with the
// vision block + rope shift, then streams generation. Streaming/detok mirrors `KitExecutor`
// (UTF-8-safe incremental decode + thinking/response segmentation via the shared profile).
//
// v1 deliberately re-prefills the whole prompt every turn (no append-only KV fast path): the
// VL decode bundle is S=1 and the image's rope shift makes cache reuse across image/text turns
// fragile — correctness first. The vision encode itself is skipped when the resident image is
// unchanged (see `VLRuntime.attach`). Multi-image and KV reuse are follow-ups.

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

public struct KitVisionExecutor: LanguageModelExecutor {
    public typealias Model = KitVisionModel

    public struct Configuration: Hashable, Sendable {
        let runtime: VLRuntime
        let modelID: String
        let profile: OutputProfile

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.modelID == rhs.modelID
        }
        public func hash(into hasher: inout Hasher) { hasher.combine(modelID) }
    }

    private let runtime: VLRuntime
    private let modelID: String
    private let profile: OutputProfile

    public init(configuration: Configuration) throws {
        self.runtime = configuration.runtime
        self.modelID = configuration.modelID
        self.profile = configuration.profile
    }

    // MARK: - Prewarm

    /// Touches the weights + compiles the sampler graph on the text-only path (the signature
    /// must be exactly `prewarm(model:transcript:)` — see KitExecutor).
    public func prewarm(model: KitVisionModel, transcript: Transcript) {
        let runtime = self.runtime
        let semaphore = DispatchSemaphore(value: 0)
        DispatchQueue(label: "coreai.kit.vl.prewarm").async {
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
        model: KitVisionModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Guided generation is not wired for the VL path: the schema grammar needs per-step
        // logits the pipelined VL engine cannot expose, and there is no honest approximation.
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "The VL model does not support guided generation."))
        }

        // 1) Render the transcript with the vision block; learn the image to attach.
        let rendered = try VLPromptRenderer.render(
            transcript: request.transcript, tokenizer: runtime.tokenizer, arch: runtime.arch)

        // 2) Bind multimodal state: encode the image into the buffers (skipped if unchanged)
        //    and set the rope shift, or revert to the text-only shift.
        if let image = rendered.image, let imageStart = rendered.imageStart {
            try await runtime.attach(
                cgImage: image.cgImage, orientation: image.orientation,
                segmentID: image.segmentID)
            runtime.setImageShift(imageStart: imageStart)
        } else {
            runtime.detach()
        }

        // 3) Full re-prefill: reset, then feed the whole prompt (pipelined engine, no logits).
        let engine = runtime.engine
        try await engine.reset()

        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        let sampling =
            request.generationOptions.temperature.map { SamplingConfiguration(temperature: $0) }
            ?? .greedy

        // 4) Stream tokens with UTF-8-safe incremental detok feeding the tag parser.
        let tokenizer = runtime.tokenizer
        var parser = StreamingTagParser(profile: profile)
        var detok = StreamingDetokenizer { tokenizer.decode(tokens: $0) }
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
                    continue  // the VL path advertises no tools
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

                let delta = detok.consume(Int(tokenId))
                if !delta.isEmpty {
                    await dispatch(parser.consume(delta))
                }
            }
        } catch InferenceRuntimeError.contextLengthExceeded(let position, let maxContext) {
            throw LanguageModelError.contextSizeExceeded(
                .init(
                    contextSize: maxContext, tokenCount: position,
                    debugDescription: "Transcript no longer fits the model context."))
        }
        await dispatch(parser.flush())

        // 5) Metadata + usage once, attached to the kind this turn produced.
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
