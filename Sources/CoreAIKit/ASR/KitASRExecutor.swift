// KitASRExecutor.swift — executor for `KitASRModel`: renders the Qwen3-ASR prompt from the
// out-of-band-attached audio-token count, generates, and streams the transcript (the text AFTER
// the `<asr_text>` marker; the `{lang}` prefix is dropped). No thinking/tools on the ASR path.
//
// The audio is attached to the runtime BEFORE this runs (the app calls `KitASRModel.attach`), so
// respond does not touch the encoder. Most apps will prefer the direct `KitASRModel.transcribe()`;
// this executor exists so an ASR model also plugs into a plain `LanguageModelSession`.

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

public struct KitASRExecutor: LanguageModelExecutor {
    public typealias Model = KitASRModel

    public struct Configuration: Hashable, Sendable {
        let runtime: ASRRuntime
        let modelID: String

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.modelID == rhs.modelID
        }
        public func hash(into hasher: inout Hasher) { hasher.combine(modelID) }
    }

    private let runtime: ASRRuntime
    private let modelID: String

    public init(configuration: Configuration) throws {
        self.runtime = configuration.runtime
        self.modelID = configuration.modelID
    }

    public func prewarm(model: KitASRModel, transcript: Transcript) {
        let runtime = self.runtime
        let semaphore = DispatchSemaphore(value: 0)
        DispatchQueue(label: "coreai.kit.asr.prewarm").async {
            Task { defer { semaphore.signal() }; try? await runtime.warmup() }
        }
        semaphore.wait()
    }

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: KitASRModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription: "The ASR model does not support guided generation."))
        }

        let tokenizer = runtime.tokenizer
        let arch = runtime.arch
        // The user prompt text (if any) is treated as a forced-language hint ("Japanese", "English"…).
        let hint = plainText(request.transcript).trimmingCharacters(in: .whitespacesAndNewlines)
        let language = hint.isEmpty ? nil : hint
        let prompt = ASRPromptRenderer.render(
            tokenizer: tokenizer, arch: arch, audioTokenCount: runtime.attachedAudioTokens,
            language: language)

        let engine = runtime.engine
        try await engine.reset()
        let maxTokens = request.generationOptions.maximumResponseTokens ?? 448

        let asrTextID = tokenizer.convertTokenToId(arch.asrText)
        let eosTokenId = tokenizer.eosTokenId
        // With a forced language the prompt already primed `<asr_text>`, so emit from the start.
        var inText = (language != nil) || (asrTextID == nil)
        var transcriptTokens: [Int] = []
        var previousDecodedText = ""
        var generatedCount = 0

        do {
            let stream = try await engine.generate(
                with: prompt, samplingConfiguration: .greedy,
                inferenceOptions: InferenceOptions(maxTokens: maxTokens))
            for try await output in stream {
                let tokenId = Int(output.tokenId)
                if let eos = eosTokenId, tokenId == eos { break }
                generatedCount += 1
                if !inText {
                    if tokenId == asrTextID { inText = true }
                    continue  // still in the `{lang}` prefix
                }
                transcriptTokens.append(tokenId)
                let decodedText = tokenizer.decode(tokens: transcriptTokens)
                if decodedText.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) { continue }
                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                previousDecodedText = decodedText
                if !delta.isEmpty {
                    await channel.send(.response(action: .appendText(delta, tokenCount: 1)))
                }
            }
        } catch InferenceRuntimeError.contextLengthExceeded(let position, let maxContext) {
            throw LanguageModelError.contextSizeExceeded(
                .init(
                    contextSize: maxContext, tokenCount: position,
                    debugDescription: "Audio + prompt no longer fits the model context."))
        }

        let metadata: [String: any Sendable & Codable & Equatable] = [
            "modelID": modelID, "requestID": request.id.uuidString,
        ]
        let usageInput = LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: prompt.count, cachedTokenCount: 0)
        let usageOutput = LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: generatedCount, reasoningTokenCount: 0)
        await channel.send(.response(action: .updateMetadata(metadata)))
        await channel.send(.response(action: .updateUsage(input: usageInput, output: usageOutput)))
        await Task.yield()
    }

    private func plainText(_ transcript: Transcript) -> String {
        var out: [String] = []
        for entry in transcript {
            if case .prompt(let prompt) = entry {
                for segment in prompt.segments {
                    if case .text(let text) = segment { out.append(text.content) }
                }
            }
        }
        return out.joined(separator: " ")
    }
}
