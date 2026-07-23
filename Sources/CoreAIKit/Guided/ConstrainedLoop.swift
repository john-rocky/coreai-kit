// ConstrainedLoop.swift — the kit's grammar-constrained decode loop, shared by
// ChatSession (guided turns) and KitExecutor (FoundationModels schema requests).
//
// Upstream's ConstrainedDecodingStrategy discards applyMask's return value, so once
// the grammar completes it samples one token from UNMASKED logits and yields it —
// the output gains a trailing artifact like `<|endoftext|>`. This loop guards every
// exit instead: grammar complete, EOS sampled, token rejected.

import CoreAILanguageModels
import Foundation
import Tokenizers

enum ConstrainedLoop {
    /// Streams schema-constrained tokens: one engine step per token
    /// (`maxTokens: 1, includeLogits: true`), logits masked through the schema's
    /// grammar (xgrammar bitmask) before sampling.
    ///
    /// `promptTokens` is the cumulative token list from position zero — the contract
    /// of the logits-capable engines (sequential, static-shape), which skip the
    /// already-processed prefix internally. The caller owns reset decisions.
    ///
    /// Yields `GenerationResult(text: delta, tokenId:)` per accepted token; the delta
    /// is incremental UTF-8-safe detok ("" while a multi-byte sequence is incomplete).
    static func stream(
        jsonSchema: String,
        promptTokens: [Int32],
        engine: any InferenceEngine,
        tokenizer: any Tokenizer,
        vocabSize: Int,
        sampling: SamplingConfiguration,
        maxTokens: Int
    ) -> AsyncThrowingStream<GenerationResult, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // EOS is grammar-legal only at terminal states (valid JSON done).
                    let stopTokenIds = tokenizer.eosTokenId.map { [Int32($0)] }
                    var session = try ConstrainedGenerationSession(
                        jsonSchema: jsonSchema,
                        tokenizer: tokenizer,
                        vocabSize: vocabSize,
                        stopTokenIds: stopTokenIds)

                    let stepOptions = InferenceOptions(maxTokens: 1, includeLogits: true)
                    let eosTokenId = tokenizer.eosTokenId
                    var tokens = promptTokens
                    var pendingTokens: [Int] = []
                    var previousDecodedText = ""

                    for _ in 0..<maxTokens {
                        if session.isTerminated { break }
                        try Task.checkCancellation()

                        var stepLogits: [LogitsScalarType]? = nil
                        let stream = try await engine.generate(
                            with: tokens,
                            samplingConfiguration: sampling,
                            inferenceOptions: stepOptions)
                        for try await output in stream {
                            stepLogits = output.logits
                            break
                        }
                        guard var logits = stepLogits else {
                            throw ConstrainedGenerationError.generationFailed(
                                "No logits returned from engine")
                        }
                        // false = grammar complete; never sample from unmasked logits.
                        guard session.applyMask(to: &logits) else { break }
                        let token = CompositeSampler.sample(from: &logits, config: sampling)
                        if let eos = eosTokenId, Int(token) == eos { break }
                        guard session.acceptToken(token) else { break }
                        // An accept that terminates the grammar is a stop token (xgrammar
                        // auto-detects them from the vocabulary — `<|endoftext|>` etc. —
                        // and allows them only once the JSON is complete; the session's
                        // stopTokenIds parameter is dropped upstream). Stop-token text
                        // must never reach the output.
                        if session.isTerminated { break }
                        tokens.append(token)

                        // Incremental UTF-8-safe detok: hold while the full decode
                        // contains U+FFFD, then retain one token of context so
                        // SentencePiece-style tokenizers keep leading spaces.
                        pendingTokens.append(Int(token))
                        let decodedText = tokenizer.decode(tokens: pendingTokens)
                        if decodedText.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                            previousDecodedText = decodedText
                            continuation.yield(
                                GenerationResult(text: "", tokenId: token, rawLogits: nil))
                            continue
                        }
                        let common = decodedText.commonPrefix(with: previousDecodedText)
                        let delta = String(decodedText.dropFirst(common.count))
                        continuation.yield(
                            GenerationResult(text: delta, tokenId: token, rawLogits: nil))
                        if let last = pendingTokens.last {
                            pendingTokens = [last]
                            previousDecodedText = tokenizer.decode(tokens: [last])
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            // Stop stepping the engine when the consumer goes away (cancel/early exit);
            // the per-step checkCancellation picks this up at the next iteration.
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
