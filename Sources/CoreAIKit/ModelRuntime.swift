// ModelRuntime.swift — loads a language bundle into ready-to-generate parts.

import CoreAILanguageModels
import Foundation
import Tokenizers

struct ModelRuntime: Sendable {
    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelName: String
    let loadSeconds: Double
    let outputProfile: OutputProfile

    /// Loads a bundle directory (metadata.json + *.aimodel/ + tokenizer/), creating the
    /// engine and tokenizer concurrently — the same path as Apple's CoreAIRunner.
    init(bundleAt url: URL) async throws {
        let start = SuspendingClock.now
        let bundle = try LanguageBundle(at: url)
        let runner = CoreAIRunner(from: bundle)
        async let engineLoad = runner.makeInferenceEngine()
        async let tokenizerLoad = bundle.loadTokenizer()
        let (engine, tokenizer) = try await (engineLoad, tokenizerLoad)
        self.engine = engine
        self.tokenizer = tokenizer
        self.modelName = bundle.name
        self.outputProfile = OutputProfile.detect(probing: tokenizer)
        self.loadSeconds = ProcessStats.seconds(from: start, to: .now)
    }
}
