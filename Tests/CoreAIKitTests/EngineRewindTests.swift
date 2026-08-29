// EngineRewindTests.swift — the fallback that keeps hybrid multi-turn alive on 0.2.3-zoo.
//
// coreai-models 0.2.3-zoo throws `invalidState` when asked for a partial reset on a hybrid
// GDN/SSM engine (upstream #132); the fork tags before it degraded to a full reset internally.
// ChatSession and KitExecutor ask for exactly that partial reset on every second turn, so
// without the fallback the second turn of every hybrid chat fails — reproduced on
// Qwen3.5-0.8B before EngineRewind.swift landed. The fake below is the smallest engine that
// behaves like the fork: partial reset throws, full reset works, everything is recorded.

import CoreAILanguageModels
import Foundation
import Synchronization
import Testing

@testable import CoreAIKit

struct EngineRewindTests {
    @Test func rewindableEngineKeepsTheRequestedIndex() async throws {
        let engine = FakeEngine(rewindable: true)
        let kept = try await engine.rewind(to: 12)
        #expect(kept == 12)
        #expect(engine.resets == [12])
    }

    @Test func hybridEngineFallsBackToAFullReset() async throws {
        let engine = FakeEngine(rewindable: false)
        let kept = try await engine.rewind(to: 12)
        #expect(kept == 0)
        #expect(engine.resets == [12, 0])
    }

    @Test func zeroIsAlwaysAFullReset() async throws {
        let engine = FakeEngine(rewindable: false)
        let kept = try await engine.rewind(to: 0)
        #expect(kept == 0)
        #expect(engine.resets == [0])
    }

    /// Only the "cannot rewind" refusal is absorbed; any other engine error must surface.
    @Test func otherErrorsAreNotSwallowed() async throws {
        let engine = FakeEngine(
            rewindable: true, failWith: InferenceRuntimeError.invalidArgument("boom"))
        await #expect(throws: InferenceRuntimeError.self) {
            _ = try await engine.rewind(to: 12)
        }
        #expect(engine.resets == [12])
    }
}

// MARK: - Fake

private final class ResetLog: Sendable {
    let calls = Mutex<[Int]>([])
}

private struct FakeEngine: InferenceEngine {
    struct Sequence: InferenceOutputSequence {
        typealias Element = InferenceOutput
        typealias Failure = any Error
        var stopReason: StopReason? { nil }
        func setStopReason(_ reason: StopReason) {}
        func makeAsyncIterator() -> AsyncThrowingStream<InferenceOutput, any Error>.AsyncIterator {
            AsyncThrowingStream<InferenceOutput, any Error> { $0.finish() }.makeAsyncIterator()
        }
    }

    let rewindable: Bool
    let failWith: (any Error)?
    private let log = ResetLog()

    init(rewindable: Bool, failWith: (any Error)? = nil) {
        self.rewindable = rewindable
        self.failWith = failWith
    }

    var resets: [Int] { log.calls.withLock { $0 } }

    func generate(
        with input: [TokenId],
        samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> Sequence {
        Sequence()
    }

    var processedTokenCount: Int { 40 }

    func reset(to tokenIndex: Int) async throws {
        log.calls.withLock { $0.append(tokenIndex) }
        if let failWith { throw failWith }
        if tokenIndex > 0, !rewindable {
            throw InferenceRuntimeError.invalidState(
                "Partial reset is not supported for hybrid models with recurrent state.")
        }
    }

    func warmup(queryLength: Int, sampling: SamplingConfiguration?) async throws {}
    var isBusy: Bool { false }
    func cancel() async throws {}
    var supportsLogits: Bool { false }
    var lastPrefixHitCount: Int { 0 }
    var config: ModelConfig {
        ModelConfig(
            name: "fake", tokenizer: "fake", vocabSize: 1, maxContextLength: 64,
            serializedModel: [], function: "main")
    }
}
