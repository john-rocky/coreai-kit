import XCTest

@testable import CoreAIKit

/// End-to-end smoke over a real local bundle. Opt-in (loads a model, runs the GPU):
///
///     KIT_SMOKE_BUNDLE=/path/to/bundle swift test --filter ChatSessionSmoke
final class ChatSessionSmokeTests: XCTestCase {
    func testTwoTurnChat() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SMOKE_BUNDLE"] else {
            throw XCTSkip("Set KIT_SMOKE_BUNDLE to a local bundle directory to run.")
        }

        var config = ChatSession.Configuration()
        config.temperature = nil  // greedy: deterministic smoke
        config.maxResponseTokens = 512
        let session = try await ChatSession(
            bundleAt: URL(fileURLWithPath: path), configuration: config)
        try await session.prewarm()

        var sawThinking = false
        var answer = ""
        for try await event in await session.streamResponse(to: "In one short sentence: what is the capital of Japan?") {
            switch event {
            case .response(let delta): answer += delta
            case .thinking: sawThinking = true
            case .stats, .complete: break
            }
        }
        let stats = await session.stats
        print("SMOKE load=\(stats.loadSeconds ?? -1)s ttft=\(stats.ttftSeconds ?? -1)s "
            + "tok/s=\(stats.tokensPerSecond ?? -1) generated=\(stats.generatedTokens) "
            + "thinking=\(sawThinking)")
        print("SMOKE answer: \(answer)")
        XCTAssertFalse(answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        XCTAssertGreaterThan(stats.generatedTokens, 0)

        let answer2 = try await session.respond(to: "Repeat your previous answer.")
        print("SMOKE turn2: \(answer2)")
        XCTAssertFalse(answer2.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        let history = await session.history
        XCTAssertEqual(history.count, 4)
        XCTAssertEqual(history[3].role, .assistant)
    }
}
