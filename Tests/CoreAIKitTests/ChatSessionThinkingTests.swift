import Tokenizers
import XCTest

@testable import CoreAIKit

/// `ChatSession.Configuration.enableThinking` against LFM3-1.7B's tokenizer and chat template. The
/// goldens are transformers 5.20.2 `apply_chat_template` of LiquidAI/LFM3-1.7B with
/// `enable_thinking=True` / `False` (one user turn, the generation prompt, the BOS once). Opt-in:
///
///     KIT_LFM3_TOKENIZER=<bundle>/tokenizer swift test --filter ChatSessionThinking
@available(macOS 27, iOS 27, *)
final class ChatSessionThinkingTests: XCTestCase {
    private static let question: [[String: any Sendable]] = [
        ["role": "user", "content": "What is 17 + 25? Answer briefly."]
    ]
    /// `…<|im_start|>assistant\n<think>`
    private static let thinkingOn: [Int32] = [
        124894, 124899, 5922, 207, 2992, 355, 229, 916, 1510, 229, 1197, 39, 41774, 21422, 22,
        124900, 207, 124899, 63514, 207, 124901,
    ]
    /// `…<|im_start|>assistant\n<think></think>`
    private static let thinkingOff: [Int32] = thinkingOn + [124902]

    private func tokenizer() async throws -> any Tokenizer {
        guard let path = ProcessInfo.processInfo.environment["KIT_LFM3_TOKENIZER"] else {
            throw XCTSkip("Set KIT_LFM3_TOKENIZER to an LFM3-1.7B bundle's tokenizer/ directory.")
        }
        return try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: path))
    }

    /// nil renders the template's default, which for LFM3 is thinking on, the same as `true`.
    func testDefaultAndOnRenderTheOpenThinkBlock() async throws {
        let tokenizer = try await tokenizer()
        XCTAssertEqual(
            try ChatSession.renderChatTemplate(Self.question, tokenizer: tokenizer, enableThinking: nil),
            Self.thinkingOn)
        XCTAssertEqual(
            try ChatSession.renderChatTemplate(Self.question, tokenizer: tokenizer, enableThinking: true),
            Self.thinkingOn)
    }

    /// `false` renders the template's closed block once: LFM3's template writes `<think></think>`
    /// itself, and nothing may add a second one.
    func testOffRendersOneClosedThinkBlock() async throws {
        let tokenizer = try await tokenizer()
        let ids = try ChatSession.renderChatTemplate(
            Self.question, tokenizer: tokenizer, enableThinking: false)
        XCTAssertEqual(ids, Self.thinkingOff)
        XCTAssertEqual(ids.filter { $0 == 124902 }.count, 1)
    }
}
