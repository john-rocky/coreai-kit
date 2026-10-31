import Tokenizers
import XCTest

@testable import CoreAIKit

/// `enableThinking = false` across the chat templates of every catalog model that thinks: whether the
/// template reads the flag (Qwen3, LFM3, MiniCPM5, GLM-4.7, Nemotron-3-Nano) or not (LFM2.5 2.6B,
/// Nanbeige4.1 3B), the thinking-off prompt of one user turn holds exactly one closed think block and
/// does not end inside one. Opt-in, with a folder holding `<id>/tokenizer/` per model:
///
///     KIT_THINKING_TOKENIZERS=<dir> swift test --filter ChatSessionThinkingTemplates
@available(macOS 27, iOS 27, *)
final class ChatSessionThinkingTemplatesTests: XCTestCase {
    private static let question: [[String: any Sendable]] = [
        ["role": "user", "content": "What is 17 + 25?"]
    ]

    func testThinkingOffClosesOneThinkBlockOnEveryTemplate() async throws {
        guard let root = ProcessInfo.processInfo.environment["KIT_THINKING_TOKENIZERS"] else {
            throw XCTSkip("Set KIT_THINKING_TOKENIZERS to a folder of <id>/tokenizer/ directories.")
        }
        let ids = try FileManager.default.contentsOfDirectory(atPath: root).sorted().filter {
            FileManager.default.fileExists(atPath: "\(root)/\($0)/tokenizer/tokenizer.json")
        }
        XCTAssertFalse(ids.isEmpty, "no <id>/tokenizer/ under \(root)")
        for id in ids {
            let tokenizer = try await AutoTokenizer.from(
                modelFolder: URL(fileURLWithPath: "\(root)/\(id)/tokenizer"))
            guard let close = tokenizer.convertTokenToId("</think>") else {
                XCTFail("\(id): no </think> token"); continue
            }
            let profile = OutputProfile.detect(probing: tokenizer)
            let raw = try tokenizer.applyChatTemplate(
                messages: Self.question, chatTemplate: nil, addGenerationPrompt: true,
                truncation: false, maxLength: nil, tools: nil,
                additionalContext: ["enable_thinking": false]
            ).map(Int32.init)
            let plain = try tokenizer.applyChatTemplate(messages: Self.question).map(Int32.init)
            let off = try ChatSession.renderChatTemplate(
                Self.question, tokenizer: tokenizer, enableThinking: false)
            let readsFlag = raw != plain
            print("thinking-off \(id): reads_flag=\(readsFlag) ids=\(off.count)",
                  "tail=\(tokenizer.decode(tokens: off.suffix(8).map(Int.init)).debugDescription)")
            XCTAssertEqual(off.filter { $0 == Int32(close) }.count, 1, "\(id): closed think blocks")
            XCTAssertNil(
                profile.openThinkingRule(afterPrompt: off, tokenizer: tokenizer),
                "\(id): the thinking-off prompt ends inside a think block")
            if readsFlag { XCTAssertEqual(off, raw, "\(id): the template's own thinking-off form") }
        }
    }
}
