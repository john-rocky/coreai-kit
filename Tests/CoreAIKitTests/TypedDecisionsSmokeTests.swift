// TypedDecisionsSmokeTests.swift — the token-level scoring API over a real local bundle: the
// caller's own prompt through the bundle's tokenizer, the whole vocabulary back, the same
// numbers `decide` reads, and the prefix reuse accounted for. Opt-in (loads a model, runs the
// GPU):
//
//     KIT_SMOKE_BUNDLE=/path/to/bundle swift test --filter TypedDecisionsSmoke

import Tokenizers
import XCTest

@testable import CoreAIKit

final class TypedDecisionsSmokeTests: XCTestCase {
    func testLogitsAgreeWithDecideAndAccountForThePrefix() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SMOKE_BUNDLE"] else {
            throw XCTSkip("Set KIT_SMOKE_BUNDLE to a local bundle directory to run.")
        }
        let decider = try await TypedDecisions(bundleAt: URL(fileURLWithPath: path))
        let state = "Customer: my order arrived with the box crushed and the screen cracked."
        let question = Decision.Question.noul("Does the customer want a replacement?")

        // The public path and `decide` read the same logits: both from a cold cache, the
        // letter softmax over `logits(for:)` is `decide`'s answer.
        let (tokens, slots) = try decider.promptTokens(state, question)
        let scored = try await decider.logits(for: tokens)
        XCTAssertGreaterThan(scored.values.count, Int(slots.max()!))
        XCTAssertEqual(scored.timing.promptTokens, tokens.count)
        XCTAssertEqual(scored.timing.reusedTokens, 0)
        try await decider.reset()
        let answer = try await decider.decide(state, question)
        let readout = DecisionPrompt.probabilities(
            logits: slots.map { Double(scored.values[Int($0)]) }, temperature: 1)
        XCTAssertEqual(readout.count, answer.probabilities.count)
        for (own, kit) in zip(readout, answer.probabilities) {
            XCTAssertEqual(own, kit, accuracy: 1e-6)
        }
        print("SMOKE vocab=\(scored.values.count) prompt=\(tokens.count) "
            + "P(yes)=\(readout[1]) decide=\(answer.noul ?? -1) "
            + "\(scored.timing.milliseconds) ms")

        // A prefilled prefix is kept on an engine that can rewind (reported in full) and
        // re-prefilled on one that cannot (reported as 0); the readout does not move.
        try await decider.reset()
        let prefix = try DecisionPrompt.statePrefix(state: state, tokenizer: decider.tokenizer)
        let prefilled = try await decider.prefill(tokens: prefix)
        XCTAssertEqual(prefilled.promptTokens, prefix.count)
        let reused = try await decider.logits(for: tokens)
        XCTAssertTrue(
            [0, prefix.count].contains(reused.timing.reusedTokens),
            "reused \(reused.timing.reusedTokens) of a \(prefix.count)-token prefix")
        let reusedReadout = DecisionPrompt.probabilities(
            logits: slots.map { Double(reused.values[Int($0)]) }, temperature: 1)
        for (own, cold) in zip(reusedReadout, readout) {
            XCTAssertEqual(own, cold, accuracy: 1e-3)
        }
        print("SMOKE prefix=\(prefix.count) reused=\(reused.timing.reusedTokens) "
            + "P(yes)=\(reusedReadout[1]) \(reused.timing.milliseconds) ms")

        // A prompt the caller renders with the bundle's tokenizer scores the same way, and
        // the label variants a different readout sums are single tokens it can look up.
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "Answer yes or no: is the sea salty?"]
        ]
        let own = try decider.tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: true,
            truncation: false, maxLength: nil, tools: nil,
            additionalContext: ["enable_thinking": false])
        let ownScored = try await decider.logits(for: own.map(Int32.init))
        XCTAssertEqual(ownScored.values.count, scored.values.count)
        XCTAssertEqual(ownScored.timing.promptTokens, own.count)
        let yes = decider.tokenizer.encode(text: "yes", addSpecialTokens: false)
        let no = decider.tokenizer.encode(text: "no", addSpecialTokens: false)
        XCTAssertEqual(yes.count, 1)
        XCTAssertEqual(no.count, 1)
        let top = ownScored.values.indices.max { ownScored.values[$0] < ownScored.values[$1] }!
        print("SMOKE own prompt=\(own.count) reused=\(ownScored.timing.reusedTokens) "
            + "top=\(decider.tokenizer.decode(tokens: [top], skipSpecialTokens: false).debugDescription) "
            + "yes=\(ownScored.values[yes[0]]) no=\(ownScored.values[no[0]])")

        // Nothing to score is an error, not a crash.
        do {
            _ = try await decider.logits(for: [])
            XCTFail("expected DecisionError.emptyPrompt")
        } catch DecisionError.emptyPrompt {}
    }
}
