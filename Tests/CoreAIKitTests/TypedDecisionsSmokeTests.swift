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

    /// Concurrent calls on one instance take turns on its engine: each reads what it would
    /// have read alone. Two alternating prompts make every call rewind the other's cache.
    func testConcurrentCallsOnOneInstanceMatchSequentialCalls() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SMOKE_BUNDLE"] else {
            throw XCTSkip("Set KIT_SMOKE_BUNDLE to a local bundle directory to run.")
        }
        let decider = try await TypedDecisions(bundleAt: URL(fileURLWithPath: path))
        let states = ["Water is wet.", "Fire is hot."]
        let question = Decision.Question.choice(
            "What is the claim about?", ["water", "fire", "air"])
        let prompts = try states.map { try decider.promptTokens($0, question) }
        // Cached and uncached runs of the fp16 engine differ by up to ~4e-3.
        let accuracy = 5e-3
        let calls = 6

        func readout(_ logits: Decision.Logits, _ slots: [Int32]) -> [Double] {
            DecisionPrompt.probabilities(
                logits: slots.map { Double(logits.values[Int($0)]) }, temperature: 1)
        }

        var logitsReference: [[Double]] = []
        var decideReference: [[Double]] = []
        for (state, prompt) in zip(states, prompts) {
            try await decider.reset()
            let logits = try await decider.logits(for: prompt.tokens)
            logitsReference.append(readout(logits, prompt.slots))
            try await decider.reset()
            decideReference.append(try await decider.decide(state, question).probabilities)
        }
        // The two prompts must read apart, or a call answered from the other's cache would pass.
        XCTAssertGreaterThan(
            zip(decideReference[0], decideReference[1]).map { abs($0 - $1) }.max()!, 10 * accuracy)

        try await decider.reset()
        let scored = try await withThrowingTaskGroup(of: (Int, Decision.Logits).self) { group in
            for index in 0..<calls {
                group.addTask { (index, try await decider.logits(for: prompts[index % 2].tokens)) }
            }
            var scored: [Int: Decision.Logits] = [:]
            for try await (index, logits) in group { scored[index] = logits }
            return scored
        }
        for index in 0..<calls {
            let probabilities = readout(scored[index]!, prompts[index % 2].slots)
            for (p, reference) in zip(probabilities, logitsReference[index % 2]) {
                XCTAssertEqual(p, reference, accuracy: accuracy, "logits(for:) call \(index)")
            }
        }

        try await decider.reset()
        let answers = try await withThrowingTaskGroup(of: (Int, Decision.Answer).self) { group in
            for index in 0..<calls {
                group.addTask { (index, try await decider.decide(states[index % 2], question)) }
            }
            var answers: [Int: Decision.Answer] = [:]
            for try await (index, answer) in group { answers[index] = answer }
            return answers
        }
        for index in 0..<calls {
            let probabilities = answers[index]!.probabilities
            XCTAssertEqual(probabilities.count, decideReference[index % 2].count)
            for (p, reference) in zip(probabilities, decideReference[index % 2]) {
                XCTAssertEqual(p, reference, accuracy: accuracy, "decide call \(index)")
            }
        }
    }
}
