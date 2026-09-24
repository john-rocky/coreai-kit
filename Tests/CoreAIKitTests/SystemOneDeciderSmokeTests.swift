// SystemOneDeciderSmokeTests.swift — the op against the decider's own API: the 13 requests of
// the zoo's `models/decider-0.8b/fixtures-decider-0.8b.json`, each sent through
// `CoreAI.systemOne(json:)` on `decider-0.8b` as a client would send it, and every answer
// compared with the author's `system_one` output recorded beside it (`api_assembly`: the
// author's Python over the fp32 model). Choice: the same option, every option's probability
// within the card's bar; noul: P(yes) within it; score: the expected level, the distribution,
// each level's fit and the fit mass within it. The author's `usage` counts unique prefix
// tokens and no output; the kit's counts every prompt and one slot per decision, so `usage` is
// not compared. Opt-in (loads the model, runs the GPU; about a minute on an M4 Max):
//
//     KIT_DECIDER_FIXTURE=/path/to/fixtures-decider-0.8b.json swift test --filter SystemOneDeciderSmoke

import Foundation
import XCTest

@testable import CoreAIKit
@testable import CoreAIOps

@available(macOS 27, iOS 27, *)
final class SystemOneDeciderSmokeTests: XCTestCase {
    /// The card's ship bar for |Δp| against the author's fp32 readout (int8hu measured 0.0084).
    static let bar = 0.02

    func testTheOpAssemblesTheAuthorsAnswers() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_DECIDER_FIXTURE"] else {
            throw XCTSkip("Set KIT_DECIDER_FIXTURE to the zoo's fixtures-decider-0.8b.json to run.")
        }
        let fixture = try JSONValue.parse(Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(fixture["schema"]?.stringValue, "coreai-decider-fixtures/1")
        let requests = try XCTUnwrap(fixture["requests"]?.elements)
        let assembly = try XCTUnwrap(fixture["api_assembly"]?.elements)
        XCTAssertEqual(requests.count, assembly.count)

        var maxDelta = 0.0
        var worst = ""
        var answers = 0
        let started = SuspendingClock.now
        for (request, expected) in zip(requests, assembly) {
            let requestID = try XCTUnwrap(request["id"]?.stringValue)
            XCTAssertEqual(expected["request_id"]?.stringValue, requestID)
            // The request as a client sends it: the fixture's state (text or structured) and
            // its questions in the wire form, nothing else.
            let body = JSONValue.object([
                .init("state", try XCTUnwrap(request["state"])),
                .init("questions", try XCTUnwrap(request["questions"])),
            ])
            let response = try await CoreAI.systemOne(json: Data(body.dumps().utf8), options: .model("decider-0.8b"))
            let reference = try XCTUnwrap(expected["system_one"]?["answers"])
            XCTAssertEqual(response.ids, reference.members?.map(\.key), "\(requestID): answer keys, in request order")

            for answer in response.answers {
                let ref = try XCTUnwrap(reference[answer.id], "\(requestID)/\(answer.id)")
                let tag = "\(requestID)/\(answer.id)"
                func compare(_ kit: Double, _ author: JSONValue?, _ what: String) throws {
                    let value = try XCTUnwrap(author?.doubleValue, "\(tag) \(what)")
                    let delta = abs(kit - value)
                    if delta > maxDelta { maxDelta = delta; worst = "\(tag) \(what)" }
                    XCTAssertLessThanOrEqual(delta, Self.bar, "\(tag) \(what): kit \(kit) vs author \(value)")
                }
                switch answer.answer.value {
                case .choice(let choice):
                    XCTAssertEqual(ref["type"]?.stringValue, "choice", tag)
                    XCTAssertEqual(choice.id, ref["choice"]?.stringValue, "\(tag) argmax")
                    for option in choice.options {
                        try compare(choice.probabilities[option] ?? 0, ref["probabilities"]?[option], "p(\(option))")
                    }
                case .noul(let p):
                    XCTAssertEqual(ref["type"]?.stringValue, "noul", tag)
                    try compare(p, ref["noul"], "noul")
                case .score(let score):
                    XCTAssertEqual(ref["type"]?.stringValue, "score", tag)
                    // The expected level moves by at most Σ level × |Δp|; a per-level bar of 0.02
                    // over five levels bounds it by 0.2, and the author rounds to two places.
                    let expectedScore = try XCTUnwrap(ref["score"]?.doubleValue, "\(tag) score")
                    XCTAssertLessThanOrEqual(abs(score.value - expectedScore), 0.1, "\(tag) score: kit \(score.value) vs author \(expectedScore)")
                    for (level, p) in score.probabilities.enumerated() {
                        try compare(p, ref["probabilities"]?[String(level)], "p(level \(level))")
                    }
                    let fit = try XCTUnwrap(score.fit, "\(tag): the decider judges each level alone")
                    for (level, f) in fit.enumerated() {
                        try compare(f, ref["level_fit"]?[String(level)], "fit(level \(level))")
                    }
                    try compare(fit.reduce(0, +), ref["fit_mass"], "fit_mass")
                    if case .score(let levels) = answer.question.kind {
                        XCTAssertEqual(
                            answer.value["legend"], .object(levels.enumerated().map { .init(String($0.offset), .string($0.element)) }),
                            "\(tag) legend")
                    }
                }
                answers += 1
            }
            XCTAssertEqual(response.usage.outputTokens, response.answers.count, requestID)
        }
        let elapsed = SuspendingClock.now - started
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        print("SMOKE decider api_assembly: \(requests.count) requests, \(answers) answers, max |Δp| \(maxDelta) at \(worst) (bar \(Self.bar)), \(String(format: "%.1f", seconds)) s")
        XCTAssertEqual(answers, 36)
    }
}
