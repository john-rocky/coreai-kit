// FoundationModelDecisionsTests.swift — the system-model backend without the system model:
// the prompt each question shape renders, the schema it generates under, the one-hot answer
// a generated value becomes, and the `metadata` a response from this backend carries through
// `SystemOneServer`. The live backend runs only when KIT_FM_SMOKE is set (Apple Intelligence on):
//
//     KIT_FM_SMOKE=1 swift test --filter FoundationModelDecisionsSmoke

import Foundation
import FoundationModels
import Testing

@testable import CoreAIKit

struct FoundationModelPromptTests {
    static let timing = Decision.Timing(promptTokens: 205, reusedTokens: 0, seconds: 0.31)

    @Test func theInstructionsHoldTheRuleThenTheState() {
        let text = FoundationModelPrompt.instructions(state: "Help! My payouts have been failing for 3 days.")
        #expect(text.hasPrefix("You answer typed questions about a state."))
        #expect(text.hasSuffix("\n\nState:\nHelp! My payouts have been failing for 3 days."))
    }

    @Test func aChoiceListsItsOptionsAsTheModelReadsThem() {
        let question = Decision.Question.choice(
            "Which queue?",
            options: [.init(id: "billing", description: "billing: invoices, payments, refunds"),
                      .init(id: "technical", description: "bugs, outages"),
                      .init("other")])
        #expect(FoundationModelPrompt.prompt(question) == """
            Question: Which queue?
            Options:
            - billing: invoices, payments, refunds
            - technical: bugs, outages
            - other
            Answer with one option name exactly as listed.
            """)
    }

    @Test func aScoreListsItsLevelsLowestFirst() {
        let question = Decision.Question.score("How urgent?", levels: ["can wait", "this week", "today"])
        #expect(FoundationModelPrompt.prompt(question) == """
            Question: How urgent?
            Levels, lowest first:
            0: can wait
            1: this week
            2: today
            Answer with the number of the level that fits.
            """)
    }

    @Test func aNoulNamesWhatEachSideMeansWhenGiven() {
        #expect(FoundationModelPrompt.prompt(.noul("Is it permitted?")) == """
            Question: Is it permitted?
            Answer true if it holds, false if it does not.
            """)
        #expect(FoundationModelPrompt.prompt(.noul("Is it permitted?", yes: "every condition holds", no: "a condition is missing")) == """
            Question: Is it permitted?
            true: every condition holds
            false: a condition is missing
            Answer true if it holds, false if it does not.
            """)
    }

    @Test func theSchemaIsTheAnswerShape() throws {
        // A schema builds for every shape; what it constrains the generation to is checked live.
        _ = try FoundationModelPrompt.schema(.choice("Which?", ["a", "b"]))
        _ = try FoundationModelPrompt.schema(.score("How?", levels: ["low", "high"]))
        _ = try FoundationModelPrompt.schema(.noul("Is it?"))
    }

    @Test func aGeneratedChoiceIsOneHot() throws {
        let question = Decision.Question.choice("Which queue?", ["billing", "technical", "other"])
        let answer = try FoundationModelPrompt.answer(question, generated: GeneratedContent("technical"), timing: Self.timing)
        #expect(answer.choice == "technical")
        #expect(answer.probabilities == [0, 1, 0])
        #expect(answer.confidence == 1)
        guard case .choice(let choice) = answer.value else { Issue.record("not a choice"); return }
        #expect(choice.certainty == 1)
        #expect(choice.ranking == ["technical", "billing", "other"])
        #expect(choice.options == ["billing", "technical", "other"])
        #expect(answer.timing == Self.timing)
        #expect(answer.abstain == nil)
        // A value outside the list is refused rather than reported.
        #expect(throws: DecisionError.refused(reason: "the model answered 'sales', which is not a listed option")) {
            try FoundationModelPrompt.answer(question, generated: GeneratedContent("sales"), timing: Self.timing)
        }
    }

    @Test func aGeneratedLevelIsTheScore() throws {
        let question = Decision.Question.score("How urgent?", levels: ["can wait", "this week", "today"])
        let answer = try FoundationModelPrompt.answer(question, generated: GeneratedContent(2), timing: Self.timing)
        #expect(answer.score == 2)
        #expect(answer.probabilities == [0, 0, 1])
        guard case .score(let score) = answer.value else { Issue.record("not a score"); return }
        #expect(score.level == 2 && score.confidence == 1 && score.certainty == 1 && score.fit == nil)
        #expect(throws: DecisionError.self) {
            try FoundationModelPrompt.answer(question, generated: GeneratedContent(3), timing: Self.timing)
        }
    }

    @Test func aGeneratedBoolIsTheNoul() throws {
        let question = Decision.Question.noul("Urgent?")
        #expect(try FoundationModelPrompt.answer(question, generated: GeneratedContent(true), timing: Self.timing).noul == 1)
        #expect(try FoundationModelPrompt.answer(question, generated: GeneratedContent(false), timing: Self.timing).noul == 0)
        #expect(try FoundationModelPrompt.answer(question, generated: GeneratedContent(false), timing: Self.timing).probabilities == [1, 0])
    }

    @Test func theUnavailableReasonsAreNamed() {
        #expect(FoundationModelDecisions.describe(.deviceNotEligible).contains("not eligible"))
        #expect(FoundationModelDecisions.describe(.appleIntelligenceNotEnabled).contains("not enabled"))
        #expect(FoundationModelDecisions.describe(.modelNotReady).contains("not ready"))
    }
}

/// A backend that answers what it is told to, for the server's path.
private actor StubBackend: DecisionBackend {
    nonisolated let id = "stub"
    nonisolated let maxOptions = 4
    var modelName: String { "a stub" }
    let metadata: JSONValue?
    let refuse: String?

    init(metadata: JSONValue?, refuse: String? = nil) {
        self.metadata = metadata
        self.refuse = refuse
    }

    func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        if let refuse { throw DecisionError.refused(reason: refuse) }
        let chosen: String
        switch question.kind {
        case .choice(let options): chosen = options[0].id
        case .score: chosen = "0"
        case .noul: chosen = "yes"
        }
        return FoundationModelPrompt.answer(question, chosen: chosen, timing: .init(promptTokens: 10, reusedTokens: 0, seconds: 0.01))
    }

    func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        var answers: [SystemOne.Answer] = []
        for (key, question) in request.questions {
            answers.append(.init(id: key, question: question, answer: try await decide(request.state, question)))
        }
        return SystemOne.Response(
            model: id, answers: answers, stateTokens: 0,
            prefill: .init(promptTokens: 0, reusedTokens: 0, seconds: 0), metadata: metadata)
    }
}

struct DecisionBackendServerTests {
    static let metadata = JSONValue.object([
        .init("backend", .string("foundation-models")),
        .init("probabilities", .string("one-hot")),
        .init("calibration", .string("none")),
    ])

    static func post(_ body: String) -> HTTPRequest {
        HTTPRequest(method: "POST", path: SystemOne.path, headers: [:], body: Data(body.utf8))
    }

    @Test func aResponseCarriesTheBackendsMetadataLast() async throws {
        let server = SystemOneServer(modelID: "apple-foundation-model", backend: StubBackend(metadata: Self.metadata)) { _ in }
        #expect(server.decider == nil)
        let response = await server.route(Self.post(
            "{\"state\": \"Help!\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"Which?\", \"criteria\": {\"a\": \"one\", \"b\": \"two\"}}}}"))
        #expect(response.status == 200)
        let text = String(decoding: response.body, as: UTF8.self)
        #expect(text.hasPrefix("{\"model\": \"apple-foundation-model\", \"answers\": {\"q\": {\"type\": \"choice\", \"choice\": \"a\", "
            + "\"probabilities\": {\"a\": 1.0, \"b\": 0.0}, \"confidence\": 1.0}}, "))
        #expect(text.hasSuffix("\"metadata\": {\"backend\": \"foundation-models\", \"probabilities\": \"one-hot\", \"calibration\": \"none\"}}"))
    }

    @Test func aTypedDecisionsResponseCarriesNoMetadata() {
        // The wire object of every other backend is unchanged: no `metadata` member.
        let question = Decision.Question.noul("Urgent?")
        let answer = FoundationModelPrompt.answer(question, chosen: "yes", timing: .init(promptTokens: 10, reusedTokens: 0, seconds: 0.01))
        let plain = SystemOne.response(model: "m", answers: [("q", question, answer)])
        #expect(plain["metadata"] == nil)
        #expect(plain.dumps().hasSuffix("\"timing_ms\": 10.0}"))
        let response = SystemOne.Response(model: "m", answers: [.init(id: "q", question: question, answer: answer)],
                                          stateTokens: 5, prefill: .init(promptTokens: 5, reusedTokens: 0, seconds: 0))
        #expect(response.metadata == nil)
        #expect(response.value == plain)
    }

    @Test func aRefusalIsA422() async throws {
        let server = SystemOneServer(modelID: "stub", backend: StubBackend(metadata: nil, refuse: "a guardrail refused it")) { _ in }
        let response = await server.route(Self.post(
            "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"Is it?\"}}}"))
        #expect(response.status == 422)
        #expect(String(decoding: response.body, as: UTF8.self).contains("The model refused this request: a guardrail refused it"))
    }

    @Test func theBackendsOptionCountBoundsAChoice() async throws {
        let server = SystemOneServer(modelID: "stub", backend: StubBackend(metadata: nil)) { _ in }
        let response = await server.route(Self.post(
            "{\"state\": \"x\", \"questions\": {\"q\": {\"type\": \"choice\", \"instructions\": \"Which?\", \"criteria\": [\"a\", \"b\", \"c\", \"d\", \"e\"]}}}"))
        #expect(response.status == 422)
        #expect(String(decoding: response.body, as: UTF8.self).contains("at most 4 options, got 5"))
    }
}

struct FoundationModelDecisionsSmokeTests {
    static var enabled: Bool { ProcessInfo.processInfo.environment["KIT_FM_SMOKE"] != nil }

    /// Three shapes on one state, shared then fresh: every answer is a listed value with all
    /// of the probability on it, and the response names the backend.
    @Test(.enabled(if: enabled)) func theSystemModelAnswersEveryShape() async throws {
        for share in [true, false] {
            var configuration = FoundationModelDecisions.Configuration()
            configuration.shareSession = share
            let decider = try FoundationModelDecisions(configuration: configuration)
            let request = SystemOne.Request(
                state: "Do not cancel my membership. Please return the duplicate charge.",
                questions: [
                    "intent": .choice("Select the primary requested action.",
                                      options: [.init(id: "cancel", description: "cancel: end the subscription"),
                                                .init(id: "refund", description: "refund: return money already charged"),
                                                .init("other")]),
                    "urgent": .score("How urgent?", levels: ["can wait", "this week", "today"]),
                    "reply": .noul("Does the customer expect a reply?"),
                ])
            let response = try await decider.systemOne(request)
            #expect(response.ids == ["intent", "urgent", "reply"])
            #expect(response["intent"]?.choice == "refund")
            #expect(response["intent"]?.confidence == 1)
            #expect((0...2).contains(Int(response["urgent"]?.score ?? -1)))
            #expect([0, 1].contains(response["reply"]?.noul))
            #expect(response.metadata?["calibration"]?.stringValue == "none")
            #expect(response.metadata?["session"]?.stringValue == (share ? "shared" : "fresh"))
            for answer in response.answers {
                #expect(answer.answer.timing.promptTokens > 0)
                #expect(answer.answer.probabilities.reduce(0, +) == 1)
            }
        }
    }
}
