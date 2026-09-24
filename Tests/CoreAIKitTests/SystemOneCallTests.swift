// SystemOneCallTests.swift — the whole-request call without weights: a request built in Swift
// keeps its order and serializes a structured state the reference way, a duplicate question id
// is refused, the typed response writes the same wire object the servers always wrote, and the
// op resolves its model in the documented order.

import Foundation
import Testing

@testable import CoreAIKit
@testable import CoreAIOps

struct SystemOneRequestTests {
    @Test func aRequestBuiltInSwiftKeepsItsOrder() {
        let request = SystemOne.Request(
            state: "Help! My payouts have been failing for 3 days.",
            questions: [
                "is_urgent": .noul("Does this convey urgency?"),
                "queue": .choice("Which queue?", ["billing", "technical", "other"]),
                "anger": .score("How upset is the customer?", levels: ["calm", "frustrated", "very angry"]),
            ])
        #expect(request.questions.map(\.id) == ["is_urgent", "queue", "anger"])
        #expect(request.model == nil)
        #expect(request.structuredState == false)
        #expect(request.questions[1].question.kind == .choice([.init("billing"), .init("technical"), .init("other")]))

        let named = SystemOne.Request(
            state: "s", model: "decider-0.8b", questions: [(id: "q", question: .noul("Refund?"))])
        #expect(named.model == "decider-0.8b")
        #expect(named.questions.map(\.id) == ["q"])
    }

    @Test func aStructuredStateSerializesTheReferenceWay() throws {
        let state = try JSONValue.parse("{\"subject\":\"Duplicate charge\",\"body\":\"Refund the duplicate.\",\"amount\":12.5}")
        let request = try SystemOne.Request(state: state, questions: [(id: "q", question: .noul("Refund?"))])
        #expect(request.structuredState == true)
        #expect(request.state == "{\"subject\": \"Duplicate charge\", \"body\": \"Refund the duplicate.\", \"amount\": 12.5}")
        // The same bytes the wire produces from the JSON form of that request.
        let wire = try SystemOne.request(from: Data(
            "{\"state\": {\"subject\":\"Duplicate charge\",\"body\":\"Refund the duplicate.\",\"amount\":12.5}, \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"Refund?\"}}}".utf8))
        #expect(wire.state == request.state)

        let plain = try SystemOne.Request(state: .string("just text"), questions: [(id: "q", question: .noul("Refund?"))])
        #expect(plain.state == "just text")
        #expect(plain.structuredState == false)
        #expect(throws: SystemOne.WireError.self) {
            try SystemOne.Request(state: .null, questions: [(id: "q", question: .noul("Refund?"))])
        }
    }

    /// A choice that came through the wire carries `key: description` as each option's
    /// description; the decider form writes that text once, as the author's builder does, and
    /// a choice built in Swift from separate fields composes to the same text. Found by the
    /// decider fixture's one described choice, where the doubled name moved P(repair) by 0.03.
    @Test func aWireChoiceRendersOnceInTheDeciderForm() throws {
        let wire = try SystemOne.request(from: Data("""
            {"state": "s", "questions": {"route": {"type": "choice", "instructions": "Which team should handle the request?",
             "criteria": {"repair": "Replace damaged parts", "billing": "Handle payments", "delivery": null}}}}
            """.utf8))
        let fromWire = DeciderPrompt.rows(for: wire.questions[0].question)
        let fromSwift = DeciderPrompt.rows(for: .choice(
            "Which team should handle the request?",
            options: [.init(id: "repair", description: "Replace damaged parts"), .init(id: "billing", description: "Handle payments"), .init("delivery")]))
        #expect(fromWire == fromSwift)
        #expect(fromWire[0].options == ["repair: Replace damaged parts", "billing: Handle payments", "delivery"])
        #expect(DeciderPrompt.optionText(.init(id: "a", description: "")) == "a")
    }

    @Test func aDuplicateQuestionIdIsRefused() {
        let body = "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"a\"}, \"q\": {\"type\": \"noul\", \"instructions\": \"b\"}}}"
        #expect(throws: SystemOne.WireError("duplicate question id 'q'")) {
            try SystemOne.request(from: Data(body.utf8))
        }
        #expect(throws: SystemOne.WireError("duplicate question id 'b'")) {
            try SystemOne.validateIDs(["a", "b", "c", "b"])
        }
        #expect(throws: Never.self) { try SystemOne.validateIDs(["a", "b", "c"]) }
    }
}

struct SystemOneResponseTests {
    static let timing = Decision.Timing(promptTokens: 120, reusedTokens: 100, seconds: 0.0421)
    static let prefill = Decision.Timing(promptTokens: 100, reusedTokens: 0, seconds: 0.2)
    static let choiceQuestion = Decision.Question.choice("Which queue?", ["billing", "technical", "other"])
    static let scoreQuestion = Decision.Question.score("How upset?", levels: ["calm", "frustrated", "very angry"])
    static let noulQuestion = Decision.Question.noul("Urgent?")

    static func response() -> SystemOne.Response {
        SystemOne.Response(
            model: "minicpm5-2b",
            answers: [
                .init(id: "queue", question: choiceQuestion,
                      answer: DecisionPrompt.answer(for: choiceQuestion, probabilities: [0.88123, 0.11877, 0.0], timing: timing)),
                .init(id: "anger", question: scoreQuestion,
                      answer: DecisionPrompt.answer(for: scoreQuestion, probabilities: [0.0, 0.95, 0.05], timing: timing)),
                .init(id: "is_urgent", question: noulQuestion,
                      answer: DecisionPrompt.answer(for: noulQuestion, probabilities: [0.05, 0.95], timing: timing)),
            ],
            stateTokens: 100, prefill: prefill)
    }

    /// The typed response and the servers' wire object are one thing: the same bytes the
    /// `SystemOne.response(model:answers:)` test pins, from the same answers.
    @Test func theResponseWritesTheWireObject() {
        let response = Self.response()
        let expected = SystemOne.response(
            model: "minicpm5-2b",
            answers: response.answers.map { ($0.id, $0.question, $0.answer) })
        #expect(response.value == expected)
        let text = response.dumps()
        #expect(text.hasPrefix("{\"model\": \"minicpm5-2b\", \"answers\": {\"queue\": {\"type\": \"choice\", \"choice\": \"billing\", "))
        #expect(text.hasSuffix("\"usage\": {\"input_tokens\": 360, \"output_tokens\": 3}, \"timing_ms\": 126.3}"))
        #expect(response.answers[0].value == expected["answers"]?["queue"])
    }

    @Test func theTypedAnswersAreReadByTheRequestsKeys() {
        let response = Self.response()
        #expect(response.ids == ["queue", "anger", "is_urgent"])
        #expect(response["queue"]?.choice == "billing")
        #expect(response["anger"]?.score == 1.05)
        #expect(response["is_urgent"]?.noul == 0.95)
        #expect(response["missing"] == nil)
        #expect(response.answers[0].choice == "billing")
        #expect(response.answers[1].score == 1.05)
        #expect(response.answers[2].noul == 0.95)
        #expect(response.answers[2].choice == nil)
    }

    @Test func usageAndTimingAreTheServersFigures() {
        let response = Self.response()
        #expect(response.usage == SystemOne.Usage(inputTokens: 360, outputTokens: 3))
        #expect(response.stateTokens == 100)
        // The whole request: the prefill's 200 ms plus three decisions of 42.1 ms.
        #expect(abs(response.milliseconds - (200 + 3 * 42.1)) < 1e-9)
        // The wire's timing_ms stays the decisions alone, as the servers have always reported it.
        #expect(response.value["timing_ms"]?.doubleValue == 126.3)
    }
}

struct SystemOneOpTests {
    /// The op's option first, then the request's own `model`, then the decision default — and
    /// the default is `CoreAI.decide`'s, the same cache and residency namespace.
    @Test func theModelResolvesInOrder() {
        #expect(CoreAI.systemOneModel(options: OpOptions(), requested: nil) == CoreAI.defaultDecisionModel)
        #expect(CoreAI.systemOneModel(options: OpOptions(), requested: "decider-0.8b") == "decider-0.8b")
        #expect(CoreAI.systemOneModel(options: .model("qwen3-0.6b"), requested: "decider-0.8b") == "qwen3-0.6b")
        #expect(CoreAI.Op.decide.defaultModelID == CoreAI.defaultDecisionModel)
    }

    /// Bytes the wire refuses never reach a model: the op throws the wire's own error first.
    @available(macOS 27, iOS 27, *)
    @Test func bytesTheWireRefusesThrowBeforeAnyModelLoads() async {
        await #expect(throws: SystemOne.WireError.self) {
            try await CoreAI.systemOne(json: Data("not json".utf8))
        }
        await #expect(throws: SystemOne.WireError("'state' is required")) {
            try await CoreAI.systemOne(json: Data("{\"questions\": {}}".utf8))
        }
        await #expect(throws: SystemOne.WireError("duplicate question id 'q'")) {
            try await CoreAI.systemOne(json: Data(
                "{\"state\": \"s\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"a\"}, \"q\": {\"type\": \"noul\", \"instructions\": \"b\"}}}".utf8))
        }
        // Nothing above loaded a model.
        let resident = await CoreAI.residentModels()
        #expect(!resident.contains { $0.hasPrefix("\(ResidentKind.decider):") })
    }
}
