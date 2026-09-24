// SystemOneCall.swift — one `/v1/systemone` request answered whole on a loaded model: the state
// prefilled once, every question decided in request order against it, and the response in the
// hosted form with the typed answers beside it.
//
// ```swift
// let decider = try await TypedDecisions(catalog: "minicpm5-2b")
// let response = try await decider.systemOne(try SystemOne.request(from: body))
// response["queue"]?.choice        // the typed answer, by the request's own key
// response.dumps()                 // the reply in the wire form — what `systemone serve` returns
// ```
//
// `TypedDecisions.systemOne(_:)` is the model-level call; `CoreAI.systemOne` (CoreAIOps) resolves
// and caches the model behind it. `SystemOneServer`, `SystemOneMCPServer` and the `systemone`
// binary all answer through this one path, so a request gets the same answer whichever door it
// came in by. The wire object is `SystemOne.response(model:answers:)`, unchanged.

import Foundation

extension SystemOne {
    /// The response's `usage`, counted as a hosted endpoint counts it.
    public struct Usage: Sendable, Equatable {
        /// The prompt tokens of every decision, summed — each question's whole prompt, the
        /// state included, whether or not the engine reused it.
        public let inputTokens: Int
        /// One answer slot per decision.
        public let outputTokens: Int

        public init(inputTokens: Int, outputTokens: Int) {
            self.inputTokens = inputTokens
            self.outputTokens = outputTokens
        }
    }

    /// One question's answer, keyed as the request keyed it.
    public struct Answer: Sendable, Equatable {
        public let id: String
        public let question: Decision.Question
        public let answer: Decision.Answer

        public init(id: String, question: Decision.Question, answer: Decision.Answer) {
            self.id = id
            self.question = question
            self.answer = answer
        }

        /// The chosen option id of a `choice` question; nil for the other shapes.
        public var choice: String? { answer.choice }
        /// The expected level of a `score` question; nil for the other shapes.
        public var score: Double? { answer.score }
        /// P(yes) of a `noul` question; nil for the other shapes.
        public var noul: Double? { answer.noul }

        /// The wire object of this answer (`SystemOne.answerValue`).
        public var value: JSONValue { SystemOne.answerValue(question, answer) }
    }

    /// A whole request's answers: typed, in request order, and in the wire form.
    public struct Response: Sendable {
        /// What the response names as the model: the catalog id, or a local bundle's directory
        /// name.
        public let model: String
        /// Every question's answer, in request order.
        public let answers: [Answer]
        /// Tokens of the state's shared prefix the model held while answering.
        public let stateTokens: Int
        /// What prefilling that prefix cost.
        public let prefill: Decision.Timing
        /// What the backend says beside its answers, or nil: `FoundationModelDecisions` names
        /// itself and says its probabilities are one-hot; `TypedDecisions` adds nothing.
        public let metadata: JSONValue?

        public init(
            model: String, answers: [Answer], stateTokens: Int, prefill: Decision.Timing, metadata: JSONValue? = nil
        ) {
            self.model = model
            self.answers = answers
            self.stateTokens = stateTokens
            self.prefill = prefill
            self.metadata = metadata
        }

        /// The answer to the question the request keyed `id`; nil for an id it did not ask.
        public subscript(id: String) -> Decision.Answer? {
            answers.first { $0.id == id }?.answer
        }

        /// The question ids, in request order.
        public var ids: [String] { answers.map(\.id) }

        /// `usage` as the wire reports it.
        public var usage: Usage {
            Usage(
                inputTokens: answers.map(\.answer.timing.promptTokens).reduce(0, +),
                outputTokens: answers.count)
        }

        /// Wall-clock milliseconds of the whole request on this machine: the prefill and every
        /// decision. The wire's `timing_ms` is the decisions alone, as the servers have always
        /// reported it.
        public var milliseconds: Double {
            prefill.milliseconds + answers.map(\.answer.timing.milliseconds).reduce(0, +)
        }

        /// The wire object — `model`, `answers`, `usage`, `timing_ms`, and `metadata` when the
        /// backend gave one — exactly as `SystemOne.response(model:answers:metadata:)` writes it.
        public var value: JSONValue {
            SystemOne.response(model: model, answers: answers.map { ($0.id, $0.question, $0.answer) }, metadata: metadata)
        }

        /// The wire object as text, the way the servers write it.
        public func dumps() -> String { value.dumps() }
    }
}

extension TypedDecisions {
    /// Answers a whole request on this model: the state prefilled once, each question decided
    /// in request order against it. Every question is checked against this model's option
    /// count, and the ids for duplicates, before anything runs. `request.model` is the caller's
    /// to resolve (`CoreAI.systemOne` does); this model answers, and the response names it.
    public func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        try SystemOne.validateIDs(request.questions.map(\.id))
        for (_, question) in request.questions {
            try DecisionPrompt.validate(question, maxOptions: maxOptions)
        }
        let prefilled = try await prefill(request.state)
        var answers: [SystemOne.Answer] = []
        answers.reserveCapacity(request.questions.count)
        for (key, question) in request.questions {
            answers.append(SystemOne.Answer(id: key, question: question, answer: try await prefilled.decide(question)))
        }
        return SystemOne.Response(model: id, answers: answers, stateTokens: prefilled.tokens, prefill: prefilled.timing)
    }
}
