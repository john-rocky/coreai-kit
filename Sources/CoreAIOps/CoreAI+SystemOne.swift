// CoreAI+SystemOne.swift — the hosted System One call, answered on this machine: a request in
// the `/v1/systemone` form — the state, the questions keyed by the caller's ids, an optional
// model — and the response in that form, the typed answers beside it. `CoreAI.decide` is the
// same decisions with Swift values in and out; this op is for code that already holds the
// hosted request or wants the hosted response: a client library's transport, an app that
// serves the route itself, a test that replays recorded requests.
//
// ```swift
// let response = try await CoreAI.systemOne(json: body)      // the bytes a client sent
// response["queue"]?.choice                                  // typed, by the request's key
// response.dumps()                                           // the reply, in the wire form
//
// let r = try await CoreAI.systemOne(
//     state: ticket,
//     questions: ["reply": .noul("Does the customer expect a response today?"),
//                 "topic": .choice("What is it about?", ["billing", "delivery", "how-to"])])
// r.answers.map(\.id)                                        // ["reply", "topic"] — request order
// ```
//
// The model is `options.model`, else the request's own `model`, else
// `CoreAI.defaultDecisionModel` — the same catalog models, the same process-wide cache and the
// same one-request-at-a-time rule as `CoreAI.decide`; `Op.decide` is the capability to check
// and the op to `prepare`.

import CoreAIKit
import Foundation

extension CoreAI {
    /// A request in the hosted form, answered whole: the state prefilled once, every question
    /// decided in request order, the response with the typed answers and the wire object
    /// (`response.value`, `response.dumps()`).
    public static func systemOne(
        _ request: SystemOne.Request, options: OpOptions = OpOptions()
    ) async throws -> SystemOne.Response {
        try await DecideOpModels.shared.run(
            catalog: systemOneModel(options: options, requested: request.model)
        ) {
            try await $0.systemOne(request)
        }
    }

    /// The same from the request's bytes, as a client sends them. A request the wire refuses —
    /// not JSON, no `state`, a duplicate question id, a choice wider than the hosted 255 —
    /// throws `SystemOne.WireError` before any model loads; one the model refuses (a choice
    /// wider than it reads, an empty instruction) throws `DecisionError`. A server answers
    /// both as a 422.
    public static func systemOne(
        json data: Data, options: OpOptions = OpOptions()
    ) async throws -> SystemOne.Response {
        try await systemOne(try SystemOne.request(from: data), options: options)
    }

    /// A state and questions keyed by ids of the caller's choosing, in the order the answers
    /// should come back.
    public static func systemOne(
        state: String, questions: KeyValuePairs<String, Decision.Question>, options: OpOptions = OpOptions()
    ) async throws -> SystemOne.Response {
        try await systemOne(SystemOne.Request(state: state, questions: questions), options: options)
    }

    /// The catalog id a request runs on: the op's option, else the request's own `model`, else
    /// the decision default.
    static func systemOneModel(options: OpOptions, requested: String?) -> String {
        options.model ?? requested ?? defaultDecisionModel
    }
}
