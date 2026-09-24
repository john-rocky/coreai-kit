// SystemOneWire.swift — the `/v1/systemone` request and answer forms over typed decisions,
// so a client written for a hosted System One endpoint can be pointed at a model on this
// machine and hand over the same JSON: `state` (a string, or structured data), `model`, and
// `questions` keyed by an id of the caller's choosing, each `{type, instructions, criteria}`;
// back come `answers` keyed the same way — `choice` with every option's probability, `score`
// with the expected level and its legend, `noul` as the probability the statement holds —
// with `confidence` and `usage`. `decide-cli serve` puts this behind an HTTP port; the same
// codec is here for an app that wants to accept or emit the form itself.
//
// The forms are those of TypeSafe's `/v1/systemone` (2026-09), with its limits: a choice
// lists at most 255 options, a score 10 levels. A server that knows its model passes the
// model's own option count (`TypedDecisions.maxOptions`) to `request(from:maxOptions:)`, so a
// list the model cannot read is refused with that count; 255 stays the ceiling either way.

import Foundation

public enum SystemOne {
    /// A parsed request: the state the model reads, the requested model (if any), the
    /// questions in request order.
    public struct Request: Sendable {
        public let state: String
        public let model: String?
        public let questions: [(id: String, question: Decision.Question)]
        /// True when `state` was structured data serialized the reference way.
        public let structuredState: Bool

        /// A request built in Swift rather than parsed: the state as text, the questions in the
        /// order their answers should come back, `model` a catalog id or nil for the caller's
        /// default.
        public init(state: String, model: String? = nil, questions: [(id: String, question: Decision.Question)]) {
            self.init(state: state, model: model, questions: questions, structuredState: false)
        }

        /// The same with the questions as a literal, in the order written:
        /// `["reply": .noul("…"), "topic": .choice("…", ["billing", "delivery"])]`.
        public init(state: String, model: String? = nil, questions: KeyValuePairs<String, Decision.Question>) {
            self.init(state: state, model: model, questions: questions.map { (id: $0.key, question: $0.value) })
        }

        /// A structured state — an object or an array — serialized the way the hosted API's
        /// Python clients serialize it (`JSONValue.dumps()`), so the model reads the same bytes.
        /// A string is the plain state; null is refused.
        public init(state: JSONValue, model: String? = nil, questions: [(id: String, question: Decision.Question)]) throws {
            switch state {
            case .string(let text):
                self.init(state: text, model: model, questions: questions, structuredState: false)
            case .null:
                throw WireError("'state' must be a string, an object or an array")
            default:
                self.init(state: state.dumps(), model: model, questions: questions, structuredState: true)
            }
        }

        init(state: String, model: String?, questions: [(id: String, question: Decision.Question)], structuredState: Bool) {
            self.state = state
            self.model = model
            self.questions = questions
            self.structuredState = structuredState
        }
    }

    /// A request the endpoint rejects — HTTP 422 with the message.
    public struct WireError: Error, LocalizedError, Equatable {
        public let message: String
        public init(_ message: String) { self.message = message }
        public var errorDescription: String? { message }
    }

    public static let path = "/v1/systemone"
    /// The widest choice the hosted API takes.
    public static let maxOptions = 255
    public static let maxScoreLevels = DecisionPrompt.maxScoreLevels

    // MARK: - Request

    /// A request from its bytes. `maxOptions` is the widest choice to accept: the hosted
    /// API's 255 by default, the loaded model's own count when the caller knows it (a larger
    /// value is held to 255).
    public static func request(from data: Data, maxOptions: Int = maxOptions) throws -> Request {
        let root: JSONValue
        do {
            root = try JSONValue.parse(data)
        } catch {
            throw WireError("invalid JSON: \(error.localizedDescription)")
        }
        return try request(from: root, maxOptions: maxOptions)
    }

    /// The same request from an already parsed value (an MCP tool's arguments carry the
    /// object rather than its bytes).
    public static func request(from root: JSONValue, maxOptions: Int = maxOptions) throws -> Request {
        guard let members = root.members else { throw WireError("the request must be a JSON object") }
        guard let stateValue = root["state"] else { throw WireError("'state' is required") }
        let state: String
        let structured: Bool
        switch stateValue {
        case .string(let s):
            state = s
            structured = false
        case .null:
            throw WireError("'state' must be a string, an object or an array")
        default:
            state = stateValue.dumps()
            structured = true
        }
        let model = root["model"]?.stringValue
        guard let questionMembers = root["questions"]?.members else {
            throw WireError("'questions' must be an object keyed by question id")
        }
        guard !questionMembers.isEmpty else { throw WireError("'questions' is empty") }
        var questions: [(id: String, question: Decision.Question)] = []
        for member in questionMembers {
            questions.append((member.key, try question(from: member.value, id: member.key, maxOptions: maxOptions)))
        }
        try validateIDs(questions.map(\.id))
        _ = members
        return Request(state: state, model: model, questions: questions, structuredState: structured)
    }

    /// Every question id once: the answers are keyed by them, and a second answer under the same
    /// key would overwrite the first in the response object.
    static func validateIDs(_ ids: [String]) throws {
        var seen: Set<String> = []
        for id in ids where !seen.insert(id).inserted {
            throw WireError("duplicate question id '\(id)'")
        }
    }

    /// One question from its wire object: `{"type": "choice" | "score" | "noul", "instructions": …, "criteria": …}`.
    /// A choice may list up to `maxOptions` options (held to the hosted API's 255).
    public static func question(from value: JSONValue, id: String, maxOptions: Int = maxOptions) throws -> Decision.Question {
        let maxOptions = min(maxOptions, Self.maxOptions)
        guard value.members != nil else { throw WireError("question '\(id)' is not an object") }
        guard let type = value["type"]?.stringValue else { throw WireError("question '\(id)' has no string 'type'") }
        guard let instructionsValue = value["instructions"] else { throw WireError("question '\(id)' has no 'instructions'") }
        let instructions: String
        switch instructionsValue {
        case .string(let s): instructions = s
        case .null: throw WireError("question '\(id)': 'instructions' must be a string, an object or an array")
        default: instructions = instructionsValue.dumps()
        }
        guard !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WireError("question '\(id)': 'instructions' is empty")
        }
        switch type {
        case "choice":
            var options: [Decision.Option] = []
            if let members = value["criteria"]?.members {
                for member in members {
                    let description: String?
                    switch member.value {
                    case .null: description = nil
                    case .string(let s): description = s.isEmpty ? nil : s
                    default: description = member.value.dumps()
                    }
                    options.append(.init(id: member.key, description: description.map { "\(member.key): \($0)" } ?? member.key))
                }
            } else if let elements = value["criteria"]?.elements {
                for element in elements {
                    let key = element.stringValue ?? element.dumps()
                    options.append(.init(key))
                }
            } else {
                throw WireError("question '\(id)': a choice needs 'criteria' (an object of option → description, or a list of options)")
            }
            guard options.count >= 2 else { throw WireError("question '\(id)': a choice needs at least 2 criteria, got \(options.count)") }
            guard options.count <= maxOptions else {
                throw WireError("question '\(id)': this engine lists at most \(maxOptions) options, got \(options.count)")
            }
            guard Set(options.map(\.id)).count == options.count else { throw WireError("question '\(id)': duplicate option") }
            return .choice(instructions, options: options)
        case "score":
            guard let elements = value["criteria"]?.elements else {
                throw WireError("question '\(id)': a score needs 'criteria', an array of level descriptions (lowest first)")
            }
            let levels = elements.map { $0.stringValue ?? $0.dumps() }
            guard levels.count >= 2, levels.count <= maxScoreLevels else {
                throw WireError("question '\(id)': a score needs 2–\(maxScoreLevels) levels, got \(levels.count)")
            }
            return .score(instructions, levels: levels)
        case "noul":
            var yes: String?
            var no: String?
            if let criteria = value["criteria"] {
                guard let members = criteria.members else {
                    throw WireError("question '\(id)': noul 'criteria' must be an object with optional 'true' and 'false'")
                }
                for member in members {
                    let text = member.value.stringValue ?? (member.value == .null ? nil : member.value.dumps())
                    switch member.key {
                    case "true": yes = text
                    case "false": no = text
                    default: break
                    }
                }
            }
            return .noul(instructions, yes: yes, no: no)
        default:
            throw WireError("question '\(id)': unknown type '\(type)' (choice | score | noul)")
        }
    }

    // MARK: - Answers

    /// Four decimals, as the reference writes them.
    static func rounded(_ value: Double) -> JSONValue {
        .double((value * 10000).rounded() / 10000)
    }

    /// The wire object of one answer. `confidence` is 1 − normalised entropy of the
    /// distribution for a choice or a score, and max(p, 1 − p) for a noul.
    public static func answerValue(_ question: Decision.Question, _ answer: Decision.Answer) -> JSONValue {
        switch answer.value {
        case .choice(let c):
            var members: [JSONValue.Member] = [
                .init("type", .string("choice")),
                .init("choice", .string(c.id)),
                .init("probabilities", .object(c.options.map { .init($0, rounded(c.probabilities[$0] ?? 0)) })),
                .init("confidence", rounded(c.certainty)),
            ]
            // A slot-head model's "none of these" mass, the way its own API adds it beside
            // the renormalised option probabilities; absent for every other model.
            if let abstain = answer.abstain { members.append(.init("abstain", rounded(abstain))) }
            return .object(members)
        case .score(let s):
            var legend: [JSONValue.Member] = []
            if case .score(let levels) = question.kind {
                legend = levels.enumerated().map { .init(String($0.offset), .string($0.element)) }
            }
            return .object([
                .init("type", .string("score")),
                .init("score", rounded(s.value)),
                .init("legend", .object(legend)),
                .init("probabilities", .object(s.probabilities.enumerated().map { .init(String($0.offset), rounded($0.element)) })),
                .init("confidence", rounded(s.certainty)),
            ])
        case .noul(let p):
            return .object([
                .init("type", .string("noul")),
                .init("noul", rounded(p)),
                .init("confidence", rounded(max(p, 1 - p))),
            ])
        }
    }

    /// The response: `model`, `answers` keyed as the request was, `usage` (`input_tokens` =
    /// the prompt tokens of every decision summed, as a hosted endpoint counts them;
    /// `output_tokens` = one answer slot per decision) and `timing_ms` (this machine's wall clock).
    public static func response(
        model: String, answers: [(id: String, question: Decision.Question, answer: Decision.Answer)]
    ) -> JSONValue {
        let inputTokens = answers.map(\.answer.timing.promptTokens).reduce(0, +)
        let milliseconds = answers.map(\.answer.timing.milliseconds).reduce(0, +)
        return .object([
            .init("model", .string(model)),
            .init("answers", .object(answers.map { .init($0.id, answerValue($0.question, $0.answer)) })),
            .init("usage", .object([
                .init("input_tokens", .int(inputTokens)),
                .init("output_tokens", .int(answers.count)),
            ])),
            .init("timing_ms", rounded(milliseconds)),
        ])
    }

    /// `GET /v1/models` in the hosted form — `models`, each `name` / `description` /
    /// `release_date` (empty for a local bundle; its identity is the pinned `revision`, given
    /// beside it) — with the OpenAI-style `object` / `data` keys after it for a client that
    /// reads that form instead.
    public static func modelsValue(id: String, description: String, revision: String?) -> JSONValue {
        .object([
            .init("models", .array([.object([
                .init("name", .string(id)),
                .init("description", .string(description)),
                .init("release_date", .string("")),
                .init("revision", .string(revision ?? "")),
            ])])),
            .init("object", .string("list")),
            .init("data", .array([.object([
                .init("id", .string(id)), .init("object", .string("model")), .init("owned_by", .string("local")),
            ])])),
        ])
    }

    /// An error body: `{"error": {"type": …, "message": …}}`.
    public static func errorValue(type: String, message: String) -> JSONValue {
        .object([.init("error", .object([.init("type", .string(type)), .init("message", .string(message))]))])
    }
}
