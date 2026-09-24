// FoundationModelDecisions.swift — Apple's on-device foundation model (the FoundationModels
// framework's `SystemLanguageModel.default`) as a decision backend. Nothing is scored: each
// question is one guided generation whose schema is the answer shape — a choice is an
// enumeration of its option ids, a noul a Bool, a score an Int within its levels — so the
// model can only answer with a listed value, and it answers in a few tokens.
//
//   let decider = try FoundationModelDecisions()             // throws when Apple Intelligence is off
//   let answer = try await decider.decide(ticket, .choice("Which queue?", ["billing", "technical"]))
//   answer.choice                                            // "billing"
//   answer.probabilities                                     // [1, 0] — one-hot, not a distribution
//
// What it is not: the framework exposes no logits, so there is no probability behind an
// answer. The answer's `probabilities` put 1 on the generated value and 0 elsewhere, its
// `confidence` and `certainty` are 1, and a `SystemOne.Response` from this backend carries
// `metadata` (`probabilities: one-hot`, `calibration: none`) so a client does not read them
// as calibrated. A benchmark's calibration axis (ECE, Brier) says nothing about this backend.
//
// The state goes into the session's instructions once; each question is a prompt on that
// session. With `Configuration.shareSession` (the default) consecutive questions on the same
// state continue one transcript, the analogue of `TypedDecisions.Configuration.sharePrefix`;
// off, every question starts a new session — `decide-cli --no-share`. The transcript is
// bounded by the model's context window (8,192 tokens on macOS 27.0, by the framework's own
// error text): a question that no longer fits on a shared transcript starts a new session and
// is asked once more; one that does not fit on its own is refused (`DecisionError.refused`),
// as are a guardrail violation and a model refusal — a server answers all three with a 422.
//
// Sampling is greedy, so the same question on the same state answers the same way.

import Foundation
import FoundationModels

public actor FoundationModelDecisions: DecisionBackend {
    public struct Configuration: Sendable {
        /// Continue one session for consecutive questions on the same state (the transcript
        /// keeps the earlier questions and answers). Off, each question is a new session
        /// holding the state alone.
        public var shareSession: Bool = true
        /// The most tokens an answer may take. Every answer shape is a few tokens; the cap is
        /// a guard, not a budget.
        public var maximumResponseTokens: Int = 64

        public init() {}
    }

    /// What responses name as the model.
    public static let modelID = "apple-foundation-model"
    nonisolated public let id = FoundationModelDecisions.modelID
    /// The hosted API's ceiling: a 255-way enumeration is answered (M4 Max, macOS 27.0, 2026-09-24).
    nonisolated public let maxOptions = SystemOne.maxOptions
    /// The configuration this instance answers with.
    nonisolated public let configuration: Configuration

    private let model: SystemLanguageModel
    private let options: GenerationOptions
    private let lock = AsyncMutex()
    private var session: LanguageModelSession?
    private var sessionState: String?
    /// Questions answered on the current session; 0 when it holds the state alone.
    private var sessionTurns = 0

    /// Throws `DecisionError.unsupportedModel` with the reason when the system model is
    /// unavailable on this device (not eligible, Apple Intelligence off, model not ready).
    public init(configuration: Configuration = Configuration()) throws {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            break
        case .unavailable(let reason):
            throw DecisionError.unsupportedModel(id: Self.modelID, reason: Self.describe(reason))
        }
        self.model = model
        self.configuration = configuration
        self.options = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: configuration.maximumResponseTokens)
    }

    /// The model and the OS build it ships with, the only identity the system model has.
    nonisolated public var modelName: String {
        "Apple on-device foundation model (FoundationModels, SystemLanguageModel.default), "
            + ProcessInfo.processInfo.operatingSystemVersionString
    }

    /// What a response from this backend says beside its answers.
    nonisolated public var metadata: JSONValue {
        .object([
            .init("backend", .string("foundation-models")),
            .init("model", .string("SystemLanguageModel.default")),
            .init("os", .string(ProcessInfo.processInfo.operatingSystemVersionString)),
            .init("probabilities", .string("one-hot")),
            .init("calibration", .string("none")),
            .init("sampling", .string("greedy")),
            .init("session", .string(configuration.shareSession ? "shared" : "fresh")),
        ])
    }

    static func describe(_ reason: SystemLanguageModel.Availability.UnavailableReason) -> String {
        switch reason {
        case .deviceNotEligible: return "this device is not eligible for Apple Intelligence"
        case .appleIntelligenceNotEnabled: return "Apple Intelligence is not enabled in System Settings"
        case .modelNotReady: return "the system model is not ready (still downloading, or the device is busy)"
        @unknown default: return "the system model is unavailable (\(reason))"
        }
    }

    // MARK: - Decide

    /// One question on one state: a guided generation on the session that holds the state.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        try DecisionPrompt.validate(question, maxOptions: maxOptions)
        return try await lock.withLock { try await answer(state, question) }
    }

    /// Several questions on one state, answered in order.
    public func decide(_ state: String, _ questions: [Decision.Question]) async throws -> [Decision.Answer] {
        var answers: [Decision.Answer] = []
        answers.reserveCapacity(questions.count)
        for question in questions {
            answers.append(try await decide(state, question))
        }
        return answers
    }

    /// Questions keyed by an id of the caller's choosing, answered in key order.
    public func decide(
        _ state: String, _ questions: [String: Decision.Question]
    ) async throws -> [String: Decision.Answer] {
        var answers: [String: Decision.Answer] = [:]
        for key in questions.keys.sorted() {
            answers[key] = try await decide(state, questions[key]!)
        }
        return answers
    }

    /// A whole request: the state in one session, every question in request order. The
    /// response's `stateTokens` and `prefill` are 0 — the framework reads the state inside
    /// each generation and reports its tokens on the answer — and its `metadata` names this
    /// backend.
    public func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        try SystemOne.validateIDs(request.questions.map(\.id))
        for (_, question) in request.questions {
            try DecisionPrompt.validate(question, maxOptions: maxOptions)
        }
        var answers: [SystemOne.Answer] = []
        answers.reserveCapacity(request.questions.count)
        for (key, question) in request.questions {
            answers.append(SystemOne.Answer(id: key, question: question, answer: try await decide(request.state, question)))
        }
        return SystemOne.Response(
            model: id, answers: answers, stateTokens: 0,
            prefill: Decision.Timing(promptTokens: 0, reusedTokens: 0, seconds: 0), metadata: metadata)
    }

    /// Drops the session: the next question starts one holding its state alone.
    public func reset() {
        session = nil
        sessionState = nil
        sessionTurns = 0
    }

    // MARK: - Session

    private func answer(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        let (session, continued) = sessionFor(state)
        do {
            return try await respond(on: session, question)
        } catch DecisionError.refused(let reason) where continued && Self.isContextOverflow(reason) {
            // The shared transcript no longer holds the question: start over with the state alone.
            reset()
            let (fresh, _) = sessionFor(state)
            return try await respond(on: fresh, question)
        }
    }

    /// The session for `state`: the current one when sharing is on and it holds this state
    /// (`continued` true), else a new one holding the state alone.
    private func sessionFor(_ state: String) -> (LanguageModelSession, continued: Bool) {
        if configuration.shareSession, let session, sessionState == state {
            return (session, sessionTurns > 0)
        }
        let session = LanguageModelSession(model: model, instructions: FoundationModelPrompt.instructions(state: state))
        self.session = session
        sessionState = state
        sessionTurns = 0
        return (session, false)
    }

    private func respond(on session: LanguageModelSession, _ question: Decision.Question) async throws -> Decision.Answer {
        let prompt = FoundationModelPrompt.prompt(question)
        let schema = try FoundationModelPrompt.schema(question)
        let start = SuspendingClock.now
        let response: LanguageModelSession.Response<GeneratedContent>
        do {
            response = try await session.respond(to: prompt, schema: schema, options: options)
        } catch let error as LanguageModelError {
            throw Self.refusal(error)
        }
        let elapsed = SuspendingClock.now - start
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        sessionTurns += 1
        let timing = Decision.Timing(
            promptTokens: response.usage.input.totalTokenCount,
            reusedTokens: response.usage.input.cachedTokenCount,
            seconds: seconds)
        return try FoundationModelPrompt.answer(question, generated: response.content, timing: timing)
    }

    static func isContextOverflow(_ reason: String) -> Bool {
        reason.hasPrefix(contextOverflow)
    }

    static let contextOverflow = "the prompt exceeds the model's context window"

    /// A generation error as the decision surface reports it: what the model refused, as a
    /// `DecisionError` (a server's 422); the rest as they are (a 500). macOS 27.0 throws
    /// `LanguageModelError` (the framework's `GenerationError` is deprecated there).
    static func refusal(_ error: LanguageModelError) -> any Error {
        switch error {
        case .contextSizeExceeded(let detail):
            return DecisionError.refused(reason: "\(contextOverflow): \(detail.debugDescription)")
        case .guardrailViolation(let detail):
            return DecisionError.refused(reason: "a guardrail refused it: \(detail.debugDescription)")
        case .refusal(let detail):
            return DecisionError.refused(reason: "the model refused it: \(detail.debugDescription)")
        case .unsupportedCapability(let detail):
            return DecisionError.refused(reason: detail.debugDescription)
        case .unsupportedTranscriptContent(let detail):
            return DecisionError.refused(reason: detail.debugDescription)
        case .unsupportedGenerationGuide(let detail):
            return DecisionError.refused(reason: detail.debugDescription)
        case .unsupportedLanguageOrLocale(let detail):
            return DecisionError.refused(reason: detail.debugDescription)
        case .rateLimited, .timeout:
            return error
        @unknown default:
            return error
        }
    }
}

/// The prompt, the schema and the answer of one question on the system model.
enum FoundationModelPrompt {
    /// The session's instructions: the rule, then the state every question shares.
    static let preamble = """
        You answer typed questions about a state. The state is the only evidence; do not assume \
        facts it does not give. Each question fixes the form of its answer — one of the listed \
        options, a level on the listed scale, or true or false — and you answer with that value only.
        """

    static func instructions(state: String) -> String {
        "\(preamble)\n\nState:\n\(state)"
    }

    /// One question as a prompt: the instructions, the options as the model reads them, and
    /// the form of the answer.
    static func prompt(_ question: Decision.Question) -> String {
        var lines = ["Question: \(question.instructions)"]
        switch question.kind {
        case .choice(let options):
            lines.append("Options:")
            for option in options {
                lines.append("- \(optionLine(option))")
            }
            lines.append("Answer with one option name exactly as listed.")
        case .score(let levels):
            lines.append("Levels, lowest first:")
            for (index, level) in levels.enumerated() {
                lines.append("\(index): \(level)")
            }
            lines.append("Answer with the number of the level that fits.")
        case .noul(let yes, let no):
            if let yes { lines.append("true: \(yes)") }
            if let no { lines.append("false: \(no)") }
            lines.append("Answer true if it holds, false if it does not.")
        }
        return lines.joined(separator: "\n")
    }

    /// An option as one list line: the id, then its description when it has one of its own
    /// (the wire form already writes `id: description`).
    static func optionLine(_ option: Decision.Option) -> String {
        if option.description == option.id || option.description.hasPrefix("\(option.id):") {
            return option.description
        }
        return "\(option.id): \(option.description)"
    }

    /// The generation schema of the answer: the option ids for a choice, the level range for a
    /// score, a Bool for a noul. The generation can only produce a value in it.
    static func schema(_ question: Decision.Question) throws -> GenerationSchema {
        switch question.kind {
        case .choice(let options):
            return GenerationSchema(type: String.self, description: "the chosen option, exactly as listed", anyOf: options.map(\.id))
        case .score(let levels):
            return try GenerationSchema(
                root: DynamicGenerationSchema(type: Int.self, guides: [.range(0...(levels.count - 1))]), dependencies: [])
        case .noul:
            return try GenerationSchema(root: DynamicGenerationSchema(type: Bool.self), dependencies: [])
        }
    }

    /// The generated value as a decision answer: the value chosen, probability 1 on it.
    static func answer(
        _ question: Decision.Question, generated: GeneratedContent, timing: Decision.Timing
    ) throws -> Decision.Answer {
        switch question.kind {
        case .choice(let options):
            let id = try generated.value(String.self)
            guard options.contains(where: { $0.id == id }) else {
                throw DecisionError.refused(reason: "the model answered '\(id)', which is not a listed option")
            }
            return answer(question, chosen: id, timing: timing)
        case .score(let levels):
            let level = try generated.value(Int.self)
            guard levels.indices.contains(level) else {
                throw DecisionError.refused(reason: "the model answered level \(level) of \(levels.count)")
            }
            return answer(question, chosen: String(level), timing: timing)
        case .noul:
            let holds = try generated.value(Bool.self)
            return answer(question, chosen: holds ? "yes" : "no", timing: timing)
        }
    }

    /// The answer with all of the probability on `chosen` (an option id, a level index as text,
    /// or `yes` / `no`).
    static func answer(_ question: Decision.Question, chosen: String, timing: Decision.Timing) -> Decision.Answer {
        let ids = question.optionIDs
        let probabilities = ids.map { $0 == chosen ? 1.0 : 0.0 }
        switch question.kind {
        case .choice:
            return Decision.Answer(
                value: .choice(Decision.Choice(
                    id: chosen, confidence: 1, certainty: 1,
                    probabilities: Dictionary(uniqueKeysWithValues: zip(ids, probabilities)),
                    ranking: [chosen] + ids.filter { $0 != chosen }, options: ids)),
                timing: timing)
        case .score:
            let level = Int(chosen) ?? 0
            return Decision.Answer(
                value: .score(Decision.Score(
                    value: Double(level), level: level, confidence: 1, certainty: 1,
                    probabilities: probabilities, fit: nil)),
                timing: timing)
        case .noul:
            return Decision.Answer(value: .noul(chosen == "yes" ? 1 : 0), timing: timing)
        }
    }
}
