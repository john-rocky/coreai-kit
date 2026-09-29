// CoreAI+Decide.swift — anchored typed decisions: a state and typed questions in, answers with
// probabilities out. Nothing is generated; each question is one scored prompt.
//
// ```swift
// let a = try await CoreAI.decide(
//     transcript,
//     ["reply":  .noul("Does the speaker want a response from the assistant?"),
//      "topic":  .choice("What is the message about?", ["billing", "delivery", "other"]),
//      "urgent": .score("How urgent is it?", levels: ["can wait", "this week", "today"])])
// a["reply"]?.noul        // P(yes)
// a["topic"]?.choice      // "delivery"
// a["urgent"]?.score      // expected level, 0…2
// ```
//
// The op is the one-call shape over `TypedDecisions`; hold a `TypedDecisions` yourself when
// an app decides continuously (a speech gate, a clipboard watcher) and wants the model warm.
//
// The same questions about an image go to decider-2b-vision (`KitVisionDecider`), every
// question of a call read in one pass:
//
// ```swift
// let b = try await CoreAI.decide(
//     image: frame, "The image shows the current game screen.",
//     ["move": .choice("Where should the paddle move?", ["up", "down", "stay"]),
//      "ball": .noul("Is the ball visible?")])
// ```

import CoreAIKit
import CoreGraphics
import Foundation

extension CoreAI {
    /// Default decision model, the same on every platform. MiniCPM5 2B is the smallest catalog
    /// chat model whose zero-shot decisions clear chance on the published fixtures;
    /// `options: .model("qwen3-0.6b")` trades accuracy for a 352 MB download. docs/SYSTEM_ONE.md
    /// ("Which model") compares the other catalog models on JevBench's items, Mac and iPhone.
    public static let defaultDecisionModel = "minicpm5-2b"

    /// One typed question about a state.
    @available(macOS 27, iOS 27, *)
    public static func decide(
        _ state: String, _ question: Decision.Question, options: OpOptions = OpOptions()
    ) async throws -> Decision.Answer {
        try await DecideOpModels.shared.run(catalog: options.model ?? defaultDecisionModel) {
            try await $0.decide(state, question)
        }
    }

    /// Several typed questions about one state, keyed by ids of the caller's choosing. The
    /// state is prefilled once; each question reuses it.
    @available(macOS 27, iOS 27, *)
    public static func decide(
        _ state: String, _ questions: [String: Decision.Question], options: OpOptions = OpOptions()
    ) async throws -> [String: Decision.Answer] {
        try await DecideOpModels.shared.run(catalog: options.model ?? defaultDecisionModel) {
            try await $0.decide(state, questions)
        }
    }

    /// Several typed questions about one state, answered in order.
    @available(macOS 27, iOS 27, *)
    public static func decide(
        _ state: String, _ questions: [Decision.Question], options: OpOptions = OpOptions()
    ) async throws -> [Decision.Answer] {
        try await DecideOpModels.shared.run(catalog: options.model ?? defaultDecisionModel) {
            try await $0.decide(state, questions)
        }
    }

    /// Default model for decisions about an image: decider-2b-vision (catalog kind
    /// `visionDecision`), 3.3 GB with the 256×256 tower.
    public static let defaultVisionDecisionModel = "decider-2b-vision"

    /// One typed question about an image and a state. `grid` is the tile the image is resized to:
    /// `.g256` for game frames and speed, `.g448` for photos (its tower downloads on first use).
    @available(macOS 27, iOS 27, *)
    public static func decide(
        image: CGImage, _ state: String, _ question: Decision.Question,
        grid: KitVisionDecider.Grid = .g256, options: OpOptions = OpOptions()
    ) async throws -> Decision.Answer {
        try await VisionDecideOpModels.shared.run(catalog: options.model ?? defaultVisionDecisionModel) {
            try await $0.decide(image: image, state: state, question: question, grid: grid)
        }
    }

    /// Several typed questions about an image and a state, keyed by ids of the caller's
    /// choosing: one row lists them in key order and one pass reads every answer.
    @available(macOS 27, iOS 27, *)
    public static func decide(
        image: CGImage, _ state: String, _ questions: [String: Decision.Question],
        grid: KitVisionDecider.Grid = .g256, options: OpOptions = OpOptions()
    ) async throws -> [String: Decision.Answer] {
        try await VisionDecideOpModels.shared.run(catalog: options.model ?? defaultVisionDecisionModel) {
            try await $0.decide(image: image, state: state, questions: questions, grid: grid)
        }
    }

    /// Several typed questions about an image and a state, in one pass, answered in order.
    @available(macOS 27, iOS 27, *)
    public static func decide(
        image: CGImage, _ state: String, _ questions: [Decision.Question],
        grid: KitVisionDecider.Grid = .g256, options: OpOptions = OpOptions()
    ) async throws -> [Decision.Answer] {
        try await VisionDecideOpModels.shared.run(catalog: options.model ?? defaultVisionDecisionModel) {
            try await $0.decide(image: image, state: state, questions: questions, grid: grid)
        }
    }
}

/// Process-wide cache of loaded decision models, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached, and calls
/// on one model serialize behind each other.
@available(macOS 27, iOS 27, *)
actor DecideOpModels {
    static let shared = DecideOpModels()

    private let deciders = ResidentCache<TypedDecisions>(kind: ResidentKind.decider)
    private var turns: [String: Task<Void, Never>] = [:]

    func run<Value: Sendable>(
        catalog id: String, _ body: @escaping @Sendable (TypedDecisions) async throws -> Value
    ) async throws -> Value {
        let decider = try await self.decider(catalog: id)
        let previous = turns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.decider, id) {
                try await body(decider)
            }
        }
        turns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func decider(catalog id: String) async throws -> TypedDecisions {
        try await deciders.value(for: id) {
            try await TypedDecisions(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}

/// The same contract for the image deciders (`KitVisionDecider`), in their own residency
/// namespace: one load per catalog id, and calls on one model serialize behind each other.
@available(macOS 27, iOS 27, *)
actor VisionDecideOpModels {
    static let shared = VisionDecideOpModels()

    private let deciders = ResidentCache<KitVisionDecider>(kind: ResidentKind.visionDecider)
    private var turns: [String: Task<Void, Never>] = [:]

    func run<Value: Sendable>(
        catalog id: String, _ body: @escaping @Sendable (KitVisionDecider) async throws -> Value
    ) async throws -> Value {
        let decider = try await self.decider(catalog: id)
        let previous = turns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.visionDecider, id) {
                try await body(decider)
            }
        }
        turns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func decider(catalog id: String) async throws -> KitVisionDecider {
        try await deciders.value(for: id) {
            try await KitVisionDecider(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}
