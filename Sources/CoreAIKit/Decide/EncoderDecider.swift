// EncoderDecider.swift — runs an encoder-type decision bundle (`Decision.Format.encoder`, laya):
// its two graph functions and the host work between them.
//
// A bundle directory holds the graph, `tokenizer/` and a metadata.json whose `decision` block
// (`head: "encoder"`) is the contract — `EncoderPrompt.Layout` reads the window, the head
// budget, the special ids and the temperatures from it; this file reads which asset and
// function hold the two graphs (`assets`, `decision.functions`) and the scalar type of the
// attention mask (`decision.inputs`). An AOT `.aimodelc` wins over the JIT `.aimodel` of the
// same name, as everywhere in the kit (`GraphBundle`).
//
// One call is one question row, batch 1, at the window the bundle was exported for:
//
//   main  input_ids [1,S] int32, right-padded with PAD; attention_mask [1,S] int32, 1 for a
//         real token and 0 for padding; qtype_onehot [1,3] float32 (choice / score / noul)
//         → token_logits [1,S] float32 (one logit per position), pooled_cls [1,768] float32
//   act   pooled_cls [1,768], feats [1,4] → act_logits [1,2]
//
// Between the two the host gathers the option logits at the markers and computes the act
// features (`EncoderReadout`). `decideRow` returns the raw numbers — the layer a parity run or
// a gate compares; `decide` is what `TypedDecisions` answers with.

import CoreAIKitVision
import Foundation

public actor EncoderDecider {
    /// The raw outputs of one question row.
    public struct Row: Sendable, Equatable {
        /// The token logits at the marker positions, in option order.
        public let optionLogits: [Float]
        /// The act head's two logits; class 0 is the direct-answer action.
        public let actLogits: [Float]
        /// The four act features the host computed from `optionLogits`.
        public let features: [Float]
        /// Wall-clock seconds of the `main` and the `act` run.
        public let mainSeconds: Double
        public let actSeconds: Double
        /// Wall-clock seconds of the whole row: both runs and the host work between them.
        public let seconds: Double
    }

    /// The graph assets, functions and input types a bundle declares, and its name.
    struct Graph: Sendable, Equatable {
        let main: URL
        let act: URL
        let mainFunction: String
        let actFunction: String
        /// The attention mask's scalar type is an integer one (the Core AI contract) rather
        /// than a float.
        let integerMask: Bool
        let name: String?

        static func read(bundleAt url: URL) throws -> Graph {
            let file = url.appendingPathComponent("metadata.json")
            let root = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] ?? [:]
            let assets = root["assets"] as? [String: String] ?? [:]
            let block = root["decision"] as? [String: Any] ?? [:]
            let functions = block["functions"] as? [String: String] ?? [:]
            let inputs = block["inputs"] as? [String: Any] ?? [:]
            let maskType = ((inputs["attention_mask"] as? [String: Any])?["dtype"] as? String) ?? "int32"
            let main = try asset(assets["main"], in: url)
            return Graph(
                main: main, act: try assets["act"].map { try asset($0, in: url) } ?? main,
                mainFunction: functions["main"] ?? "main", actFunction: functions["act"] ?? "act",
                integerMask: maskType.hasPrefix("int"), name: root["name"] as? String)
        }

        /// A named asset, its AOT form first; the bundle's only graph when none is named.
        static func asset(_ name: String?, in url: URL) throws -> URL {
            guard let name else { return try GraphBundle.resolve(in: url) }
            let named = url.appendingPathComponent(name)
            let compiled = name.hasSuffix(".aimodel") ? url.appendingPathComponent(name + "c") : named
            for candidate in [compiled, named] where FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            throw KitBundleError.graphMissing(url)
        }
    }

    nonisolated public let layout: EncoderPrompt.Layout
    /// The row builder over the bundle's tokenizer.
    nonisolated public let prompt: EncoderPrompt
    /// The bundle's `name`, or its directory name.
    nonisolated public let modelName: String
    private let graph: Graph
    private let main: GraphModel
    private let act: GraphModel
    /// The last state tokenized with sharing on, and its tokens.
    private var cachedState: (text: String, tokens: [Int32])?

    /// The cached tokens when they are `state`'s, byte for byte: String's `==` also matches
    /// canonically equivalent text, which the tokenizer (no Unicode normalizer) cuts differently.
    private func cachedTokens(for state: String) -> [Int32]? {
        guard let cachedState, cachedState.text.utf8.elementsEqual(state.utf8) else { return nil }
        return cachedState.tokens
    }

    /// Loads an encoder bundle directory. `actComputeUnits` runs the small act head elsewhere
    /// than the encoder (`nil`: on the same units).
    public init(
        bundleAt url: URL, computeUnits: GraphModel.ComputeUnits = .gpu,
        actComputeUnits: GraphModel.ComputeUnits? = nil
    ) async throws {
        guard let layout = try EncoderPrompt.Layout.read(bundleAt: url) else {
            throw DecisionError.unsupportedModel(
                id: url.lastPathComponent, reason: "its metadata.json declares no encoder head ('decision' block)")
        }
        try await self.init(bundleAt: url, layout: layout, computeUnits: computeUnits, actComputeUnits: actComputeUnits)
    }

    init(
        bundleAt url: URL, layout: EncoderPrompt.Layout, computeUnits: GraphModel.ComputeUnits,
        actComputeUnits: GraphModel.ComputeUnits? = nil
    ) async throws {
        let graph = try Graph.read(bundleAt: url)
        let prompt = try await EncoderPrompt(tokenizerFolder: url.appendingPathComponent("tokenizer"), layout: layout)
        let main = try await GraphModel(contentsOf: graph.main, function: graph.mainFunction, computeUnits: computeUnits)
        let act = try await GraphModel(
            contentsOf: graph.act, function: graph.actFunction, computeUnits: actComputeUnits ?? computeUnits)
        let bundle = url.lastPathComponent
        for (model, inputs, outputs) in [
            (main, ["input_ids", "attention_mask", "qtype_onehot"], ["token_logits", "pooled_cls"]),
            (act, ["pooled_cls", "feats"], ["act_logits"]),
        ] where !(Set(inputs).isSubset(of: model.inputNames) && Set(outputs).isSubset(of: model.outputNames)) {
            throw DecisionError.unsupportedModel(
                id: bundle,
                reason: "its graph takes \(model.inputNames) and gives \(model.outputNames), not \(inputs) → \(outputs)")
        }
        guard let shape = main.shape(ofInput: "input_ids"), shape.last == layout.window else {
            throw DecisionError.unsupportedModel(
                id: bundle, reason: "its graph's input_ids shape \(main.shape(ofInput: "input_ids") ?? []) is not the declared window \(layout.window)")
        }
        self.layout = layout
        self.prompt = prompt
        self.modelName = graph.name ?? bundle
        self.graph = graph
        self.main = main
        self.act = act
    }

    // MARK: - One row

    /// Runs one question row: `ids` unpadded (at most `layout.window` tokens), `markers` the
    /// option positions inside it, `qtype` the question's type (`EncoderReadout.qtype`).
    public func decideRow(ids: [Int32], markers: [Int32], qtype: Int) async throws -> Row {
        let window = layout.window
        guard ids.count <= window else { throw DecisionError.promptTooLong(tokens: ids.count, max: window) }
        precondition(EncoderReadout.questionTypes.indices.contains(qtype), "qtype \(qtype) is not a question type")
        precondition(markers.allSatisfy { $0 >= 0 && Int($0) < ids.count }, "a marker lies outside the row")
        let padding = window - ids.count
        let mask = [Int32](repeating: 1, count: ids.count) + [Int32](repeating: 0, count: padding)
        var onehot = [Float](repeating: 0, count: EncoderReadout.questionTypes.count)
        onehot[qtype] = 1
        let start = SuspendingClock.now
        let outputs = try await main.run([
            "input_ids": .int32(ids + [Int32](repeating: layout.padTokenID, count: padding), shape: [1, window]),
            "attention_mask": graph.integerMask
                ? .int32(mask, shape: [1, window]) : .float32(mask.map(Float.init), shape: [1, window]),
            "qtype_onehot": .float32(onehot, shape: [1, onehot.count]),
        ])
        let mainEnd = SuspendingClock.now
        guard let tokenLogits = outputs["token_logits"], let pooled = outputs["pooled_cls"] else {
            throw DecisionError.noLogits
        }
        let logits = EncoderReadout.optionLogits(tokenLogits: tokenLogits.floats(), markers: markers)
        let features = EncoderReadout.actFeatures(logits: logits)
        let actStart = SuspendingClock.now
        let actOutputs = try await act.run(["pooled_cls": pooled, "feats": .float32(features, shape: [1, features.count])])
        let end = SuspendingClock.now
        guard let actLogits = actOutputs["act_logits"]?.floats() else { throw DecisionError.noLogits }
        return Row(
            optionLogits: logits, actLogits: actLogits, features: features,
            mainSeconds: ProcessStats.seconds(from: start, to: mainEnd),
            actSeconds: ProcessStats.seconds(from: actStart, to: end),
            seconds: ProcessStats.seconds(from: start, to: end))
    }

    // MARK: - Decisions

    /// One question on one state, read at `temperature`. With `sharePrefix`, a state tokenized
    /// by the previous call or by `prefill` is reused, and the timing counts its tokens as
    /// reused.
    func decide(
        _ state: String, _ question: Decision.Question, temperature: Double, sharePrefix: Bool
    ) async throws -> Decision.Answer {
        let cached = sharePrefix ? cachedTokens(for: state) : nil
        let stateTokens = cached ?? prompt.contextTokens(state: state)
        if sharePrefix { cachedState = (state, stateTokens) }
        let rendered = try prompt.render(stateTokens: stateTokens, question: question)
        let row = try await decideRow(ids: rendered.tokens, markers: rendered.markers, qtype: rendered.qtype)
        let timing = Decision.Timing(
            promptTokens: rendered.tokens.count, reusedTokens: cached == nil ? 0 : rendered.stateTokens,
            seconds: row.seconds)
        return DecisionPrompt.answer(
            for: question, probabilities: EncoderReadout.probabilities(logits: row.optionLogits, temperature: temperature),
            timing: timing)
    }

    /// Tokenizes `state` once for the decisions that follow on it: the whole of what an encoder
    /// shares between questions, since every question is its own forward pass.
    func prefill(_ state: String, sharePrefix: Bool) -> (tokens: Int, timing: Decision.Timing) {
        if sharePrefix, let cached = cachedTokens(for: state) {
            return (cached.count, Decision.Timing(promptTokens: cached.count, reusedTokens: cached.count, seconds: 0))
        }
        let start = SuspendingClock.now
        let tokens = prompt.contextTokens(state: state)
        let timing = Decision.Timing(
            promptTokens: tokens.count, reusedTokens: 0, seconds: ProcessStats.seconds(from: start, to: .now))
        if sharePrefix { cachedState = (state, tokens) }
        return (tokens.count, timing)
    }

    /// Forgets the tokenized state.
    func reset() {
        cachedState = nil
    }
}
