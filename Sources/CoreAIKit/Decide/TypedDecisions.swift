// TypedDecisions.swift — typed decisions over a Core AI chat bundle: one prompt scored at its
// last position, the answer read as probabilities over the listed options.
//
// ```swift
// let decider = try await TypedDecisions(catalog: "minicpm5-2b")
// let answer = try await decider.decide(
//     "Customer: my order arrived with the box crushed and the screen cracked.",
//     .noul("Does the customer want a replacement?"))
// answer.noul   // P(yes)
// ```
//
// ## One prefill, N decisions
//
// Every decision on the same state shares the prompt prefix up to the end of the state (the
// system line and the state itself); only the question and the options differ. The engine
// keeps its KV cache between calls, so a second question on the same state rewinds to that
// shared prefix and prefills just the tail. `prefill(_:)` runs the prefix on its own so the
// first question is as cheap as the rest; `Decision.Timing.reusedTokens` reports what was
// kept. Engines that cannot rewind mid-sequence (recurrent hybrids — Qwen3.5, LFM2.5,
// Granite 4) checkpoint instead: the prefix pass — `prefill(_:)`, or the first question on a
// new state — saves the recurrent state after it (`InferenceEngine.checkpoint()`), and each
// later question on the state returns there and prefills just the tail, bit-identical to a
// full re-prefill on the three hybrid decision fixtures (Mac, 2026-09-24). A rewind below the
// checkpoint still falls back to a full re-prefill, losslessly; the timing says so.
// A chat question past 26 options is read under its own system line (`DecisionPrompt`'s
// wide rendering), so it keeps only the start of the shared prefix and prefills the state again.
//
// ## Which prompt
//
// A chat model reads the request as JSON in one user turn under its chat template
// (`Decision.Format.chat`); a model trained for decisions (`decider-0.8b`, catalog kind
// `decision`) reads the plain `Context:` / `Question:` / `Options:` / `Answer: (` form it was
// trained on (`Decision.Format.decider`, `DeciderPrompt.swift`). Both are read at one label
// token per option, from tables built once at load (`LabelTable.swift`): the decision model at
// its author's A–Z, AA, AB, … (255 labels), a chat model at A–Z and, past 26 options, at the
// numbers 1, 2, … where its tokenizer writes them as single tokens — 255 on MiniCPM5, none past
// 9 on the Qwen tokenizers — so `maxOptions` is 255 there and 26 elsewhere. A slot-head model
// (OpenThai-SystemOne) reads its control-token layout and is read at a 256-way head
// (`Decision.Format.slot`, `SlotPrompt.swift`); a model trained on the `Shared state:` + JSON
// task turn (APUS's decision model, apus-decision-v1-4b) is read at the letters under its chat template
// (`Decision.Format.sharedState`, `SharedStatePrompt.swift`); a scalar-head model (the
// System One scorer) is one row per option, read by its head at each row's last token
// (`Decision.Format.scalar`, `ScalarPrompt.swift`); the lettered option list under
// the chat template is read at the bare letters A–Z a–z at its helper's temperature
// (`Decision.Format.letterList`, `LetterListPrompt.swift`). A slot-head, scalar-head or
// letter-list bundle declares itself in its metadata.json (`decision.head` / `readout`,
// with its temperature); otherwise the catalog entry's `format` decides, then the catalog
// kind. `Configuration.format` overrides all of them.
//
// An encoder-type model (laya) is not a language bundle at all: one forward pass reads the
// whole question and each option is read at its mask marker (`Decision.Format.encoder`,
// `EncoderPrompt.swift`, `EncoderDecider.swift`). Its bundle declares `decision.head ==
// "encoder"`, and both initialisers look for that before anything else.
//
// ## Which engine
//
// The answer needs the logits at the answer slot, which the default GPU-pipelined engine does
// not expose (it samples on the GPU). A catalog model loads on the sequential engine, or on
// the static-shape engine for a chunked static (Neural Engine) bundle — the two engines that
// return logits. Decode speed does not matter here: a decision generates nothing.
//
// On iPhone keep a prompt under 1024 tokens: the on-device compiler miscompiles the growing
// KV cache of a dynamic bundle once it reaches 2048 positions (the same guard the pipelined
// engine enforces). A choice of 255 options does not fit that — 1,965 tokens for the decider's
// fixture row, 3,100–4,200 for a chat model's JSON of short options — so it is a Mac call; on
// a phone the ceiling is what fits in 1024 tokens (about 60 short options in the chat form,
// estimated from the Mac prompt sizes, not measured on a phone).

import CoreAIKitVision
import CoreAILanguageModels
import Foundation
import Tokenizers

/// A loaded decision model. One decision at a time; calls on the same instance serialize.
public actor TypedDecisions {
    public struct Configuration: Sendable {
        /// Softmax temperature over the answer-slot logits, for every question type. `nil` uses
        /// the catalog entry's calibration when it has one (`CatalogEntry.calibration`, fitted by
        /// the maintainer), else the model's own: 1 for a chat model (its raw distribution), the
        /// card's calibration temperature for a decision model (1.03 for `decider-0.8b`).
        public var temperature: Double? = nil
        /// How the prompt is rendered. `nil` follows the bundle — `.slot` when its
        /// metadata.json declares a slot head — then the catalog kind (`.decider` for a
        /// `decision` entry, `.chat` otherwise) and, for a local bundle, the bundle name.
        public var format: Decision.Format? = nil
        /// Reuse the engine's KV cache across decisions on the same state (rewind to the shared
        /// prefix, prefill only the question). Off, every decision re-prefills its whole prompt
        /// — the `direct` column of a benchmark.
        public var sharePrefix: Bool = true
        /// Engine to load the bundle with. `.auto` picks the static-shape engine for a bundle
        /// the catalog marks `static-shape` and the sequential engine for everything else.
        public var engineVariant: EngineVariant = .auto
        /// Prefill one token per step. `nil` detects a decode-only (S=1) graph from the bundle
        /// name; set it explicitly for a local bundle the heuristic cannot see.
        public var singleTokenPrefill: Bool? = nil
        /// Compute units for an encoder bundle's graphs (`Format.encoder`); the language
        /// formats load on their engine and ignore it. `.neuralEngine` is refused at load: the
        /// shipped encoder graph's answers are wrong there.
        public var computeUnits: GraphModel.ComputeUnits = .gpu

        public init() {}
    }

    /// What answers: a language bundle's engine, or an encoder bundle's graphs.
    enum Backend: Sendable {
        case language(ModelRuntime)
        case encoder(EncoderDecider)
    }

    private let backend: Backend
    /// The language runtime. Every path that reads it belongs to a language format; an
    /// encoder bundle (`format == .encoder`) branches off before reaching any of them.
    nonisolated private var runtime: ModelRuntime {
        guard case .language(let runtime) = backend else {
            preconditionFailure("an encoder bundle has no language runtime")
        }
        return runtime
    }
    private let configuration: Configuration
    /// The prompt form this model is scored with.
    nonisolated public let format: Decision.Format
    /// The temperature applied to the answer-slot logits. A slot-head model carries one per
    /// question type; this is its choice temperature, and `temperature(for:)` has the rest.
    nonisolated public let temperature: Double
    /// The temperature per question type (`DecisionCalibration.swift` has the order).
    nonisolated let temperatures: DecisionTemperatures
    /// What a slot-head bundle declares about its head (`Format.slot`); nil otherwise.
    nonisolated let slotLayout: SlotPrompt.Layout?
    /// What a scalar-head bundle declares about its head (`Format.scalar`); nil otherwise.
    nonisolated let scalarLayout: ScalarPrompt.Layout?
    /// What a letter-list bundle declares about its readout (`Format.letterList`); nil otherwise.
    nonisolated let letterLayout: LetterListPrompt.Layout?
    /// The tokenizer path of a slot-head bundle (control tokens by id, text cut the
    /// reference way); nil for the other formats.
    nonisolated private let slotEncoder: SlotPrompt.Encoder?
    /// The answer labels, built from the tokenizer at load: the author's table for the decider
    /// form, the letters A–Z for the chat form; empty for the other formats.
    nonisolated let labels: LabelTable
    /// The chat form's labels past the letters: the run of single-token numbers "1", "2", …;
    /// empty for the other formats.
    nonisolated let numbers: LabelTable
    /// Options a choice may list on this model: 255 for the decider form (its label table) and
    /// for a chat model whose tokenizer writes the numbers to 255 as single tokens (minicpm5-2b),
    /// 26 for another chat model (qwen3-0.6b), 16 for the `Shared state:` form, 26 for the
    /// decision-function form, 52 for the letter list, 255 rows for a scalar head, every slot
    /// but the abstain one (255 for OpenThai-SystemOne) for a slot head, 20 for an encoder. A
    /// score keeps 10 levels. `maxOptions(format:labels:numbers:slot:)` is the rule.
    nonisolated public let maxOptions: Int
    /// The catalog id, or the bundle directory name for a local bundle.
    public let id: String
    /// The bundle's `max_context_length`; a prompt must leave one slot for the answer. An
    /// encoder bundle's window: the length every row is cut and padded to.
    public let maxContextLength: Int
    /// Exact token sequence the engine's KV cache holds (the last scored prompt).
    private var kvTokens: [Int32] = []
    /// The state prefix the engine's checkpoint holds (`InferenceEngine.checkpoint()`, taken
    /// after a prefix pass on an engine that cannot rewind into a state); empty when none.
    private var checkpointed: [Int32] = []
    /// Timing of the last decision or prefill.
    public private(set) var lastTiming: Decision.Timing?

    /// Display name from the bundle metadata.
    public var modelName: String {
        switch backend {
        case .language(let runtime): runtime.modelName
        case .encoder(let encoder): encoder.modelName
        }
    }

    /// Whether this catalog entry can answer typed questions here: a `chat` or `decision`
    /// model on a runtime that exposes logits. The Gemma 4 pairs and the raw-Metal pack sample
    /// on the GPU and are refused by `init(catalog:)` with `DecisionError.unsupportedModel`.
    public static func supports(_ entry: CatalogEntry) -> Bool {
        (entry.kind == .chat || entry.kind == .decision) && entry.modelID != nil
            && entry.id != Gemma4MetalRuntime.catalogID && GemmaModelID.byCatalogID[entry.id] == nil
    }

    /// Loads a model by its catalog id (`kind: chat`); downloads on first use.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        configuration: Configuration = Configuration(),
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id)
        guard entry.kind == .chat || entry.kind == .decision else {
            throw CoreAIKitError.catalogKindMismatch(
                id: id, expected: "chat or decision", found: entry.kind.rawValue)
        }
        guard let model = entry.modelID else {
            throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
        }
        // The Gemma 4 pairs and the raw-Metal pack load through their own runtimes, which
        // sample on the GPU and expose no logits.
        if entry.id == Gemma4MetalRuntime.catalogID || GemmaModelID.byCatalogID[entry.id] != nil {
            throw DecisionError.unsupportedModel(
                id: id, reason: "its runtime samples on the GPU and exposes no logits")
        }
        let url = try await store.download(model, progress: downloadProgress)
        if let layout = try EncoderPrompt.Layout.read(bundleAt: url) {
            try await self.init(encoderAt: url, layout: layout, configuration: configuration, id: id)
            return
        }
        Self.configureSingleTokenPrefill(
            bundleName: model.resolvedPath, override: configuration.singleTokenPrefill)
        let bundle = try LanguageBundle(at: url)
        let layout = try SlotPrompt.Layout.read(bundleAt: url)
        let scalar = try ScalarPrompt.Layout.read(bundleAt: url)
        let letters = try LetterListPrompt.Layout.read(bundleAt: url)
        let runtime = try await ModelRuntime(
            bundleAt: url,
            engineVariant: Self.resolveEngine(configuration.engineVariant, hint: entry.engine))
        try self.init(
            runtime: runtime, configuration: configuration, id: id,
            maxContextLength: bundle.maxContextLength,
            format: configuration.format
                ?? (layout != nil
                    ? .slot
                    : scalar != nil
                        ? .scalar
                        : letters != nil
                            ? .letterList
                            : entry.format.flatMap(Decision.Format.init(rawValue:))
                                ?? (entry.kind == .decision ? .decider : .chat)),
            layout: layout, scalar: scalar, letters: letters, calibration: entry.calibration)
    }

    /// Loads a local bundle directory (metadata.json + *.aimodel/ + tokenizer/). The prompt
    /// format follows the bundle name (`decider` → `.decider`) unless the configuration sets it.
    public init(bundleAt url: URL, configuration: Configuration = Configuration()) async throws {
        if let layout = try EncoderPrompt.Layout.read(bundleAt: url) {
            try await self.init(encoderAt: url, layout: layout, configuration: configuration, id: url.lastPathComponent)
            return
        }
        Self.configureSingleTokenPrefill(
            bundleName: url.lastPathComponent, override: configuration.singleTokenPrefill)
        let bundle = try LanguageBundle(at: url)
        let runtime = try await ModelRuntime(
            bundleAt: url, engineVariant: Self.resolveEngine(configuration.engineVariant, hint: nil))
        let name = url.lastPathComponent.lowercased()
        let layout = try SlotPrompt.Layout.read(bundleAt: url)
        let scalar = try ScalarPrompt.Layout.read(bundleAt: url)
        let letters = try LetterListPrompt.Layout.read(bundleAt: url)
        try self.init(
            runtime: runtime, configuration: configuration, id: url.lastPathComponent,
            maxContextLength: bundle.maxContextLength,
            format: configuration.format
                ?? (layout != nil
                    ? .slot
                    : scalar != nil
                        ? .scalar
                        : letters != nil
                            ? .letterList
                            : name.contains("decider")
                                ? .decider
                                : name.contains("apus") ? .sharedState : name.contains("decision") ? .decisionFunction : .chat),
            layout: layout, scalar: scalar, letters: letters, calibration: nil)
    }

    private init(
        runtime: ModelRuntime, configuration: Configuration, id: String, maxContextLength: Int,
        format: Decision.Format, layout: SlotPrompt.Layout?, scalar: ScalarPrompt.Layout?,
        letters: LetterListPrompt.Layout?, calibration: CatalogEntry.Calibration?
    ) throws {
        guard runtime.engine.supportsLogits else {
            throw DecisionError.engineWithoutLogits(model: id)
        }
        // The first two letters cover every question shape; a tokenizer that cannot slot
        // them fails here, at load, not on the first decision. The chat and decider forms
        // build their whole label table here, once. A slot-head model has no letters: its
        // control tokens must each be one token, and its bundle must say so.
        var encoder: SlotPrompt.Encoder? = nil
        var labels = LabelTable(names: [], ids: [])
        var numbers = LabelTable(names: [], ids: [])
        switch format {
        case .encoder:
            throw DecisionError.unsupportedModel(
                id: id, reason: "its metadata.json declares no encoder head ('decision' block), which Format.encoder needs")
        case .chat:
            labels = try Self.labels(LabelTable.letters, .chat, tokenizer: runtime.tokenizer)
            numbers = LabelTable.build(LabelTable.numbers, rule: .chat, run: true, tokenizer: runtime.tokenizer)
        case .sharedState:
            _ = try DecisionPrompt.slotTokens(names: Array(SharedStatePrompt.letters.prefix(2)), tokenizer: runtime.tokenizer)
        case .decider: labels = try Self.labels(LabelTable.candidates, .decider, tokenizer: runtime.tokenizer)
        case .decisionFunction: _ = try DecisionFunctionPrompt.labelTokens(count: 2, tokenizer: runtime.tokenizer)
        case .letterList:
            // The temperature and the yes/no calibration are the helper's contract: the
            // bundle must declare them.
            guard letters != nil else {
                throw DecisionError.unsupportedModel(
                    id: id, reason: "its metadata.json declares no letter readout ('decision' block), which Format.letterList needs")
            }
            _ = try LetterListPrompt.labelTokens(count: 2, tokenizer: runtime.tokenizer)
        case .scalar:
            // No letters: the head is one number per row, and the bundle must declare it.
            guard scalar != nil else {
                throw DecisionError.unsupportedModel(
                    id: id, reason: "its metadata.json declares no scalar head ('decision' block), which Format.scalar needs")
            }
        case .slot:
            guard let layout else {
                throw DecisionError.unsupportedModel(
                    id: id, reason: "its metadata.json declares no slot head ('decision' block), which Format.slot needs")
            }
            encoder = try SlotPrompt.Encoder(tokenizer: runtime.tokenizer, layout: layout)
        }
        let temperatures = try Self.resolveTemperatures(
            id: id, configured: configuration.temperature, catalog: calibration, format: format,
            slot: layout, scalar: scalar, letters: letters)
        self.backend = .language(runtime)
        self.configuration = configuration
        self.id = id
        self.maxContextLength = maxContextLength
        self.format = format
        self.slotLayout = format == .slot ? layout : nil
        self.scalarLayout = format == .scalar ? scalar : nil
        self.letterLayout = format == .letterList ? letters : nil
        self.slotEncoder = encoder
        self.labels = labels
        self.numbers = numbers
        self.maxOptions = Self.maxOptions(format: format, labels: labels, numbers: numbers, slot: layout)
        self.temperatures = temperatures
        self.temperature = temperatures.choice
    }

    /// The temperature a question's logits are read at: the configured one when set, else the
    /// catalog's calibration, else the model's own — per question type for a slot-head model
    /// or a calibration fitted per type, by type and option count for an encoder, one value
    /// otherwise.
    nonisolated public func temperature(for question: Decision.Question) -> Double {
        if case .encoder(let encoder) = backend {
            return configuration.temperature ?? encoder.layout.temperatures.temperature(for: question)
        }
        return temperatures.temperature(for: question.kind)
    }

    /// `.auto` → the static-shape engine for a chunked static bundle, else the sequential
    /// engine. An explicit choice is kept, and refused at load if it cannot return logits.
    static func resolveEngine(_ requested: EngineVariant, hint: String?) -> EngineVariant {
        guard requested == .auto else { return requested }
        return EngineVariant(catalogHint: hint) == .staticShape ? .staticShape : .sequential
    }

    /// Zoo decode-only ports are S=1 graphs (their bundle names carry `_decode_`): every
    /// prefill chunk must be one token, on every engine. The runtime reads the threshold from
    /// the environment per generation, so this is process-wide, exactly as the kit's pipelined
    /// loads set it; a value already in the environment always wins.
    static func configureSingleTokenPrefill(bundleName: String, override: Bool?) {
        let single = override ?? bundleName.contains("_decode_")
        if single, getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
    }

    /// The options a choice may list on a model read in `format`: the decider's label table's
    /// count, a chat model's number run where it reaches past its letters (else the letters'),
    /// a slot head's slot count, the form's own letter set otherwise (16, 26, 52), the hosted
    /// API's 255 rows for a scalar head, and an encoder's 20.
    static func maxOptions(
        format: Decision.Format, labels: LabelTable, numbers: LabelTable, slot: SlotPrompt.Layout?
    ) -> Int {
        switch format {
        case .chat: return DecisionPrompt.maxOptions(letters: labels, numbers: numbers)
        case .decider: return labels.count
        case .sharedState: return SharedStatePrompt.maxOptions
        case .decisionFunction: return DecisionFunctionPrompt.maxOptions
        case .letterList: return LetterListPrompt.maxOptions
        case .scalar: return ScalarPrompt.maxOptions
        case .slot: return slot?.maxOptions ?? DecisionPrompt.maxOptions
        case .encoder: return EncoderPrompt.maxOptions
        }
    }

    /// The label table a letter readout reads its options at, refused at load when the
    /// tokenizer cannot label even two options.
    static func labels(_ names: [String], _ rule: LabelTable.Rule, tokenizer: any Tokenizer) throws -> LabelTable {
        let table = LabelTable.build(names, rule: rule, tokenizer: tokenizer)
        guard table.count >= 2 else {
            throw DecisionError.answerSlotNotSingleToken(letter: names.first { !table.names.contains($0) } ?? names[0])
        }
        return table
    }

    /// An encoder bundle (`Format.encoder`): its graphs, its tokenizer and the contract its
    /// metadata declares, no language runtime. It reads as `.encoder` only.
    private init(
        encoderAt url: URL, layout: EncoderPrompt.Layout, configuration: Configuration, id: String
    ) async throws {
        if let requested = configuration.format, requested != .encoder {
            throw DecisionError.unsupportedModel(
                id: id, reason: "it is an encoder bundle, which reads as Format.encoder, not .\(requested.rawValue)")
        }
        // `EncoderDecider` itself still takes the Neural Engine, for measuring it.
        if configuration.computeUnits == .neuralEngine {
            throw DecisionError.unsupportedModel(
                id: id,
                reason: "with a Neural Engine preference this graph's answers fall outside the 1e-3 bar and change "
                    + "from run to run (2026-09-23, Mac GPU exact); use the GPU")
        }
        self.backend = .encoder(
            try await EncoderDecider(bundleAt: url, layout: layout, computeUnits: configuration.computeUnits))
        self.configuration = configuration
        self.id = id
        self.maxContextLength = layout.window
        self.format = .encoder
        // One per type for `temperature` and the calibration tools; `temperature(for:)` reads the
        // bundle's table by type and option count.
        let byType = layout.temperatures.byType
        self.temperatures = configuration.temperature.map { DecisionTemperatures(choice: $0, score: $0, noul: $0) }
            ?? DecisionTemperatures(choice: byType[0], score: byType[1], noul: byType[2])
        self.temperature = temperatures.choice
        self.slotLayout = nil
        self.scalarLayout = nil
        self.letterLayout = nil
        self.slotEncoder = nil
        self.labels = LabelTable(names: [], ids: [])
        self.numbers = LabelTable(names: [], ids: [])
        self.maxOptions = EncoderPrompt.maxOptions
    }

    // MARK: - Decide

    /// One question on one state. Under `.decider` a score question is several rows (one
    /// per level) and under `.scalar` every question is one row per option; the answer's
    /// timing is their sum.
    ///
    /// On an engine that cannot rewind into a state (a recurrent hybrid on the sequential
    /// engine), the first question on a state runs the state's shared prefix on its own and
    /// checkpoints after it, so the next question on that state returns there instead of
    /// replaying from the start. At one token per step that is the same number of steps. The
    /// question's timing includes the prefix pass, and its reused tokens are what the pass
    /// itself reused.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        try DecisionPrompt.validate(question, maxOptions: maxOptions)
        guard let prefix = try await checkpointState(state) else {
            return try await answerQuestion(state, question)
        }
        let answer = try await answerQuestion(state, question)
        let processed = prefix.processedTokens + answer.timing.processedTokens
        let timing = Decision.Timing(
            promptTokens: answer.timing.promptTokens,
            reusedTokens: max(0, answer.timing.promptTokens - processed),
            seconds: prefix.seconds + answer.timing.seconds)
        lastTiming = timing
        return Decision.Answer(value: answer.value, timing: timing, abstain: answer.abstain)
    }

    /// Answers one validated question on one state.
    private func answerQuestion(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        switch format {
        case .chat:
            let rendered = try DecisionPrompt.render(
                state: state, question: question, letters: labels, numbers: numbers, tokenizer: runtime.tokenizer)
            let (probabilities, timing) = try await readout(rendered, temperature: temperature(for: question))
            return DecisionPrompt.answer(for: question, probabilities: probabilities, timing: timing)
        case .sharedState:
            let rendered = try SharedStatePrompt.render(
                state: state, question: question, tokenizer: runtime.tokenizer)
            let (letterOrder, timing) = try await readout(rendered, temperature: temperature(for: question))
            return DecisionPrompt.answer(
                for: question, probabilities: SharedStatePrompt.probabilities(kitOrder: letterOrder, for: question),
                timing: timing)
        case .decisionFunction:
            let rendered = try DecisionFunctionPrompt.render(
                state: state, question: question, tokenizer: runtime.tokenizer)
            let (letterOrder, timing) = try await readout(rendered, temperature: temperature(for: question))
            return DecisionPrompt.answer(
                for: question, probabilities: DecisionFunctionPrompt.probabilities(kitOrder: letterOrder, for: question),
                timing: timing)
        case .letterList:
            guard let layout = letterLayout else { throw DecisionError.noLogits }
            let rendered = try LetterListPrompt.render(state: state, question: question, tokenizer: runtime.tokenizer)
            let (letterOrder, timing) = try await readout(rendered, temperature: temperature(for: question))
            return DecisionPrompt.answer(
                for: question,
                probabilities: LetterListPrompt.probabilities(kitOrder: letterOrder, for: question, layout: layout),
                timing: timing)
        case .scalar:
            // One row per option; the head's one logit per row, softmaxed across the rows.
            guard let layout = scalarLayout else { throw DecisionError.noLogits }
            var scalars: [Double] = []
            var total = Decision.Timing(promptTokens: 0, reusedTokens: 0, seconds: 0)
            for tokens in ScalarPrompt.render(state: state, question: question, layout: layout, tokenizer: runtime.tokenizer) {
                let (logits, timing) = try await score(tokens)
                guard let scalar = logits.first else { throw DecisionError.noLogits }
                scalars.append(Double(scalar))
                total = Decision.Timing(
                    promptTokens: total.promptTokens + timing.promptTokens,
                    reusedTokens: total.reusedTokens + timing.reusedTokens,
                    seconds: total.seconds + timing.seconds)
            }
            let p = DecisionPrompt.probabilities(logits: scalars, temperature: temperature(for: question))
            return DecisionPrompt.answer(
                for: question, probabilities: ScalarPrompt.probabilities(kitOrder: p, for: question), timing: total)
        case .slot:
            guard let layout = slotLayout, let encoder = slotEncoder else { throw DecisionError.noLogits }
            let rendered = try SlotPrompt.render(state: state, question: question, encoder: encoder)
            let (logits, timing) = try await score(rendered.tokens)
            guard logits.count >= layout.slots else { throw DecisionError.noLogits }
            let (probabilities, abstain) = SlotPrompt.readout(
                logits: logits.map(Double.init), options: rendered.slots.count,
                temperature: temperature(for: question), layout: layout)
            return DecisionPrompt.answer(
                for: question, probabilities: probabilities, timing: timing, abstain: abstain)
        case .decider:
            var fit: [Double] = []
            var last: [Double] = []
            var total = Decision.Timing(promptTokens: 0, reusedTokens: 0, seconds: 0)
            for row in DeciderPrompt.rows(for: question) {
                let rendered = try DeciderPrompt.render(state: state, row: row, labels: labels, tokenizer: runtime.tokenizer)
                let (probabilities, timing) = try await readout(rendered, temperature: temperature(for: question))
                last = probabilities
                fit.append(probabilities[1])
                total = Decision.Timing(
                    promptTokens: total.promptTokens + timing.promptTokens,
                    reusedTokens: total.reusedTokens + timing.reusedTokens,
                    seconds: total.seconds + timing.seconds)
            }
            if case .score = question.kind {
                return DecisionPrompt.answer(
                    for: question, probabilities: DeciderPrompt.combine(fit: fit), timing: total, fit: fit)
            }
            return DecisionPrompt.answer(for: question, probabilities: last, timing: total)
        case .encoder:
            // One forward pass per question; the options read at their markers.
            guard case .encoder(let encoder) = backend else { throw DecisionError.noLogits }
            let answer = try await encoder.decide(
                state, question, temperature: temperature(for: question), sharePrefix: configuration.sharePrefix)
            lastTiming = answer.timing
            return answer
        }
    }

    /// Scores one rendered row and reads the option probabilities at its answer slot, at the
    /// question type's temperature.
    private func readout(
        _ rendered: DecisionPrompt.Rendered, temperature: Double
    ) async throws -> ([Double], Decision.Timing) {
        let (logits, timing) = try await score(rendered.tokens)
        let slotLogits = rendered.slots.map { Double(logits[Int($0)]) }
        return (DecisionPrompt.probabilities(logits: slotLogits, temperature: temperature), timing)
    }

    /// Several questions on one state, answered in order; the state is prefilled once and
    /// each question rewinds to it.
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

    /// Runs the part of the prompt every question on `state` shares, so the first decision
    /// pays only for its question. Returns a handle whose `decide` calls score against it.
    public func prefill(_ state: String) async throws -> PrefilledState {
        if format == .encoder {
            // An encoder shares only the state's tokens between questions: tokenized once here.
            guard case .encoder(let encoder) = backend else { throw DecisionError.noLogits }
            let (tokens, timing) = await encoder.prefill(state, sharePrefix: configuration.sharePrefix)
            lastTiming = timing
            return PrefilledState(state: state, tokens: tokens, timing: timing, decider: self)
        }
        let prefix = try statePrefix(state)
        let timing = try await scorePrefix(prefix)
        return PrefilledState(state: state, tokens: prefix.count, timing: timing, decider: self)
    }

    /// The part of the prompt every question on `state` shares, in a language format.
    private func statePrefix(_ state: String) throws -> [Int32] {
        switch format {
        case .chat: return try DecisionPrompt.statePrefix(state: state, tokenizer: runtime.tokenizer)
        case .decider: return DeciderPrompt.contextTokens(state: state, tokenizer: runtime.tokenizer)
        case .slot:
            guard let encoder = slotEncoder else { throw DecisionError.noLogits }
            return try SlotPrompt.contextTokens(state: state, encoder: encoder)
        case .sharedState: return try SharedStatePrompt.statePrefix(state: state, tokenizer: runtime.tokenizer)
        case .decisionFunction: return DecisionFunctionPrompt.statePrefix(state: state, tokenizer: runtime.tokenizer)
        case .scalar:
            guard let layout = scalarLayout else { throw DecisionError.noLogits }
            return ScalarPrompt.statePrefix(state: state, layout: layout, tokenizer: runtime.tokenizer)
        case .letterList: return try LetterListPrompt.statePrefix(state: state, tokenizer: runtime.tokenizer)
        case .encoder: throw DecisionError.noLogits  // an encoder bundle has no language prompt
        }
    }

    /// Runs a state prefix and, on an engine that cannot rewind into it
    /// (`InferenceEngine.supportsCheckpoint`), checkpoints the engine after it; the timing
    /// covers both. A prefix the checkpoint already holds costs nothing.
    private func scorePrefix(_ prefix: [Int32]) async throws -> Decision.Timing {
        let engine = runtime.engine
        let checkpoints = configuration.sharePrefix && engine.supportsCheckpoint
        if checkpoints, !prefix.isEmpty, prefix == checkpointed {
            let timing = Decision.Timing(promptTokens: prefix.count, reusedTokens: prefix.count, seconds: 0)
            lastTiming = timing
            return timing
        }
        let (_, timing) = try await score(prefix, includeLogits: false)
        guard checkpoints else { return timing }
        let start = SuspendingClock.now
        try await engine.checkpoint()
        checkpointed = prefix
        let total = Decision.Timing(
            promptTokens: timing.promptTokens, reusedTokens: timing.reusedTokens,
            seconds: timing.seconds + ProcessStats.seconds(from: start, to: .now))
        lastTiming = total
        return total
    }

    /// Before the first question on a new state, on an engine that checkpoints: runs the
    /// state's prefix and checkpoints after it. Returns that pass's timing; nil when nothing ran.
    private func checkpointState(_ state: String) async throws -> Decision.Timing? {
        guard format != .encoder, configuration.sharePrefix, runtime.engine.supportsCheckpoint else { return nil }
        let prefix = try statePrefix(state)
        guard !prefix.isEmpty, prefix != checkpointed else { return nil }
        return try await scorePrefix(prefix)
    }

    /// The exact token sequences a question on a state is scored as — one per row, with the
    /// answer-slot token per option — for checking the rendering against a reference token
    /// file without a decision. One row, except a score question under `.decider` (one row
    /// per level) and every question under `.scalar` (one row per option, no slot tokens).
    nonisolated public func promptRows(
        _ state: String, _ question: Decision.Question
    ) throws -> [(tokens: [Int32], slots: [Int32])] {
        try DecisionPrompt.validate(question, maxOptions: maxOptions)
        switch format {
        case .chat:
            let rendered = try DecisionPrompt.render(
                state: state, question: question, letters: labels, numbers: numbers, tokenizer: runtime.tokenizer)
            return [(rendered.tokens, rendered.slots)]
        case .decider:
            return try DeciderPrompt.rows(for: question).map { row in
                let rendered = try DeciderPrompt.render(state: state, row: row, labels: labels, tokenizer: runtime.tokenizer)
                return (rendered.tokens, rendered.slots)
            }
        case .slot:
            guard let encoder = slotEncoder else { throw DecisionError.noLogits }
            let rendered = try SlotPrompt.render(state: state, question: question, encoder: encoder)
            return [(rendered.tokens, rendered.slots)]
        case .sharedState:
            let rendered = try SharedStatePrompt.render(state: state, question: question, tokenizer: runtime.tokenizer)
            return [(rendered.tokens, rendered.slots)]
        case .decisionFunction:
            let rendered = try DecisionFunctionPrompt.render(state: state, question: question, tokenizer: runtime.tokenizer)
            return [(rendered.tokens, rendered.slots)]
        case .scalar:
            guard let layout = scalarLayout else { throw DecisionError.noLogits }
            return ScalarPrompt.render(state: state, question: question, layout: layout, tokenizer: runtime.tokenizer)
                .map { ($0, []) }
        case .letterList:
            let rendered = try LetterListPrompt.render(state: state, question: question, tokenizer: runtime.tokenizer)
            return [(rendered.tokens, rendered.slots)]
        case .encoder:
            // One row; the slots are the option markers' positions in it.
            guard case .encoder(let encoder) = backend else { throw DecisionError.noLogits }
            let rendered = try encoder.prompt.render(state: state, question: question)
            return [(rendered.tokens, rendered.markers)]
        }
    }

    /// Drops the cached prompt: the next decision prefills from scratch.
    public func reset() async throws {
        kvTokens = []
        checkpointed = []
        switch backend {
        case .language(let runtime): try await runtime.engine.reset()
        case .encoder(let encoder): await encoder.reset()
        }
    }

    // MARK: - Engine

    /// Feeds `tokens` and returns the logits at the last position. Rewinds to the longest
    /// prefix shared with the previous prompt first (unless sharing is off), so only the
    /// tail is processed.
    private func score(
        _ tokens: [Int32], includeLogits: Bool = true
    ) async throws -> ([LogitsScalarType], Decision.Timing) {
        guard tokens.count < maxContextLength else {
            throw DecisionError.promptTooLong(tokens: tokens.count, max: maxContextLength - 1)
        }
        let engine = runtime.engine
        let start = SuspendingClock.now
        let wanted = configuration.sharePrefix
            ? min(
                DecisionPrompt.commonPrefixLength(tokens, kvTokens),
                max(0, tokens.count - 1),
                engine.processedTokenCount)
            : 0
        // A mirror in an unknown state must not be trusted: clear it until the call lands.
        kvTokens = []
        let held = checkpointed
        checkpointed = []
        let kept = try await engine.rewind(to: wanted)
        let stream = try await engine.generate(
            with: tokens,
            samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1, includeLogits: includeLogits))
        var logits: [LogitsScalarType]? = nil
        var sawOutput = false
        for try await output in stream {
            sawOutput = true
            logits = output.logits
            break
        }
        guard sawOutput, let logits = includeLogits ? logits : (logits ?? []) else {
            throw DecisionError.noLogits
        }
        // The engine consumed exactly the prompt: the one sampled token was never fed back.
        kvTokens = tokens
        // A rewind that kept the checkpointed prefix kept the checkpoint; one below it, or a
        // full reset, discarded it.
        if kept >= held.count { checkpointed = held }
        let timing = Decision.Timing(
            promptTokens: tokens.count, reusedTokens: kept,
            seconds: ProcessStats.seconds(from: start, to: .now))
        lastTiming = timing
        return (logits, timing)
    }
}

/// A state whose shared prompt prefix the engine already holds. Decisions made through it
/// pay for their question only.
public struct PrefilledState: Sendable {
    public let state: String
    /// Tokens of the shared prefix the engine holds.
    public let tokens: Int
    /// What the prefill cost.
    public let timing: Decision.Timing
    let decider: TypedDecisions

    public func decide(_ question: Decision.Question) async throws -> Decision.Answer {
        try await decider.decide(state, question)
    }

    public func decide(_ questions: [Decision.Question]) async throws -> [Decision.Answer] {
        try await decider.decide(state, questions)
    }

    public func decide(_ questions: [String: Decision.Question]) async throws -> [String: Decision.Answer] {
        try await decider.decide(state, questions)
    }
}
