// CoreAI.swift — anchored operations over local catalog models.
//
// Ops are the stable API; models are catalog entries resolved behind them. Anchored ops
// only — no free-generation API by design: each op fixes its own prompt contract and
// output shape, so the model behind it can change (catalog update, `options: .model(...)`)
// without breaking callers.
//
// ```swift
// let text = try await CoreAI.transcribe(voiceMemoURL)               // speech -> text
// let tldr = try await CoreAI.summarize(text, style: .bullets)
// let todo = try await CoreAI.extract(text, as: ActionItems.self)    // @Generable-typed
// let en   = try await CoreAI.translate(text, to: .english)
// let neat = try await CoreAI.proofread(text)
// ```
//
// The other ops extend `CoreAI` from sibling files: CoreAI+Image (caption / detect /
// read / upscale / estimateDepth), CoreAI+Audio (transcribeMeeting / describeAudio /
// compose / separate), CoreAI+Speech (speak), CoreAI+Search (search), CoreAI+Video
// (recognizeAction), CoreAI+Forecast (forecast), CoreAI+Redact (redact / extractEntities).
//
// Every op rides `ChatSession`: its incremental detokenizer survives the raw-byte
// tokens long CJK output can contain, which the FM executor's current decode hold does
// not (a long Japanese translation starves the event stream and the turn ends with no
// response). `extract` shapes the reply with the `@Generable` type's generation schema
// and parses it through `GeneratedContent(json:)` — measured against grammar-guided
// decoding on the sequential engine, the schema-shaped prompt on the 4B default is the
// configuration that held up (guided on 0.6B oscillates between merging and shredding
// items; 4B cannot load sequentially under the engine's 5s drain watchdog). Loaded
// models are cached per catalog id for the life of the process, and ops on the same
// model serialize behind each other.

import CoreAIKit
import Foundation
import FoundationModels

/// How `CoreAI.summarize` shapes its output.
public enum SummaryStyle: String, Sendable, CaseIterable {
    /// 2–3 plain sentences.
    case concise
    /// Markdown bullet list of the key points.
    case bullets
    /// A single sentence.
    case oneLine

    var instruction: String {
        switch self {
        case .concise: "Reply with a summary of 2-3 plain sentences."
        case .bullets: "Reply with a Markdown bullet list of the key points."
        case .oneLine: "Reply with a single-sentence summary."
        }
    }
}

/// Target language for `CoreAI.translate`. Use the presets or name any language:
/// `Language("Swahili")`.
public struct Language: Sendable, Hashable {
    /// English name of the language, as spelled in the prompt contract.
    public let name: String

    public init(_ name: String) { self.name = name }

    public static let english = Language("English")
    public static let japanese = Language("Japanese")
    public static let chinese = Language("Simplified Chinese")
    public static let korean = Language("Korean")
    public static let spanish = Language("Spanish")
    public static let french = Language("French")
    public static let german = Language("German")
}

/// Per-call knobs for an op. The default resolves the op's default model from the catalog;
/// `.model("qwen3-4b")` runs the same op on another catalog model of the same kind.
public struct OpOptions: Sendable {
    /// Catalog id of the model to run on (nil = the op's default).
    public var model: String?

    public init(model: String? = nil) { self.model = model }

    /// Run the op on this catalog model instead of the default.
    public static func model(_ id: String) -> OpOptions { OpOptions(model: id) }
}

/// Anchored operations over catalog models. The text ops live here, each a thin prompt
/// contract; the rest extend this type from the sibling `CoreAI+*` files.
public enum CoreAI {
    /// Default model for the free-text ops (summarize/translate/proofread). 4B is the
    /// quality floor at which every op holds up — 0.6B passes summarize but fails
    /// translate outright. Drop to `options: .model("qwen3-0.6b")` when speed matters
    /// more than fidelity.
    public static let defaultModel = "qwen3-4b"


    /// Text → short summary in the given style. First use downloads and loads the model
    /// (cached afterwards); later calls reuse the loaded engine.
    public static func summarize(
        _ text: String, style: SummaryStyle = .concise, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await OpModels.shared.respondText(
            catalog: options.model ?? defaultModel,
            prompt: """
                You summarize text. \(style.instruction) \
                Do not add preambles, commentary, or questions.

                Summarize the following text:

                \(text)
                """)
    }

    /// Text → typed value: the `@Generable` type's generation schema shapes the reply,
    /// and the framework parses it back into the type — same type, same `@Guide`
    /// descriptions as Apple's `respond(generating:)`.
    public static func extract<Value: Generable>(
        _ text: String, as type: Value.Type = Value.self, options: OpOptions = OpOptions()
    ) async throws -> Value {
        let raw = try await OpModels.shared.respondText(
            catalog: options.model ?? defaultModel,
            prompt: """
                You extract structured data from text as JSON. Reply with ONLY a JSON \
                object matching this schema — no code fences, no commentary. Fill every \
                field from the text; use the closest supported value when a field is \
                not stated verbatim.

                \(Value.generationSchema)

                Extract the requested data from the following text:

                \(text)
                """)
        let content = try GeneratedContent(json: Self.jsonSlice(of: raw))
        if let value = try? Value(content) { return value }
        // Models sometimes nest the object under the schema's root name — unwrap a
        // single-key wrapper and retry before surfacing the original mismatch.
        if case .structure(let properties, let orderedKeys) = content.kind,
            orderedKeys.count == 1, let inner = properties[orderedKeys[0]],
            let value = try? Value(inner)
        {
            return value
        }
        return try Value(content)
    }

    /// The first top-level JSON value in a reply — models wrap JSON in code fences or
    /// prose no matter how firmly told not to.
    static func jsonSlice(of text: String) -> String {
        guard let start = text.firstIndex(where: { $0 == "{" || $0 == "[" }) else {
            return text
        }
        let closer: Character = text[start] == "{" ? "}" : "]"
        guard let end = text.lastIndex(of: closer), end > start else { return text }
        return String(text[start...end])
    }

    /// Default speech-to-text model: multilingual and robust. Overridable per call, e.g.
    /// `options: .model("parakeet-tdt-0.6b-v3")` for faster English-family transcription.
    public static let defaultSpeechModel = "whisper-large-v3-turbo"

    /// Audio file → plain-text transcript. Any audio format the system decodes;
    /// resampled to 16 kHz mono internally. `language` nil = auto-detect.
    public static func transcribe(
        _ audioURL: URL, language: String? = nil, options: OpOptions = OpOptions()
    ) async throws -> String {
        let id = options.model ?? defaultSpeechModel
        let transcriber = try await OpModels.shared.transcriber(catalog: id)
        let samples = try AudioFile.pcm16kMono(audioURL)
        return try await withPinnedModel(ResidentKind.transcriber, id) {
            let transcription = try await transcriber.transcribe(
                samples: samples, language: language)
            return transcription.text.trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    /// Text → translation into the target language, preserving meaning, tone, and formatting.
    public static func translate(
        _ text: String, to language: Language, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await OpModels.shared.respondText(
            catalog: options.model ?? defaultModel,
            prompt: """
                You translate text into \(language.name). Preserve meaning, tone, and \
                formatting. Reply with only the translation — no preambles or notes.

                Translate the following text into \(language.name):

                \(text)
                """)
    }

    /// Text → corrected text: grammar, spelling, and punctuation fixed without changing
    /// meaning or language.
    public static func proofread(
        _ text: String, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await OpModels.shared.respondText(
            catalog: options.model ?? defaultModel,
            prompt: """
                You proofread text: fix grammar, spelling, and punctuation while keeping \
                the meaning, language, and wording as close to the original as possible. \
                Reply with only the corrected text.

                Proofread the following text:

                \(text)
                """)
    }
}

/// Process-wide cache of loaded op models, keyed by catalog id — ops are static calls, so
/// this is what keeps the second op call from reloading the weights. Concurrent first
/// calls share one load; a failed load is not cached (a later call retries); a model idle
/// long enough to be the least-recently-used one is dropped when the next load needs the
/// room (`ModelResidency`).
actor OpModels {
    static let shared = OpModels()

    private let chats = ResidentCache<ChatSession>(kind: ResidentKind.chat)
    private let transcribers = ResidentCache<KitTranscriber>(kind: ResidentKind.transcriber)
    private var chatTurns: [String: Task<Void, Never>] = [:]


    /// One stateless free-text turn: reset the cached `ChatSession`, send the op's full
    /// prompt (contract + input in one user message — the session's system prompt stays
    /// empty so every op shares one loaded model), return the trimmed response. Turns on
    /// the same model serialize behind each other so concurrent ops don't collide on the
    /// single-generation session.
    func respondText(catalog id: String, prompt: String) async throws -> String {
        let chat = try await self.chat(catalog: id)
        let previous = chatTurns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.chat, id) {
                // One retry: a thinking model occasionally spends the whole budget in the
                // think block and yields an empty answer — a fresh sample usually lands.
                for attempt in 0..<2 {
                    await chat.reset()
                    var text = try await chat.respond(to: prompt)
                    // A thinking model occasionally skips its opening <think> tag, so the
                    // deliberation leaks into the response ending in a bare closing tag —
                    // keep only what follows the last close.
                    if let close = text.range(of: "</think>", options: .backwards) {
                        text = String(text[close.upperBound...])
                    }
                    text = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !text.isEmpty || attempt == 1 { return text }
                }
                return ""
            }
        }
        chatTurns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func chat(catalog id: String) async throws -> ChatSession {
        try await chats.value(for: id) {
            var configuration = ChatSession.Configuration()
            configuration.temperature = 0.7  // greedy makes qwen3 thinking models loop
            // Thinking counts against this budget and routinely runs 1-2k tokens on a
            // hard input; a tight cap truncates the turn before the answer starts.
            configuration.maxResponseTokens = 4096
            return try await ChatSession(
                catalog: id, configuration: configuration,
                downloadProgress: OpDownloads.forward)
        }
    }

    func transcriber(catalog id: String) async throws -> KitTranscriber {
        try await transcribers.value(for: id) {
            try await KitTranscriber(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}
