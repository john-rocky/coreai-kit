// KitTextNormalizer.swift — S1-mini by Superwhisper behind one `normalize(_:)` call: a raw
// speech-to-text transcript in, written text out. Fillers dropped, false starts resolved to
// whatever the speaker landed on, punctuation and capitalization applied, and spoken numbers,
// dates, times, currency and email addresses rendered the way a person would type them.
//
//   let tidier = try await KitTextNormalizer(catalog: "s1-mini")
//   let text = try await tidier.normalize(rawTranscript, styling: .semiFormal)
//
// This is NOT a chat model and it does not ride `ChatSession`, for two reasons that are both
// properties of the model rather than preferences:
//
//   • **The system prompt is part of the trained input format.** The upstream card is explicit
//     that changing its wording makes the model hallucinate or emit garbled output, and the
//     op layer's shared chat path deliberately sends an EMPTY system prompt so every text op
//     can share one loaded model. That is outside the format.
//   • **`enable_thinking=False` is mandatory.** Leave thinking on and the model emits an empty
//     `<think>` block and stops — every call returns "", which looks like a working pipeline
//     producing nothing. `promptTokens` renders the closed-think tail itself and checks for it
//     rather than trusting the flag to have been honoured.
//
// And the empty string is a CORRECT answer here: filler-only input ("um") is supposed to
// normalize to nothing. Anything that retries on empty output burns a second generation on
// every legitimately empty one.
//
// **Chunking is the substance of this type, not a convenience.** On iOS the shipped engine
// caps a growing KV cache at 1024 tokens and truncates generation to
// `1024 - processed - prompt.count` (`CoreAIPipelinedEngine`, guarding an on-device compiler
// bug at seq ≥ 2048), so a whole meeting transcript does not merely run slowly — it stops
// mid-sentence. Measured on iPhone 17 Pro: a 611-token transcript whose Mac rewrite runs 603
// tokens produced 413 tokens, every one token-identical to the Mac, and stopped at absolute
// position exactly 1024. The rewrite runs roughly as long as its input, so the input is cut to
// `chunkTokens` (450 by default, inside the card's ~450–500 recommendation) and the pieces are
// stitched back together.

import CoreAILanguageModels
import Foundation
import Tokenizers

/// Register the rewrite is written in — S1-mini's `Styling` control axis.
public enum TranscriptStyling: String, Sendable, CaseIterable {
    /// Leaves the speaker's own wording and casing alone; still drops fillers.
    case casual = "casual"
    /// Contractions kept, sentences and capitalization applied.
    case semiCasual = "semi-casual"
    /// The default: full sentences, contractions kept.
    case semiFormal = "semi-formal"
    /// Contractions expanded ("I am", "cannot"), written register.
    case formal = "formal"
}

/// Shape of the rewrite — S1-mini's `Structure` control axis.
public enum TranscriptStructure: String, Sendable, CaseIterable {
    /// Paragraphs of running text.
    case prose
    /// Enumerations in the speech become a Markdown bullet list.
    case lists
}

/// What the text is for — S1-mini's `Context` control axis.
public enum TranscriptContext: String, Sendable, CaseIterable {
    /// Plain written text.
    case general
    /// Email shape: greeting, body and sign-off broken into their own blocks.
    case email
}

public enum KitTextNormalizerError: Error, LocalizedError, Sendable {
    /// A single unbreakable run of text is longer than the model's context on this platform.
    case chunkTooLong(promptTokens: Int, contextTokens: Int)

    public var errorDescription: String? {
        switch self {
        case .chunkTooLong(let prompt, let context):
            return "A \(prompt)-token chunk does not fit this platform's \(context)-token "
                + "context. Lower `chunkTokens`, or split the transcript before calling."
        }
    }
}

/// S1-mini by Superwhisper — an ASR text normalizer, one `normalize(_:)` call.
///
/// An actor because a chunked rewrite is many generations over one engine: two concurrent
/// callers sharing a `KitTextNormalizer` would interleave their prompts into the same KV cache.
public actor KitTextNormalizer {
    /// The catalog id this loads by default.
    public static let defaultCatalogID = "s1-mini"

    /// The system prompt S1-mini was trained with, verbatim. It is part of the input format,
    /// not an instruction to tune: the upstream card says rewording it makes the model
    /// hallucinate or emit garbled output.
    public static let systemPrompt =
        "You are a text normalizer for speech-to-text transcripts. The input begins with a "
        + "control line specifying the styling, structure, and context settings; clean the "
        + "transcript to match those settings and output only the cleaned text."

    /// Input tokens per chunk. The card recommends inputs up to ~1000 tokens, but on iPhone a
    /// ~1000-token prompt leaves no room for the rewrite under the engine's 1024-token cap, so
    /// the shipped default is the ~450–500 the device evidence calls for. Same on both
    /// platforms on purpose: the same transcript should not normalize differently on a Mac.
    public static let defaultChunkTokens = 450

    /// What the engine will hold. iOS: `CoreAIPipelinedEngine` caps a growing KV cache at 1024
    /// and throws `contextLengthExceeded` for a prompt that leaves no budget. macOS: the
    /// bundle's own declared context.
    static var contextTokens: Int {
        #if os(iOS)
        return 1024
        #else
        return 4096
        #endif
    }

    private let runtime: ModelRuntime

    /// The catalog id this normalizer was loaded from.
    public let id: String

    /// Loads by catalog id — the id shown on the model's card. Downloads on first use
    /// (progress via `downloadProgress`), then loads from the local cache.
    public init(
        catalog id: String = KitTextNormalizer.defaultCatalogID,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .textNormalizer)
        guard let model = entry.modelID else {
            throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
        }
        let url = try await store.download(model, progress: downloadProgress)
        self.runtime = try await ModelRuntime(
            bundleAt: url, engineVariant: EngineVariant(catalogHint: entry.engine))
        self.id = id
    }

    /// Loads a local bundle directory (metadata.json + *.aimodel/ + tokenizer/).
    public init(bundleAt url: URL, engineVariant: EngineVariant = .pipelined) async throws {
        self.runtime = try await ModelRuntime(bundleAt: url, engineVariant: engineVariant)
        self.id = url.lastPathComponent
    }

    /// Display name from the bundle metadata.
    public var modelName: String { runtime.modelName }

    // MARK: - Normalize

    /// Rewrite a raw transcript as written text.
    ///
    /// The three arguments are the model's own control axes and are the whole steering surface
    /// — there is no free-text instruction to give it. Long input is chunked to `chunkTokens`
    /// at word boundaries and the rewrites are stitched; `onPartial` is called with the running
    /// result after each chunk, so a UI can fill in as it goes.
    ///
    /// Returns the **empty string** for input that is nothing but filler. That is the model
    /// working, not a failure.
    ///
    /// English only: S1-mini is trained on English transcripts and has no language control.
    public func normalize(
        _ transcript: String,
        styling: TranscriptStyling = .semiFormal,
        structure: TranscriptStructure = .prose,
        context: TranscriptContext = .general,
        chunkTokens: Int = KitTextNormalizer.defaultChunkTokens,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        let source = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !source.isEmpty else { return "" }

        let joiner = Self.joiner(structure: structure, context: context)
        var parts: [String] = []
        for piece in chunks(of: source, budget: chunkTokens) {
            let rewritten = try await rewrite(
                piece, styling: styling, structure: structure, context: context)
            guard !rewritten.isEmpty else { continue }
            parts.append(rewritten)
            onPartial?(parts.joined(separator: joiner))
        }
        return parts.joined(separator: joiner)
    }

    /// One chunk through the model, greedy, EOS-terminated.
    private func rewrite(
        _ piece: String, styling: TranscriptStyling, structure: TranscriptStructure,
        context: TranscriptContext
    ) async throws -> String {
        let prompt = try Self.promptTokens(
            for: piece, styling: styling, structure: structure, context: context,
            tokenizer: runtime.tokenizer)
        // Reserve one token: the engine's own cap is `context - processed - prompt.count`, and
        // a generation that fills the cache to the last slot has nowhere to put its EOS.
        let headroom = Self.contextTokens - prompt.count - 1
        guard headroom > 0 else {
            throw KitTextNormalizerError.chunkTooLong(
                promptTokens: prompt.count, contextTokens: Self.contextTokens)
        }
        // A clean rewrite runs about as long as its input (measured: 611 in, 603 out). Twice
        // that plus a floor is generous headroom for one that expands; the cap is what stops a
        // degenerate repeat from running to the end of the context.
        let inputTokens = runtime.tokenizer.encode(text: piece).count
        let maxTokens = max(1, min(headroom, 2 * inputTokens + 64))

        try await runtime.engine.reset()
        let eos = runtime.tokenizer.eosTokenId
        let turnEnd = runtime.tokenizer.convertTokenToId("<|im_end|>")
        var ids: [Int] = []
        let stream = try await runtime.engine.generate(
            with: prompt, samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        for try await output in stream {
            let id = Int(output.tokenId)
            if id == eos || id == turnEnd { break }
            ids.append(id)
        }
        return runtime.tokenizer.decode(tokens: ids)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Prompt

    /// The card's exact input format: system prompt, then a control line and the raw transcript
    /// as one user turn, then an assistant turn whose thinking block is already closed.
    ///
    /// Static and tokenizer-in so the rendering can be checked against the token sequence the
    /// device gate ran, without loading 796 MB of weights.
    static func promptTokens(
        for transcript: String, styling: TranscriptStyling, structure: TranscriptStructure,
        context: TranscriptContext, tokenizer: any Tokenizer
    ) throws -> [Int32] {
        let control =
            "[Styling: \(styling.rawValue)] [Structure: \(structure.rawValue)] "
            + "[Context: \(context.rawValue)]"
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": "\(control)\n\(transcript)"],
        ]
        var ids = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: nil, addGenerationPrompt: true,
            truncation: false, maxLength: nil, tools: nil,
            additionalContext: ["enable_thinking": false]
        ).map(Int32.init)

        // `enable_thinking=False` is what appends `<think>\n\n</think>\n\n`. If the template
        // ignored the flag the model would open its own think block, spend the budget in it,
        // and return "" — the failure that looks like a working pipeline. Append the tail
        // ourselves rather than trust it.
        let tail = closedThink(tokenizer)
        if !tail.isEmpty, !ids.suffix(tail.count).elementsEqual(tail) { ids += tail }
        return ids
    }

    /// `<think>\n\n</think>\n\n` as token ids, or empty if this tokenizer has no think markers.
    private static func closedThink(_ tokenizer: any Tokenizer) -> [Int32] {
        guard let open = tokenizer.convertTokenToId("<think>"),
            let close = tokenizer.convertTokenToId("</think>")
        else { return [] }
        let blank = tokenizer.encode(text: "\n\n").map(Int32.init)
        guard !blank.isEmpty else { return [] }
        return [Int32(open)] + blank + [Int32(close)] + blank
    }

    // MARK: - Chunking

    /// How the rewritten chunks are joined back together. Each chunk comes back as finished
    /// sentences, so prose only needs a space; the block structures keep their blank line.
    static func joiner(structure: TranscriptStructure, context: TranscriptContext) -> String {
        if structure == .lists { return "\n" }
        return context == .email ? "\n\n" : " "
    }

    /// Split a transcript into pieces of at most `budget` tokens, cut at word boundaries.
    func chunks(of transcript: String, budget: Int) -> [String] {
        let tokenizer = runtime.tokenizer
        let ids = tokenizer.encode(text: transcript)
        let ranges = Self.chunkRanges(count: ids.count, budget: budget) { index in
            // A token that begins a word decodes with its leading space; cutting in front of
            // one is the only cut that cannot split a word in half.
            tokenizer.decode(tokens: [ids[index]]).first.map(\.isWhitespace) ?? false
        }
        if ranges.count <= 1 { return [transcript] }
        return ranges.map {
            tokenizer.decode(tokens: Array(ids[$0]))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        .filter { !$0.isEmpty }
    }

    /// Token index ranges for `count` tokens cut into pieces of at most `budget`.
    ///
    /// The pieces are **evened out** rather than packed: a 500-token transcript at a 450-token
    /// budget becomes 250 + 250, not 450 + 50. A 50-token tail is a chunk the model rewrites
    /// with almost nothing around it, and it reads like one.
    ///
    /// `isBoundary(i)` says whether token `i` starts a new word. Split-out and given plain
    /// arguments so the arithmetic is testable without a tokenizer or a model.
    static func chunkRanges(
        count: Int, budget: Int, isBoundary: (Int) -> Bool
    ) -> [Range<Int>] {
        guard count > 0 else { return [] }
        guard budget > 0, count > budget else { return [0..<count] }
        let pieces = (count + budget - 1) / budget
        let even = (count + pieces - 1) / pieces

        var ranges: [Range<Int>] = []
        var start = 0
        while start < count {
            let hard = min(start + even, count)
            if hard >= count { ranges.append(start..<count); break }
            var end = hard
            while end > start + 1, !isBoundary(end) { end -= 1 }
            // One word longer than the budget: cut through it rather than loop forever.
            if end <= start { end = hard }
            ranges.append(start..<end)
            start = end
        }
        return ranges
    }
}
