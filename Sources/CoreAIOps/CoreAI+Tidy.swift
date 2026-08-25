// CoreAI+Tidy.swift — the dictation post-processor: a raw speech-to-text transcript in,
// written text out.
//
// ```swift
// let raw   = try await CoreAI.transcribe(voiceMemoURL)          // Apple's transcriber
// let clean = try await CoreAI.tidyTranscript(raw)               // S1-mini by Superwhisper
// ```
//
// **Why this is its own op and not a flag on `proofread`.** `proofread` is a generic prompt
// over the 4B default, contracted as "keep the wording as close to the original as possible".
// S1-mini *deletes* fillers and false starts, resolves self-corrections to whatever the
// speaker landed on, rewrites spoken numbers and dates in written form, is English-only, and
// returns the **empty string** for filler-only input. No caller of `proofread` expects any of
// that, so overloading it would change an op's contract under everyone using it.
//
// The three arguments are the model's own trained control axes — there is no free-text
// instruction, and the styling axis is the difference between "hmm im gonna be late" surviving
// as it was spoken and "I am going to be late."

import CoreAIKit
import Foundation

extension CoreAI {
    /// Default ASR text normalizer: S1-mini by Superwhisper, the only `textNormalizer` in the
    /// catalog. 796 MB, iPhone-verified.
    public static let defaultNormalizerModel = "s1-mini"

    /// Raw transcript → written text: fillers dropped, false starts resolved, punctuation and
    /// capitalization applied, and spoken numbers, dates, times, currency and email addresses
    /// rendered the way a person would type them.
    ///
    /// ```swift
    /// try await CoreAI.tidyTranscript(
    ///     "so um i need to like send the the report by uh friday no wait make that thursday")
    /// // "I need to send the report by Thursday."
    /// ```
    ///
    /// **English only** — the model has no language control, and there is no other
    /// `textNormalizer` in the catalog to switch to.
    ///
    /// **Long input is chunked and stitched.** The rewrite runs about as long as its input and
    /// iOS caps prompt + generated at 1024 tokens, so a transcript past ~450 tokens is cut at
    /// word boundaries and rewritten piece by piece. `onPartial` receives the running result
    /// after each piece.
    ///
    /// Returns the **empty string** when the input is nothing but filler. That is the model
    /// doing its job, not a failure — do not retry it.
    ///
    /// First use downloads and loads the model (cached afterwards); calls on the same model
    /// serialize behind each other.
    public static func tidyTranscript(
        _ transcript: String,
        styling: TranscriptStyling = .semiFormal,
        structure: TranscriptStructure = .prose,
        context: TranscriptContext = .general,
        options: OpOptions = OpOptions(),
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> String {
        try await TidyOpModels.shared.tidy(
            catalog: options.model ?? defaultNormalizerModel, transcript: transcript,
            styling: styling, structure: structure, context: context, onPartial: onPartial)
    }
}

/// Process-wide cache of loaded normalizers, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached, and calls
/// on one engine serialize behind each other.
actor TidyOpModels {
    static let shared = TidyOpModels()

    private let normalizers = ResidentCache<KitTextNormalizer>(kind: ResidentKind.normalizer)
    private var turns: [String: Task<Void, Never>] = [:]

    func tidy(
        catalog id: String, transcript: String, styling: TranscriptStyling,
        structure: TranscriptStructure, context: TranscriptContext,
        onPartial: (@Sendable (String) -> Void)?
    ) async throws -> String {
        let normalizer = try await self.normalizer(catalog: id)
        let previous = turns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.normalizer, id) {
                try await normalizer.normalize(
                    transcript, styling: styling, structure: structure, context: context,
                    onPartial: onPartial)
            }
        }
        turns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func normalizer(catalog id: String) async throws -> KitTextNormalizer {
        try await normalizers.value(for: id) {
            try await KitTextNormalizer(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}
