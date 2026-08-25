// QuickStart.swift — the take-home core of this runner: a raw speech-to-text transcript in,
// written text out, one typed function, no UI. Both the GUI app and the CLI call exactly this
// function (the view model is a display shell, `CLI/main.swift` an argument shell). Want
// dictation cleanup in your own app? This file is the part you copy; the model card's 💻
// snippet is the marked block below.

import CoreAIKit
import Foundation

/// Rewrite a raw ASR transcript as written text with S1-mini by Superwhisper: fillers dropped,
/// false starts resolved to whatever the speaker landed on, punctuation and capitalization
/// applied, and spoken numbers, dates, times, currency and email addresses written out.
///
/// The three axes are the model's own trained controls — there is no free-text instruction.
/// English only. Filler-only input returns the **empty string**: that is the model working.
/// First use downloads the model (progress via `downloadProgress`), later runs load from cache.
func tidy(
    transcript: String,
    model id: String = "s1-mini",
    styling: TranscriptStyling = .semiFormal,
    structure: TranscriptStructure = .prose,
    context: TranscriptContext = .general,
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    // CARD-SNIPPET-BEGIN
    let tidier = try await KitTextNormalizer(catalog: id, downloadProgress: downloadProgress)
    // Long input is cut at word boundaries into ~450-token chunks and the rewrites stitched:
    // on iPhone the engine caps prompt + generated at 1024 tokens, so a whole meeting
    // transcript passed in one call would stop mid-sentence.
    return try await tidier.normalize(transcript, styling: styling, structure: structure, context: context)
    // CARD-SNIPPET-END
}
