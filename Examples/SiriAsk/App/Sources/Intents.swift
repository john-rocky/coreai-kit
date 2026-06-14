// Intents — the single App Intent Siri invokes: ask the on-device Gemma 4 model anything.
//
// This is the ask-anything rework. One read-only intent: Siri prompts for a free-text question,
// `perform()` runs a TOOL-LESS `LanguageModelSession` on our Gemma 4 model, and the answer comes
// back as an `IntentDialog`. The model has no tool to send/post/delete and no retrieval feeds it
// untrusted data — so the WWDC 347 "act/communicate" leg is absent by construction.
//
// IMPORTANT (device finding S1): the Siri crash is a MEMORY-limit issue, not a GPU one (GPU
// inference runs fine backgrounded — proven by a 2B VLM under Visual Intelligence). A backgrounded
// Siri-intent process gets a LOWER jetsam limit than a foreground app; the increased-memory
// entitlement's ~6.44 GB applies in the FOREGROUND. The old `_tbl` runtime held the ~2.36 GB PLE
// tables as OWNED DIRTY buffers (peak ~4.4 GB) → fits foreground (G1 passed) but killed backgrounded.
//
// FIX: `GemmaRuntime` now maps the PLE tables copy-on-write (clean, evictable pages) — footprint
// drops to ~VLM class and the cold load is faster (lazy fault, no 2.36 GB upfront read). So we go
// for HANDS-FREE: `openAppWhenRun = false` — Siri answers without foregrounding the app. Warm the
// app once first (so `ModelHost.shared` is resident) for the snappiest call.
//
// FALLBACK: if the device still crashes (jetsam) or times out (cold separate process) on the Siri
// call, set `openAppWhenRun = true` — foregrounding restores the full 6.44 GB limit + warm reuse
// (now also lighter thanks to the COW mapping); the trade-off is the app comes to front.

import AppIntents
import Foundation

/// "Hey Siri, Ask Gemma." Siri prompts for the question, brings the app to the foreground, and the
/// on-device Gemma 4 E2B answers it from its own knowledge.
struct AskGemmaIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask Gemma"
    static let description = IntentDescription(
        "Ask your on-device Gemma 4 model anything.", categoryName: "Ask")

    /// Hands-free: answer without foregrounding the app. Viable now that `GemmaRuntime` maps the
    /// PLE tables copy-on-write (low, evictable footprint). Flip to `true` if the device still
    /// crashes/times out on the Siri call (see the file header's FALLBACK note).
    static let openAppWhenRun = false

    @Parameter(title: "Question", requestValueDialog: "What do you want to ask Gemma?")
    var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        // The app is foregrounded (openAppWhenRun) before this runs, so the Metal engine is allowed
        // and ModelHost.shared is the warm, resident instance.
        let answer = try await ModelHost.shared.ask(question: question)
        return .result(dialog: IntentDialog("\(answer)"))
    }
}
