// Intents — the single App Intent Siri invokes: ask the on-device Gemma 4 model anything.
//
// This is the ask-anything rework. One read-only intent: Siri prompts for a free-text question,
// `perform()` runs a TOOL-LESS `LanguageModelSession` on our Gemma 4 model, and the answer comes
// back as an `IntentDialog`. The model has no tool to send/post/delete and no retrieval feeds it
// untrusted data — so the WWDC 347 "act/communicate" leg is absent by construction.
//
// DEVICE FINDING (confirmed by the debugger + real usage): GPU submission is permitted while the app
// is FOREGROUND or just `.inactive` (e.g. a Siri overlay over a recently-used app) — in that window
// "Hey Siri, Ask Gemma" answers hands-free, repeatedly. It is NOT permitted once iOS has suspended
// the app to `.background`: the engine then gets
//   IOGPUMetalError: Insufficient Permission (to submit GPU work from background)
//   (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted)
// and fatalErrors in drain() → the app crashes and wedges. So the hands-free window is real but
// bounded by the app's lifecycle state.
//
// STRATEGY: keep hands-free (`openAppWhenRun = false`) for the working `.inactive`/foreground window,
// but GUARD the GPU call — if the app is fully `.background`, decline gracefully (speak a short "open
// me" dialog) instead of submitting GPU work that iOS will reject and the engine will crash on. This
// preserves the experience that worked while removing the crash. (The deeper fix is in the engine:
// it should surface a recoverable error on a rejected GPU submit instead of fatalError — see STATE
// FOR-ORCHESTRATOR. Bulletproof fallback if any crash recurs: set `openAppWhenRun = true` to always
// foreground.)

import AppIntents
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// "Hey Siri, Ask Gemma." Siri prompts for the question; the on-device Gemma 4 E2B answers it from
/// its own knowledge — hands-free while the app is foreground/recently-used.
struct AskGemmaIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask Gemma"
    static let description = IntentDescription(
        "Ask your on-device Gemma 4 model anything.", categoryName: "Ask")

    /// Hands-free: don't force the app to the foreground. GPU work is allowed in the foreground/
    /// `.inactive` window (the answer streams via Siri); the `.background` case is handled in
    /// perform() so it declines gracefully instead of crashing.
    static let openAppWhenRun = false

    @Parameter(title: "Question", requestValueDialog: "What do you want to ask Gemma?")
    var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        #if os(iOS)
        // iOS rejects GPU work from a fully-backgrounded process (and the engine fatalErrors on the
        // rejection). Only run the model when the app is foreground or `.inactive` (Siri overlay over
        // a recently-used app) — the window where it reliably answered. If suspended, ask the user to
        // reopen Gemma (which warms it) rather than submitting GPU work that would crash the app.
        if UIApplication.shared.applicationState == .background {
            return .result(dialog: IntentDialog(
                "Open Gemma so it stays loaded, then ask again."))
        }
        #endif
        let answer = try await ModelHost.shared.ask(question: question)
        return .result(dialog: IntentDialog("\(answer)"))
    }
}
