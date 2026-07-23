// Intents — the single App Intent Siri invokes: ask the on-device Gemma 4 model anything.
//
// This is the ask-anything rework. One read-only intent: Siri prompts for a free-text question,
// `perform()` runs a TOOL-LESS `LanguageModelSession` on our Gemma 4 model, and the answer comes
// back as an `IntentDialog`. The model has no tool to send/post/delete and no retrieval feeds it
// untrusted data — so the WWDC 347 "act/communicate" leg is absent by construction.
//
// GOAL: hands-free in the BACKGROUND (`openAppWhenRun = false`) — "Hey Siri, Ask Gemma" answers
// without opening the app. iOS permits GPU work in a window (recently-foreground / not yet
// suspended), which is why it answers hands-free repeatedly. Two device realities are handled so
// that window is as wide as possible and missing it never crashes:
//
//   1) WIDEN THE WINDOW: hold a background-execution assertion (`beginBackgroundTask`) around the
//      call so iOS keeps the process running (and GPU-eligible) for the whole short generation
//      instead of suspending it mid-answer.
//   2) FAIL SOFT: if iOS still rejects GPU work (fully backgrounded), the engine errors; ModelHost
//      drops the wedged runtime (so the next ask rebuilds fresh) and we speak a short retry message
//      instead of crashing. (Previously a rejection fatalError'd the engine and killed the app.)
//
// Reliable always-on background GPU isn't guaranteed by iOS, but this makes the common case
// hands-free and the edge case graceful — no app foregrounding, no crash.

import AppIntents
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// "Hey Siri, Ask Gemma." Siri prompts for the question; the on-device Gemma 4 E2B answers it from
/// its own knowledge — hands-free, without opening the app.
struct AskGemmaIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask Gemma"
    static let description = IntentDescription(
        "Ask your on-device Gemma 4 model anything.", categoryName: "Ask")

    /// Hands-free: do NOT foreground the app. The GPU window is widened by the background-task
    /// assertion in perform(), and a rejection is handled gracefully there.
    static let openAppWhenRun = false

    @Parameter(title: "Question", requestValueDialog: "What do you want to ask Gemma?")
    var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        #if os(iOS)
        let app = UIApplication.shared
        let stateName: String
        switch app.applicationState {
        case .active: stateName = "active"
        case .inactive: stateName = "inactive"
        case .background: stateName = "background"
        @unknown default: stateName = "unknown"
        }
        let bgTask = app.beginBackgroundTask(withName: "AskGemma")
        defer { if bgTask != .invalid { app.endBackgroundTask(bgTask) } }
        #else
        let stateName = "macos"
        #endif

        do {
            let answer = try await ModelHost.shared.ask(question: question)
            return .result(dialog: IntentDialog("\(answer)"))
        } catch {
            // DIAGNOSTIC (temporary): speak the real error + app state so we can see exactly why the
            // Siri path fails (GPU rejection? load failure? something else?) instead of Siri's
            // generic "something went wrong". Revert to a friendly message once the cause is known.
            return .result(dialog: IntentDialog("Gemma failed in state \(stateName): \(error)"))
        }
    }
}
