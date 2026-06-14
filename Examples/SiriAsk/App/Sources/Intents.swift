// Intents — the single App Intent Siri invokes: ask the on-device Gemma 4 model anything.
//
// This is the ask-anything rework. One read-only intent: Siri prompts for a free-text question,
// `perform()` runs a TOOL-LESS `LanguageModelSession` on our Gemma 4 model, and the answer comes
// back as an `IntentDialog`. The model has no tool to send/post/delete and no retrieval feeds it
// untrusted data — so the WWDC 347 "act/communicate" leg is absent by construction.
//
// DEVICE FINDING (confirmed by the debugger): a Siri intent that runs the GPU engine while the app
// is backgrounded is killed by iOS —
//   IOGPUMetalError: Insufficient Permission (to submit GPU work from background)
//   (kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted)
// and the engine then wedges (the failed generate leaves the tokenSequence Task stuck, so
// CoreAIPipelinedEngine.drain() spins and fatalErrors → the app crashes and won't recover until
// relaunch). So GPU inference CANNOT run hands-free in the background here. (The earlier "it worked
// backgrounded" was a fluke — the app was still foreground-active behind the Siri overlay at submit
// time; the VLM under Visual Intelligence runs with the VI UI on screen = foreground.)
//
// FIX: `openAppWhenRun = true` — the system foregrounds the app to run the intent, where GPU work is
// permitted (and the warm, resident `ModelHost.shared` is reused). The trade-off is that the app
// comes to front: "Hey Siri, Ask Gemma" opens the app and answers, rather than a pure voice overlay.
// Hands-free background GPU inference for a third-party Metal model isn't possible on current iOS.
// (`GemmaRuntime`'s copy-on-write PLE mapping still helps here: lower footprint + faster cold load
// inside Siri's intent budget.)

import AppIntents
import Foundation

/// "Hey Siri, Ask Gemma." Siri prompts for the question, brings the app to the foreground, and the
/// on-device Gemma 4 E2B answers it from its own knowledge.
struct AskGemmaIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask Gemma"
    static let description = IntentDescription(
        "Ask your on-device Gemma 4 model anything.", categoryName: "Ask")

    /// REQUIRED: foreground the app to run the intent. The GPU engine cannot submit work from the
    /// background (iOS returns BackgroundExecutionNotPermitted, which then wedges the engine). With
    /// the app foregrounded, GPU work is allowed and the warm `ModelHost.shared` is reused.
    static let openAppWhenRun = true

    @Parameter(title: "Question", requestValueDialog: "What do you want to ask Gemma?")
    var question: String

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog {
        // The app is foregrounded (openAppWhenRun) before this runs, so GPU submission is permitted
        // and ModelHost.shared is the warm, resident instance.
        let answer = try await ModelHost.shared.ask(question: question)
        return .result(dialog: IntentDialog("\(answer)"))
    }
}
