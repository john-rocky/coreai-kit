// Intents — the decisions as Shortcuts actions. A shortcut can now branch on meaning: "Ask
// yes/no" returns P(yes) for any text and question, "Classify text" returns the chosen option.
// Both run fully on device through `CoreAI.decide`, which loads the model once per process
// and keeps it warm for the next action.
//
// Free-text parameters cannot be interpolated into a spoken phrase (App Intents accepts only
// `AppEntity` / `AppEnum` there), so the phrase is the trigger and Siri prompts for the
// question; in the Shortcuts editor every parameter is a field.

import AppIntents
import CoreAIOps
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Runs a decision with the process kept alive for its duration when invoked in the background.
private func withBackgroundTime<Value: Sendable>(_ body: () async throws -> Value) async rethrows -> Value {
    #if os(iOS)
    let app = await UIApplication.shared
    let task = await app.beginBackgroundTask(withName: "Decide")
    defer { if task != .invalid { Task { @MainActor in app.endBackgroundTask(task) } } }
    #endif
    return try await body()
}

struct AskYesNoIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask yes/no"
    static let description = IntentDescription(
        "Answers a yes/no question about a piece of text with the probability of yes, on device.",
        categoryName: "Decide")
    static let openAppWhenRun = false

    @Parameter(title: "Text", requestValueDialog: "What text should I look at?")
    var text: String

    @Parameter(title: "Question", requestValueDialog: "What do you want to know about it?")
    var question: String

    static var parameterSummary: some ParameterSummary {
        Summary("Ask \(\.$question) about \(\.$text)")
    }

    func perform() async throws -> some IntentResult & ReturnsValue<Double> & ProvidesDialog {
        let answer = try await withBackgroundTime {
            try await CoreAI.decide(text, .noul(question))
        }
        let p = answer.noul ?? 0
        let word = p >= 0.5 ? "Yes" : "No"
        return .result(
            value: p,
            dialog: IntentDialog("\(word) — \(Int((p * 100).rounded()))% yes, decided in \(Int(answer.timing.milliseconds.rounded())) ms."))
    }
}

struct ClassifyTextIntent: AppIntent {
    static let title: LocalizedStringResource = "Classify text"
    static let description = IntentDescription(
        "Picks one of your options for a piece of text, on device.", categoryName: "Decide")
    static let openAppWhenRun = false

    @Parameter(title: "Text", requestValueDialog: "What text should I look at?")
    var text: String

    @Parameter(title: "Question", requestValueDialog: "What do you want to decide about it?")
    var question: String

    @Parameter(title: "Options", requestValueDialog: "What are the options?")
    var options: [String]

    static var parameterSummary: some ParameterSummary {
        Summary("Decide \(\.$question) about \(\.$text) among \(\.$options)")
    }

    func perform() async throws -> some IntentResult & ReturnsValue<String> & ProvidesDialog {
        let answer = try await withBackgroundTime {
            try await CoreAI.decide(text, .choice(question, options))
        }
        let chosen = answer.choice ?? ""
        return .result(
            value: chosen,
            dialog: IntentDialog("\(chosen) — \(Int((answer.confidence * 100).rounded()))%, decided in \(Int(answer.timing.milliseconds.rounded())) ms."))
    }
}

struct DecideShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: AskYesNoIntent(),
            phrases: ["Ask \(.applicationName) a yes or no question"],
            shortTitle: "Ask yes/no",
            systemImageName: "questionmark.circle")
        AppShortcut(
            intent: ClassifyTextIntent(),
            phrases: ["Classify with \(.applicationName)"],
            shortTitle: "Classify text",
            systemImageName: "tag")
    }
}
