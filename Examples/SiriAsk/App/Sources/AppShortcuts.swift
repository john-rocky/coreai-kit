// AppShortcuts — the spoken phrases that route Siri to our intent. Siri only does phrase matching
// here; the actual answering is our on-device Gemma 4 model inside the intent's perform(). Custom
// shortcut phrases must contain the app name (\(.applicationName)) — the app's display name is
// "Gemma" (see project.yml CFBundleDisplayName), so "Ask \(.applicationName)" speaks as "Ask Gemma".
//
// NOTE (App Intents constraint, confirmed by appintentsmetadataprocessor): only `AppEntity` /
// `AppEnum` parameters may be interpolated into a phrase — a free-text `String` like `question`
// cannot. So the phrase is the trigger only; Siri then prompts for the question via the parameter's
// `requestValueDialog` ("What do you want to ask Gemma?"). That two-step flow is the correct pattern
// for free-text input by voice.

import AppIntents

struct GemmaShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: AskGemmaIntent(),
            phrases: [
                "Ask \(.applicationName)",
                "Ask \(.applicationName) a question",
                "Ask \(.applicationName) something",
            ],
            shortTitle: "Ask Gemma",
            systemImageName: "sparkles")
    }
}
