// PhraseCard — the two cards of the ask-anything coach:
//
//   AskBox          the in-app "Ask Gemma" box. Type a question, tap Ask, and the on-device Gemma 4
//                   answers — no Siri, no network. This is the make-or-break G1 surface: it proves
//                   the AOT bundle + entitlement load and the model answers free-text on the device.
//                   Running it also warms the shared model for the live Siri demo.
//
//   SiriPhraseCard  teaches the real Siri phrase ("Hey Siri, Ask Gemma") — the thing you say in the
//                   recording. Siri can't take free text in a phrase (App Intents limitation), so it
//                   prompts for the question; the card explains that two-step flow.

import SwiftUI

// MARK: - In-app ask box (the G1 surface)

struct AskBox: View {
    let modelReady: Bool
    @State private var question = ""
    @State private var run = AskRun()
    @FocusState private var focused: Bool

    /// One-tap sample questions so the demo is legible and the first answer is fast to trigger.
    private let samples = [
        "What is the capital of France?",
        "Explain photosynthesis in two sentences.",
        "Give me three ideas for a rainy-day activity.",
    ]

    var body: some View {
        DemoCard(accent: Demo.accent) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 6) {
                    Image(systemName: "sparkles").foregroundStyle(Demo.accent).imageScale(.small)
                    Text("Ask Gemma — on this device, no Siri needed")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Spacer()
                }

                TextField("Ask anything…", text: $question, axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.plain)
                    .focused($focused)
                    .onSubmit(askNow)
                    .padding(10)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))

                // Sample chips — tap to fill the field.
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(samples, id: \.self) { sample in
                            Button { question = sample } label: {
                                Text(sample)
                                    .font(.caption)
                                    .lineLimit(1)
                                    .padding(.horizontal, 10).padding(.vertical, 6)
                                    .background(Demo.accent.opacity(0.12), in: Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }

                Button(action: askNow) {
                    Label(run.isRunning ? "Answering…" : "Ask Gemma", systemImage: "play.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(Demo.accent)
                .controlSize(.large)
                .disabled(!canAsk)

                if !modelReady {
                    Text("Loading the model on device… the button enables when it’s warm.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                AskResult(run: run)
            }
        }
    }

    private var canAsk: Bool {
        modelReady && !run.isRunning
            && !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func askNow() {
        guard canAsk else { return }
        focused = false
        run.ask(question)
    }
}

// MARK: - Ask result (trace → answer → timing)

/// The visible proof: the live trace, the answer, and how long it took (the cold first answer
/// includes the model load; warm answers are the real on-device generate speed).
struct AskResult: View {
    var run: AskRun

    var body: some View {
        if run.hasResult {
            VStack(alignment: .leading, spacing: 10) {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(run.trace) { TraceRow(line: $0) }
                }
                if !run.answer.isEmpty {
                    Text(run.answer)
                        .font(.callout)
                        .textSelection(.enabled)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    if let s = run.elapsed {
                        Text(String(format: "answered in %.1fs · entirely on device", s))
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.top, 2)
        }
    }
}

// MARK: - Siri coach card

/// "Hey Siri, Ask Gemma" — the phrase you say in the recording. Rendered large and in quotes
/// because it is literally the line the presenter reads aloud.
struct SiriPhraseCard: View {
    var body: some View {
        DemoCard(accent: .blue) {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 6) {
                    Image(systemName: "mic.fill").foregroundStyle(.blue).imageScale(.small)
                    Text("Say it to Siri")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Spacer()
                }
                Text("“Hey Siri, Ask \(Demo.appName)”")
                    .font(.title3.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Text("Siri routes to this app and asks “What do you want to ask Gemma?” — say your "
                    + "question and your \(Demo.modelName) answers out loud. Siri’s own model never "
                    + "sees it.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Label("Free-text questions can’t live in a Siri phrase, so Siri prompts for it "
                    + "(App Intents).", systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
