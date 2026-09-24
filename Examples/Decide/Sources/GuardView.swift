import CoreAIOps
import SwiftUI

/// The agent's command log judged under a plain-words policy: refused on top, then the ones
/// to ask about, then the ones that just run.
struct GuardView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = GuardModel()
    @State private var editing = false

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Command guard",
                subtitle: "A policy in plain words; every command an agent wants to run gets run / ask / refuse")
            controlsLayout {
                HStack(spacing: 10) {
                    Button("Sample log") { model.loadSample() }
                    Button(editing ? "Hide policy" : "Policy…") { editing.toggle() }
                }
                HStack(spacing: 10) {
                    Spacer()
                    Button("Check") { model.run(runtime) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.working || model.commands.isEmpty)
                }
            }
            .disabled(model.working)
            if editing {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Policy").font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $model.policy).font(.callout)
                        .frame(height: 80)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                }
            }
            if model.verdicts.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Commands, one per line").font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $model.commandsText).font(.callout.monospaced())
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                }
            }
            Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
            if !model.verdicts.isEmpty {
                List(model.ordered) { verdict in
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Label(verdict.decision, systemImage: icon(verdict.decision))
                            .font(.callout.bold())
                            .foregroundStyle(tint(verdict.decision))
                            .frame(width: isPhone ? 110 : 150, alignment: .leading)
                        Text(verdict.command).font(.callout.monospaced()).lineLimit(2)
                        Spacer()
                        Text(verdict.answer.confidence.formatted(.number.precision(.fractionLength(2))))
                            .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                    }
                    .padding(.vertical, 2)
                }
                Button("Edit the commands") { model.verdicts = [] }.font(.caption)
            }
        }
        .padding()
        .task {
            await autoplay.run(.guard, runtime: runtime, status: { model.working ? model.status : model.status + " || " + model.detail }) {
                model.loadSample()
                model.run(runtime)
            }
        }
    }

    private func tint(_ decision: String) -> Color {
        switch decision {
        case "refuse": return .red
        case "ask the user first": return .orange
        default: return .green
        }
    }

    private func icon(_ decision: String) -> String {
        switch decision {
        case "refuse": return "xmark.octagon.fill"
        case "ask the user first": return "hand.raised.fill"
        default: return "checkmark.circle.fill"
        }
    }
}
