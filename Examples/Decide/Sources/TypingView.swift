import CoreAIOps
import SwiftUI

/// A text field with three chips under it — tone, intent, emoji — read from the text at
/// every pause in typing.
struct TypingView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = TypingModel()

    var body: some View {
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Typing",
                subtitle: "Three decisions on the text so far at every pause — tone, intent, the emoji that fits")
            TextEditor(text: $model.text)
                .font(.title3)
                .frame(minHeight: 90, maxHeight: 140)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(.quaternary))
                .onChange(of: model.text) { model.changed(runtime) }
            HStack(spacing: 10) {
                chip("Tone", value: model.toneLevel, tint: toneTint, confidence: model.tone?.confidence)
                chip("Intent", value: model.intent?.choice, tint: .blue, confidence: model.intent?.confidence)
                Button {
                    model.acceptEmoji()
                } label: {
                    HStack(spacing: 6) {
                        Text(model.emoji?.choice ?? "·").font(.title)
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Emoji").font(.caption2).foregroundStyle(.secondary)
                            Text(model.emoji.map { $0.confidence.formatted(.number.precision(.fractionLength(2))) } ?? "")
                                .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                        }
                    }
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.4)))
                }
                .buttonStyle(.plain)
                .disabled(model.emoji == nil)
                Spacer()
            }
            if let tone = model.tone, case .score(let s) = tone.value {
                HStack(spacing: 4) {
                    ForEach(Array(s.probabilities.enumerated()), id: \.offset) { index, p in
                        RoundedRectangle(cornerRadius: 3)
                            .fill([Color.red, Color.gray, Color.green][min(index, 2)].opacity(0.25 + 0.75 * p))
                            .frame(height: 8)
                    }
                }
                .frame(maxWidth: 360)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(model.status).font(.caption).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            Spacer()
        }
        .padding()
        .task {
            await autoplay.run(.typing, runtime: runtime, status: { model.status + " || " + model.detail }) {
                await model.type(TypingModel.sample, runtime: runtime)
            }
        }
    }

    private var toneTint: Color {
        guard let value = model.tone?.score else { return .secondary }
        return value < 0.67 ? .red : (value > 1.33 ? .green : .gray)
    }

    private func chip(_ title: String, value: String?, tint: Color, confidence: Double?) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Text(value ?? "·").font(.callout.bold()).foregroundStyle(value == nil ? Color.secondary : tint)
                if let confidence {
                    Text(confidence.formatted(.number.precision(.fractionLength(2))))
                        .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                }
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.4)))
    }
}
