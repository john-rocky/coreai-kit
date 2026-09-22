import CoreAIOps
import SwiftUI

struct ChecklistView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = ChecklistModel()
    @State private var importing = false

    var body: some View {
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Checklist",
                subtitle: "One document prefilled once, then every question costs only its own tail")
            VStack(alignment: .leading, spacing: 4) {
                Text("Document (\(model.documentName))").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $model.document)
                    .font(.callout)
                    .frame(minHeight: 70, maxHeight: 96)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Questions (one per line: noul / choice / score)").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $model.questionsText)
                    .font(.callout.monospaced())
                    .frame(minHeight: 70, maxHeight: 96)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            }
            HStack {
                Button("Open…") { importing = true }
                Button("Sample") { model.loadSample() }
                Spacer()
                if let prefill = model.prefillMilliseconds {
                    Text("prefill \(model.prefillTokens) tokens · \(ms(prefill))")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button("Run") { model.run(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working || model.document.isEmpty)
            }
            .disabled(model.working)
            .fileImporter(isPresented: $importing, allowedContentTypes: DocumentText.readableTypes) { result in
                if case .success(let url) = result { model.load(url) }
            }
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List(model.items) { item in
                VStack(alignment: .leading, spacing: 4) {
                    Text(item.question.instructions).font(.callout)
                    HStack {
                        answerLine(item.answer)
                        Spacer()
                        Text(item.answer.timingLine).font(.caption).foregroundStyle(.secondary)
                    }
                    switch item.answer.value {
                    case .noul(let p):
                        ProbabilityBar(value: p, tint: p >= 0.5 ? .green : .gray)
                    case .choice(let c):
                        ProbabilityBar(value: c.confidence, tint: .blue)
                    case .score(let s):
                        ProbabilityBar(value: s.value / Double(max(1, s.probabilities.count - 1)), tint: .orange)
                    }
                }
                .padding(.vertical, 2)
            }
        }
        .padding()
        .task {
            await autoplay.run(.checklist, runtime: runtime) {
                model.loadSample()
                model.run(runtime)
            }
        }
    }

    @ViewBuilder
    private func answerLine(_ answer: Decision.Answer) -> some View {
        switch answer.value {
        case .noul(let p):
            Label(p >= 0.5 ? "Yes" : "No", systemImage: p >= 0.5 ? "checkmark.circle.fill" : "xmark.circle")
                .foregroundStyle(p >= 0.5 ? .green : .secondary)
            Text("P(yes) \(p.formatted(.number.precision(.fractionLength(2))))").font(.caption.monospacedDigit())
        case .choice(let c):
            Label(c.id, systemImage: "tag.fill").foregroundStyle(.blue)
            Text("\(c.confidence.formatted(.number.precision(.fractionLength(2))))").font(.caption.monospacedDigit())
        case .score(let s):
            Label("level \(s.level) of \(s.probabilities.count - 1)", systemImage: "chart.bar.fill").foregroundStyle(.orange)
            Text("expected \(s.value.formatted(.number.precision(.fractionLength(2))))").font(.caption.monospacedDigit())
        }
    }
}
