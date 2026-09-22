import CoreAIOps
import SwiftUI

/// A checklist over one document: open it, pick the questions, read the answers as a list of
/// verdicts. The editors sit behind a disclosure so the screen is the checklist, not a form.
struct ChecklistView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = ChecklistModel()
    @State private var importing = false
    @State private var editing = false

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Contract check",
                subtitle: "Open a document, ask it your questions — read once, every answer with its probability")
            HStack(spacing: 10) {
                Button("Open…") { importing = true }
                Button("Sample lease") { model.loadSample() }
                Button(editing ? "Hide questions" : "Edit questions") { editing.toggle() }
                Spacer()
                Text(model.documentName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Button("Check") { model.run(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working || model.document.isEmpty)
            }
            .disabled(model.working)
            .fileImporter(isPresented: $importing, allowedContentTypes: DocumentText.readableTypes) { result in
                if case .success(let url) = result { model.load(url) }
            }
            if editing {
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Document").font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $model.document).font(.callout)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Questions — one per line: noul: / choice: q | a | b / score: q | low | high")
                            .font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $model.questionsText).font(.callout.monospaced())
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    }
                }
                .frame(height: 160)
            }
            if !model.items.isEmpty, let prefill = model.prefillMilliseconds {
                HStack {
                    Text("\(model.items.count) answers in \(ms(model.totalMilliseconds)) · the document read once (\(model.prefillTokens) tokens, \(ms(prefill)))")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
            } else {
                Text(model.status).font(.callout).foregroundStyle(.secondary)
            }
            List(model.items) { item in
                HStack(alignment: .firstTextBaseline, spacing: 10) {
                    verdict(item.answer)
                        .frame(width: 150, alignment: .leading)
                    Text(item.question.instructions).font(.body)
                    Spacer()
                    Text(item.answer.confidence.formatted(.number.precision(.fractionLength(2))))
                        .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                }
                .padding(.vertical, 3)
            }
        }
        .padding()
        .task {
            await autoplay.run(.checklist, runtime: runtime, status: { model.status }) {
                model.loadSample()
                model.run(runtime)
            }
        }
    }

    @ViewBuilder
    private func verdict(_ answer: Decision.Answer) -> some View {
        switch answer.value {
        case .noul(let p):
            Label(p >= 0.5 ? "Yes" : "No", systemImage: p >= 0.5 ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.body.bold())
                .foregroundStyle(p >= 0.5 ? Color.green : Color.secondary)
        case .choice(let c):
            Label(c.id, systemImage: "tag.fill").font(.body.bold()).foregroundStyle(.blue)
        case .score(let s):
            Label(model.levelName(for: s) ?? "level \(s.level)", systemImage: "chart.bar.fill")
                .font(.body.bold()).foregroundStyle(.orange)
        }
    }
}
