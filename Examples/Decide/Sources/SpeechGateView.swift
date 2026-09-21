import CoreAIOps
import SwiftUI

struct SpeechGateView: View {
    @Environment(DecideRuntime.self) private var runtime
    @State private var model = SpeechGateModel()

    var body: some View {
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Speech gate",
                subtitle: "One yes/no decision per utterance — only the ones that pass go to a language model")
            TextField("Gate question", text: $model.question, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .disabled(model.working)
            HStack {
                Button(model.recording ? "Stop" : "Record") { model.toggleRecord(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working)
                Button("Sample") { model.runSample(runtime) }
                    .disabled(model.working || model.recording)
                Spacer()
                if !model.utterances.isEmpty {
                    Text("\(model.passed)/\(model.utterances.count) pass · median \(ms(model.medianMilliseconds))")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List(model.utterances) { utterance in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top) {
                        Image(systemName: utterance.passes ? "arrow.right.circle.fill" : "xmark.circle")
                            .foregroundStyle(utterance.passes ? .green : .secondary)
                        Text(utterance.text)
                    }
                    ProbabilityBar(value: utterance.answer.noul ?? 0, tint: utterance.passes ? .green : .gray)
                    Text("\(utterance.answer.summaryLine) · \(utterance.answer.timingLine)")
                        .font(.caption).foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
            }
        }
        .padding()
    }
}
