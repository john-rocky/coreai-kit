import CoreAIOps
import SwiftUI

/// The transcript's tool results with a relevance bar each; the dropped ones fold to one
/// grey line, and the header says what the context shrank to.
struct ContextView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = ContextModel()
    @State private var expanded: Set<UUID> = []

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Context",
                subtitle: "An agent's tool results scored against the question — the unrelated ones drop out of the context")
            controlsLayout {
                HStack(spacing: 10) {
                    Button("Sample transcript") { model.loadSample() }
                }
                HStack(spacing: 10) {
                    Spacer()
                    Button("Compress") { model.run(runtime) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.working || model.items.isEmpty)
                }
            }
            .disabled(model.working)
            VStack(alignment: .leading, spacing: 4) {
                Text("The question").font(.caption).foregroundStyle(.secondary)
                TextField("What the user asked", text: $model.question, axis: .vertical)
                    .textFieldStyle(.roundedBorder).font(.callout)
                    .disabled(model.working)
            }
            HStack {
                if model.decided {
                    Text("\(model.items.count) tool results, \(model.totalTokens) tokens → kept \(model.keptItems.count), \(model.keptTokens) tokens")
                        .font(.callout.bold())
                    Text("· \(model.items.count) decisions in \(ms(model.totalMilliseconds))").font(.caption).foregroundStyle(.secondary)
                } else {
                    Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
            }
            List {
                ForEach(model.items) { item in
                    let kept = item.kept(at: model.threshold)
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 8) {
                            Image(systemName: kept == nil ? "doc.text" : (kept! ? "checkmark.circle.fill" : "minus.circle"))
                                .foregroundStyle(kept == nil ? Color.secondary : (kept! ? Color.green : Color.secondary))
                            Text(item.tool).font(.callout.bold())
                                .foregroundStyle(kept == false ? .secondary : .primary)
                            if item.tokens > 0 {
                                Text("\(item.tokens) tokens").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if let relevance = item.relevance {
                                ProbabilityBar(value: relevance / 2, tint: kept == true ? .green : .gray).frame(width: 80)
                                Text(relevance.formatted(.number.precision(.fractionLength(2))))
                                    .font(.caption.monospacedDigit()).foregroundStyle(.tertiary)
                            }
                            if kept == false {
                                Text("dropped").font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        if kept != false || expanded.contains(item.id) {
                            Text(item.text).font(.caption.monospaced())
                                .foregroundStyle(kept == false ? .tertiary : .secondary)
                                .lineLimit(expanded.contains(item.id) ? nil : 3)
                        }
                    }
                    .padding(.vertical, 2)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        if expanded.contains(item.id) { expanded.remove(item.id) } else { expanded.insert(item.id) }
                    }
                }
            }
        }
        .padding()
        .task {
            await autoplay.run(.context, runtime: runtime, status: { model.working ? model.status : model.status + " || " + model.detail }) {
                model.loadSample()
                model.run(runtime)
            }
        }
    }
}
