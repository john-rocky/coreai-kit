import CoreAIOps
import SwiftUI

struct ClipboardView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(ClipboardModel.self) private var model
    @Environment(Autoplay.self) private var autoplay

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Clipboard",
                subtitle: "What is on the clipboard, and is it the thing you need right now?")
            VStack(alignment: .leading, spacing: 4) {
                Text("What you need (is this text …? | what \"partly\" means | what \"completely\" means)").font(.caption).foregroundStyle(.secondary)
                TextField("a shipping address | only a name or a postal code | a complete address", text: $model.need)
                    .textFieldStyle(.roundedBorder)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Clipboard").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $model.clipboard)
                    .font(.callout)
                    .frame(minHeight: 90, maxHeight: 140)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            }
            HStack {
                Button("Read clipboard") { model.readClipboard() }
                Button("Sample") { model.loadSample() }
                Toggle("Watch", isOn: Binding(
                    get: { model.watching },
                    set: { model.setWatching($0, runtime: runtime) }))
                    .toggleStyle(.switch)
                    .disabled(!runtime.isReady)
                Spacer()
                Button("Decide") { model.decide(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working || model.clipboard.isEmpty)
            }
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List {
                if !model.history.isEmpty {
                    Section("Watched copies (newest first)") {
                        ForEach(model.history) { verdict in
                            HStack(alignment: .top) {
                                Image(systemName: verdict.isSecret ? "hand.raised.fill" : (verdict.fitLevel == 2 ? "checkmark.circle.fill" : "doc.on.clipboard"))
                                    .foregroundStyle(verdict.isSecret ? .red : (verdict.fitLevel == 2 ? .green : .secondary))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(verdict.line).font(.callout.bold())
                                    Text(verdict.text.replacingOccurrences(of: "\n", with: " ")).font(.caption).lineLimit(1)
                                    Text("\(ms(verdict.milliseconds)) for 2 decisions · kind \(verdict.kindConfidence.formatted(.number.precision(.fractionLength(2)))) · fit \(verdict.fitConfidence.formatted(.number.precision(.fractionLength(2))))")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
                if let kind = model.answers["kind"], case .choice(let c) = kind.value {
                    Section("What is it?") {
                        ForEach(c.ranking.prefix(4), id: \.self) { id in
                            HStack {
                                Text(id).frame(width: 190, alignment: .leading)
                                ProbabilityBar(value: c.probabilities[id] ?? 0, tint: id == c.id ? .blue : .gray)
                                Text((c.probabilities[id] ?? 0).formatted(.number.precision(.fractionLength(2))))
                                    .font(.caption.monospacedDigit()).frame(width: 40)
                            }
                        }
                        Text(kind.timingLine).font(.caption).foregroundStyle(.secondary)
                    }
                }
                if let fit = model.answers["fit"], case .score(let s) = fit.value {
                    Section("Is it \(model.needSpec.need)?") {
                        ForEach(Array(s.probabilities.enumerated()), id: \.offset) { level, p in
                            HStack {
                                Text(model.needSpec.levels[level])
                                    .frame(width: 190, alignment: .leading)
                                ProbabilityBar(value: p, tint: level == s.level ? .blue : .gray)
                                Text(p.formatted(.number.precision(.fractionLength(2))))
                                    .font(.caption.monospacedDigit()).frame(width: 40)
                            }
                        }
                        Text("expected level \(s.value.formatted(.number.precision(.fractionLength(2)))) · \(fit.timingLine)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Text("In Shortcuts: “Ask yes/no” and “Classify text” run these decisions on any text without opening the app.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding()
        .task {
            await autoplay.run(.clipboard, runtime: runtime) {
                model.setWatching(true, runtime: runtime)
            }
        }
    }
}
