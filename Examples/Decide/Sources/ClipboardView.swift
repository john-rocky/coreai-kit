import CoreAIOps
import SwiftUI

struct ClipboardView: View {
    @Environment(DecideRuntime.self) private var runtime
    @State private var model = ClipboardModel()

    var body: some View {
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Clipboard",
                subtitle: "What is on the clipboard, and does it fit what you are about to do?")
            VStack(alignment: .leading, spacing: 4) {
                Text("Purpose").font(.caption).foregroundStyle(.secondary)
                TextField("What you are about to do", text: $model.purpose)
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
                Spacer()
                Button("Decide") { model.decide(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working || model.clipboard.isEmpty)
            }
            .disabled(model.working)
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List {
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
                if let fits = model.answers["fits"] {
                    Section("Is it what the purpose needs?") {
                        HStack {
                            Text(fits.summaryLine)
                            ProbabilityBar(value: fits.noul ?? 0, tint: (fits.noul ?? 0) >= 0.5 ? .green : .orange)
                        }
                        Text(fits.timingLine).font(.caption).foregroundStyle(.secondary)
                    }
                }
                if let safe = model.answers["safe"], case .score(let s) = safe.value {
                    Section("Paste as-is?") {
                        ForEach(Array(s.probabilities.enumerated()), id: \.offset) { level, p in
                            HStack {
                                Text(["do not paste", "paste part of it", "paste as-is"][level])
                                    .frame(width: 190, alignment: .leading)
                                ProbabilityBar(value: p, tint: level == s.level ? .blue : .gray)
                                Text(p.formatted(.number.precision(.fractionLength(2))))
                                    .font(.caption.monospacedDigit()).frame(width: 40)
                            }
                        }
                        Text("expected level \(s.value.formatted(.number.precision(.fractionLength(2)))) · \(safe.timingLine)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            Text("In Shortcuts: “Ask yes/no” and “Classify text” run these decisions on any text without opening the app.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .padding()
    }
}
