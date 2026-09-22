import CoreAIOps
import SwiftUI

struct SorterView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = SorterModel()
    @State private var importing = false

    var body: some View {
        VStack(spacing: 12) {
            ScreenHeader(
                title: "Sorter",
                subtitle: "Every file read once, two decisions each: which folder, and does it need you")
            VStack(alignment: .leading, spacing: 4) {
                Text("Folders (one per line: name: what goes in it)").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $model.binsText)
                    .font(.callout.monospaced())
                    .frame(minHeight: 60, maxHeight: 76)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("What it needs from you (question | nothing | ask | ask…)").font(.caption).foregroundStyle(.secondary)
                TextField("Need question", text: $model.needText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)
            }
            HStack {
                Button("Open folder…") { importing = true }
                Button("Sample folder") { model.makeSampleFolder() }
                Spacer()
                Button("Sort") { model.sort(runtime) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.working || model.entries.isEmpty)
                Button("Apply") { model.apply() }
                    .disabled(model.working || model.sorted == 0 || model.applied)
            }
            .disabled(model.working)
            .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { model.open(url) }
            }
            if let folder = model.folder {
                Text(folder.path(percentEncoded: false)).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
            }
            Text(model.status).font(.callout).foregroundStyle(.secondary)
            List(model.entries) { entry in
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: entry.needsYou ? "exclamationmark.circle.fill" : "doc.text")
                        .foregroundStyle(entry.needsYou ? .orange : .secondary)
                        .frame(width: 18)
                    VStack(alignment: .leading, spacing: 3) {
                        HStack {
                            Text(entry.name).font(.callout.bold())
                            Spacer()
                            if let chosen = entry.chosen, let bin = entry.bin {
                                Text(chosen)
                                    .font(.caption.bold())
                                    .padding(.horizontal, 8).padding(.vertical, 2)
                                    .background(Capsule().fill(.blue.opacity(0.15)))
                                Text(bin.confidence.formatted(.number.precision(.fractionLength(2))))
                                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            }
                        }
                        Text(entry.excerpt.replacingOccurrences(of: "\n", with: " "))
                            .font(.caption).foregroundStyle(.secondary).lineLimit(2)
                        if let need = entry.need, let label = entry.needLabel {
                            HStack {
                                Text(entry.needsYou ? "needs you: \(label)" : "nothing to do")
                                    .font(.caption).foregroundStyle(entry.needsYou ? .orange : .secondary)
                                    .lineLimit(1)
                                ProbabilityBar(value: need.confidence, tint: entry.needsYou ? .orange : .gray)
                                    .frame(width: 60)
                                Text("\(ms(entry.milliseconds)) for 2 · \(need.timing.reusedTokens) tokens reused")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        }
        .padding()
        .task {
            await autoplay.run(.sorter, runtime: runtime) {
                model.makeSampleFolder()
                try? await Task.sleep(for: .seconds(autoplay.delay))
                model.sort(runtime)
                while model.working { try? await Task.sleep(for: .milliseconds(100)) }
                try? await Task.sleep(for: .seconds(autoplay.delay))
                model.apply()
            }
        }
    }
}
