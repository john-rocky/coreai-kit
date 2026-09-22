import CoreAIOps
import SwiftUI

/// A folder sorted by meaning: what needs you first, then everything filed by folder.
/// Folder names and the "what it needs" question sit behind a disclosure.
struct SorterView: View {
    @Environment(DecideRuntime.self) private var runtime
    @Environment(Autoplay.self) private var autoplay
    @State private var model = SorterModel()
    @State private var importing = false
    @State private var editing = false

    private var needsYou: [SorterModel.Entry] { model.entries.filter(\.needsYou) }
    private var filed: [(folder: String, files: [SorterModel.Entry])] {
        let groups = Dictionary(grouping: model.entries.filter { $0.bin != nil && !$0.needsYou }) { $0.chosen ?? "" }
        return model.bins.compactMap { bin in
            guard let files = groups[bin.name], !files.isEmpty else { return nil }
            return (bin.name, files)
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            ScreenHeader(
                title: "Sort a folder",
                subtitle: "Every file read once — which folder it belongs in, and whether it needs you")
            controlsLayout {
                HStack(spacing: 10) {
                    Button("Open folder…") { importing = true }
                    Button("Sample folder") { model.makeSampleFolder() }
                    Button(editing ? "Hide folders" : "Folders…") { editing.toggle() }
                }
                HStack(spacing: 10) {
                    Spacer()
                    Button("Sort") { model.sort(runtime) }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.working || model.entries.isEmpty)
                    Button("Move files") { model.apply() }
                        .disabled(model.working || model.sorted == 0 || model.applied)
                }
            }
            .disabled(model.working)
            .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { model.open(url) }
            }
            if editing {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Folders (one per line: name: what goes in it)").font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $model.binsText).font(.callout.monospaced())
                        .frame(height: 76)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    Text("What it needs from you (question | nothing | ask | ask…)").font(.caption).foregroundStyle(.secondary)
                    TextField("Need question", text: $model.needText, axis: .vertical)
                        .textFieldStyle(.roundedBorder).font(.callout)
                }
            }
            HStack {
                if let folder = model.folder {
                    Text(folder.lastPathComponent).font(.caption.bold())
                }
                Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
                if model.sorted > 0 {
                    Text("\(model.entries.count) files · \(model.entries.count * 2) decisions · \(ms(model.totalMilliseconds))")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            List {
                if !needsYou.isEmpty {
                    Section("Needs you (\(needsYou.count))") {
                        ForEach(needsYou) { entry in
                            HStack(alignment: .top, spacing: 8) {
                                Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.orange)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(entry.name).font(.body)
                                    if isPhone {
                                        Text(entry.needLabel ?? "").font(.caption).foregroundStyle(.orange)
                                    }
                                }
                                Spacer()
                                if !isPhone {
                                    Text(entry.needLabel ?? "").font(.caption).foregroundStyle(.orange)
                                }
                                Text(entry.chosen ?? "").font(.caption.bold())
                                    .padding(.horizontal, 8).padding(.vertical, 2)
                                    .background(Capsule().fill(.blue.opacity(0.15)))
                            }
                        }
                    }
                }
                ForEach(filed, id: \.folder) { group in
                    Section("\(group.folder) (\(group.files.count))") {
                        ForEach(group.files) { entry in
                            HStack(spacing: 8) {
                                Image(systemName: "doc.text").foregroundStyle(.secondary)
                                Text(entry.name).font(.body)
                                Spacer()
                                if !isPhone {
                                    Text(entry.excerpt.replacingOccurrences(of: "\n", with: " "))
                                        .font(.caption).foregroundStyle(.tertiary).lineLimit(1)
                                        .frame(maxWidth: 360, alignment: .trailing)
                                }
                            }
                        }
                    }
                }
                if model.sorted == 0 {
                    ForEach(model.entries) { entry in
                        HStack(spacing: 8) {
                            Image(systemName: "doc").foregroundStyle(.secondary)
                            Text(entry.name)
                            Spacer()
                            Text(entry.excerpt.replacingOccurrences(of: "\n", with: " "))
                                .font(.caption).foregroundStyle(.tertiary).lineLimit(1)
                                .frame(maxWidth: 360, alignment: .trailing)
                        }
                    }
                }
            }
        }
        .padding()
        .task {
            await autoplay.run(.sorter, runtime: runtime, status: { model.status }) {
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
