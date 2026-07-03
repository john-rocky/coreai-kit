import CoreAIKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
@Observable
final class ReadDocModel {
    var markdown = ""
    var status = "Pick a document image."
    var busy = false
    /// OCR entries in the catalog; the picker reloads on a new choice.
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?

    private var reader: KitDocReader?
    private var loadedID: String?

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        entries = await ModelCatalog.load().available(.ocr)
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func read(_ url: URL) async {
        guard let entry = selectedEntry, !busy else { return }
        busy = true
        defer { busy = false }
        do {
            if reader == nil || loadedID != entry.id {
                status = "Loading \(entry.name)…"
                // Same gesture as the model card: the catalog id resolves the bundles.
                reader = try await KitDocReader(catalog: entry.id) { progress in
                    Task { @MainActor in
                        self.status = "Downloading… \(Int(progress.fraction * 100))%"
                    }
                }
                loadedID = entry.id
            }
            status = "Reading…"
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            markdown = try await reader!.read(imageAt: url)
            status = "Done — \(markdown.count) characters."
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}

struct ReadDocView: View {
    @State private var model = ReadDocModel()
    @State private var showImporter = false

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Picker("Model", selection: $model.selectedEntry) {
                    ForEach(model.entries) { entry in
                        Text(entry.name).tag(Optional(entry))
                    }
                }
                .fixedSize()
                Button("Open Image…") { showImporter = true }
                    .disabled(model.busy || model.selectedEntry == nil)
                Spacer()
            }
            .padding(.horizontal, 12)
            ScrollView {
                Text(model.markdown.isEmpty ? "The document's markdown appears here." : model.markdown)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 8)
            HStack {
                Text(model.status)
                if model.busy { ProgressView().controlSize(.small) }
                Spacer()
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
        }
        .padding(.vertical, 8)
        .task { await model.loadCatalog() }
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.image]) { result in
            if case .success(let url) = result {
                Task { await model.read(url) }
            }
        }
    }
}
