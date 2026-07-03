import CoreAIKit
import SwiftUI

@MainActor
@Observable
final class DiffuseChatModel {
    var prompt = ""
    var canvas = ""
    var status = "Type a prompt — the reply denoises in place."
    var busy = false
    /// dLLM entries in the catalog; the picker reloads on a new choice.
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?

    private var dlm: KitDiffusionLM?
    private var loadedID: String?

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        entries = await ModelCatalog.load().available(.dllm)
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func send() async {
        guard let entry = selectedEntry, !busy, !prompt.isEmpty else { return }
        busy = true
        defer { busy = false }
        do {
            if dlm == nil || loadedID != entry.id {
                status = "Loading \(entry.name)…"
                // Same gesture as the model card: the catalog id resolves the bundle.
                dlm = try await KitDiffusionLM(catalog: entry.id) { progress in
                    Task { @MainActor in
                        self.status = "Downloading… \(Int(progress.fraction * 100))%"
                    }
                }
                loadedID = entry.id
            }
            status = "Denoising…"
            let reply = try await dlm!.reply(to: prompt) { step in
                Task { @MainActor in
                    self.canvas = step.text
                    self.status = "Denoising… \(step.committed) tokens, \(step.forwards) forwards"
                }
            }
            canvas = reply
            status = "Done."
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}

struct DiffuseChatView: View {
    @State private var model = DiffuseChatModel()

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 10) {
                Picker("Model", selection: $model.selectedEntry) {
                    ForEach(model.entries) { entry in
                        Text(entry.name).tag(Optional(entry))
                    }
                }
                .fixedSize()
                Spacer()
            }
            .padding(.horizontal, 12)
            ScrollView {
                Text(model.canvas.isEmpty ? "The canvas denoises here — ░ are still-masked positions." : model.canvas)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            }
            .background(Color.gray.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 8)
            HStack(spacing: 8) {
                TextField("Prompt", text: $model.prompt)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await model.send() } }
                Button("Send") { Task { await model.send() } }
                    .disabled(model.busy || model.prompt.isEmpty || model.selectedEntry == nil)
            }
            .padding(.horizontal, 12)
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
    }
}
