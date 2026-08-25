import CoreAIKit
import SwiftUI

struct TidyView: View {
    @State private var model = TidyModel()

    var body: some View {
        VStack(spacing: 12) {
            header
            axes
            editor(
                title: "Raw transcript", text: $model.transcript,
                placeholder: "Paste dictation output here.")
            controls
            resultBox
        }
        .padding()
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 620)
        #endif
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("Tidy").font(.title2.bold())
            Text("Raw speech-to-text → written text, on device")
                .font(.caption).foregroundStyle(.secondary)
            Text(model.status.label).font(.callout).foregroundStyle(statusColor)
            if let f = model.downloadFraction {
                ProgressView(value: f).frame(maxWidth: 280)
            }
        }
    }

    /// The model's three trained control axes — the whole steering surface; there is no
    /// free-text instruction to give it.
    private var axes: some View {
        VStack(spacing: 6) {
            Picker("Model", selection: $model.selectedID) {
                ForEach(model.models) { entry in Text(entry.name).tag(entry.id) }
            }
            Picker("Styling", selection: $model.styling) {
                ForEach(TranscriptStyling.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            Picker("Structure", selection: $model.structure) {
                ForEach(TranscriptStructure.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            Picker("Context", selection: $model.context) {
                ForEach(TranscriptContext.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
        }
        .pickerStyle(.menu)
        .disabled(model.isBusy)
    }

    private func editor(
        title: String, text: Binding<String>, placeholder: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            TextEditor(text: text)
                .font(.callout)
                .frame(minHeight: 110)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                .overlay(alignment: .topLeading) {
                    if text.wrappedValue.isEmpty {
                        Text(placeholder)
                            .font(.callout).foregroundStyle(.tertiary)
                            .padding(.horizontal, 6).padding(.vertical, 8)
                            .allowsHitTesting(false)
                    }
                }
        }
    }

    private var controls: some View {
        HStack {
            Button("Sample") { model.loadSample() }.disabled(model.isBusy)
            Spacer()
            Button("Tidy") { model.run() }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canRun)
        }
    }

    private var resultBox: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Written text").font(.caption).foregroundStyle(.secondary)
            ScrollView {
                Text(model.result.isEmpty ? "The rewrite will appear here." : model.result)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .foregroundStyle(model.result.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .padding()
            }
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private var statusColor: Color {
        switch model.status {
        case .error: return .red
        case .idle: return .green
        default: return .secondary
        }
    }
}
