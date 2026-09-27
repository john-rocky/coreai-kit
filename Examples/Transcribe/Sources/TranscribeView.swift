import SwiftUI
import UniformTypeIdentifiers

struct TranscribeView: View {
    @State private var model = TranscribeModel()
    @State private var autoplay = Autoplay()
    @State private var showImporter = false

    var body: some View {
        VStack(spacing: 12) {
            header
            picker
            controls
            if model.takesHotwords {
                hotwordField
            }
            transcriptBox
            if let measured = model.measuredLine {
                Text(measured).font(.footnote).foregroundStyle(.secondary)
            }
            if !model.detectedLanguage.isEmpty {
                Text("Detected language: \(model.detectedLanguage)")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding()
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { model.loadFile(url) }
        }
        .onChange(of: model.selectedID) { Task { await model.load() } }
        .task { await autoplay.run(model) }
        #if os(macOS)
            .frame(minWidth: 480, minHeight: 560)
        #endif
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("Transcribe").font(.title2.bold())
            Text("On-device speech-to-text — any ASR model in the catalog")
                .font(.caption).foregroundStyle(.secondary)
            Text(model.statusLabel).font(.callout).foregroundStyle(statusColor)
            if let f = model.downloadFraction {
                ProgressView(value: f).frame(maxWidth: 280)
            }
            Text(model.clipName).font(.caption).foregroundStyle(.secondary)
        }
    }

    private var picker: some View {
        Picker("Model", selection: $model.selectedID) {
            if model.selectedID == nil {
                Text("Choose a model").tag(String?.none)
            }
            ForEach(model.models) { entry in
                Text(entry.name).tag(Optional(entry.id))
            }
        }
        .pickerStyle(.menu)
        .disabled(model.isBusy || model.recording)
    }

    /// One row when it fits; on a narrow phone, Transcribe gets a row of its own.
    private var controls: some View {
        ViewThatFits(in: .horizontal) {
            HStack {
                clipButtons
                Spacer()
                transcribeButton
            }
            VStack(spacing: 10) {
                HStack(spacing: 16) { clipButtons }
                transcribeButton
            }
        }
    }

    @ViewBuilder private var clipButtons: some View {
        Button(model.recording ? "Stop" : "Record") { model.toggleRecord() }
            .disabled(model.isBusy)
        Button("Choose…") { showImporter = true }.disabled(model.isBusy)
        Button("Demo") { model.loadDemo() }.disabled(model.isBusy)
        Button(model.playing ? "Stop" : "Play") { model.togglePlay() }
            .disabled(!model.hasClip || model.recording)
    }

    private var transcribeButton: some View {
        Button("Transcribe") { Task { await model.transcribe() } }
            .buttonStyle(.borderedProminent)
            .disabled(!model.canTranscribe)
    }

    private var hotwordField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Hotwords").font(.subheadline).foregroundStyle(.secondary)
            // Wraps on a phone, so every name typed stays visible (the list is what the model gets).
            TextField(
                "Names to spell right, comma-separated", text: $model.hotwordsText, axis: .vertical
            )
            .lineLimit(1...3)
            .textFieldStyle(.roundedBorder)
            .autocorrectionDisabled()
            .disabled(model.isBusy)
        }
    }

    private var transcriptBox: some View {
        ScrollView {
            Text(
                model.transcript.isEmpty
                    ? AttributedString("Transcript will appear here.") : model.emphasizedTranscript
            )
            #if os(iOS)
            .font(.title3)
            #endif
            .frame(maxWidth: .infinity, alignment: .leading)
            .foregroundStyle(model.transcript.isEmpty ? .secondary : .primary)
            .textSelection(.enabled)
            .padding()
        }
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }

    private var statusColor: Color {
        switch model.status {
        case .loading, .transcribing: return .secondary
        case _ where model.playing: return .secondary
        case .error: return .red
        case .ready: return .green
        case .idle: return .secondary
        }
    }
}
