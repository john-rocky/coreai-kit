// AudioChatView.swift — pick or generate a clip, ask "what do you hear?", see the local model's
// answer. Audio UNDERSTANDING (sounds / events / emotion), not transcription.

import SwiftUI
import UniformTypeIdentifiers

struct AudioChatView: View {
    @StateObject private var model = AudioChatModel()
    @State private var question = "What do you hear?"
    @State private var showImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("AudioChat — on-device audio understanding")
                .font(.title2).bold()
            Text("Qwen2.5-Omni Thinker on Core AI. The model describes the *sounds* it hears — "
                + "not a transcript.")
                .font(.callout).foregroundStyle(.secondary)

            if !model.loaded {
                Button {
                    Task { await model.load() }
                } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }.disabled(model.busy)
            }

            HStack(spacing: 12) {
                Button {
                    model.toggleRecord()
                } label: {
                    Label(model.recording ? "Stop" : "Record",
                          systemImage: model.recording ? "stop.circle.fill" : "mic.circle")
                }
                .disabled(!model.loaded || model.busy)
                .tint(model.recording ? .red : .accentColor)

                Button {
                    showImporter = true
                } label: {
                    Label("Choose…", systemImage: "waveform")
                }.disabled(!model.loaded || model.busy || model.recording)

                Button {
                    model.loadDemoNoise()
                } label: {
                    Label("Demo", systemImage: "speaker.wave.2")
                }.disabled(!model.loaded || model.busy || model.recording)
            }

            Text(model.clipName).font(.footnote).foregroundStyle(.secondary)

            HStack {
                TextField("Question", text: $question)
                    .textFieldStyle(.roundedBorder)
                Button("Ask") {
                    Task { await model.ask(question) }
                }.disabled(!model.loaded || model.busy)
            }

            if model.busy { ProgressView().controlSize(.small) }

            ScrollView {
                Text(model.answer.isEmpty ? " " : model.answer)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(10)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
            }.frame(minHeight: 120)

            Text(model.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 460)  // macOS window sizing; on iOS it overflows
        #endif
        .fileImporter(
            isPresented: $showImporter, allowedContentTypes: [.audio], allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first { model.loadFile(url) }
        }
        .task {
            // Headless device measurement: `devicectl ... process launch ... --selftest`.
            if CommandLine.arguments.contains("--selftest") { await model.selfTest() }
        }
    }
}
