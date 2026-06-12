import CoreAIKitUI
import SwiftUI

struct ChatView: View {
    @State private var model = ChatModel()
    @State private var input = ""

    var body: some View {
        VStack(spacing: 0) {
            ModelPickerBar(
                entries: model.entries,
                selection: $model.selectedEntry,
                isBusy: model.isBusy,
                statusText: model.status.label,
                downloadFraction: model.downloadFraction,
                loadTitle: model.status == .idle ? "Download & Load" : "Reload"
            ) {
                model.load()
            }
            Divider()
            ChatTranscriptView(bubbles: model.bubbles)
            Divider()
            StatsBar(stats: model.stats)
            inputBar
        }
        .task { await model.loadCatalog() }
        #if os(macOS)
        .frame(minWidth: 480, minHeight: 560)
        #endif
    }

    private var inputBar: some View {
        HStack {
            TextField("Message", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
                .disabled(model.status != .ready)
            if model.status == .generating {
                Button("Stop") { model.stop() }
            } else {
                Button("Send", action: send)
                    .disabled(model.status != .ready || input.isEmpty)
            }
            Button("New") { model.newChat() }
                .disabled(model.isBusy)
        }
        .padding(10)
    }

    private func send() {
        let text = input
        input = ""
        model.send(text)
    }
}
