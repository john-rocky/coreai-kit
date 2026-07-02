// VLChatView — pick a photo, ask a local VLM about it, watch the answer stream in.

import CoreAIKit
import CoreAIKitUI
import PhotosUI
import SwiftUI

struct VLChatView: View {
    @State private var model = VLChatModel()
    @State private var pickerItem: PhotosPickerItem?
    @State private var input = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            imageStrip
            Divider()
            ChatTranscriptView(bubbles: model.bubbles)
            Divider()
            inputBar
        }
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 640)
        #endif
        .onChange(of: pickerItem) { _, item in
            if let item { model.setImage(from: item) }
        }
        .task { await model.loadCatalog() }
    }

    // MARK: - Header (model picker + load)

    private var header: some View {
        HStack(spacing: 10) {
            Picker("Model", selection: $model.selectedEntry) {
                ForEach(model.entries) { entry in
                    Text("\(entry.name)\(sizeSuffix(entry))")
                        .tag(Optional(entry))
                }
            }
            .fixedSize()
            .disabled(model.isBusy)

            Button(model.status == .idle ? "Download & Load" : "Reload") { model.load() }
                .disabled(model.isBusy)

            if let fraction = model.downloadFraction {
                ProgressView(value: fraction).frame(width: 90)
            } else if model.isBusy {
                ProgressView().controlSize(.small)
            }

            Text(model.status.label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Spacer()
        }
        .padding(10)
    }

    // MARK: - Picked image

    @ViewBuilder private var imageStrip: some View {
        HStack(spacing: 12) {
            if let picked = model.picked {
                Image(decorative: picked.cgImage.upright(picked.orientation), scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 160)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(.quaternary)
                    .frame(height: 120)
                    .overlay(
                        Text("No photo yet").foregroundStyle(.secondary).font(.callout))
            }
            VStack(alignment: .leading, spacing: 8) {
                PhotosPicker(
                    selection: $pickerItem, matching: .images, photoLibrary: .shared()
                ) {
                    Label(model.picked == nil ? "Choose Photo" : "Change Photo",
                          systemImage: "photo.on.rectangle")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.status == .idle)

                Text("Runs fully on-device. The photo never leaves your device.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(10)
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask about the photo (blank = describe it)", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
                .disabled(!model.canSend)
            if model.status == .generating {
                Button("Stop") { model.stop() }
            } else {
                Button("Ask", action: send).disabled(!model.canSend)
            }
            Button("New") { model.newChat() }
                .disabled(model.isBusy || model.bubbles.isEmpty)
        }
        .padding(10)
    }

    private func send() {
        let text = input
        input = ""
        model.send(text)
    }

    private func sizeSuffix(_ entry: CatalogEntry) -> String {
        guard let mb = entry.variant?.sizeMB else { return "" }
        return String(format: " (%.1f GB)", Double(mb) / 1000)
    }
}
