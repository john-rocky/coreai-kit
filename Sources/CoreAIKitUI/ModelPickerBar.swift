// ModelPickerBar.swift — catalog-driven model picker with a load button, status label,
// and download progress.

import SwiftUI

public struct ModelPickerBar: View {
    @Binding private var selection: CatalogEntry?
    private let entries: [CatalogEntry]
    private let isBusy: Bool
    private let statusText: String
    private let downloadFraction: Double?
    private let loadTitle: String
    private let onLoad: () -> Void

    public init(
        entries: [CatalogEntry],
        selection: Binding<CatalogEntry?>,
        isBusy: Bool,
        statusText: String,
        downloadFraction: Double? = nil,
        loadTitle: String = "Load",
        onLoad: @escaping () -> Void
    ) {
        self.entries = entries
        self._selection = selection
        self.isBusy = isBusy
        self.statusText = statusText
        self.downloadFraction = downloadFraction
        self.loadTitle = loadTitle
        self.onLoad = onLoad
    }

    public var body: some View {
        HStack {
            Picker("Model", selection: $selection) {
                ForEach(entries) { entry in
                    Text(label(for: entry)).tag(Optional(entry))
                }
            }
            .fixedSize()
            .disabled(isBusy)
            Button(loadTitle, action: onLoad)
                .disabled(isBusy || selection == nil)
            Spacer()
            if let downloadFraction {
                ProgressView(value: downloadFraction).frame(width: 80)
            }
            Text(statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(10)
    }

    private func label(for entry: CatalogEntry) -> String {
        if let size = entry.variant?.sizeMB {
            return "\(entry.name) (\(size) MB)"
        }
        return entry.name
    }
}
