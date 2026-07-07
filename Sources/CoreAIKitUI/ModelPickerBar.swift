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
        HStack(spacing: 10) {
            // A Menu (not a Picker) so the trigger can stay compact — just the model name,
            // truncating when narrow — while the dropdown still lists the full "name (size)".
            // A `.fixedSize()` Picker showing "Gemma 4 E2B (4931 MB)" overflows iPhone width
            // and shoves the load button and status off-screen.
            Menu {
                ForEach(entries) { entry in
                    Button {
                        selection = entry
                    } label: {
                        if entry == selection {
                            Label(label(for: entry), systemImage: "checkmark")
                        } else {
                            Text(label(for: entry))
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(selection?.name ?? "Model").lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .disabled(isBusy)

            Button(loadTitle, action: onLoad)
                .fixedSize()
                .disabled(isBusy || selection == nil)

            Spacer(minLength: 8)

            if let downloadFraction {
                ProgressView(value: downloadFraction).frame(width: 60)
            }
            Text(statusText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize()
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
