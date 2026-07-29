// The whole app. A receipt photo goes in, a row appears. Nothing leaves the device, so there is
// no key to store, no bill to watch, and nothing to declare about where the data went.

import CoreAIOps
import PhotosUI
import SwiftUI

@main
struct ScanToTypeApp: App {
    var body: some Scene { WindowGroup { ScanView() } }
}

struct ScanView: View {
    @State private var scanned: [Receipt] = []
    @State private var picked: PhotosPickerItem?
    @State private var status: String?

    var body: some View {
        NavigationStack {
            List {
                if let status { Text(status).foregroundStyle(.secondary) }
                ForEach(Array(scanned.enumerated()), id: \.offset) { _, r in
                    LabeledContent(r.merchant) {
                        Text("\(r.total, format: .number) · \(r.date)")
                    }
                    
                }
            }
            .navigationTitle("Scanned")
            .toolbar {
                PhotosPicker(selection: $picked, matching: .images) { Image(systemName: "camera") }
            }
            .onChange(of: picked) { _, item in Task { await scan(item) } }
            .task {
                // Fetch the models before the first photo, so the first scan is not the download.
                CoreAI.onDownload { p in
                    status = "Preparing… \(Int(p.fraction * 100))%"
                }
                try? await CoreAI.prepare(.read, .extract)
                status = nil
            }
        }
    }

    private func scan(_ item: PhotosPickerItem?) async {
        guard let item,
              let data = try? await item.loadTransferable(type: Data.self),
              let image = CGImage.from(data)
        else { return }
        status = "Reading…"
        do {
            scanned.append(try await scan(image, as: Receipt.self))
            status = nil
        } catch {
            status = error.localizedDescription
        }
    }
}

extension CGImage {
    static func from(_ data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}
