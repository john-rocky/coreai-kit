// ScanScreen.swift — a video file as a timeline, and the one place where the offline
// contract is visible: a scan delivers every sample it was asked for. Nothing is dropped,
// because nothing is on screen waiting for it.

import CoreAIOps
import PhotosUI
import SwiftUI

@MainActor
@Observable
final class ScanModel {
    struct Row: Identifiable {
        let id = UUID()
        let time: TimeInterval
        let labels: String
    }

    var rows: [Row] = []
    var status = "Pick a video"
    var isScanning = false
    var framesPerSecond = 1.0
    var skipStaticFrames = false
    var elapsed: Duration?

    private var task: Task<Void, Never>?
    private var videoURL: URL?

    /// Copies the picked item out of the photo library into a file the readers can open.
    func load(_ item: PhotosPickerItem?) {
        guard let item else { return }
        status = "Loading video…"
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self) else {
                status = "Could not read that item"
                return
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("scan-\(UUID().uuidString).mov")
            do {
                try data.write(to: url)
                videoURL = url
                status = "Ready — tap Scan"
            } catch {
                status = "Could not save the clip: \(error.localizedDescription)"
            }
        }
    }

    func scan() {
        guard let videoURL, task == nil else { return }
        rows = []
        elapsed = nil
        isScanning = true
        status = "Scanning…"
        let rate = framesPerSecond
        let change: Float? = skipStaticFrames ? 0.02 : nil
        task = Task {
            let clock = ContinuousClock()
            let started = clock.now
            do {
                for try await entry in scanVideo(
                    at: videoURL, framesPerSecond: rate, minimumChange: change)
                {
                    let names = Set(entry.value.map(\.label)).sorted()
                    rows.append(
                        Row(
                            time: entry.time,
                            labels: names.isEmpty ? "—" : names.joined(separator: ", ")))
                }
                elapsed = clock.now - started
                status = "\(rows.count) samples"
            } catch is CancellationError {
                status = "Cancelled"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
            isScanning = false
            task = nil
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        isScanning = false
    }
}

struct ScanScreen: View {
    @State private var model = ScanModel()
    @State private var pick: PhotosPickerItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            PhotosPicker(selection: $pick, matching: .videos) {
                Label("Choose a video", systemImage: "film")
            }
            .onChange(of: pick) { model.load(pick) }

            HStack {
                Text("\(model.framesPerSecond, format: .number.precision(.fractionLength(1))) samples/s")
                    .font(.caption.monospaced())
                Slider(value: $model.framesPerSecond, in: 0.2...10, step: 0.2)
            }
            Toggle("Skip frames that barely changed", isOn: $model.skipStaticFrames)
                .font(.callout)

            HStack {
                Button(model.isScanning ? "Cancel" : "Scan") {
                    model.isScanning ? model.cancel() : model.scan()
                }
                .buttonStyle(.borderedProminent)
                Spacer()
                if let elapsed = model.elapsed {
                    Text(String(format: "%.1f s", elapsed.seconds))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            Text(model.status).font(.footnote).foregroundStyle(.secondary)

            List(model.rows) { row in
                HStack {
                    Text(String(format: "%6.2f s", row.time))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                    Text(row.labels).font(.callout)
                }
            }
            .listStyle(.plain)
        }
        .padding()
        .onDisappear { model.cancel() }
    }
}
