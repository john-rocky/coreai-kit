import CoreAIKitCore
import CoreAIKitVision
import SwiftUI

@MainActor
@Observable
final class DepthCameraModel {
    var cameraImage: CGImage?
    var depthImage: CGImage?
    var status = "Loading model…"
    var inferenceMS: Double?
    /// Depth entries in the catalog; the picker restarts the feed on a new choice.
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?

    private var feed: CameraFeed?
    private var started = false

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        entries = await ModelCatalog.load().available(.depth)
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func restart() {
        stop()
        started = false
        Task { await start() }
    }

    func start() async {
        guard !started, let entry = selectedEntry else { return }
        started = true
        do {
            // Same gesture as the model card: the catalog id resolves the bundle.
            let estimator = try await DepthEstimator(catalog: entry.id) { progress in
                Task { @MainActor in
                    self.status = "Downloading… \(Int(progress.fraction * 100))%"
                }
            }
            status = "Starting camera…"
            let feed = CameraFeed(framesPerSecond: 5)
            self.feed = feed
            for await frame in try await feed.start() {
                let start = SuspendingClock.now
                let map = try await estimator.estimateDepth(for: frame)
                let elapsed = (SuspendingClock.now - start).components
                inferenceMS =
                    Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
                cameraImage = frame
                depthImage = map.cgImage()
                status = "Live"
            }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    func stop() {
        feed?.stop()
    }
}

struct DepthCameraView: View {
    @State private var model = DepthCameraModel()

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
            frame(model.cameraImage, label: "Camera")
            frame(model.depthImage, label: "Depth")
            HStack {
                Text(model.status)
                if let ms = model.inferenceMS {
                    Text(String(format: "· %.0f ms/frame", ms))
                }
                Spacer()
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
        }
        .padding(.vertical, 8)
        .task {
            await model.loadCatalog()
            await model.start()
        }
        .onChange(of: model.selectedEntry) { old, _ in
            if old != nil { model.restart() }
        }
        .onDisappear { model.stop() }
    }

    private func frame(_ image: CGImage?, label: String) -> some View {
        ZStack(alignment: .topLeading) {
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .scaledToFit()
            } else {
                Rectangle().fill(Color.gray.opacity(0.15)).aspectRatio(1, contentMode: .fit)
            }
            Text(label)
                .font(.caption2)
                .padding(4)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
                .padding(6)
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal, 8)
    }
}
