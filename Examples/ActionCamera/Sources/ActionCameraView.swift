import CoreAIKitCore
import CoreAIKitVision
import SwiftUI

@MainActor
@Observable
final class ActionCameraModel {
    var cameraImage: CGImage?
    var predictions: [ActionRecognizer.Prediction] = []
    var status = "Loading model…"
    var inferenceMS: Double?
    /// Video entries in the catalog; the picker restarts the feed on a new choice.
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?

    private var feed: CameraFeed?
    private var started = false

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        entries = await ModelCatalog.load().available(.video)
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
            let recognizer = try await ActionRecognizer(catalog: entry.id) { progress in
                Task { @MainActor in
                    self.status = "Downloading… \(Int(progress.fraction * 100))%"
                }
            }
            status = "Starting camera…"
            let feed = CameraFeed(framesPerSecond: 5)
            self.feed = feed
            // Rolling clip: the last 16 frames (~3 s at 5 fps); classify whenever the
            // recognizer is free, on the newest window.
            var clip: [CGImage] = []
            var busy = false
            for await frame in try await feed.start() {
                cameraImage = frame
                clip.append(frame)
                if clip.count > ActionRecognizer.frameCount { clip.removeFirst() }
                guard clip.count == ActionRecognizer.frameCount, !busy else { continue }
                busy = true
                let window = clip
                Task {
                    let start = SuspendingClock.now
                    let actions = try? await recognizer.classify(frames: window)
                    let elapsed = (SuspendingClock.now - start).components
                    await MainActor.run {
                        if let actions { self.predictions = actions }
                        self.inferenceMS =
                            Double(elapsed.seconds) * 1000
                            + Double(elapsed.attoseconds) / 1e15
                        self.status = "Live"
                        busy = false
                    }
                }
            }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    func stop() {
        feed?.stop()
    }
}

struct ActionCameraView: View {
    @State private var model = ActionCameraModel()

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
            ZStack(alignment: .topLeading) {
                if let image = model.cameraImage {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFit()
                } else {
                    Rectangle().fill(Color.gray.opacity(0.15)).aspectRatio(1, contentMode: .fit)
                }
                Text("Camera")
                    .font(.caption2)
                    .padding(4)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 6))
                    .padding(6)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 8)
            VStack(alignment: .leading, spacing: 4) {
                ForEach(model.predictions) { action in
                    HStack {
                        Text(action.label).lineLimit(1)
                        Spacer()
                        Text(String(format: "%.0f%%", action.probability * 100))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    .font(
                        action.id == model.predictions.first?.id
                            ? .callout.weight(.semibold) : .callout)
                }
            }
            .padding(.horizontal, 12)
            HStack {
                Text(model.status)
                if let ms = model.inferenceMS {
                    Text(String(format: "· %.0f ms/clip", ms))
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
}
