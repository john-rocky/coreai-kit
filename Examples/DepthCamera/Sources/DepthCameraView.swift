import CoreAIKitVision
import SwiftUI

@MainActor
@Observable
final class DepthCameraModel {
    var cameraImage: CGImage?
    var depthImage: CGImage?
    var status = "Loading model…"
    var inferenceMS: Double?

    private var feed: CameraFeed?
    private var started = false

    func start() async {
        guard !started else { return }
        started = true
        do {
            let estimator = try await DepthEstimator { progress in
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
        .task { await model.start() }
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
