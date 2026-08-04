// DepthScreen.swift — live monocular depth, the capability Apple ships no API for, at
// 54 MB. `AVDepthData` needs two cameras; this needs one.

import CoreAIOps
import SwiftUI

@MainActor
@Observable
final class DepthModel {
    var depthImage: CGImage?
    var stats: LiveStats?
    var status = "Tap Start"
    var isRunning = false

    private var task: Task<Void, Never>?
    private var watch: LiveWatch<DepthMap>?

    func start() {
        guard task == nil else { return }
        isRunning = true
        status = "Loading Depth Anything 3…"
        task = Task {
            do {
                let watch = try await liveDepth()
                self.watch = watch
                status = "Live"
                for try await frame in watch {
                    // `cgImage()` normalizes the map to the frame's own range, so a wall
                    // and a room both use the full grey ramp.
                    depthImage = frame.value.cgImage()
                    stats = frame.stats
                }
            } catch is CancellationError {
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
            isRunning = false
        }
    }

    func stop() {
        watch?.stop()
        task?.cancel()
        task = nil
        watch = nil
        depthImage = nil
        stats = nil
        isRunning = false
        status = "Stopped"
    }
}

struct DepthScreen: View {
    @State private var model = DepthModel()

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                if let depthImage = model.depthImage {
                    Image(decorative: depthImage, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                VStack {
                    HStack {
                        StatsBadge(stats: model.stats)
                        Spacer()
                    }
                    .padding()
                    Spacer()
                }
            }
            .clipped()

            VStack(spacing: 12) {
                Text(model.status).font(.footnote).foregroundStyle(.secondary)
                Button(model.isRunning ? "Stop" : "Start") {
                    model.isRunning ? model.stop() : model.start()
                }
                .buttonStyle(.borderedProminent)
                .frame(maxWidth: .infinity)
            }
            .padding()
        }
        .onDisappear { model.stop() }
    }
}
