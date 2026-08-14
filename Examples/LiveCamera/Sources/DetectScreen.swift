// DetectScreen.swift — live object detection. The model is a picker, because the whole
// claim of the op layer is that the model behind a task can change without the app changing.

import AVFoundation
import CoreAIOps
import SwiftUI

@MainActor
@Observable
final class DetectModel {
    var detections: [Detection] = []
    var stats: LiveStats?
    var status = "Tap Start"
    var modelID = "rf-detr"
    var isRunning = false
    var session: AVCaptureSession?

    private var task: Task<Void, Never>?
    private var watch: LiveWatch<[Detection]>?

    var availableModels: [String] {
        ModelCatalog.builtin.available(.detection).map(\.id)
    }

    func start() {
        guard task == nil else { return }
        isRunning = true
        status = "Loading \(modelID)…"
        task = Task { [modelID] in
            do {
                let watch = try await liveDetections(model: modelID)
                self.watch = watch
                session = watch.captureSession
                status = "Live · \(modelID)"
                for try await frame in watch {
                    detections = frame.value
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
        session = nil
        detections = []
        stats = nil
        isRunning = false
        status = "Stopped"
    }

    func restart() {
        stop()
        start()
    }
}

struct DetectScreen: View {
    @State private var model = DetectModel()

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                CameraPreview(session: model.session)
                DetectionOverlay(detections: model.detections)
                VStack {
                    HStack {
                        StatsBadge(stats: model.stats)
                        Spacer()
                    }
                    .padding()
                    Spacer()
                }
            }
            .background(.black)
            .clipped()

            VStack(alignment: .leading, spacing: 12) {
                Text(model.status).font(.footnote).foregroundStyle(.secondary)
                Picker("Model", selection: $model.modelID) {
                    ForEach(model.availableModels, id: \.self) { Text($0).tag($0) }
                }
                .pickerStyle(.segmented)
                .onChange(of: model.modelID) { if model.isRunning { model.restart() } }

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
