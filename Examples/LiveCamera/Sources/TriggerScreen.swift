// TriggerScreen.swift — the two-stage shape, which is the only one in which a
// vision-language model belongs near a live camera.
//
// A 103 MB detector watches every frame and decides. The 3.3 GB VLM runs when you tap a
// moment it kept — deliberately not automatically, because that download and that second
// per call are the app's decision to spend, not the kit's.

import AVFoundation
import CoreAIOps
import SwiftUI

@MainActor
@Observable
final class TriggerModel {
    struct Moment: Identifiable {
        let id = UUID()
        let image: CGImage
        let labels: String
        var caption: String?
        var isDescribing = false
    }

    var moments: [Moment] = []
    var stats: LiveStats?
    var status = "Tap Start"
    var label = "person"
    var isRunning = false
    var session: AVCaptureSession?

    private var task: Task<Void, Never>?
    private var watch: LiveWatch<WatchedMoment>?

    /// The labels this detector actually knows. A trigger on a label the model cannot emit
    /// never fires, and that failure is silent — so pick from the list.
    let labels = ["person", "cat", "dog", "cup", "laptop", "cell phone", "book", "chair"]

    func start() {
        guard task == nil else { return }
        isRunning = true
        status = "Loading detector…"
        task = Task { [label] in
            do {
                let watch = try await triggeredMoments(label: label)
                self.watch = watch
                session = watch.captureSession
                status = "Watching for “\(label)”"
                for try await fired in watch {
                    let names = Set(fired.value.detections.map(\.label)).sorted()
                    moments.insert(
                        Moment(
                            image: fired.value.image,
                            labels: names.joined(separator: ", ")),
                        at: 0)
                    if moments.count > 12 { moments.removeLast() }
                    stats = fired.stats
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
        isRunning = false
        status = "Stopped"
    }

    func describe(_ id: UUID) {
        guard let index = moments.firstIndex(where: { $0.id == id }),
            moments[index].caption == nil, !moments[index].isDescribing
        else { return }
        moments[index].isDescribing = true
        let moment = moments[index]
        Task {
            let text: String
            do {
                text = try await CoreAI.caption(moment.image)
            } catch {
                text = "Error: \(error.localizedDescription)"
            }
            guard let now = moments.firstIndex(where: { $0.id == id }) else { return }
            moments[now].caption = text
            moments[now].isDescribing = false
        }
    }
}

struct TriggerScreen: View {
    @State private var model = TriggerModel()

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                CameraPreview(session: model.session)
                VStack {
                    HStack {
                        StatsBadge(stats: model.stats)
                        Spacer()
                    }
                    .padding()
                    Spacer()
                }
            }
            .frame(height: 220)
            .background(.black)
            .clipped()

            HStack {
                Picker("Watch for", selection: $model.label) {
                    ForEach(model.labels, id: \.self) { Text($0).tag($0) }
                }
                Spacer()
                Button(model.isRunning ? "Stop" : "Start") {
                    model.isRunning ? model.stop() : model.start()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal)

            Text(model.status)
                .font(.footnote).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)

            List(model.moments) { moment in
                HStack(alignment: .top, spacing: 12) {
                    Image(decorative: moment.image, scale: 1)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 72, height: 72)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    VStack(alignment: .leading, spacing: 4) {
                        Text(moment.labels).font(.caption.weight(.medium))
                        if let caption = moment.caption {
                            Text(caption).font(.caption).foregroundStyle(.secondary)
                        } else if moment.isDescribing {
                            Text("Describing…").font(.caption).foregroundStyle(.secondary)
                        } else {
                            Button("Describe with a VLM") { model.describe(moment.id) }
                                .font(.caption)
                                .buttonStyle(.bordered)
                        }
                    }
                }
            }
            .listStyle(.plain)
        }
        .onDisappear { model.stop() }
    }
}
