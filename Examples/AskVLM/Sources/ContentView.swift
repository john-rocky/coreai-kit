// ContentView.swift — a coach + mirror for the on-device VLM Visual Intelligence app.
//   1. IDENTITY — "Your VLM inside Visual Intelligence", on-device / offline / no-cloud badges.
//   2. COACH — how to trigger the system Visual Intelligence flow; the answer appears in the
//      "Qwen3-VL" tab (the system shows a loading state while the model thinks, then the answer).
//   3. MIRROR — "Ask the VLM": runs the SAME engine path in the foreground (full memory), shows a
//      live "thinking…" state then the whole-image answer. Doubles as the one-time download + warm
//      step the background VI launch depends on.

import CoreGraphics
import PhotosUI
import SwiftUI

@MainActor
@Observable
final class ScreenModel {
    var pickedImage: Image?
    var pickedAspect: CGFloat = 4.0 / 3.0
    var answer: String?
    var milliseconds: Double?
    var running = false
    var status = ""
    private var lastPickedData: Data?

    func onPick(_ data: Data) {
        lastPickedData = data
        answer = nil
        milliseconds = nil
        status = ""
        if let cg = VLEngine.uprightCGImage(from: data) {
            pickedImage = Image(decorative: cg, scale: 1, orientation: .up)
            pickedAspect = CGFloat(cg.width) / CGFloat(max(cg.height, 1))
        }
    }

    /// Runs the on-device VLM on the picked image, in the foreground (full memory). Doubles as the
    /// download/warm step the background Visual Intelligence launch depends on.
    func ask() async {
        guard let data = lastPickedData, !running else { return }
        running = true
        answer = nil
        milliseconds = nil
        status = "Loading Qwen3-VL on-device… (first run downloads ~2.3 GB, then it's cached)"
        defer { running = false }
        guard let cg = VLEngine.uprightCGImage(from: data) else {
            status = "Could not decode that image."
            return
        }
        do {
            let result = try await VLEngine.shared.caption(cg)
            answer = result.text
            milliseconds = result.milliseconds
            status = MemoryProbe.snapshotLine()
        } catch {
            status = "VLM failed: \(error.localizedDescription)"
        }
    }
}

struct ContentView: View {
    @Environment(VLRouter.self) private var router
    @State private var model = ScreenModel()
    @State private var pickerItem: PhotosPickerItem?

    var body: some View {
        @Bindable var router = router
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    coachCard
                    tryItCard
                }
                .padding()
                .frame(maxWidth: 640)
                .frame(maxWidth: .infinity)
            }
            .background(backgroundWash.ignoresSafeArea())
            .navigationTitle("Qwen3-VL")
            .sheet(item: Binding(
                get: { router.presentedAnswer.map { IdentifiedString($0) } },
                set: { if $0 == nil { router.presentedAnswer = nil } }
            )) { item in
                answerDetail(answer: item.value)
            }
        }
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self) {
                    model.onPick(data)
                    await model.ask()
                }
            }
        }
    }

    private var backgroundWash: some View {
        LinearGradient(
            colors: [.indigo.opacity(0.12), .blue.opacity(0.06), .clear],
            startPoint: .top, endPoint: .center)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 30, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 54, height: 54)
                    .background(Self.gradient, in: RoundedRectangle(cornerRadius: 14))
                    .shadow(color: .indigo.opacity(0.35), radius: 8, y: 3)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Your VLM inside\nVisual Intelligence")
                        .font(.title2.bold())
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Qwen3-VL — running on your device")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                BadgePill("Qwen3-VL", system: "cpu")
                BadgePill("on-device", system: "iphone")
                BadgePill("no cloud", system: "airplane")
            }
            Text(
                "Apple's Visual Intelligence “ask” sends your photo to ChatGPT in the cloud. This "
                + "answers with your converted Qwen3-VL on the device GPU — offline, even in airplane mode.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var coachCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                Label("Use it from the camera or a screenshot", systemImage: "hand.point.up.left.fill")
                    .font(.headline)
                Text(
                    "Trigger Apple's Visual Intelligence, then open the **Qwen3-VL** tab. The system "
                    + "shows a thinking state while your model runs, then your on-device answer — "
                    + "with this app closed.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                VStack(spacing: 10) {
                    ForEach(Self.triggers, id: \.device) { t in TriggerStepRow(trigger: t) }
                }
            }
        }
    }

    private var tryItCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Label("Ask it here first", systemImage: "wand.and.stars")
                        .font(.headline)
                    Spacer()
                    if let ms = model.milliseconds {
                        Text(String(format: "%.0f ms · on-device", ms))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Text(
                    "Runs the exact same engine path the Visual Intelligence query uses. Do this once "
                    + "so the model is downloaded + warmed before you trigger the system flow.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let image = model.pickedImage {
                    image
                        .resizable()
                        .aspectRatio(model.pickedAspect, contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.quaternary, lineWidth: 1))
                }

                if model.running {
                    HStack(spacing: 10) {
                        ProgressView().controlSize(.small)
                        Text(model.status.isEmpty ? "Thinking…" : model.status)
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
                } else if let answer = model.answer {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(answer)
                            .font(.callout)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                        if !model.status.isEmpty {
                            Text(model.status)
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.tertiary)
                        }
                    }
                }

                let pickerTitle = model.pickedImage == nil ? "Pick a photo to ask the VLM" : "Try another photo"
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Label(pickerTitle, systemImage: "photo.on.rectangle.angled")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.running)
            }
        }
    }

    private func answerDetail(answer: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile").font(.system(size: 48))
            Text("On-device answer").font(.title2.bold())
            Text(answer).font(.body).multilineTextAlignment(.center).padding()
            Text("Computed by your Qwen3-VL on the device GPU — opened from a Visual Intelligence result.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }.padding()
    }

    static let gradient = LinearGradient(
        colors: [.indigo, .blue, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)

    struct Trigger { let icon: String; let device: String; let action: String; let tint: Color }
    static let triggers: [Trigger] = [
        .init(icon: "camera.aperture", device: "iPhone 16 / 17",
              action: "Press and hold the Camera Control button.", tint: .blue),
        .init(icon: "iphone", device: "Other iPhone",
              action: "Action button, or add Visual Intelligence to Control Center.", tint: .indigo),
        .init(icon: "camera.viewfinder", device: "Any screenshot",
              action: "Side + Volume Up, then tap Visual Intelligence.", tint: .purple),
    ]
}

// MARK: - Reusable bits

private struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
            .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(.quaternary, lineWidth: 1))
    }
}

private struct BadgePill: View {
    let text: String
    let system: String
    init(_ text: String, system: String) {
        self.text = text
        self.system = system
    }
    var body: some View {
        Label(text, systemImage: system)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 9).padding(.vertical, 5)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.quaternary, lineWidth: 0.5))
    }
}

private struct TriggerStepRow: View {
    let trigger: ContentView.Trigger
    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: trigger.icon)
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(trigger.tint.gradient, in: RoundedRectangle(cornerRadius: 12))
            VStack(alignment: .leading, spacing: 2) {
                Text(trigger.device).font(.subheadline.bold())
                Text(trigger.action)
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(trigger.tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct IdentifiedString: Identifiable {
    let value: String
    var id: String { value }
    init(_ value: String) { self.value = value }
}
