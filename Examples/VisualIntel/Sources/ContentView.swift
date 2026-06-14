// ContentView.swift — the in-app surface, designed as a *coach + mirror*, not an experience.
//
// VisualIntel's value lives in the system: your converted RF-DETR / CLIP models answer Apple's
// Visual Intelligence with the app closed. So this screen does three legible jobs:
//   1. IDENTITY — say plainly what runs ("Your own model inside Visual Intelligence") + a standing
//      "on-device / offline / YOUR models, not Apple Intelligence" badge row.
//   2. COACH — show, large, HOW to trigger the real flow (Camera Control / Action button /
//      screenshot) on each platform, and what you'll see.
//   3. MIRROR — a "Try it" panel that runs the EXACT same engine path the Visual Intelligence
//      query uses (RF-DETR boxes drawn over your image + CLIP photo matches), so you can verify
//      the models on a Mac today without invoking the system.
// It also hosts the screens that `OpenIntent` / "Continue in app" route to (router sheets).

import CoreGraphics
import ImageIO
import PhotosUI
import SwiftUI

@MainActor
@Observable
final class ScreenModel {
    var status = "Pick any photo — your models will detect objects and find similar shots."
    /// The decoded, upright image shown in the mirror (same pixels the engine analyzed).
    var pickedImage: Image?
    var pickedAspect: CGFloat = 4.0 / 3.0
    var detections: [DetectionResult] = []
    var photos: [PhotoResult] = []
    var searchMilliseconds: Double?
    var phase: AnalysisPhase?
    var isRunning = false

    var indexedCount = VisualIntelEngine.shared.index.count
    var indexing = false
    var indexProgress = ""

    // On-device VLM (Qwen3-VL-2B) — the foreground control: the EXACT same model the Visual
    // Intelligence query would run, but with the full app memory budget. Proves the path and is the
    // jetsam control for the background measurement.
    var vlmAnswer: String?
    var vlmMilliseconds: Double?
    var vlmRunning = false
    var vlmStatus = ""
    /// Mirrors the persisted experimental flag so the toggle reflects/sets it.
    var runVLMInVI = VLMVISettings.runInVisualIntelligence
    /// The last picked image's bytes, so "Ask the VLM" runs on the same photo shown above.
    private var lastPickedData: Data?

    func setRunVLMInVI(_ on: Bool) {
        runVLMInVI = on
        VLMVISettings.runInVisualIntelligence = on
    }

    /// Runs the on-device VLM on the picked image, in the foreground (full memory). Doubles as the
    /// one-time download/cache step the background Visual Intelligence launch depends on.
    func askVLM(prompt: String = "What is in this image? Answer in one short sentence.") async {
        guard let data = lastPickedData, !vlmRunning else { return }
        vlmRunning = true
        vlmAnswer = nil
        vlmMilliseconds = nil
        vlmStatus = "Loading Qwen3-VL on-device… (first run downloads ~2.3 GB, then it's cached)"
        defer { vlmRunning = false }
        guard let cg = Self.uprightCGImage(from: data) else {
            vlmStatus = "Could not decode that image."
            return
        }
        do {
            let answer = try await VisualIntelEngine.shared.caption(cg, prompt: prompt)
            vlmAnswer = answer.text
            vlmMilliseconds = answer.milliseconds
            vlmStatus = MemoryProbe.snapshotLine()
        } catch {
            vlmStatus = "VLM failed: \(error.localizedDescription)"
        }
    }

    func runSearch(on data: Data) async {
        lastPickedData = data
        vlmAnswer = nil
        vlmMilliseconds = nil
        vlmStatus = ""
        // Display image: an upright copy stays on the main actor inside the SwiftUI `Image`.
        guard let displayCG = Self.uprightCGImage(from: data) else {
            status = "Could not decode that image."
            return
        }
        pickedImage = Image(decorative: displayCG, scale: 1, orientation: .up)
        pickedAspect = CGFloat(displayCG.width) / CGFloat(max(displayCG.height, 1))
        detections = []
        photos = []
        searchMilliseconds = nil
        isRunning = true
        phase = .detecting
        defer { isRunning = false }

        // A second, pixel-identical instance is the one we *send* into the engine (clean region
        // isolation — it is never touched again on this actor). The VI query uses this same call.
        guard let cg = Self.uprightCGImage(from: data) else { return }
        let start = SuspendingClock.now
        let analysis =
            (try? await VisualIntelEngine.shared.analyze(cg, topPhotos: 6, minimumPhotoScore: 0) {
                newPhase in
                Task { @MainActor in self.phase = newPhase }
            }) ?? AnalysisResult(detections: [], photos: [])
        let elapsed = (SuspendingClock.now - start).components
        searchMilliseconds = Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
        detections = analysis.detections
        photos = analysis.photos
        phase = nil
        let objs = analysis.detections.count
        status =
            objs == 0
            ? "No objects detected — try a photo with a clear subject."
            : "\(objs) object\(objs == 1 ? "" : "s") detected"
            + (indexedCount == 0
                ? " · index your photos to enable similarity"
                : " · \(analysis.photos.count) similar photo\(analysis.photos.count == 1 ? "" : "s")")
    }

    func indexPhotos() async {
        guard !indexing else { return }
        indexing = true
        defer { indexing = false }
        let indexer = PhotoLibraryIndexer()
        do {
            try await indexer.index(into: VisualIntelEngine.shared) { done, total in
                Task { @MainActor in
                    self.indexProgress = "Indexing \(done)/\(total)"
                    self.indexedCount = VisualIntelEngine.shared.index.count
                }
            }
            indexProgress = ""
            indexedCount = VisualIntelEngine.shared.index.count
            status = "Indexed \(indexedCount) photos with your CLIP model."
        } catch {
            status = "Indexing failed: \(error.localizedDescription)"
        }
    }

    /// Decodes `data` to an *upright* (EXIF-applied) CGImage capped at `maxPixel` on the long side,
    /// so the mirror's overlay boxes and the displayed image share one coordinate space.
    static func uprightCGImage(from data: Data, maxPixel: Int = 2048) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

struct ContentView: View {
    @Environment(AppRouter.self) private var router
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
                    vlmCard
                    indexCard
                }
                .padding()
                .frame(maxWidth: 640)
                .frame(maxWidth: .infinity)
            }
            .background(backgroundWash.ignoresSafeArea())
            .navigationTitle("VisualIntel")
            .sheet(item: Binding(
                get: { router.presentedDetectionLabel.map { IdentifiedString($0) } },
                set: { if $0 == nil { router.presentedDetectionLabel = nil } }
            )) { item in
                detectionDetail(label: item.value)
            }
            .sheet(item: Binding(
                get: { router.presentedPhotoID.map { IdentifiedString($0) } },
                set: { if $0 == nil { router.presentedPhotoID = nil } }
            )) { item in
                photoDetail(id: item.value)
            }
            .sheet(isPresented: $router.showResults) {
                resultsSheet
            }
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
                    await model.runSearch(on: data)
                }
            }
        }
    }

    private var backgroundWash: some View {
        LinearGradient(
            colors: [.purple.opacity(0.10), .blue.opacity(0.06), .clear],
            startPoint: .top, endPoint: .center)
    }

    // MARK: 1. Identity header

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                Image(systemName: "sparkle.magnifyingglass")
                    .font(.system(size: 30, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 54, height: 54)
                    .background(Self.viGradient, in: RoundedRectangle(cornerRadius: 14))
                    .shadow(color: .purple.opacity(0.35), radius: 8, y: 3)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Your own model inside\nVisual Intelligence")
                        .font(.title2.bold())
                        .fixedSize(horizontal: false, vertical: true)
                    Text("RF-DETR · CLIP — running on your device")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                BadgePill("RF-DETR + CLIP", system: "cube.transparent")
                BadgePill("on-device", system: "cpu")
                BadgePill("offline", system: "airplane")
            }
            Text(
                "Your converted models — not Apple Intelligence — answer the system's visual search. "
                + "Everything runs locally, even in airplane mode.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: 2. Coach — how to trigger the real Visual Intelligence flow

    private var coachCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                Label("Use it from the camera or a screenshot", systemImage: "hand.point.up.left.fill")
                    .font(.headline)
                Text(
                    "VisualIntel has no search box. You trigger Apple's Visual Intelligence, and "
                    + "your models answer inside the system's own UI — with this app closed.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                VStack(spacing: 10) {
                    ForEach(Self.triggers, id: \.device) { t in
                        TriggerStepRow(trigger: t)
                    }
                }

                Divider()

                VStack(alignment: .leading, spacing: 8) {
                    Text("Then your models answer right in Apple's result list:")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        ExampleChip(system: "viewfinder", text: "cat · 94%", tint: .blue)
                        ExampleChip(system: "photo.stack", text: "3 similar photos", tint: .teal)
                    }
                }
            }
        }
    }

    // MARK: 3. Mirror — "Try it" with the identical engine path

    private var tryItCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Label("Try it here first", systemImage: "wand.and.stars")
                        .font(.headline)
                    Spacer()
                    if let ms = model.searchMilliseconds {
                        Text(String(format: "%.0f ms · on-device", ms))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Text("Runs the exact same RF-DETR + CLIP path the Visual Intelligence query uses.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                if let image = model.pickedImage {
                    AnnotatedImageView(
                        image: image, aspectRatio: model.pickedAspect, detections: model.detections)
                }

                if model.isRunning {
                    PhaseTrace(phase: model.phase ?? .detecting)
                } else if model.pickedImage != nil {
                    resultsSummary
                }

                // Capture a Sendable `String` (PhotosPicker's label closure is `@Sendable`, so it
                // can't read the main-actor `model` directly).
                let pickerTitle =
                    model.pickedImage == nil ? "Pick a photo to run your models" : "Try another photo"
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Label(pickerTitle, systemImage: "photo.on.rectangle.angled")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.isRunning)

                Text(model.status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    // MARK: 3b. On-device VLM — the cloud-free "ask" (Apple's VI "ask" calls ChatGPT in the cloud)

    private var vlmCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Label("Ask on-device — no cloud", systemImage: "brain.head.profile")
                        .font(.headline)
                    Spacer()
                    if let ms = model.vlmMilliseconds {
                        Text(String(format: "%.0f ms · on-device", ms))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Text(
                    "Apple's Visual Intelligence “ask” sends your photo to ChatGPT in the cloud. "
                    + "This runs your converted **Qwen3-VL** on the device GPU instead — same idea, "
                    + "no network, even in airplane mode.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                Button {
                    Task { await model.askVLM() }
                } label: {
                    Label(
                        model.vlmRunning ? "Asking the VLM…" : "Ask the VLM about the photo above",
                        systemImage: "sparkles")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(model.pickedImage == nil || model.vlmRunning)

                if model.vlmRunning {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(model.vlmStatus).font(.caption).foregroundStyle(.secondary)
                    }
                } else if let answer = model.vlmAnswer {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(answer)
                            .font(.callout)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                        if !model.vlmStatus.isEmpty {
                            Text(model.vlmStatus)
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.tertiary)
                        }
                    }
                } else if model.pickedImage == nil {
                    Text("Pick a photo above first, then ask.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Text(
                    "This is an in-app demo. The VLM has its own Visual Intelligence tab — see the "
                    + "Qwen3-VL (AskVLM) example.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func answerDetail(answer: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile").font(.system(size: 48))
            Text("On-device answer").font(.title2.bold())
            Text(answer)
                .font(.body)
                .multilineTextAlignment(.center)
                .padding()
            Text("Computed by your Qwen3-VL on the device GPU — opened from a Visual Intelligence result.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }.padding()
    }

    @ViewBuilder private var resultsSummary: some View {
        if !model.detections.isEmpty {
            FlowChips(
                items: model.detections.enumerated().map { idx, d in
                    (AnnotatedImageView.palette[idx % AnnotatedImageView.palette.count],
                     "\(d.label.capitalized) \(Int((d.score * 100).rounded()))%")
                })
        }
        if !model.photos.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Text("Similar photos (CLIP)").font(.caption.bold()).foregroundStyle(.secondary)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(model.photos, id: \.id) { p in
                            photoThumb(p)
                        }
                    }
                }
            }
        }
    }

    private func photoThumb(_ p: PhotoResult) -> some View {
        ZStack(alignment: .bottomTrailing) {
            thumbnail(p.thumbnailJPEG)
                .frame(width: 84, height: 84)
                .clipShape(RoundedRectangle(cornerRadius: 10))
            Text("\(Int((p.score * 100).rounded()))%")
                .font(.caption2.bold())
                .padding(.horizontal, 5).padding(.vertical, 2)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(4)
        }
    }

    // MARK: 4. CLIP index — light up the similarity surface

    private var indexCard: some View {
        Card {
            VStack(alignment: .leading, spacing: 10) {
                Label("Light up “similar photos”", systemImage: "square.stack.3d.up.fill")
                    .font(.headline)
                Text(
                    "The similar-photos surface stays empty until you embed your library with your "
                    + "CLIP model. Indexed on-device; nothing leaves your phone.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack {
                    Image(systemName: model.indexedCount > 0 ? "checkmark.circle.fill" : "circle.dashed")
                        .foregroundStyle(model.indexedCount > 0 ? .green : .secondary)
                    Text("\(model.indexedCount) photos indexed")
                        .font(.subheadline)
                    Spacer()
                }
                if model.indexing {
                    ProgressView(model.indexProgress.isEmpty ? "Indexing…" : model.indexProgress)
                } else {
                    Button {
                        Task { await model.indexPhotos() }
                    } label: {
                        Label(
                            model.indexedCount > 0 ? "Re-index my photos" : "Index my photos",
                            systemImage: "square.stack.3d.up")
                    }
                    .buttonStyle(.bordered)
                }
            }
        }
    }

    // MARK: Shared bits + router landings

    @ViewBuilder private func thumbnail(_ data: Data?) -> some View {
        if let data, let image = platformImage(data) {
            image.resizable().scaledToFill()
        } else {
            RoundedRectangle(cornerRadius: 8).fill(.quaternary)
        }
    }

    private func resultRow(thumb: Data?, title: String, subtitle: String) -> some View {
        HStack(spacing: 12) {
            thumbnail(thumb).frame(width: 52, height: 52)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading) {
                Text(title).font(.body)
                Text(subtitle).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private func detectionDetail(label: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "viewfinder").font(.system(size: 48))
            Text(label.capitalized).font(.title.bold())
            Text("Opened from a Visual Intelligence result via OpenIntent.")
                .font(.caption).foregroundStyle(.secondary).multilineTextAlignment(.center)
        }.padding()
    }

    private func photoDetail(id: String) -> some View {
        VStack(spacing: 16) {
            if let data = VisualIntelEngine.shared.index.thumbnailData(for: id),
                let image = platformImage(data) {
                image.resizable().scaledToFit().frame(maxHeight: 320)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            Text("Matching photo").font(.title2.bold())
            Text(id).font(.caption2).foregroundStyle(.tertiary).lineLimit(1)
        }.padding()
    }

    private var resultsSheet: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Full results").font(.title2.bold())
                if let r = router.results {
                    if let answer = r.vlmAnswer, !answer.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Label("On-device answer (Qwen3-VL)", systemImage: "brain.head.profile")
                                .font(.headline)
                            Text(answer)
                                .font(.callout)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                            if let ms = r.vlmMilliseconds {
                                Text(String(format: "%.0f ms · on-device, no cloud", ms))
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        Divider()
                    }
                    ForEach(r.detections, id: \.label) { d in
                        resultRow(thumb: d.thumbnailJPEG, title: d.label.capitalized,
                                  subtitle: "\(Int((d.score * 100).rounded()))% confidence")
                    }
                    ForEach(r.photos, id: \.id) { p in
                        resultRow(thumb: p.thumbnailJPEG, title: "Library photo",
                                  subtitle: "\(Int((p.score * 100).rounded()))% similar")
                    }
                }
            }.padding()
        }
    }

    // MARK: Static content

    static let viGradient = LinearGradient(
        colors: [.purple, .blue, .teal], startPoint: .topLeading, endPoint: .bottomTrailing)

    struct Trigger {
        let icon: String
        let device: String
        let action: String
        let tint: Color
    }

    static let triggers: [Trigger] = [
        .init(icon: "camera.aperture", device: "iPhone 16 / 17",
              action: "Press and hold the Camera Control button.", tint: .blue),
        .init(icon: "iphone", device: "Other iPhone",
              action: "Action button, or add Visual Intelligence to Control Center.", tint: .indigo),
        .init(icon: "camera.viewfinder", device: "Any screenshot",
              action: "Side + Volume Up, then tap Visual Intelligence.", tint: .purple),
        .init(icon: "desktopcomputer", device: "iPad / Mac",
              action: "Take a screenshot → Search with Visual Intelligence.", tint: .teal),
    ]
}

// MARK: - Reusable building blocks

/// A soft material card — the shared visual shell across the demo apps.
private struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
            .overlay(
                RoundedRectangle(cornerRadius: 20).strokeBorder(.quaternary, lineWidth: 1))
    }
}

/// Standing capability badge (`<thing> · on-device · offline`).
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

/// One big, tappable-looking "how to launch" row in the coach card.
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
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(trigger.tint.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct ExampleChip: View {
    let system: String
    let text: String
    let tint: Color
    var body: some View {
        Label(text, systemImage: system)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(tint.opacity(0.15), in: Capsule())
            .foregroundStyle(tint)
    }
}

/// Compact wrap of colored result chips (one per detection).
private struct FlowChips: View {
    let items: [(Color, String)]
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(spacing: 5) {
                        Circle().fill(item.0).frame(width: 8, height: 8)
                        Text(item.1).font(.caption.weight(.semibold))
                    }
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(item.0.opacity(0.12), in: Capsule())
                }
            }
        }
    }
}

/// Live two-step trace of the engine work, mirroring the VI query's phases.
private struct PhaseTrace: View {
    let phase: AnalysisPhase
    private var rank: Int {
        switch phase {
        case .detecting: 0
        case .matching: 1
        case .done: 2
        }
    }
    private static let steps: [(rank: Int, icon: String, text: String)] = [
        (0, "viewfinder", "RF-DETR finding objects"),
        (1, "photo.stack", "CLIP matching your photos"),
    ]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Self.steps, id: \.rank) { step in
                HStack(spacing: 10) {
                    if rank > step.rank {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                    } else if rank == step.rank {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "circle").foregroundStyle(.tertiary)
                    }
                    Label(step.text, systemImage: step.icon)
                        .font(.subheadline)
                        .foregroundStyle(rank >= step.rank ? .primary : .secondary)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 14))
    }
}

/// The mirror's hero: the picked image with RF-DETR boxes drawn in the image's own coordinate
/// space (normalized, top-left origin — matches SwiftUI), each labeled with class + confidence.
struct AnnotatedImageView: View {
    let image: Image
    let aspectRatio: CGFloat
    let detections: [DetectionResult]

    static let palette: [Color] = [.blue, .green, .orange, .pink, .purple, .teal, .yellow, .cyan]

    var body: some View {
        image
            .resizable()
            .aspectRatio(aspectRatio, contentMode: .fit)
            .overlay {
                GeometryReader { geo in
                    ForEach(Array(detections.enumerated()), id: \.offset) { idx, d in
                        boxOverlay(d, color: Self.palette[idx % Self.palette.count], in: geo.size)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.quaternary, lineWidth: 1))
    }

    private func boxOverlay(_ d: DetectionResult, color: Color, in size: CGSize) -> some View {
        let w = d.box.width * size.width
        let h = d.box.height * size.height
        return ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 6).strokeBorder(color, lineWidth: 3)
            Text("\(d.label.capitalized) \(Int((d.score * 100).rounded()))%")
                .font(.caption2.bold())
                .lineLimit(1)
                .fixedSize()
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(color, in: Capsule())
                .foregroundStyle(.white)
                .padding(4)
        }
        .frame(width: max(w, 1), height: max(h, 1))
        .position(x: d.box.midX * size.width, y: d.box.midY * size.height)
    }
}

/// Wraps a `String` as `Identifiable` for `.sheet(item:)`.
private struct IdentifiedString: Identifiable {
    let value: String
    var id: String { value }
    init(_ value: String) { self.value = value }
}

#if canImport(UIKit)
import UIKit
func platformImage(_ data: Data) -> Image? { UIImage(data: data).map { Image(uiImage: $0) } }
#elseif canImport(AppKit)
import AppKit
func platformImage(_ data: Data) -> Image? { NSImage(data: data).map { Image(nsImage: $0) } }
#endif
