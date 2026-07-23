// OpDetailView.swift — one screen per op: gather the input its kind needs, run the
// one-line call, show the result. Download progress from `CoreAI.onDownload` surfaces
// here automatically on first use.

import Charts
import CoreAIOps
import SwiftUI
import UniformTypeIdentifiers

@MainActor @Observable
final class OpRunModel {
    var isRunning = false
    var result: OpResult?
    var errorMessage: String?
    var elapsed: Double?

    func run(_ op: CoreAI.Op, _ snapshot: OpInputSnapshot) {
        guard !isRunning else { return }
        isRunning = true
        errorMessage = nil
        result = nil
        Task {
            let start = SuspendingClock.now
            do {
                result = try await runOp(op, snapshot)
                let parts = (SuspendingClock.now - start).components
                elapsed = Double(parts.seconds) + Double(parts.attoseconds) / 1e18
            } catch {
                errorMessage = error.localizedDescription
            }
            isRunning = false
        }
    }
}

struct OpDetailView: View {
    let op: CoreAI.Op

    @State private var model = OpRunModel()
    @State private var text = ""
    @State private var query = Samples.searchQuery
    @State private var documentsText = Samples.searchDocuments.joined(separator: "\n")
    @State private var seriesText = Samples.series.map { String(format: "%.1f", $0) }
        .joined(separator: ", ")
    @State private var image: CGImage?
    @State private var mediaURL: URL?
    @State private var showingImporter = false

    private var downloads = DownloadHub.shared

    init(op: CoreAI.Op) {
        self.op = op
        _text = State(initialValue: op.sampleText)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(op.summary).font(.subheadline).foregroundStyle(.secondary)

                inputSection

                Button(action: { model.run(op, snapshot()) }) {
                    Label(
                        model.isRunning ? "Running…" : "Run CoreAI.\(op.name)",
                        systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isRunning)

                if let progress = downloads.latest, downloads.isActive {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: progress.fraction)
                        Text("downloading \(progress.currentFile)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                if model.isRunning { ProgressView() }
                if let message = model.errorMessage {
                    Text(message).font(.callout).foregroundStyle(.red)
                }
                if let result = model.result {
                    ResultView(result: result, inputImage: image)
                    if let elapsed = model.elapsed {
                        Text(String(format: "%.1f s", elapsed))
                            .font(.caption2.monospaced())
                            .foregroundStyle(.tertiary)
                    }
                }
            }
            .padding()
        }
        .navigationTitle("CoreAI.\(op.name)")
        .fileImporter(
            isPresented: $showingImporter, allowedContentTypes: importerTypes
        ) { pick in
            guard case .success(let url) = pick else { return }
            importFile(url)
        }
    }

    // MARK: - Input

    @ViewBuilder
    private var inputSection: some View {
        switch op.inputKind {
        case .text:
            TextEditor(text: $text)
                .font(.callout)
                .frame(minHeight: 120)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
        case .image:
            HStack {
                Button("Pick image…") { showingImporter = true }
                Button("Use sample") { image = Samples.image() }
            }
            .buttonStyle(.bordered)
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        case .audioFile, .videoFile:
            HStack {
                Button(op.inputKind == .audioFile ? "Pick audio…" : "Pick video…") {
                    showingImporter = true
                }
                .buttonStyle(.bordered)
                if let mediaURL {
                    Text(mediaURL.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        case .series:
            TextEditor(text: $seriesText)
                .font(.caption.monospaced())
                .frame(minHeight: 80)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            Text("Comma-separated numbers; the forecast continues the series.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        case .queryAndDocs:
            TextField("Query", text: $query)
                .textFieldStyle(.roundedBorder)
            TextEditor(text: $documentsText)
                .font(.callout)
                .frame(minHeight: 110)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
            Text("One document per line — ranked against the query.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
    }

    private var importerTypes: [UTType] {
        switch op.inputKind {
        case .image: [.image]
        case .audioFile: [.audio]
        case .videoFile: [.movie, .video]
        default: []
        }
    }

    /// File-importer URLs are security-scoped; copy into temp so the op (and a later
    /// re-run) can read freely.
    private func importFile(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent("opsgallery-in-\(UUID().uuidString)")
            .appendingPathExtension(url.pathExtension)
        do {
            try FileManager.default.copyItem(at: url, to: copy)
            switch op.inputKind {
            case .image: image = try uprightImage(at: copy)
            default: mediaURL = copy
            }
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }

    private func snapshot() -> OpInputSnapshot {
        var snapshot = OpInputSnapshot()
        snapshot.text = text
        snapshot.image = image
        snapshot.mediaURL = mediaURL
        snapshot.query = query
        snapshot.documents = documentsText
            .split(separator: "\n").map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        snapshot.series = seriesText
            .split(whereSeparator: { ",;\n ".contains($0) })
            .compactMap { Float($0) }
        return snapshot
    }
}

// MARK: - Results

struct ResultView: View {
    let result: OpResult
    let inputImage: CGImage?

    var body: some View {
        switch result {
        case .text(let string):
            Text(string)
                .font(.callout)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
        case .image(let cg):
            Image(decorative: cg, scale: 1)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 320)
                .clipShape(RoundedRectangle(cornerRadius: 8))
        case .boxes(let detections):
            if let inputImage {
                Image(decorative: inputImage, scale: 1)
                    .resizable()
                    .scaledToFit()
                    .overlay { BoxOverlay(detections: detections) }
                    .frame(maxHeight: 320)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            ForEach(detections.prefix(8)) { detection in
                Text("\(detection.label)  \(Int(detection.score * 100))%")
                    .font(.callout.monospaced())
            }
            if detections.isEmpty { Text("Nothing above threshold.").font(.callout) }
        case .audio(let url, let seconds):
            PlayRow(title: String(format: "%.1f s clip", seconds), url: url)
        case .stems(let vocals, let instrumental):
            PlayRow(title: "Vocals", url: vocals)
            PlayRow(title: "Instrumental", url: instrumental)
        case .hits(let hits):
            ForEach(hits) { hit in
                VStack(alignment: .leading, spacing: 2) {
                    Text(String(format: "%.2f", hit.score)).font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                    Text(hit.document).font(.callout)
                }
                .padding(.vertical, 2)
            }
        case .actions(let predictions):
            ForEach(predictions) { prediction in
                Text("\(prediction.label)  \(Int(prediction.probability * 100))%")
                    .font(.callout)
            }
        case .forecast(let history, let mean):
            Chart {
                ForEach(Array(history.enumerated()), id: \.offset) { i, v in
                    LineMark(x: .value("t", i), y: .value("v", v))
                        .foregroundStyle(by: .value("series", "history"))
                }
                ForEach(Array(mean.enumerated()), id: \.offset) { i, v in
                    LineMark(x: .value("t", history.count + i), y: .value("v", v))
                        .foregroundStyle(by: .value("series", "forecast"))
                }
            }
            .frame(height: 220)
        }
    }
}

struct PlayRow: View {
    let title: String
    let url: URL

    private var player = AudioPlayer.shared

    init(title: String, url: URL) {
        self.title = title
        self.url = url
    }

    var body: some View {
        Button(action: { player.toggle(url) }) {
            Label(
                title,
                systemImage: player.playingURL == url ? "stop.fill" : "play.fill")
        }
        .buttonStyle(.bordered)
    }
}

/// Normalized top-left boxes → strokes over the fitted image.
struct BoxOverlay: View {
    let detections: [Detection]

    var body: some View {
        GeometryReader { geo in
            ForEach(detections) { detection in
                let box = detection.box
                Rectangle()
                    .stroke(.red, lineWidth: 2)
                    .frame(
                        width: box.width * geo.size.width,
                        height: box.height * geo.size.height)
                    .position(
                        x: box.midX * geo.size.width,
                        y: box.midY * geo.size.height)
            }
        }
    }
}
