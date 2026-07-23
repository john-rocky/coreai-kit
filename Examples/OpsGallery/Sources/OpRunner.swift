// OpRunner.swift — the gallery's engine room: per-op display metadata (icon, input
// kind) layered over the kit's own `CoreAI.Op` registry (`summary` / `defaultModelID`
// come from there), plus the one switch that turns gathered inputs into an op call.

import CoreAIOps
import CoreGraphics
import Foundation
import FoundationModels

/// What the detail screen must collect before the op can run.
enum OpInputKind {
    case text
    case image
    case audioFile
    case videoFile
    case series
    case queryAndDocs
}

extension CoreAI.Op {
    var name: String { String(describing: self) }

    var inputKind: OpInputKind {
        switch self {
        case .summarize, .extract, .translate, .proofread, .redact, .extractEntities,
            .speak, .compose:
            .text
        case .caption, .detect, .read, .upscale, .estimateDepth:
            .image
        case .transcribe, .transcribeMeeting, .describeAudio, .separate:
            .audioFile
        case .recognizeAction:
            .videoFile
        case .forecast:
            .series
        case .search:
            .queryAndDocs
        }
    }

    var icon: String {
        switch self {
        case .summarize: "text.alignleft"
        case .extract: "curlybraces"
        case .translate: "globe"
        case .proofread: "checkmark.seal"
        case .redact: "eye.slash"
        case .extractEntities: "person.text.rectangle"
        case .transcribe: "waveform"
        case .transcribeMeeting: "person.2.wave.2"
        case .describeAudio: "ear"
        case .speak: "speaker.wave.2"
        case .compose: "music.note"
        case .separate: "music.mic"
        case .caption: "text.below.photo"
        case .detect: "viewfinder"
        case .read: "doc.viewfinder"
        case .upscale: "arrow.up.left.and.arrow.down.right"
        case .estimateDepth: "square.3.layers.3d"
        case .recognizeAction: "figure.run"
        case .search: "magnifyingglass"
        case .forecast: "chart.line.uptrend.xyaxis"
        }
    }

    /// Text prefilled into the input editor (text-input ops only).
    var sampleText: String {
        switch self {
        case .speak: Samples.speakLine
        case .compose: Samples.composePrompt
        default: Samples.article
        }
    }
}

/// Sendable snapshot of the inputs a run needs — gathered on the main actor, consumed
/// off it.
struct OpInputSnapshot: Sendable {
    var text = ""
    var image: CGImage?
    var mediaURL: URL?
    var query = ""
    var documents: [String] = []
    var series: [Float] = []
}

/// What a run produced, shaped for display.
enum OpResult: Sendable {
    case text(String)
    case image(CGImage)
    case boxes([Detection])
    case audio(URL, seconds: Double)
    case stems(vocals: URL, instrumental: URL)
    case hits([SearchHit])
    case actions([ActionRecognizer.Prediction])
    case forecast(history: [Float], mean: [Float])
}

enum GalleryError: LocalizedError {
    case missingInput(String)

    var errorDescription: String? {
        switch self {
        case .missingInput(let what): what
        }
    }
}

/// The demo `@Generable` type behind the `extract` card — the same shape OpsDemo uses.
@Generable
struct GalleryActionItems {
    @Guide(description: "One entry per distinct task, as a short imperative sentence")
    var tasks: [String]
}

/// Runs one op on the gathered inputs. Every case is exactly the call the Cookbook
/// documents — this switch is the app.
func runOp(_ op: CoreAI.Op, _ s: OpInputSnapshot) async throws -> OpResult {
    switch op {
    case .summarize:
        return .text(try await CoreAI.summarize(s.text))
    case .extract:
        let items = try await CoreAI.extract(s.text, as: GalleryActionItems.self)
        return .text(items.tasks.map { "• \($0)" }.joined(separator: "\n"))
    case .translate:
        return .text(try await CoreAI.translate(s.text, to: .japanese))
    case .proofread:
        return .text(try await CoreAI.proofread(s.text))
    case .redact:
        return .text(try await CoreAI.redact(s.text))
    case .extractEntities:
        let found = try await CoreAI.extractEntities(
            from: s.text, labels: ["person", "organization", "location", "email"])
        guard !found.isEmpty else { return .text("No entities found.") }
        return .text(
            found.sorted { $0.key < $1.key }
                .map { "\($0.key): \($0.value.joined(separator: ", "))" }
                .joined(separator: "\n"))

    case .transcribe:
        let url = try s.requireMedia("Pick an audio file first.")
        return .text(try await CoreAI.transcribe(url))
    case .transcribeMeeting:
        let url = try s.requireMedia("Pick an audio file first.")
        return .text(try await CoreAI.transcribeMeeting(url).text)
    case .describeAudio:
        let url = try s.requireMedia("Pick an audio file first.")
        return .text(try await CoreAI.describeAudio(url))
    case .separate:
        let url = try s.requireMedia("Pick an audio file first.")
        let stems = try await CoreAI.separate(url)
        return .stems(
            vocals: try writeWAV(
                channels: stems.vocals, sampleRate: stems.sampleRate, name: "vocals"),
            instrumental: try writeWAV(
                channels: stems.instrumental, sampleRate: stems.sampleRate,
                name: "instrumental"))
    case .speak:
        let audio = try await CoreAI.speak(s.text)
        let url = try writeWAV(
            interleaved: audio.samples, sampleRate: audio.sampleRate, channelCount: 1,
            name: "speech")
        return .audio(url, seconds: audio.seconds)
    case .compose:
        let audio = try await CoreAI.compose(s.text)  // 44.1 kHz stereo, interleaved
        let url = try writeWAV(
            interleaved: audio.samples, sampleRate: audio.sampleRate, channelCount: 2,
            name: "music")
        return .audio(url, seconds: audio.seconds / 2)

    case .caption:
        return .text(try await CoreAI.caption(try s.requireImage()))
    case .detect:
        return .boxes(try await CoreAI.detect(in: try s.requireImage()))
    case .read:
        return .text(try await CoreAI.read(try s.requireImage()))
    case .upscale:
        return .image(try await CoreAI.upscale(try s.requireImage()))
    case .estimateDepth:
        let map = try await CoreAI.estimateDepth(in: try s.requireImage())
        guard let rendered = map.cgImage() else {
            throw GalleryError.missingInput("Depth map could not be rendered.")
        }
        return .image(rendered)

    case .recognizeAction:
        let url = try s.requireMedia("Pick a video file first.")
        return .actions(try await CoreAI.recognizeAction(videoAt: url))

    case .search:
        return .hits(try await CoreAI.search(s.query, in: s.documents))
    case .forecast:
        guard !s.series.isEmpty else {
            throw GalleryError.missingInput("Enter a few comma-separated numbers first.")
        }
        let forecast = try await CoreAI.forecast(s.series)
        return .forecast(history: s.series, mean: Array(forecast.mean.prefix(32)))
    }
}

extension OpInputSnapshot {
    func requireImage() throws -> CGImage {
        guard let image else {
            throw GalleryError.missingInput("Pick an image (or use the sample) first.")
        }
        return image
    }

    func requireMedia(_ message: String) throws -> URL {
        guard let mediaURL else { throw GalleryError.missingInput(message) }
        return mediaURL
    }
}
