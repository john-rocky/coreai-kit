// CoreAI+Prepare.swift — first-run ergonomics for the ops: one process-wide download
// observer (`CoreAI.onDownload`) and an explicit prefetch (`CoreAI.prepare`), so the
// first call to an op never turns into a silent multi-gigabyte download.
//
// ```swift
// CoreAI.onDownload { print("\($0.currentFile): \(Int($0.fraction * 100))%") }
// try await CoreAI.prepare(.transcribe, .summarize)   // download + load ahead of use
// ```

import CoreAIKit
import Foundation

extension CoreAI {
    /// Observes every op model download in the process — one hook instead of a progress
    /// parameter on twenty ops. Set it once at launch (before the first op call, or any
    /// time: events read the current handler). Pass `nil` to stop observing.
    public static func onDownload(_ handler: (@Sendable (DownloadProgress) -> Void)?) {
        OpDownloads.shared.set(handler)
    }

    /// The ops, as values — what `prepare(_:)` takes, and a machine-readable registry:
    /// `summary` / `defaultModelID` are the same one-liners the docs use, so an op
    /// picker or gallery renders straight from `Op.allCases`.
    public enum Op: Sendable, Hashable, CaseIterable {
        case summarize, extract, translate, proofread
        case transcribe, transcribeMeeting, describeAudio, speak, compose, separate
        case caption, detect, read, upscale, estimateDepth
        case recognizeAction, search, forecast
        case redact, extractEntities
        /// Live camera and video-file ops. They were missing from this enum until
        /// `coreai-doctor` reported an app using `watch()` as having nothing to download —
        /// the mapping is what `capability`, `prepare` and the doctor all walk, so an op
        /// absent from here is an op the kit cannot answer questions about.
        case watch, watchDepth, scan

        /// One-line "input → output" contract of the op.
        public var summary: String {
            switch self {
            case .summarize: "Text → short summary"
            case .extract: "Text → typed value (@Generable)"
            case .translate: "Text → translation in a named language"
            case .proofread: "Text → corrected text"
            case .transcribe: "Audio file → plain-text transcript"
            case .transcribeMeeting: "Audio file → speaker-attributed transcript"
            case .describeAudio: "Audio file → description of the sounds"
            case .speak: "Text → synthesized speech (PCM)"
            case .compose: "Prompt → generated music (PCM)"
            case .separate: "Song → vocal and instrumental stems"
            case .caption: "Image → description"
            case .detect: "Image → labeled bounding boxes"
            case .read: "Document image → markdown text"
            case .upscale: "Image → ×4 upscaled image"
            case .estimateDepth: "Image → relative depth map"
            case .recognizeAction: "Video clip → ranked action labels"
            case .search: "Query + strings → ranked matches"
            case .forecast: "Number series → 128-step forecast"
            case .redact: "Text → text with PII replaced by labels"
            case .extractEntities: "Text → entities by zero-shot label"
            case .watch: "Live camera → labeled boxes per frame"
            case .watchDepth: "Live camera → relative depth per frame"
            case .scan: "Video file → a time-stamped detection timeline"
            }
        }

        /// Catalog id of the model the op resolves by default — what `options: .model(...)`
        /// overrides. `nil` for the GLiNER2 ops, which load a pinned non-catalog preset.
        public var defaultModelID: String? {
            switch self {
            case .summarize, .extract, .translate, .proofread: CoreAI.defaultModel
            case .transcribe, .transcribeMeeting: CoreAI.defaultSpeechModel
            case .describeAudio: CoreAI.defaultAudioModel
            case .speak: CoreAI.defaultVoiceModel
            case .compose: CoreAI.defaultMusicModel
            case .separate: CoreAI.defaultSeparationModel
            case .caption: CoreAI.defaultVisionModel
            case .detect: CoreAI.defaultDetectionModel
            case .read: CoreAI.defaultOCRModel
            case .upscale: CoreAI.defaultUpscaleModel
            case .estimateDepth: CoreAI.defaultDepthModel
            case .recognizeAction: CoreAI.defaultActionModel
            case .search: CoreAI.defaultEmbeddingModel
            case .forecast: CoreAI.defaultForecastModel
            case .redact, .extractEntities: nil
            case .watch, .scan: CoreAI.defaultDetectionModel
            case .watchDepth: CoreAI.defaultDepthModel
            }
        }
    }

    /// Downloads and loads the models behind `ops` ahead of their first call — run it
    /// behind your loading UI (progress lands on the `onDownload` observer) and the
    /// first real call starts instantly. Already-cached models just load. `options.model`
    /// applies to every listed op; prepare ops with different overrides in separate calls.
    public static func prepare(_ ops: Op..., options: OpOptions = OpOptions()) async throws {
        try await withThrowingTaskGroup(of: Void.self) { group in
            for op in Set(ops) {
                group.addTask { try await prepare(op, options: options) }
            }
            try await group.waitForAll()
        }
    }

    private static func prepare(_ op: Op, options: OpOptions) async throws {
        switch op {
        case .summarize, .extract, .translate, .proofread:
            _ = try await OpModels.shared.chat(catalog: options.model ?? defaultModel)
        case .transcribe:
            _ = try await OpModels.shared.transcriber(
                catalog: options.model ?? defaultSpeechModel)
        case .transcribeMeeting:
            _ = try await AudioOpModels.shared.meetingTranscriber(
                asr: options.model ?? defaultSpeechModel)
        case .describeAudio:
            _ = try await AudioOpModels.shared.audioModel(
                catalog: options.model ?? defaultAudioModel)
        case .speak:
            _ = try await SpeechOpModels.shared.speaker(
                catalog: options.model ?? defaultVoiceModel)
        case .compose:
            _ = try await AudioOpModels.shared.musician(
                catalog: options.model ?? defaultMusicModel)
        case .separate:
            _ = try await AudioOpModels.shared.separator(
                catalog: options.model ?? defaultSeparationModel)
        case .caption:
            _ = try await ImageOpModels.shared.visionModel(
                catalog: options.model ?? defaultVisionModel)
        case .detect:
            _ = try await ImageOpModels.shared.detector(
                catalog: options.model ?? defaultDetectionModel)
        case .read:
            _ = try await ImageOpModels.shared.reader(catalog: options.model ?? defaultOCRModel)
        case .upscale:
            _ = try await ImageOpModels.shared.upscaler(
                catalog: options.model ?? defaultUpscaleModel)
        case .estimateDepth:
            _ = try await ImageOpModels.shared.depthEstimator(
                catalog: options.model ?? defaultDepthModel)
        case .recognizeAction:
            _ = try await VideoOpModels.shared.recognizer(
                catalog: options.model ?? defaultActionModel)
        case .search:
            _ = try await SearchOpModels.shared.embedder(
                catalog: options.model ?? defaultEmbeddingModel)
        case .forecast:
            _ = try await ForecastOpModels.shared.forecaster(
                catalog: options.model ?? defaultForecastModel)
        case .redact, .extractEntities:
            _ = try await RedactOpModels.shared.extractor()
        case .watch, .scan:
            _ = try await ImageOpModels.shared.detector(
                catalog: options.model ?? defaultDetectionModel)
        case .watchDepth:
            _ = try await ImageOpModels.shared.depthEstimator(
                catalog: options.model ?? defaultDepthModel)
        }
    }
}

/// Holds the process-wide `onDownload` handler. Every op loader passes
/// `OpDownloads.forward` as its driver's `downloadProgress:` — the closure reads the
/// current handler per event, so an observer set after a load began still sees it.
final class OpDownloads: @unchecked Sendable {
    static let shared = OpDownloads()

    private let lock = NSLock()
    private var handler: (@Sendable (DownloadProgress) -> Void)?

    func set(_ handler: (@Sendable (DownloadProgress) -> Void)?) {
        lock.lock()
        defer { lock.unlock() }
        self.handler = handler
    }

    static let forward: @Sendable (DownloadProgress) -> Void = { progress in
        shared.current()?(progress)
    }

    private func current() -> (@Sendable (DownloadProgress) -> Void)? {
        lock.lock()
        defer { lock.unlock() }
        return handler
    }
}
