// CoreAI+Audio.swift — anchored audio operations beyond `transcribe` (CoreAI.swift) and
// `speak` (CoreAI+Speech.swift): speaker-attributed meeting transcripts, audio
// understanding, text-to-music, and vocal/instrumental separation.
//
// ```swift
// let meeting = try await CoreAI.transcribeMeeting(recordingURL)   // who said what
// let scene   = try await CoreAI.describeAudio(clipURL)            // sounds, not just words
// let jingle  = try await CoreAI.compose("upbeat lo-fi loop")      // text -> music
// let stems   = try await CoreAI.separate(songURL)                 // vocals + instrumental
// ```

import CoreAIKit
import Foundation
import FoundationModels

extension CoreAI {
    /// Audio file → speaker-attributed transcript: Sortformer diarization finds the turns,
    /// a transcriber does each one. `MeetingTranscript.text` is the ready-to-print form.
    ///
    /// **This is the op the kit exists for in speech, and it costs 238 MB.** Who spoke when is
    /// the one speech capability Apple ships nothing for — `Speech.framework` has no speaker
    /// API at all — so the diarizer is a real download, while the transcription it is paired
    /// with is Apple's and free. `options: .model(...)` swaps in a catalog ASR and takes the
    /// feature from 238 MB to over 3 GB; do it when Apple's locale coverage or determinism is
    /// genuinely not enough.
    public static func transcribeMeeting(
        _ audioURL: URL, language: String? = nil, options: OpOptions = OpOptions()
    ) async throws -> MeetingTranscript {
        let samples = try AudioFile.pcm16kMono(audioURL)
        guard let id = options.model else {
            let transcriber = try await AudioOpModels.shared.systemMeetingTranscriber(
                language: language)
            return try await withPinnedModel(ResidentKind.meeting, "system") {
                try await transcriber.transcribe(samples: samples, language: language)
            }
        }
        let transcriber = try await AudioOpModels.shared.meetingTranscriber(asr: id)
        return try await withPinnedModel(ResidentKind.meeting, id) {
            try await transcriber.transcribe(samples: samples, language: language)
        }
    }

    /// Default audio-understanding model.
    public static let defaultAudioModel = "qwen2.5-omni-3b-audio"

    /// Audio file → description of the sounds and setting, not just a transcript.
    public static func describeAudio(
        _ audioURL: URL, options: OpOptions = OpOptions()
    ) async throws -> String {
        try await AudioOpModels.shared.describe(
            catalog: options.model ?? defaultAudioModel,
            samples: try AudioFile.pcm16kMono(audioURL))
    }

    /// Default text-to-music model.
    public static let defaultMusicModel = "stable-audio-open-small"

    /// Prompt → generated music (44.1 kHz stereo, interleaved L/R). Prompts name
    /// genre, instruments, and mood — "warm lo-fi hip hop loop, vinyl crackle, 90 BPM".
    public static func compose(
        _ prompt: String, seconds: Float = 11, options: OpOptions = OpOptions()
    ) async throws -> SpokenAudio {
        try await AudioOpModels.shared.compose(
            catalog: options.model ?? defaultMusicModel, prompt: prompt, seconds: seconds)
    }

    /// Default source-separation model (vocal / instrumental).
    public static let defaultSeparationModel = "melband-roformer-vocal"

    /// Song → vocal and instrumental stems (44.1 kHz stereo).
    public static func separate(
        _ audioURL: URL, options: OpOptions = OpOptions()
    ) async throws -> Stems {
        let id = options.model ?? defaultSeparationModel
        let separator = try await AudioOpModels.shared.separator(catalog: id)
        let mix = try AudioFile.pcmStereo(audioURL)
        return try await withPinnedModel(ResidentKind.separator, id) {
            try await separator.separate(mix)
        }
    }
}

/// Process-wide cache of loaded audio-op models — same contract as `OpModels`: concurrent
/// first calls share one load, a failed load is not cached. `describeAudio` and `compose`
/// turns on one model serialize behind each other (one attached clip / one generation at
/// a time); the separator is an actor over an immutable graph and needs no turn chain.
actor AudioOpModels {
    static let shared = AudioOpModels()

    private let meetings = ResidentCache<MeetingTranscriber>(kind: ResidentKind.meeting)
    private let audioModels = ResidentCache<KitAudioModel>(kind: ResidentKind.audio)
    private let musicians = ResidentCache<KitMusician>(kind: ResidentKind.musician)
    private let separators = ResidentCache<KitSeparator>(kind: ResidentKind.separator)
    private var audioTurns: [String: Task<Void, Never>] = [:]
    private var musicianTurns: [String: Task<Void, Never>] = [:]

    func describe(catalog id: String, samples: [Float]) async throws -> String {
        let model = try await audioModel(catalog: id)
        let previous = audioTurns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.audio, id) {
                try await model.attach(samples: samples)
                let session = LanguageModelSession(model: model)
                let reply = try await session.respond(
                    to: "Describe this audio clip — the sounds, any speech or music, and "
                        + "the setting. Reply with only the description.")
                return reply.content.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        audioTurns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func compose(catalog id: String, prompt: String, seconds: Float) async throws -> SpokenAudio {
        let musician = try await musician(catalog: id)
        let previous = musicianTurns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await withPinnedModel(ResidentKind.musician, id) {
                try await musician.generate(prompt, seconds: seconds)
            }
        }
        musicianTurns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    /// Sortformer plus Apple's transcriber — the default pairing.
    func systemMeetingTranscriber(language: String?) async throws -> MeetingTranscriber {
        let locale = language.map { Locale(identifier: $0) } ?? .current
        return try await meetings.value(for: "system:\(locale.identifier)") {
            try await MeetingTranscriber(
                locale: locale, downloadProgress: OpDownloads.forward)
        }
    }

    func meetingTranscriber(asr id: String) async throws -> MeetingTranscriber {
        try await meetings.value(for: id) {
            try await MeetingTranscriber(
                asr: KitTranscriber(catalog: id, downloadProgress: OpDownloads.forward),
                downloadProgress: OpDownloads.forward)
        }
    }

    func separator(catalog id: String) async throws -> KitSeparator {
        try await separators.value(for: id) {
            try await KitSeparator(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func audioModel(catalog id: String) async throws -> KitAudioModel {
        try await audioModels.value(for: id) {
            try await KitAudioModel(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }

    func musician(catalog id: String) async throws -> KitMusician {
        try await musicians.value(for: id) {
            try await KitMusician(catalog: id, downloadProgress: OpDownloads.forward)
        }
    }
}
