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
    /// Audio file → speaker-attributed transcript: Sortformer diarization finds the
    /// turns, the ASR model transcribes each one. `language` nil = auto-detect.
    /// `options: .model(...)` overrides the ASR model (the diarizer is the catalog's
    /// only one); `MeetingTranscript.text` is the ready-to-print form.
    public static func transcribeMeeting(
        _ audioURL: URL, language: String? = nil, options: OpOptions = OpOptions()
    ) async throws -> MeetingTranscript {
        let id = options.model ?? defaultSpeechModel
        let transcriber = try await AudioOpModels.shared.meetingTranscriber(asr: id)
        let samples = try AudioFile.pcm16kMono(audioURL)
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

    func meetingTranscriber(asr id: String) async throws -> MeetingTranscriber {
        try await meetings.value(for: id) {
            try await MeetingTranscriber(asr: id, downloadProgress: OpDownloads.forward)
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
