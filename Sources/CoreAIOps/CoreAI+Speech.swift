// CoreAI+Speech.swift — anchored speech synthesis over local catalog models.
//
// ```swift
// let audio = try await CoreAI.speak("Hello from Core AI.")   // text -> PCM + sample rate
// ```

import CoreAIKit
import Foundation

extension CoreAI {
    /// Default text-to-speech model: VoxCPM 0.5B — the catalog TTS published for both
    /// iOS and macOS. On a Mac, `options: .model("kokoro-82m")` is a 341 MB download
    /// against VoxCPM's ~1.4 GB.
    public static let defaultVoiceModel = "voxcpm-0.5b"

    /// Text → synthesized speech: mono PCM in [-1, 1] plus its sample rate — hand it to
    /// `AVAudioEngine` or write it out as a file. First use downloads and loads the model
    /// (cached afterwards); utterances on the same model serialize behind each other.
    public static func speak(
        _ text: String, options: OpOptions = OpOptions()
    ) async throws -> SpokenAudio {
        try await SpeechOpModels.shared.speak(
            catalog: options.model ?? defaultVoiceModel, text: text)
    }
}

/// Process-wide cache of loaded speakers, keyed by catalog id — same contract as
/// `OpModels`: concurrent first calls share one load, a failed load is not cached,
/// utterances on one engine serialize behind each other.
actor SpeechOpModels {
    static let shared = SpeechOpModels()

    private var speakerLoads: [String: Task<KitSpeaker, Error>] = [:]
    private var speakerTurns: [String: Task<Void, Never>] = [:]

    func speak(catalog id: String, text: String) async throws -> SpokenAudio {
        let speaker = try await speaker(catalog: id)
        let previous = speakerTurns[id]
        let turn = Task { [previous] in
            await previous?.value
            return try await speaker.synthesize(text)
        }
        speakerTurns[id] = Task { _ = try? await turn.value }
        return try await turn.value
    }

    func speaker(catalog id: String) async throws -> KitSpeaker {
        if let load = speakerLoads[id] { return try await load.value }
        let load = Task<KitSpeaker, Error> {
            try await KitSpeaker(catalog: id, downloadProgress: OpDownloads.forward)
        }
        speakerLoads[id] = load
        do {
            return try await load.value
        } catch {
            speakerLoads[id] = nil
            throw error
        }
    }
}
