// TranscribeModel — the display shell over `transcribe(audio:model:)` (Sources/QuickStart.swift):
// status, progress, and transcript state for the UI. All model work happens in that one shared
// function — the same one the CLI runs — so the model loads per call and this type never touches
// an engine type. If your app transcribes repeatedly, keep a `KitTranscriber` loaded instead.

import CoreAIKit
import Foundation
import Observation

@MainActor
@Observable
final class TranscribeModel {
    enum Status: Equatable {
        case idle
        case downloading(Double)
        case transcribing
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pick a clip — record, choose, or demo."
            case .downloading(let f): return "Downloading model… \(Int(f * 100))%"
            case .transcribing: return "Transcribing…"
            case .error(let m): return "Error: \(m)"
            }
        }
    }

    /// Speech-to-text entries published for this platform (macOS: Whisper / Qwen3-ASR /
    /// Parakeet; iOS: Whisper) — the picker's content, ids straight off the model cards.
    let models = ModelCatalog.builtin.available(.asr)
    var selectedID: String

    var status: Status = .idle
    var clipName = "No audio loaded."
    var transcript = ""
    var detectedLanguage = ""
    var recording = false

    private var clipURL: URL?
    private let recorder = MicRecorder()

    init() {
        selectedID = models.first?.id ?? "whisper-large-v3-turbo"
    }

    var isBusy: Bool {
        switch status {
        case .downloading, .transcribing: return true
        default: return false
        }
    }

    var downloadFraction: Double? {
        if case .downloading(let f) = status { return f }
        return nil
    }

    var canTranscribe: Bool { !isBusy && clipURL != nil }

    func loadFile(_ url: URL) {
        setClip(url, name: url.lastPathComponent)
    }

    func loadDemo() {
        guard let url = Bundle.main.url(forResource: "sample", withExtension: "wav") else {
            clipName = "sample.wav missing from the app bundle."
            return
        }
        setClip(url, name: "Demo: sample.wav (5s, spoken)")
    }

    func toggleRecord() {
        if recording {
            recording = false
            if let url = recorder.stopFile() {
                setClip(url, name: "Mic clip")
            } else {
                clipName = "No audio captured."
            }
        } else {
            Task {
                do {
                    try await recorder.start()
                    recording = true
                    clipName = "Recording… tap Stop when done."
                } catch {
                    clipName = error.localizedDescription
                }
            }
        }
    }

    func transcribeClip() {
        guard let clipURL, canTranscribe else { return }
        status = .transcribing
        transcript = ""
        detectedLanguage = ""
        Task {
            do {
                let result = try await transcribe(
                    audio: clipURL, model: selectedID,
                    downloadProgress: { progress in
                        Task { @MainActor in
                            self.status =
                                progress.fraction < 1
                                ? .downloading(progress.fraction) : .transcribing
                        }
                    },
                    onPartial: { partial in
                        Task { @MainActor in self.transcript = partial }
                    })
                self.transcript = result.text
                self.detectedLanguage = result.language
                self.status = .idle
            } catch {
                self.status = .error(error.localizedDescription)
            }
        }
    }

    private func setClip(_ url: URL, name: String) {
        clipURL = url
        transcript = ""
        detectedLanguage = ""
        clipName = name
    }
}
