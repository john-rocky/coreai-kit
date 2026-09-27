// TranscribeModel — the display shell over `SpeechModel` (Sources/QuickStart.swift): status,
// progress, clip and transcript state for the UI. The model picked in the picker loads once and
// is kept, so each Transcribe press times the transcription alone; all model work goes through
// QuickStart.swift, the same code the CLI runs.

import CoreAIKit
import Foundation
import Observation

@MainActor
@Observable
final class TranscribeModel {
    enum Status: Equatable {
        case idle
        /// Loading the named model; `fraction` while its first download runs.
        case loading(String, fraction: Double?)
        case ready
        case transcribing
        case error(String)
    }

    /// One Transcribe press, timed around the transcription call (the model is already loaded).
    struct Run: Equatable {
        let hotwords: [String]
        let audioSeconds: Double
        let seconds: Double
        let text: String
    }

    /// Speech-to-text entries published for this platform (macOS: Whisper / Qwen3-ASR /
    /// Parakeet / Fun-ASR-Nano / …; iOS: Whisper / Fun-ASR-Nano / …) — the picker's content, ids
    /// straight off the model cards. Nothing is picked at launch, so nothing downloads until a
    /// model is chosen.
    let models = ModelCatalog.builtin.available(.asr)
    var selectedID: String?

    var status: Status = .idle
    var clipName = "No audio loaded."
    var transcript = ""
    var detectedLanguage = ""
    /// The hotword field's text, comma-separated (shown for a model that takes hotwords).
    var hotwordsText = ""
    var recording = false
    private(set) var playing = false
    /// The finished run whose text the transcript shows; nil before a run and while one runs.
    private(set) var lastRun: Run?
    private(set) var loadedID: String?

    private var speech: SpeechModel?
    private var pendingLoad: (id: String, task: Task<Void, Never>)?
    private var runID = 0
    private var clipURL: URL?
    private var scopedURL: URL?
    private let recorder = MicRecorder()
    private let player = ClipPlayer()

    init() {
        player.onStop = { [weak self] in self?.playing = false }
    }

    var isReady: Bool { speech != nil && loadedID == selectedID }

    var isBusy: Bool {
        switch status {
        case .loading, .transcribing: return true
        default: return false
        }
    }

    var downloadFraction: Double? {
        if case .loading(_, let fraction?) = status { return fraction }
        return nil
    }

    var hasClip: Bool { clipURL != nil }
    var canTranscribe: Bool { isReady && !isBusy && clipURL != nil && !recording }
    var takesHotwords: Bool { selectedID.map(SpeechModel.takesHotwords) ?? false }

    var statusLabel: String {
        switch status {
        case .loading(let name, let fraction):
            return fraction.map { "Loading \(name)… \(Int($0 * 100))%" } ?? "Loading \(name)…"
        case .transcribing: return "Transcribing…"
        case _ where playing: return "Playing…"
        case .error(let message): return "Error: \(message)"
        case .ready: return "Ready"
        case .idle: return "Pick a model to load."
        }
    }

    /// "10.5 s of audio · transcribed in 0.93 s" for the last run.
    var measuredLine: String? {
        lastRun.map {
            String(format: "%.1f s of audio · transcribed in %.2f s", $0.audioSeconds, $0.seconds)
        }
    }

    /// The transcript with every hotword of its run in bold — the only emphasis on screen.
    var emphasizedTranscript: AttributedString {
        Self.emphasize(transcript, lastRun?.hotwords ?? [])
    }

    /// Loads the picked model once and keeps it; picking the loaded one again does nothing.
    func load() async {
        guard let id = selectedID, loadedID != id || speech == nil else { return }
        if let pendingLoad, pendingLoad.id == id { return await pendingLoad.task.value }
        let name = models.first { $0.id == id }?.name ?? id
        speech = nil
        loadedID = nil
        status = .loading(name, fraction: nil)
        let task = Task {
            do {
                let loaded = try await SpeechModel(catalog: id) { progress in
                    Task { @MainActor in
                        guard case .loading = self.status else { return }
                        self.status = .loading(
                            name, fraction: progress.fraction < 1 ? progress.fraction : nil)
                    }
                }
                guard self.selectedID == id else { return }
                self.speech = loaded
                self.loadedID = id
                self.status = .ready
            } catch {
                if self.selectedID == id { self.status = .error(error.localizedDescription) }
            }
        }
        pendingLoad = (id, task)
        await task.value
        if pendingLoad?.id == id { pendingLoad = nil }
    }

    func loadFile(_ url: URL) {
        // A file picked on iOS is security-scoped: keep access open while it is the clip.
        let scoped = url.startAccessingSecurityScopedResource()
        setClip(url, name: url.lastPathComponent)
        if scoped { scopedURL = url }
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
            player.stop()
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

    /// Plays the clip out loud, or stops it; returns the clip's length when it starts.
    @discardableResult
    func togglePlay() -> Double? {
        if playing {
            player.stop()
            return nil
        }
        guard let clipURL else { return nil }
        do {
            let seconds = try player.play(clipURL)
            playing = true
            return seconds
        } catch {
            status = .error(error.localizedDescription)
            return nil
        }
    }

    /// Returns once the clip has played to its end (or was stopped).
    func playbackFinished() async {
        await player.finished()
    }

    /// Transcribes the clip with the loaded model. The clock runs around the transcription call
    /// alone: the model is loaded before and the file is read before.
    func transcribe() async {
        guard canTranscribe, let clipURL, let speech else { return }
        let hotwords = takesHotwords ? hotwordList(hotwordsText) : []
        runID += 1
        let run = runID
        status = .transcribing
        transcript = ""
        detectedLanguage = ""
        lastRun = nil
        do {
            let samples = try AudioFile.pcm16kMono(clipURL)
            let clock = ContinuousClock()
            let start = clock.now
            let result = try await speech.transcribe(
                samples: samples, hotwords: hotwords,
                onPartial: { partial in
                    Task { @MainActor in
                        if self.runID == run, self.status == .transcribing {
                            self.transcript = partial
                        }
                    }
                })
            let elapsed = (clock.now - start).components
            let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            transcript = result.text
            detectedLanguage = result.language
            lastRun = Run(
                hotwords: hotwords, audioSeconds: Double(samples.count) / 16000,
                seconds: seconds, text: result.text)
            status = .ready
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    private func setClip(_ url: URL, name: String) {
        player.stop()
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = nil
        clipURL = url
        transcript = ""
        detectedLanguage = ""
        lastRun = nil
        clipName = name
    }

    /// Bold on every case-insensitive occurrence of a hotword; overlapping matches merge.
    static func emphasize(_ text: String, _ words: [String]) -> AttributedString {
        var found: [Range<String.Index>] = []
        for word in words where !word.isEmpty {
            var from = text.startIndex
            while from < text.endIndex,
                let range = text.range(of: word, options: .caseInsensitive, range: from..<text.endIndex)
            {
                found.append(range)
                from = range.upperBound
            }
        }
        var merged: [Range<String.Index>] = []
        for range in found.sorted(by: { $0.lowerBound < $1.lowerBound }) {
            if let last = merged.last, range.lowerBound <= last.upperBound {
                merged[merged.count - 1] = last.lowerBound..<max(last.upperBound, range.upperBound)
            } else {
                merged.append(range)
            }
        }
        var out = AttributedString()
        var cursor = text.startIndex
        for range in merged {
            out += AttributedString(String(text[cursor..<range.lowerBound]))
            var bold = AttributedString(String(text[range]))
            bold.inlinePresentationIntent = .stronglyEmphasized
            out += bold
            cursor = range.upperBound
        }
        out += AttributedString(String(text[cursor...]))
        return out
    }
}
