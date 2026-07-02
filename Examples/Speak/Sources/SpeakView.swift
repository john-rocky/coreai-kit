// SpeakView — type a sentence, an on-device TTS model speaks it. Everything (LM, diffusion,
// vocoder) runs locally; the text never leaves the device.

import AVFoundation
import CoreAIKit
import SwiftUI

@MainActor
@Observable
final class SpeakModel {
    enum Status: Equatable {
        case idle
        case downloading(Double)
        case loading
        case ready
        case speaking
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pick a voice and load it"
            case .downloading(let f): return "Downloading… \(Int(f * 100))%"
            case .loading: return "Loading…"
            case .ready: return "Ready"
            case .speaking: return "Synthesizing…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    var status: Status = .idle
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?
    var lastSeconds: Double?

    private var speaker: KitSpeaker?
    private var player: AVAudioPlayer?

    var isBusy: Bool {
        switch status {
        case .downloading, .loading, .speaking: return true
        default: return false
        }
    }
    var downloadFraction: Double? {
        if case .downloading(let f) = status { return f }
        return nil
    }

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard entries.isEmpty else { return }
        entries = await ModelCatalog.load().available(.tts)
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func load() {
        guard !isBusy, let entry = selectedEntry else { return }
        status = .loading
        speaker = nil
        Task {
            do {
                // Same gesture as the model card: the catalog id resolves the voice.
                let speaker = try await KitSpeaker(catalog: entry.id) { progress in
                    Task { @MainActor in
                        self.status = progress.fraction < 1
                            ? .downloading(progress.fraction) : .loading
                    }
                }
                self.speaker = speaker
                status = .ready
            } catch {
                status = .error(error.localizedDescription)
            }
        }
    }

    func speak(_ text: String) {
        guard let speaker, status == .ready else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        status = .speaking
        Task {
            do {
                let audio = try await speaker.synthesize(trimmed)
                lastSeconds = audio.seconds
                let wav = WAVFile.data(samples: audio.samples, sampleRate: audio.sampleRate)
                let player = try AVAudioPlayer(data: wav)
                self.player = player
                player.play()
                status = .ready
            } catch {
                status = .error(error.localizedDescription)
            }
        }
    }
}

struct SpeakView: View {
    @State private var model = SpeakModel()
    @State private var input = "Hello! I am a text to speech model running fully on this device."

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {
                Picker("Voice", selection: $model.selectedEntry) {
                    ForEach(model.entries) { entry in
                        Text(entry.name).tag(Optional(entry))
                    }
                }
                .fixedSize()
                .disabled(model.isBusy)

                Button(model.status == .idle ? "Download & Load" : "Reload") { model.load() }
                    .disabled(model.isBusy)

                if let fraction = model.downloadFraction {
                    ProgressView(value: fraction).frame(width: 90)
                } else if model.isBusy {
                    ProgressView().controlSize(.small)
                }

                Text(model.status.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
            }

            TextEditor(text: $input)
                .font(.body)
                .frame(minHeight: 100)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))

            HStack {
                Button("Speak") { model.speak(input) }
                    .buttonStyle(.borderedProminent)
                    .disabled(model.status != .ready)
                if let s = model.lastSeconds {
                    Text(String(format: "%.1f s of audio", s))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text("Runs fully on-device. The text never leaves your device.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(14)
        #if os(macOS)
        .frame(minWidth: 460, minHeight: 260)
        #endif
        .task { await model.loadCatalog() }
    }
}
