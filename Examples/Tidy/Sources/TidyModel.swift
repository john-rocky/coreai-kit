// TidyModel — the display shell over `tidy(transcript:model:)` (Sources/QuickStart.swift):
// status, progress, the three control axes, and the rewritten text. All model work happens in
// that one shared function — the same one the CLI runs — so the model loads per call and this
// type never touches an engine type. If your app cleans transcripts repeatedly, keep a
// `KitTextNormalizer` loaded instead.

import CoreAIKit
import Foundation
import Observation

@MainActor
@Observable
final class TidyModel {
    enum Status: Equatable {
        case idle
        case downloading(Double)
        case working
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Paste a raw transcript, or use the sample."
            case .downloading(let f): return "Downloading model… \(Int(f * 100))%"
            case .working: return "Rewriting…"
            case .error(let m): return "Error: \(m)"
            }
        }
    }

    /// What dictation actually produces: fillers, a false start corrected mid-sentence, spoken
    /// numbers and a spoken date, and no punctuation anywhere.
    static let sample =
        "so um i need to like send the the report by uh friday no wait make that thursday "
        + "and the invoice came to twenty three thousand four hundred and fifty dollars "
        + "it's due on march third twenty twenty six"

    /// Text normalizers published for this platform — the picker's content, ids straight off
    /// the model cards. One today (S1-mini); the picker is what makes a second one free.
    let models = ModelCatalog.builtin.available(.textNormalizer)
    var selectedID: String

    var status: Status = .idle
    var transcript = TidyModel.sample
    var result = ""

    var styling: TranscriptStyling = .semiFormal
    var structure: TranscriptStructure = .prose
    var context: TranscriptContext = .general

    init() {
        selectedID = models.first?.id ?? "s1-mini"
    }

    var isBusy: Bool {
        switch status {
        case .downloading, .working: return true
        default: return false
        }
    }

    var downloadFraction: Double? {
        if case .downloading(let f) = status { return f }
        return nil
    }

    var canRun: Bool { !isBusy && !transcript.trimmingCharacters(in: .whitespaces).isEmpty }

    func loadSample() {
        transcript = Self.sample
        result = ""
    }

    func run() {
        guard canRun else { return }
        status = .working
        result = ""
        Task {
            do {
                let clean = try await tidy(
                    transcript: transcript, model: selectedID, styling: styling,
                    structure: structure, context: context,
                    downloadProgress: { progress in
                        Task { @MainActor in
                            self.status =
                                progress.fraction < 1 ? .downloading(progress.fraction) : .working
                        }
                    })
                // Filler-only input normalizes to nothing. Say so, rather than leaving a blank
                // box that reads like a failure.
                self.result = clean.isEmpty ? "(nothing but filler — empty by design)" : clean
                self.status = .idle
            } catch {
                self.status = .error(error.localizedDescription)
            }
        }
    }
}
