// DecideRuntime — the one loaded decision model the three screens share, and the status the
// UI shows while it downloads and loads. Held once so a screen that decides continuously (the
// speech gate, the clipboard watcher) never reloads the weights; the take-home function in
// QuickStart.swift is the per-call shape of the same thing.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class DecideRuntime {
    enum Status: Equatable {
        case idle
        case downloading(Double)
        case loading
        case ready
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pick a model to load."
            case .downloading(let f): return "Downloading model… \(Int(f * 100))%"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .error(let m): return "Error: \(m)"
            }
        }
    }

    /// Chat models published for this platform, then the decision models — the picker's
    /// content, ids straight off the model cards. Every one loads on the logits-capable
    /// engine; the default is the chat model whose zero-shot decisions clear chance on the
    /// published fixtures.
    let models = ModelCatalog.builtin.available(.chat) + ModelCatalog.builtin.available(.decision)
    var selectedID = CoreAI.defaultDecisionModel
    var status: Status = .idle
    private(set) var decider: TypedDecisions?
    private(set) var loadedID: String?

    var isReady: Bool { status == .ready && decider != nil }

    var downloadFraction: Double? {
        if case .downloading(let f) = status { return f }
        return nil
    }

    /// Loads the selected model once; a second call with the same id is a no-op.
    func load() async {
        guard loadedID != selectedID || decider == nil else { return }
        decider = nil
        loadedID = nil
        status = .loading
        let id = selectedID
        do {
            let loaded = try await TypedDecisions(catalog: id) { progress in
                Task { @MainActor in
                    self.status = progress.fraction < 1 ? .downloading(progress.fraction) : .loading
                }
            }
            decider = loaded
            loadedID = id
            status = .ready
        } catch {
            status = .error(error.localizedDescription)
        }
    }

    /// The loaded model, loading it first if needed.
    func ready() async throws -> TypedDecisions {
        if decider == nil || loadedID != selectedID { await load() }
        guard let decider else { throw DecideRuntimeError.notLoaded(status.label) }
        return decider
    }
}

enum DecideRuntimeError: LocalizedError {
    case notLoaded(String)

    var errorDescription: String? {
        if case .notLoaded(let why) = self { return "The model is not loaded: \(why)" }
        return nil
    }
}

extension Decision.Answer {
    /// One line for a table cell: the answer and where its tokens went.
    var summaryLine: String {
        switch value {
        case .noul(let p): return "P(yes) \(p.formatted(.number.precision(.fractionLength(2))))"
        case .choice(let c):
            return "\(c.id) · \(c.confidence.formatted(.number.precision(.fractionLength(2))))"
        case .score(let s):
            return "\(s.value.formatted(.number.precision(.fractionLength(2)))) (level \(s.level))"
        }
    }

    var timingLine: String {
        "\(Int(timing.milliseconds.rounded())) ms · \(timing.promptTokens) tokens, \(timing.reusedTokens) reused"
    }
}
