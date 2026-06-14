// DemoModels — the view-side state for the ask-anything coach screen. Two small @Observable types:
//
//   ModelStatus  one shared, @MainActor mirror of ModelHost's load/warm state, so the header can
//                show "Downloading… / Loading… / Ready" and the ask box can disable itself until the
//                on-device model is warm. Warming it once (on launch) is the latency mitigation for
//                Siri: a Siri-invoked intent then reuses the warm session instead of cold-loading a
//                multi-GB model inside Siri's time budget.
//
//   AskRun       the runner behind the in-app "Ask" box. It calls the EXACT code path the
//                AskGemmaIntent uses (`ModelHost.shared.ask`) and records a tiny visible trace
//                (asked → answered on device) + the answer + how long it took. This is the G1
//                surface: it proves Gemma 4 loads and answers free-text on the device WITHOUT Siri,
//                and it shares ModelHost.shared with Siri so a run also warms the model for the
//                voice demo.

import Foundation
import Observation

// MARK: - Shared model status (drives the header)

@MainActor
@Observable
final class ModelStatus {
    /// One per process — the same warm model backs the in-app ask box and the Siri intent.
    static let shared = ModelStatus()

    enum Phase: Equatable {
        case idle, downloading(Double), loading, ready, error(String)
    }

    private(set) var phase: Phase = .idle
    private(set) var loadSeconds: Double?

    var isReady: Bool { if case .ready = phase { return true }; return false }

    private init() {}

    /// Load + warm the shared model once. Idempotent — safe to call from `.task` on every appear.
    func prepare() {
        guard case .idle = phase else { return }
        phase = .loading
        let started = Date()
        Task {
            do {
                try await ModelHost.shared.ensureReady { progress in
                    Task { @MainActor in
                        if progress.fraction < 1 { self.phase = .downloading(progress.fraction) }
                    }
                }
                loadSeconds = Date().timeIntervalSince(started)
                phase = .ready
            } catch {
                phase = .error(error.localizedDescription)
            }
        }
    }
}

// MARK: - One line of the visible pipeline trace

struct TraceLine: Identifiable {
    let id = UUID()
    var icon: String
    var text: String
    var done: Bool = false
    var failed: Bool = false
}

// MARK: - The in-app ask runner (the G1 surface)

@MainActor
@Observable
final class AskRun {
    private(set) var trace: [TraceLine] = []
    private(set) var answer: String = ""
    private(set) var isRunning = false
    private(set) var didRun = false
    /// Wall-clock for the answer (cold first run includes load; warm runs are the real generate cost).
    private(set) var elapsed: Double?

    var hasResult: Bool { didRun }

    func reset() {
        trace = []; answer = ""; elapsed = nil; didRun = false
    }

    /// Ask the on-device model a free-text question — the exact `AskGemmaIntent` path. Records each
    /// step so the user watches the model answer entirely on device.
    func ask(_ question: String) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isRunning, !trimmed.isEmpty else { return }
        reset()
        isRunning = true
        didRun = true

        append(.init(icon: "questionmark.bubble", text: "“\(trimmed)”", done: true))
        let answering = append(.init(
            icon: "sparkles", text: "\(Demo.modelName) answering on device…"))
        let started = Date()

        Task {
            defer { isRunning = false }
            do {
                let result = try await ModelHost.shared.ask(question: trimmed)
                elapsed = Date().timeIntervalSince(started)
                answer = result
                complete(answering, text: "Answered on device — no network")
            } catch {
                answer = ""
                fail(answering, text: "Couldn't answer: \(error.localizedDescription)")
            }
        }
    }

    // MARK: trace helpers

    @discardableResult
    private func append(_ line: TraceLine) -> UUID {
        trace.append(line)
        return line.id
    }

    private func complete(_ id: UUID, text: String) {
        guard let i = trace.firstIndex(where: { $0.id == id }) else { return }
        trace[i].done = true
        trace[i].text = text
    }

    private func fail(_ id: UUID, text: String) {
        guard let i = trace.firstIndex(where: { $0.id == id }) else { return }
        trace[i].failed = true
        trace[i].text = text
    }
}
