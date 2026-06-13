// ChatModel.swift — the SwiftUI view model. Owns the RagEngine, indexes the notes, and turns
// the engine's live callbacks into observable UI state. One conversation, one question at a time;
// each question is an independent grounded retrieval (a fresh LanguageModelSession), so the
// transcript never outgrows the 4k window and the result is always freshly searched.

import Foundation
import Observation
import SwiftUI

/// One exchange in the conversation: the question, the retrieval chain it triggered, and the
/// grounded answer.
struct ChatTurn: Identifiable {
    enum Phase {
        case retrieving  // searching / reading notes
        case answering  // answer text streaming
        case done
        case failed
    }

    let id = UUID()
    let question: String
    var found: [FoundNote] = []  // notes Spotlight surfaced (deduped, in order)
    var sources: [TrailNote] = []  // notes actually read → the citations
    var answer: String = ""
    var queries: [String] = []  // the model's own Spotlight queries (post-hoc)
    var phase: Phase = .retrieving
    var errorText: String?
}

@MainActor
@Observable
final class ChatModel {
    enum Status: Equatable {
        case starting  // indexing + downloading/loading + warming
        case downloading(Double)
        case ready
        case answering
        case error(String)

        var label: String {
            switch self {
            case .starting: return "Preparing…"
            case .downloading(let f): return "Downloading model… \(Int(f * 100))%"
            case .ready: return "Ready · on-device"
            case .answering: return "Thinking…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    var status: Status = .starting
    var turns: [ChatTurn] = []
    var modelName = "Qwen3 4B"
    var loadSeconds: Double?
    var footprintBytes: UInt64 = 0

    /// A note presented full-text in a sheet when a source chip is tapped.
    var inspectedNote: TrailNote?

    let suggestions = [
        "What did I write about the night hike?",
        "Where did I see eagles?",
        "Which hike had a waterfall?",
        "What should I pack next time?",
    ]

    private var engine: RagEngine?
    private var answerTask: Task<Void, Never>?

    var isReady: Bool { if case .ready = status { return true } else { return false } }
    var isBusy: Bool {
        switch status {
        case .ready, .error: return false
        default: return true
        }
    }
    var downloadFraction: Double? {
        if case .downloading(let f) = status { return f }
        return nil
    }

    /// Index the notes and load the model. Safe to call once on appear.
    func start() async {
        guard engine == nil else { return }
        let clock = ContinuousClock()
        let begin = clock.now
        do {
            status = .starting
            try await indexNotes()
            let engine = try await RagEngine.makeDefault { fraction in
                Task { @MainActor in
                    if fraction < 1 { self.status = .downloading(fraction) }
                    else if case .downloading = self.status { self.status = .starting }
                }
            }
            self.modelName = engine.modelName
            await engine.prewarm()
            self.engine = engine
            let elapsed = clock.now - begin
            self.loadSeconds =
                Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            self.footprintBytes = MemoryFootprint.current()
            self.status = .ready
        } catch {
            self.status = .error(error.localizedDescription)
        }
    }

    /// Ask a question. Appends a turn and streams the retrieval chain + answer into it.
    func ask(_ text: String) {
        guard let engine, isReady else { return }
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }

        turns.append(ChatTurn(question: question))
        let index = turns.count - 1
        status = .answering

        answerTask = Task {
            do {
                let answer = try await engine.answer(
                    to: question,
                    onFound: { found in
                        Task { @MainActor in self.mergeFound(found, at: index) }
                    },
                    onReading: { id in
                        Task { @MainActor in self.addSource(id, at: index) }
                    },
                    onAnswer: { text in
                        Task { @MainActor in self.setAnswer(text, at: index) }
                    })
                guard index < turns.count else { return }
                turns[index].queries = answer.queries
                turns[index].answer = answer.text
                turns[index].phase = .done
                footprintBytes = MemoryFootprint.current()
            } catch is CancellationError {
                if index < turns.count { turns[index].phase = .done }
            } catch {
                if index < turns.count {
                    turns[index].errorText = Self.friendly(error)
                    turns[index].phase = .failed
                }
            }
            if case .answering = status { status = .ready }
        }
    }

    func cancel() {
        answerTask?.cancel()
    }

    func newChat() {
        cancel()
        turns = []
    }

    // MARK: - Live-callback mutators (main actor)

    private func mergeFound(_ found: [FoundNote], at index: Int) {
        guard index < turns.count else { return }
        var existing = Set(turns[index].found.map(\.id))
        for note in found where !existing.contains(note.id) {
            turns[index].found.append(note)
            existing.insert(note.id)
        }
    }

    private func addSource(_ id: String, at index: Int) {
        guard index < turns.count, let note = notesByID[id] else { return }
        if !turns[index].sources.contains(note) {
            turns[index].sources.append(note)
        }
    }

    private func setAnswer(_ text: String, at index: Int) {
        guard index < turns.count else { return }
        turns[index].answer = text
        if turns[index].phase == .retrieving, !text.isEmpty {
            turns[index].phase = .answering
        }
    }

    private static func friendly(_ error: Error) -> String {
        let text = error.localizedDescription
        if text.contains("without producing a response") {
            return "The model finished without an answer — try rephrasing the question."
        }
        return text
    }
}

/// Resident memory of this process, for the on-device footprint readout.
enum MemoryFootprint {
    static func current() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
}
