import CoreAIKit
import Foundation
import Observation

struct Starter: Identifiable, Hashable {
    let name: String
    let model: ModelID
    let sizeMB: Int?
    var id: String { model.repo }

    var label: String {
        sizeMB.map { "\(name) (\($0) MB)" } ?? name
    }
}

struct Bubble: Identifiable, Equatable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var thinking = ""
    var content = ""
    var isStreaming = false
}

@MainActor
@Observable
final class ChatModel {
    enum Status: Equatable {
        case idle
        case downloading(Double)
        case loading
        case warming
        case ready
        case generating
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pick a model"
            case .downloading(let f): return "Downloading… \(Int(f * 100))%"
            case .loading: return "Loading…"
            case .warming: return "Warming up…"
            case .ready: return "Ready"
            case .generating: return "Generating…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    var status: Status = .idle
    var bubbles: [Bubble] = []
    var stats = GenerationStats()
    var starters: [Starter] = []
    var selectedStarter: Starter?

    private var session: ChatSession?

    var isBusy: Bool {
        switch status {
        case .downloading, .loading, .warming, .generating: return true
        default: return false
        }
    }

    /// Live catalog with the built-in snapshot as offline fallback.
    func loadCatalog() async {
        guard starters.isEmpty else { return }
        let catalog = await ModelCatalog.load()
        starters = catalog.available(.chat).compactMap { entry in
            entry.modelID.map {
                Starter(name: entry.name, model: $0, sizeMB: entry.variant?.sizeMB)
            }
        }
        if selectedStarter == nil { selectedStarter = starters.first }
    }

    func load() {
        guard !isBusy, let starter = selectedStarter else { return }
        status = .loading
        bubbles = []
        session = nil
        Task {
            do {
                let session = try await ChatSession(model: starter.model) { progress in
                    Task { @MainActor in
                        self.status = progress.fraction < 1
                            ? .downloading(progress.fraction) : .loading
                    }
                }
                self.status = .warming
                try await session.prewarm()
                self.session = session
                self.stats = await session.stats
                self.status = .ready
            } catch {
                self.status = .error(error.localizedDescription)
            }
        }
    }

    func send(_ text: String) {
        guard let session, status == .ready else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        bubbles.append(Bubble(role: .user, content: trimmed))
        var reply = Bubble(role: .assistant)
        reply.isStreaming = true
        bubbles.append(reply)
        status = .generating

        Task {
            do {
                for try await event in await session.streamResponse(to: trimmed) {
                    switch event {
                    case .response(let delta):
                        bubbles[bubbles.count - 1].content += delta
                    case .thinking(let delta):
                        bubbles[bubbles.count - 1].thinking += delta
                    case .stats(let s):
                        stats = s
                    case .complete:
                        break
                    }
                }
            } catch {
                status = .error(error.localizedDescription)
            }
            bubbles[bubbles.count - 1].isStreaming = false
            if status == .generating { status = .ready }
        }
    }

    func stop() {
        Task { await session?.cancelGeneration() }
    }

    func newChat() {
        guard let session else { return }
        bubbles = []
        Task { await session.reset() }
    }
}
