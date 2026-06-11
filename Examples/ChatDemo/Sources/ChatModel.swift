import CoreAIKit
import Foundation
import Observation

struct Starter: Identifiable, Hashable {
    let name: String
    let model: ModelID
    var id: String { model.repo }

    static var all: [Starter] {
        var list = [
            Starter(name: "Qwen3 0.6B", model: .qwen3_0_6B),
            Starter(name: "Qwen3 4B", model: .qwen3_4B),
        ]
        #if os(macOS)
        list.append(Starter(name: "Mistral 7B v0.3", model: .mistral_7B))
        list.append(Starter(name: "Gemma 3 4B", model: .gemma3_4B))
        #endif
        return list
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
        case ready
        case generating
        case error(String)

        var label: String {
            switch self {
            case .idle: return "Pick a model"
            case .downloading(let f): return "Downloading… \(Int(f * 100))%"
            case .loading: return "Loading…"
            case .ready: return "Ready"
            case .generating: return "Generating…"
            case .error(let message): return "Error: \(message)"
            }
        }
    }

    var status: Status = .idle
    var bubbles: [Bubble] = []
    var stats = GenerationStats()
    var selectedStarter: Starter = Starter.all[0]

    private var session: ChatSession?

    var isBusy: Bool {
        switch status {
        case .downloading, .loading, .generating: return true
        default: return false
        }
    }

    func load() {
        guard !isBusy else { return }
        let starter = selectedStarter
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
