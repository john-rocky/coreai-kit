import CoreAIKitUI
import Foundation
import Observation

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
    var bubbles: [ChatBubble] = []
    var stats = GenerationStats()
    var entries: [CatalogEntry] = []
    var selectedEntry: CatalogEntry?

    private var session: ChatSession?

    var isBusy: Bool {
        switch status {
        case .downloading, .loading, .warming, .generating: return true
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
        entries = await ModelCatalog.load().available(.chat)
        if selectedEntry == nil { selectedEntry = entries.first }
    }

    func load() {
        guard !isBusy, let entry = selectedEntry else { return }
        status = .loading
        bubbles = []
        session = nil
        Task {
            do {
                // Same gesture as the model card: the catalog id resolves the platform
                // variant and the engine hint (zoo decode-only ports ride "pipelined").
                var config = ChatSession.Configuration()
                #if os(iOS)
                // iOS-only: the pipelined engine's growing KV cache mis-respecializes
                // on-device when a turn pre-grows capacity to >=2048
                // (history+prompt+maxTokens): output is corrupt from token 1 while the
                // Mac handles the same growth fine. 896 keeps a fresh turn inside the
                // device-verified 1024-capacity envelope until the engine fix lands
                // (device-bisected 2026-07-24, nanbeige4.2-3b: capacity 1024 clean /
                // 2048 corrupt). NB: long multi-turn history can still cross 1024 —
                // start a New chat for a fresh budget.
                config.maxResponseTokens = 896
                #endif
                let session = try await ChatSession(catalog: entry.id, configuration: config) { progress in
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

        bubbles.append(ChatBubble(role: .user, content: trimmed))
        bubbles.append(ChatBubble(role: .assistant, isStreaming: true))
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
