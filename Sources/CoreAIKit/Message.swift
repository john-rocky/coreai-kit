import Foundation

/// One conversation turn.
public struct Message: Identifiable, Sendable, Equatable {
    public enum Role: String, Sendable {
        case system, user, assistant
    }

    public let id: UUID
    public let role: Role
    public var content: String
    /// Chain-of-thought text (qwen3 `<think>` blocks, gpt-oss analysis channel). Surfaced
    /// for display; never fed back into subsequent turns.
    public var thinking: String

    public init(role: Role, content: String, thinking: String = "") {
        self.id = UUID()
        self.role = role
        self.content = content
        self.thinking = thinking
    }
}

public enum ChatSessionError: Error, LocalizedError, Sendable {
    case generationInProgress
    /// The loaded engine cannot expose per-step logits (the GPU-pipelined engine
    /// samples on-GPU), which constrained decoding requires.
    case guidedGenerationUnsupported
    /// The constrained output did not decode into the requested type. Carries the
    /// generated JSON text for inspection.
    case malformedGuidedOutput(String)

    public var errorDescription: String? {
        switch self {
        case .generationInProgress:
            return "A generation is already running on this session — "
                + "consume it or call cancelGeneration() first."
        case .guidedGenerationUnsupported:
            return "This engine cannot expose per-step logits. Load the session with "
                + "configuration.engineVariant = .sequential for guided generation."
        case .malformedGuidedOutput(let json):
            return "Guided output did not decode into the requested type: "
                + "\(json.prefix(200))"
        }
    }
}
