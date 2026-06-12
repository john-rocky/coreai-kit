// ChatTranscriptView.swift — drop-in chat transcript: bubbles with thinking disclosure
// and auto-scroll while streaming.

import SwiftUI

@_exported import CoreAIKit

/// One display bubble. Apps append a user bubble and a streaming assistant bubble, then
/// feed `ChatSession.Event` deltas into the last element.
public struct ChatBubble: Identifiable, Equatable, Sendable {
    public enum Role: Sendable {
        case user, assistant
    }

    public let id: UUID
    public let role: Role
    public var thinking: String
    public var content: String
    public var isStreaming: Bool

    public init(
        role: Role, content: String = "", thinking: String = "", isStreaming: Bool = false
    ) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.thinking = thinking
        self.isStreaming = isStreaming
    }
}

public struct ChatTranscriptView: View {
    private let bubbles: [ChatBubble]

    public init(bubbles: [ChatBubble]) {
        self.bubbles = bubbles
    }

    public var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(bubbles) { bubble in
                        ChatBubbleView(bubble: bubble).id(bubble.id)
                    }
                }
                .padding(12)
            }
            .onChange(of: bubbles.last?.content) {
                if let id = bubbles.last?.id {
                    proxy.scrollTo(id, anchor: .bottom)
                }
            }
        }
    }
}

public struct ChatBubbleView: View {
    private let bubble: ChatBubble

    public init(bubble: ChatBubble) {
        self.bubble = bubble
    }

    public var body: some View {
        VStack(alignment: bubble.role == .user ? .trailing : .leading, spacing: 4) {
            if !bubble.thinking.isEmpty {
                DisclosureGroup("Thinking") {
                    Text(bubble.thinking)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .font(.caption)
            }
            Text(bubble.content.isEmpty && bubble.isStreaming ? "…" : bubble.content)
                .textSelection(.enabled)
                .padding(10)
                .background(
                    bubble.role == .user
                        ? Color.accentColor.opacity(0.15) : Color.gray.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 10))
        }
        .frame(maxWidth: .infinity, alignment: bubble.role == .user ? .trailing : .leading)
    }
}
