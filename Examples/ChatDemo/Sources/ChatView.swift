import CoreAIKit
import SwiftUI

struct ChatView: View {
    @State private var model = ChatModel()
    @State private var input = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messages
            Divider()
            statsBar
            inputBar
        }
        #if os(macOS)
        .frame(minWidth: 480, minHeight: 560)
        #endif
    }

    private var header: some View {
        HStack {
            Picker("Model", selection: $model.selectedStarter) {
                ForEach(Starter.all) { starter in
                    Text(starter.name).tag(starter)
                }
            }
            .fixedSize()
            .disabled(model.isBusy)
            Button(model.status == .idle ? "Download & Load" : "Reload") { model.load() }
                .disabled(model.isBusy)
            Spacer()
            if case .downloading(let fraction) = model.status {
                ProgressView(value: fraction).frame(width: 80)
            }
            Text(model.status.label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(10)
    }

    private var messages: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(model.bubbles) { bubble in
                        BubbleView(bubble: bubble).id(bubble.id)
                    }
                }
                .padding(12)
            }
            .onChange(of: model.bubbles.last?.content) {
                if let id = model.bubbles.last?.id {
                    proxy.scrollTo(id, anchor: .bottom)
                }
            }
        }
    }

    private var statsBar: some View {
        HStack(spacing: 14) {
            if let load = model.stats.loadSeconds {
                stat("load", String(format: "%.1fs", load))
            }
            if let ttft = model.stats.ttftSeconds {
                stat("TTFT", String(format: "%.2fs", ttft))
            }
            if let tps = model.stats.tokensPerSecond {
                stat("tok/s", String(format: "%.1f", tps))
            }
            stat("tokens", "\(model.stats.promptTokens)→\(model.stats.generatedTokens)")
            stat("mem", ByteCountFormatter.string(
                fromByteCount: Int64(model.stats.footprintBytes), countStyle: .memory))
            Spacer()
        }
        .font(.caption.monospacedDigit())
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        HStack(spacing: 3) {
            Text(label).foregroundStyle(.secondary)
            Text(value)
        }
    }

    private var inputBar: some View {
        HStack {
            TextField("Message", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
                .disabled(model.status != .ready)
            if model.status == .generating {
                Button("Stop") { model.stop() }
            } else {
                Button("Send", action: send)
                    .disabled(model.status != .ready || input.isEmpty)
            }
            Button("New") { model.newChat() }
                .disabled(model.isBusy)
        }
        .padding(10)
    }

    private func send() {
        let text = input
        input = ""
        model.send(text)
    }
}

struct BubbleView: View {
    let bubble: Bubble

    var body: some View {
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
