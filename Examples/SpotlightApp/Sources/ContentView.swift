// ContentView.swift — "ask your notes" chat. The differentiator over a plain chat box is that
// the retrieval chain is visible: while the model works you see it search and read your notes,
// and every answer carries the source notes it was grounded in (tap to read the full text).

import SwiftUI

struct ContentView: View {
    @State private var model = ChatModel()
    @State private var input = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            Divider()
            inputBar
        }
        .task { await model.start() }
        .sheet(item: $model.inspectedNote) { note in
            NoteDetailView(note: note)
        }
        #if os(macOS)
        .frame(minWidth: 520, minHeight: 620)
        #endif
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Ask your notes").font(.headline)
                    Text(model.modelName).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                offlineBadge
            }
            statusLine
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var offlineBadge: some View {
        Label("Works offline", systemImage: "airplane")
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.green.opacity(0.15), in: Capsule())
            .foregroundStyle(.green)
    }

    @ViewBuilder
    private var statusLine: some View {
        switch model.status {
        case .ready:
            HStack(spacing: 10) {
                Label("On-device", systemImage: "lock.shield").labelStyle(.titleAndIcon)
                if let load = model.loadSeconds {
                    Text(String(format: "loaded %.1fs", load))
                }
                if model.footprintBytes > 0 {
                    Text(
                        ByteCountFormatter.string(
                            fromByteCount: Int64(model.footprintBytes), countStyle: .memory))
                }
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
        case .error(let message):
            Text(message).font(.caption).foregroundStyle(.red).lineLimit(2)
        default:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(model.status.label).font(.caption).foregroundStyle(.secondary)
                if let fraction = model.downloadFraction {
                    ProgressView(value: fraction).frame(width: 90)
                }
            }
        }
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    if model.turns.isEmpty { emptyState }
                    ForEach(model.turns) { turn in
                        TurnView(turn: turn) { note in model.inspectedNote = note }
                            .id(turn.id)
                    }
                }
                .padding(14)
            }
            .onChange(of: model.turns.last?.answer) { scrollToEnd(proxy) }
            .onChange(of: model.turns.count) { scrollToEnd(proxy) }
        }
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        guard let id = model.turns.last?.id else { return }
        withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .bottom) }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Five trail notes are indexed on this device. Ask about them — the answer is "
                + "retrieved and grounded locally, no network.")
                .font(.callout)
                .foregroundStyle(.secondary)
            FlowChips(model.suggestions) { suggestion in
                guard model.isReady else { return }
                model.ask(suggestion)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 8)
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask about your notes…", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
                .disabled(!model.isReady)
            if case .answering = model.status {
                Button("Stop") { model.cancel() }
            } else {
                Button("Ask", action: send)
                    .disabled(!model.isReady || input.trimmingCharacters(in: .whitespaces).isEmpty)
                    .keyboardShortcut(.return, modifiers: [])
            }
            if !model.turns.isEmpty {
                Button { model.newChat() } label: { Image(systemName: "trash") }
                    .help("New conversation")
            }
        }
        .padding(10)
    }

    private func send() {
        let text = input
        input = ""
        model.ask(text)
    }
}

// MARK: - One turn

struct TurnView: View {
    let turn: ChatTurn
    let onInspect: (TrailNote) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Question (trailing bubble).
            Text(turn.question)
                .padding(10)
                .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
                .frame(maxWidth: .infinity, alignment: .trailing)

            // Assistant side.
            VStack(alignment: .leading, spacing: 8) {
                if turn.phase == .retrieving {
                    liveStatus
                }
                if !turn.answer.isEmpty {
                    Text(turn.answer)
                        .textSelection(.enabled)
                        .padding(10)
                        .background(
                            Color.gray.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
                }
                if let error = turn.errorText {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .font(.callout).foregroundStyle(.orange)
                }
                if !turn.sources.isEmpty {
                    sources
                }
                if turn.phase == .done {
                    RetrievalTrace(turn: turn)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var liveStatus: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(turn.sources.last.map { "Reading “\($0.title)”…" } ?? "Searching your notes…")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    private var sources: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Sources").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            FlowChips(turn.sources.map(\.title)) { title in
                if let note = turn.sources.first(where: { $0.title == title }) { onInspect(note) }
            }
        }
    }
}

/// The visible 2-tool chain: what the model searched, what Spotlight returned, what it read.
struct RetrievalTrace: View {
    let turn: ChatTurn

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 4) {
                if !turn.queries.isEmpty {
                    row("magnifyingglass", "Searched", turn.queries.joined(separator: ", "))
                }
                if !turn.found.isEmpty {
                    row(
                        "doc.text.magnifyingglass", "Found",
                        turn.found.map(\.title).joined(separator: ", "))
                }
                if !turn.sources.isEmpty {
                    row("text.book.closed", "Read", turn.sources.map(\.title).joined(separator: ", "))
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.top, 2)
        } label: {
            Text("How it answered").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
        }
    }

    private func row(_ symbol: String, _ label: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: symbol).frame(width: 16)
            Text(label).fontWeight(.semibold)
            Text(value)
            Spacer(minLength: 0)
        }
    }
}

// MARK: - Note inspector

struct NoteDetailView: View {
    let note: TrailNote
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(note.title).font(.title2.weight(.semibold))
                Spacer()
                Button("Done") { dismiss() }
            }
            Text(note.text).font(.body)
            HStack(spacing: 6) {
                ForEach(note.keywords, id: \.self) { keyword in
                    Text(keyword)
                        .font(.caption2)
                        .padding(.horizontal, 7).padding(.vertical, 3)
                        .background(Color.gray.opacity(0.15), in: Capsule())
                }
            }
            Spacer()
        }
        .padding(20)
        .frame(minWidth: 320, minHeight: 240)
    }
}

// MARK: - Simple wrapping chip row

/// A lightweight wrapping row of tappable chips (no external layout dependency).
struct FlowChips: View {
    private let items: [String]
    private let onTap: (String) -> Void

    init(_ items: [String], onTap: @escaping (String) -> Void) {
        self.items = items
        self.onTap = onTap
    }

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(items, id: \.self) { item in
                Button { onTap(item) } label: {
                    Text(item)
                        .font(.callout)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(Color.accentColor.opacity(0.12), in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
    }
}

/// Minimal flow layout: lays children left-to-right, wrapping to the next line.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += lineHeight + spacing
                lineHeight = 0
            }
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: maxWidth == .infinity ? x : maxWidth, height: y + lineHeight)
    }

    func placeSubviews(
        in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout Void
    ) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX, x > bounds.minX {
                x = bounds.minX
                y += lineHeight + spacing
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), anchor: .topLeading, proposal: .unspecified)
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}
