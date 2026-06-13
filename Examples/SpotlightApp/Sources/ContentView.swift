// ContentView.swift — "ask your files": a chat over the user's OWN documents, answered by their
// own local model, offline. The design follows the demo-legibility bar: a one-line identity, a
// persistent "<model> · on-device · offline" badge, an honest "your model, not Apple Intelligence"
// note, and — the centerpiece — a visible RAG trace (search → read "<real file>" → answer) so the
// "magic" is on screen, not hidden. There is never an empty state: sample documents are seeded, and
// the user can point the app at a real folder or files with the picker.

import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

struct ContentView: View {
    @State private var model = ChatModel()
    @State private var input = ""
    @State private var showFolderPicker = false
    @State private var showFilePicker = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            libraryBar
            Divider()
            transcript
            Divider()
            inputBar
        }
        .task { await model.start() }
        .sheet(item: $model.inspectedFile) { file in
            FileDetailView(file: file)
        }
        .fileImporter(
            isPresented: $showFolderPicker, allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { handlePick($0) }
        .fileImporter(
            isPresented: $showFilePicker, allowedContentTypes: FileLibrary.readableTypes,
            allowsMultipleSelection: true
        ) { handlePick($0) }
        #if os(macOS)
        .frame(minWidth: 560, minHeight: 680)
        #endif
    }

    private func handlePick(_ result: Result<[URL], Error>) {
        guard case .success(let urls) = result, !urls.isEmpty else { return }
        model.selectRoots(urls)
    }

    // MARK: - Header (identity + persistent badge)

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Ask your files").font(.title2.weight(.bold))
            Text("Answered by \(model.modelName) on your device — works offline")
                .font(.subheadline).foregroundStyle(.secondary)
            badges
            statusLine
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var badges: some View {
        HStack(spacing: 6) {
            Badge(model.modelName, "cpu", .blue)
            Badge("on-device", "lock.shield", .green)
            Badge("offline", "airplane", .green)
            Spacer(minLength: 0)
            Text("YOUR model, not Apple Intelligence")
                .font(.caption2).foregroundStyle(.tertiary)
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        switch model.status {
        case .ready:
            HStack(spacing: 10) {
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

    // MARK: - Library bar ("Searching N files in <folder>" + pickers)

    private var libraryBar: some View {
        HStack(spacing: 10) {
            Image(systemName: model.isSampleLibrary ? "doc.on.doc" : "folder.fill")
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    if model.isIndexing { ProgressView().controlSize(.small) }
                    Text(libraryHeadline).font(.callout.weight(.medium))
                }
                Text(model.isSampleLibrary ? "Sample documents — tap to use your own" : model.libraryLabel)
                    .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 0)
            Menu {
                Button("Choose a folder…") { showFolderPicker = true }
                Button("Choose files…") { showFilePicker = true }
                if !model.isSampleLibrary {
                    Divider()
                    Button("Use sample documents") { model.useSampleLibrary() }
                }
            } label: {
                Label("Choose", systemImage: "folder.badge.plus")
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var libraryHeadline: String {
        let n = model.fileCount
        let unit = "file\(n == 1 ? "" : "s")"
        return model.isIndexing ? "Indexing \(n) \(unit)…" : "Searching \(n) \(unit)"
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if model.turns.isEmpty { emptyState }
                    ForEach(model.turns) { turn in
                        TurnView(turn: turn) { file in model.inspectedFile = file }
                            .id(turn.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: model.turns.last?.answer) { scrollToEnd(proxy) }
            .onChange(of: model.turns.count) { scrollToEnd(proxy) }
        }
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        guard let id = model.turns.last?.id else { return }
        withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .bottom) }
    }

    private var emptyStateText: String {
        if model.isSampleLibrary {
            let n = model.fileCount
            return "\(n) sample documents are on this device. Ask about them — or choose your own "
                + "folder above. Every answer is retrieved and grounded locally, no network."
        }
        return "Searching \(model.fileCount) files in “\(model.libraryLabel)”. Ask anything about "
            + "them — retrieved and answered on device."
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(emptyStateText)
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("Try a question").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            FlowChips(model.suggestions) { suggestion in
                guard model.canAsk else { return }
                model.ask(suggestion)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 4)
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask about your files…", text: $input)
                .textFieldStyle(.roundedBorder)
                .onSubmit(send)
                .disabled(!model.canAsk)
            if case .answering = model.status {
                Button("Stop") { model.cancel() }
            } else {
                Button("Ask", action: send)
                    .disabled(!model.canAsk || input.trimmingCharacters(in: .whitespaces).isEmpty)
                    .keyboardShortcut(.return, modifiers: [])
            }
            if !model.turns.isEmpty {
                Button { model.newChat() } label: { Image(systemName: "trash") }
                    .help("New conversation")
            }
        }
        .padding(12)
    }

    private func send() {
        let text = input
        input = ""
        model.ask(text)
    }
}

// MARK: - Persistent badge

struct Badge: View {
    let text: String
    let symbol: String
    let tint: Color

    init(_ text: String, _ symbol: String, _ tint: Color) {
        self.text = text
        self.symbol = symbol
        self.tint = tint
    }

    var body: some View {
        Label(text, systemImage: symbol)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(tint.opacity(0.15), in: Capsule())
            .foregroundStyle(tint)
    }
}

// MARK: - One turn (question + the visible RAG chain + answer)

struct TurnView: View {
    let turn: ChatTurn
    let onInspect: (LibraryFile) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Question (trailing bubble).
            Text(turn.question)
                .padding(10)
                .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
                .frame(maxWidth: .infinity, alignment: .trailing)

            // Assistant side: the RAG trace is the star, then the grounded answer + citations.
            VStack(alignment: .leading, spacing: 10) {
                RetrievalTraceCard(turn: turn)

                if !turn.answer.isEmpty {
                    Text(turn.answer)
                        .textSelection(.enabled)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
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
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var sources: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("Sources — tap to open", systemImage: "doc.text.magnifyingglass")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            FlowChips(turn.sources.map(\.displayName)) { name in
                if let file = turn.sources.first(where: { $0.displayName == name }) {
                    onInspect(file)
                }
            }
        }
    }
}

/// The visible 2-tool chain — the demo centerpiece. Three steps light up in order: search the
/// files, read the ones that matter, answer from them. Stays on screen after the turn completes so
/// the whole "how it answered" is legible at a glance (and in a screenshot / recording).
struct RetrievalTraceCard: View {
    let turn: ChatTurn

    enum StepState { case pending, active, done }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            step(
                state: searchState, symbol: "magnifyingglass", title: "Searching your files",
                detail: turn.queries.isEmpty ? nil : turn.queries.joined(separator: ", "))
            connector
            step(
                state: readState, symbol: "doc.text", title: readTitle,
                detail: turn.sources.isEmpty ? nil : turn.sources.map(\.displayName).joined(separator: ", "))
            connector
            step(
                state: answerState, symbol: "checkmark.seal", title: "Answer",
                detail: nil)
        }
        .padding(12)
        .background(Color.gray.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12).strokeBorder(Color.gray.opacity(0.18)))
    }

    private var readTitle: String {
        if let last = turn.sources.last, readState == .active { return "Reading “\(last.displayName)”" }
        return turn.sources.isEmpty ? "Reading files" : "Read \(turn.sources.count) file\(turn.sources.count == 1 ? "" : "s")"
    }

    private var hasOutput: Bool { !turn.answer.isEmpty }

    private var searchState: StepState {
        if !turn.found.isEmpty || !turn.sources.isEmpty || hasOutput || turn.phase != .retrieving {
            return .done
        }
        return turn.phase == .retrieving ? .active : .pending
    }

    private var readState: StepState {
        if !turn.sources.isEmpty && (turn.phase == .done || turn.phase == .answering || hasOutput) {
            return .done
        }
        if turn.phase == .retrieving && (!turn.found.isEmpty || !turn.sources.isEmpty) { return .active }
        return .pending
    }

    private var answerState: StepState {
        if turn.phase == .done { return .done }
        if hasOutput || turn.phase == .answering { return .active }
        return .pending
    }

    @ViewBuilder
    private func step(state: StepState, symbol: String, title: String, detail: String?) -> some View {
        HStack(alignment: .top, spacing: 10) {
            icon(for: state, symbol: symbol)
                .frame(width: 22, height: 22)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.callout.weight(state == .pending ? .regular : .semibold))
                    .foregroundStyle(state == .pending ? .secondary : .primary)
                if let detail, !detail.isEmpty {
                    Text(detail).font(.caption).foregroundStyle(.secondary)
                        .lineLimit(2).truncationMode(.middle)
                }
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private func icon(for state: StepState, symbol: String) -> some View {
        switch state {
        case .pending:
            Image(systemName: symbol).foregroundStyle(.tertiary)
        case .active:
            ProgressView().controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        }
    }

    private var connector: some View {
        Rectangle().fill(Color.gray.opacity(0.2))
            .frame(width: 1.5, height: 12)
            .padding(.leading, 10)
    }
}

// MARK: - File inspector

struct FileDetailView: View {
    let file: LibraryFile
    @Environment(\.dismiss) private var dismiss
    @State private var content: String?
    @State private var loading = true

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(file.displayName).font(.title3.weight(.semibold))
                    Text(file.url.path).font(.caption2).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                }
                Spacer()
                openButton
                Button("Done") { dismiss() }
            }
            Divider()
            ScrollView {
                if loading {
                    ProgressView().frame(maxWidth: .infinity).padding()
                } else {
                    Text(content ?? "(could not read this file's text)")
                        .font(.body).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(20)
        .frame(minWidth: 380, minHeight: 320)
        .task {
            content = FileLibrary.text(of: file.url)
            loading = false
        }
    }

    @ViewBuilder
    private var openButton: some View {
        #if canImport(AppKit)
        Button("Open") { NSWorkspace.shared.open(file.url) }
        #else
        ShareLink(item: file.url) { Image(systemName: "square.and.arrow.up") }
        #endif
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
