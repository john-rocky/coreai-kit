// ChatModel.swift — the SwiftUI view model. Owns the RagEngine and the active file library, and
// turns the engine's live callbacks into observable UI state. One conversation, one question at a
// time; each question is an independent grounded retrieval (a fresh LanguageModelSession over the
// current library), so the transcript never outgrows the context window and the answer is always
// freshly searched.

import Foundation
import Observation
import SwiftUI

/// One exchange: the question, the retrieval chain it triggered over the user's files, and the
/// grounded answer.
struct ChatTurn: Identifiable {
    enum Phase {
        case retrieving  // searching / reading files
        case answering  // answer text streaming
        case done
        case failed
    }

    let id = UUID()
    let question: String
    var found: [FoundFile] = []  // files Spotlight surfaced (deduped, in order)
    var sources: [LibraryFile] = []  // files actually read → the citations
    var answer: String = ""
    var queries: [String] = []  // the model's own Spotlight queries (post-hoc)
    var phase: Phase = .retrieving
    var errorText: String?
}

@MainActor
@Observable
final class ChatModel {
    enum Status: Equatable {
        case starting  // seeding + downloading/loading + warming
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

    /// The active corpus: the real files the model searches. Seeded with sample documents, then
    /// replaced when the user picks their own folder/files. The files are indexed into the app's
    /// Core Spotlight index; `isIndexing` is true while that runs after a selection.
    var library: LibrarySnapshot = .empty
    var libraryLabel: String { library.displayName }
    var fileCount: Int { library.files.count }
    var isSampleLibrary = true
    var isIndexing = false

    /// A file presented full-text in a sheet when a source chip is tapped.
    var inspectedFile: LibraryFile?

    /// Starter questions. The sample-document set has known answers; once the user points at their
    /// own files we offer neutral prompts that work on any document collection.
    var suggestions: [String] {
        isSampleLibrary
            ? [
                "What did I write about the night hike?",
                "Which file mentions a waterfall?",
                "What do my notes say about eagles?",
                "What did I learn at Granite Pass?",
            ]
            : [
                "What do these files say about …?",
                "Which file mentions …?",
                "Summarize the file about …",
            ]
    }

    private var engine: RagEngine?
    private var answerTask: Task<Void, Never>?
    private var reindexTask: Task<Void, Never>?
    private var accessedRoots: [URL] = []  // security-scoped roots we hold open
    private static let bookmarksKey = "SpotlightApp.libraryBookmarks"

    var isReady: Bool { if case .ready = status { return true } else { return false } }
    /// Ready to take a question: model loaded and not mid-reindex of a freshly picked library.
    var canAsk: Bool { isReady && !isIndexing }
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

    /// Seed the sample documents, restore the last library (or fall back to the sample folder), and
    /// load the model. Safe to call once on appear.
    func start() async {
        guard engine == nil else { return }
        let clock = ContinuousClock()
        let begin = clock.now
        do {
            status = .starting
            restoreLibrary()  // sets `library` to the saved folder or the seeded samples
            // Index the corpus and download/load the model concurrently — the first question needs
            // both, and they are independent.
            async let indexing: Void = indexQuietly(library.files)
            let engine = try await RagEngine.makeDefault { fraction in
                Task { @MainActor in
                    if fraction < 1 { self.status = .downloading(fraction) }
                    else if case .downloading = self.status { self.status = .starting }
                }
            }
            await indexing
            self.modelName = engine.modelName
            await engine.prewarm()
            self.engine = engine
            let elapsed = clock.now - begin
            self.loadSeconds =
                Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            self.footprintBytes = MemoryFootprint.current()
            self.status = .ready
            // Dev hooks: SPOTLIGHT_SHOT renders a screenshot after the first answer and exits;
            // SPOTLIGHT_AUTOASK just kicks off the first sample question (for a live demo).
            let env = ProcessInfo.processInfo.environment
            if let path = env["SPOTLIGHT_SHOT"] {
                Task { await captureAfterAnswer(to: path) }
            } else if env["SPOTLIGHT_AUTOASK"] != nil, let first = suggestions.first {
                ask(first)
            }
        } catch {
            self.status = .error(error.localizedDescription)
        }
    }

    /// Screenshot helper: ask the first sample question, wait for the answer + trace to render, then
    /// write a PNG of the window and exit. macOS only (in-process view render, no capture permission).
    private func captureAfterAnswer(to path: String) async {
        if let first = suggestions.first { ask(first) }
        await answerTask?.value
        try? await Task.sleep(for: .seconds(2))  // let SwiftUI settle the final frame
        #if os(macOS)
        Screenshotter.capture(to: path)
        #endif
        exit(0)
    }

    // MARK: - Library selection

    /// Point the app at a folder or a set of files the user picked (from the picker). Switches the
    /// corpus and re-indexes it in the background.
    func selectRoots(_ urls: [URL]) {
        setLibrary(roots: urls, isSample: false, persist: true)
        reindexLive()
    }

    /// Reset to the bundled sample documents (user action) and re-index.
    func useSampleLibrary() {
        seedSampleAndSet()
        UserDefaults.standard.removeObject(forKey: Self.bookmarksKey)
        reindexLive()
    }

    /// Set the active corpus: hold security-scoped access to the new roots (releasing the old),
    /// rebuild the searchable snapshot, and (best effort) save a bookmark so the choice survives
    /// relaunch. Pure — indexing is separate so `start()` can do it concurrently with the load.
    private func setLibrary(roots: [URL], isSample: Bool, persist: Bool) {
        for url in accessedRoots { url.stopAccessingSecurityScopedResource() }
        accessedRoots = roots.filter { $0.startAccessingSecurityScopedResource() }

        library = FileLibrary.snapshot(forRoots: roots)
        isSampleLibrary = isSample
        if persist { saveBookmarks(for: roots) }
    }

    private func seedSampleAndSet() {
        let folder = FileLibrary.seedSampleDocumentsIfNeeded()
        setLibrary(roots: [folder], isSample: true, persist: false)
    }

    /// On launch: restore the saved folder if it still has readable files, else the samples. Does
    /// not index — `start()` indexes once, concurrently with the model load.
    private func restoreLibrary() {
        if let roots = restoreBookmarks(), !roots.isEmpty {
            setLibrary(roots: roots, isSample: false, persist: false)
            if !library.files.isEmpty { return }  // saved folder still has readable files
        }
        seedSampleAndSet()
    }

    /// Re-index the current corpus into Core Spotlight, blocking new questions until it's searchable.
    private func reindexLive() {
        reindexTask?.cancel()
        let files = library.files
        isIndexing = true
        reindexTask = Task {
            await indexQuietly(files)
            isIndexing = false
        }
    }

    private func indexQuietly(_ files: [LibraryFile]) async {
        try? await FileLibrary.index(files)
    }

    // MARK: - Asking

    /// Ask a question over the current library. Appends a turn and streams the chain + answer.
    func ask(_ text: String) {
        guard let engine, canAsk else { return }
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }

        turns.append(ChatTurn(question: question))
        let index = turns.count - 1
        status = .answering
        let library = library  // capture the corpus for this question

        answerTask = Task {
            do {
                let answer = try await engine.answer(
                    to: question,
                    library: library,
                    onFound: { found in
                        Task { @MainActor in self.mergeFound(found, at: index) }
                    },
                    onReading: { file in
                        Task { @MainActor in self.addSource(file, at: index) }
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

    private func mergeFound(_ found: [FoundFile], at index: Int) {
        guard index < turns.count else { return }
        var existing = Set(turns[index].found.map(\.id))
        for file in found where !existing.contains(file.id) {
            turns[index].found.append(file)
            existing.insert(file.id)
        }
    }

    private func addSource(_ file: LibraryFile, at index: Int) {
        guard index < turns.count else { return }
        if !turns[index].sources.contains(file) {
            turns[index].sources.append(file)
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

    // MARK: - Security-scoped bookmark persistence (best effort)

    private func saveBookmarks(for roots: [URL]) {
        let datas: [Data] = roots.compactMap { url in
            #if os(macOS)
            return try? url.bookmarkData(
                options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
            #else
            return try? url.bookmarkData(
                options: [], includingResourceValuesForKeys: nil, relativeTo: nil)
            #endif
        }
        UserDefaults.standard.set(datas, forKey: Self.bookmarksKey)
    }

    private func restoreBookmarks() -> [URL]? {
        guard let datas = UserDefaults.standard.array(forKey: Self.bookmarksKey) as? [Data] else {
            return nil
        }
        let urls: [URL] = datas.compactMap { data in
            var stale = false
            #if os(macOS)
            return try? URL(
                resolvingBookmarkData: data, options: [.withSecurityScope], relativeTo: nil,
                bookmarkDataIsStale: &stale)
            #else
            return try? URL(
                resolvingBookmarkData: data, options: [], relativeTo: nil,
                bookmarkDataIsStale: &stale)
            #endif
        }
        return urls.isEmpty ? nil : urls
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
