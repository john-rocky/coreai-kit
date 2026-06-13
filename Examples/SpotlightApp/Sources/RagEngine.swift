// RagEngine.swift — the verified 2-tool RAG, headless, now over the user's real files. Owns the
// loaded zoo model and runs one grounded answer per call: SpotlightSearchTool's `.files` source
// (retrieve over the chosen folder/files) + FetchFileTool (read the chosen file's text) behind a
// KitLanguageModel. This is the same chain Examples/SpotlightChat proved on the CLI; the SwiftUI
// layer (ChatModel) and the headless self-test both call straight into it. The only change from
// the sample-notes version is the search source: `.files(FileSource(scopes:))` instead of an
// app-private Core Spotlight index — so the corpus is whatever real files the user granted.

import CoreAIKit
import CoreSpotlight
import Foundation
import FoundationModels
import Synchronization

/// Loads and holds one zoo language bundle, and answers questions grounded in the active file
/// library. Stateless about the corpus — the library is passed per question, so switching folders
/// needs no reload.
final class RagEngine: Sendable {
    let model: KitLanguageModel
    let modelName: String

    private init(model: KitLanguageModel, modelName: String) {
        self.model = model
        self.modelName = modelName
    }

    /// Downloads (first run) and loads the default zoo model. qwen3-4b: ChatML tokenizer, so the
    /// kit declares `.toolCalling`; large enough to hold the rich SpotlightSearchTool schema where
    /// the 0.6B loops. Cached afterwards — fully offline on later launches.
    static func makeDefault(
        onProgress: @escaping @Sendable (Double) -> Void
    ) async throws -> RagEngine {
        let model = try await KitLanguageModel(model: .qwen3_4B) { progress in
            onProgress(progress.fraction)
        }
        return RagEngine(model: model, modelName: "Qwen3 4B")
    }

    /// One generation at a time per engine (the underlying engine is serial). Disabling qwen3's
    /// chain-of-thought with `/no_think` keeps the search→fetch chain inside the token budget;
    /// without it the reasoning can run to the cap and the framework reports "ended without
    /// producing a response" (harmless to non-qwen models, which ignore the directive).
    private static let instructions = """
        You help the user ask questions about their own files, using two tools. For EVERY question: \
        (1) ALWAYS call spotlight_search first, with a few concise keywords from the question — \
        never answer from memory, never ask the user for more detail, and never claim you can't \
        access the files (you can, through the tools). (2) Then call fetch_file with the identifier \
        of the most relevant result to read its full text. (3) Answer briefly, quoting only what \
        you actually read. /no_think
        """

    struct Answer: Sendable {
        var text: String
        var readFiles: [LibraryFile]
        var queries: [String]
    }

    /// Runs one grounded answer over `library`. Callbacks stream the retrieval chain live (all
    /// `@Sendable` — the framework drives tools and the result stream off the main actor): `onFound`
    /// per Spotlight result batch, `onReading` when a file's text is fetched, `onAnswer` with the
    /// cumulative answer text. Returns the final answer plus the read files and the model's queries.
    func answer(
        to question: String,
        library: LibrarySnapshot,
        onFound: @escaping @Sendable ([FoundFile]) -> Void = { _ in },
        onReading: @escaping @Sendable (LibraryFile) -> Void = { _ in },
        onAnswer: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Answer {
        // Search the user's files via the app's own Core Spotlight index (ChatModel indexed the
        // picked files into the `user-files` domain). This is the verified `.coreSpotlight` round
        // trip behind a third-party model; the `.files` source does not honor its scopes in the
        // 27.0 beta (see FileLibrary). We ask for the identity attributes the UI + fetch tool need
        // (name + the URL to open the file); the body is read on demand by fetch_file.
        let tool = SpotlightSearchTool(
            configuration: .init(
                sources: [
                    .coreSpotlight(
                        CoreSpotlightSource(
                            searchableIndexDelegate: FileIndexDelegate(),
                            fetchAttributes: [.displayName, .title, .path, .contentURL]))
                ],
                guide: SpotlightSearchTool.Guide(level: .focused(.items), format: .compact)))

        let readFiles = Mutex<[LibraryFile]>([])
        let fetchTool = FetchFileTool(library: library) { file in
            readFiles.withLock { if !$0.contains(file) { $0.append(file) } }
            onReading(file)
        }

        // Observe the retriever's results live, separate from the model's prose.
        let observer = Task {
            for await reply in tool.searchResults {
                if let found = foundFiles(from: reply), !found.isEmpty { onFound(found) }
            }
        }
        defer { observer.cancel() }

        // Two tools, one model: spotlight_search finds files, fetch_file reads their text.
        let session = LanguageModelSession(
            model: model, tools: [tool, fetchTool], instructions: Self.instructions)

        var finalText = ""
        for try await snapshot in session.streamResponse(to: question) {
            finalText = snapshot.content
            onAnswer(finalText)
        }

        return Answer(
            text: finalText,
            readFiles: readFiles.withLock { $0 },
            queries: searchQueries(in: session.transcript))
    }

    /// Compiles the sampler graph and touches the weights so the first real question is snappy.
    /// A tool-free 1-token turn is enough; the per-question sessions reset from it cleanly.
    func prewarm() async {
        let warm = LanguageModelSession(model: model)
        _ = try? await warm.respond(
            to: "Hi", options: GenerationOptions(maximumResponseTokens: 1))
    }
}
