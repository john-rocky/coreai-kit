// RagEngine.swift — the verified 2-tool RAG, headless. Owns the loaded zoo model and runs one
// grounded answer per call: SpotlightSearchTool (retrieve) + FetchNoteTool (hydrate) behind a
// KitLanguageModel. This is the exact path Examples/SpotlightChat proved on the CLI; the SwiftUI
// layer (ChatModel) and the headless self-test both call straight into it.

import CoreAIKit
import CoreSpotlight
import Foundation
import FoundationModels
import Synchronization

/// Loads and holds one zoo language bundle, and answers questions grounded in the indexed notes.
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
        Answer questions about the user's trail notes. First call spotlight_search to find \
        relevant notes, then call fetch_note with a note id to read its full text, then answer \
        briefly quoting only what you read. /no_think
        """

    struct Answer: Sendable {
        var text: String
        var readNoteIDs: [String]
        var queries: [String]
    }

    /// Runs one grounded answer. Callbacks stream the retrieval chain live (all `@Sendable` — the
    /// framework drives tools and the result stream off the main actor): `onFound` per Spotlight
    /// result batch, `onReading` when a note body is fetched, `onAnswer` with the cumulative
    /// answer text. Returns the final answer plus the read ids and the model's own queries.
    func answer(
        to question: String,
        onFound: @escaping @Sendable ([FoundNote]) -> Void = { _ in },
        onReading: @escaping @Sendable (String) -> Void = { _ in },
        onAnswer: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Answer {
        // Guidance defaults to `.complete` (~13k tokens of instructions — beyond a 4k-context
        // model). `.focused(.items)` + `.compact` keeps the tool prompt within budget.
        let tool = SpotlightSearchTool(
            configuration: .init(
                sources: [.coreSpotlight(CoreSpotlightSource(searchableIndexDelegate: NotesIndexDelegate()))],
                guide: SpotlightSearchTool.Guide(level: .focused(.items), format: .compact)))

        let readIDs = Mutex<[String]>([])
        let fetchTool = FetchNoteTool(onFetch: { id in
            readIDs.withLock { $0.append(id) }
            onReading(id)
        })

        // Observe the retriever's results live, separate from the model's prose.
        let observer = Task {
            for await reply in tool.searchResults {
                if let found = foundNotes(from: reply), !found.isEmpty { onFound(found) }
            }
        }
        defer { observer.cancel() }

        // Two tools, one model: spotlight_search finds notes, fetch_note reads their bodies.
        let session = LanguageModelSession(
            model: model, tools: [tool, fetchTool], instructions: Self.instructions)

        var finalText = ""
        for try await snapshot in session.streamResponse(to: question) {
            finalText = snapshot.content
            onAnswer(finalText)
        }

        return Answer(
            text: finalText,
            readNoteIDs: readIDs.withLock { $0 },
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
