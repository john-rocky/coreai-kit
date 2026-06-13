// SpotlightChat — conversational search over your own Spotlight-indexed data, driven by
// YOUR model. The WWDC26 `SpotlightSearchTool` (a local RAG retriever exposed as one
// FoundationModels `Tool`) is model-agnostic: it rides behind any `LanguageModel`. Here it
// runs behind a Core AI zoo bundle via `KitLanguageModel` instead of the system model.
//
// The pattern is the realistic one for an on-device app:
//   1. `spotlight_search` (Apple's system tool) finds candidate notes in the Spotlight index.
//   2. `fetch_note` (this app's tool) reads a note's full body from the app's own store.
//   3. The model grounds its answer in the text it read — all on device, on your model.
//
// Usage:
//   swift run SpotlightChat                       # downloads qwen3-4B, asks the default question
//   swift run SpotlightChat --ask "What did I write about the night hike?"
//   swift run SpotlightChat --system              # SystemLanguageModel baseline (Apple Intelligence)
//   swift run SpotlightChat --model <bundle-dir>  # a local Core AI bundle (plain-transformer;
//                                                 # qwen3.5 hybrids need a patched engine — see README)
//
// The process embeds an Info.plist (see Package.swift) so it has the bundle identity that
// CoreSpotlight requires; the notes are indexed under that identity on first run.

import CoreAIKit
import CoreSpotlight
import Foundation
import FoundationModels

// MARK: - Sample data (stands in for your app's real content)

struct TrailNote {
    let id: String
    let title: String
    let text: String
    let keywords: [String]
    let daysAgo: Int
}

let sampleNotes: [TrailNote] = [
    TrailNote(
        id: "note-001", title: "Eagle Ridge loop",
        text: "Clear morning, 14 km loop. The summit wind was brutal but the view over "
            + "the valley was worth it. Saw two golden eagles near the ridge line.",
        keywords: ["trail", "summit", "eagles"], daysAgo: 25),
    TrailNote(
        id: "note-002", title: "Silver Falls out-and-back",
        text: "Short hike to the waterfall. The silver falls were at full flow after the "
            + "rain; the spray soaked the lower viewpoint. Slippery boardwalk — bring grippy shoes next time.",
        keywords: ["trail", "waterfall", "rain"], daysAgo: 18),
    TrailNote(
        id: "note-003", title: "Pine Hollow night hike",
        text: "First night hike of the season. Headlamp died halfway — note to self: pack "
            + "spare batteries. Owls were loud near the hollow. Cold but calm.",
        keywords: ["trail", "night", "owls"], daysAgo: 12),
    TrailNote(
        id: "note-004", title: "Granite Pass attempt",
        text: "Turned around 2 km before the pass — late snowfield too steep without an "
            + "ice axe. The marmots near the boulder field were fearless. Try again in July.",
        keywords: ["trail", "snow", "marmots"], daysAgo: 7),
    TrailNote(
        id: "note-005", title: "River bend picnic walk",
        text: "Easy flat walk with the family. Herons fishing at the river bend. The new "
            + "boots felt great — no blisters after 8 km.",
        keywords: ["trail", "river", "herons", "boots"], daysAgo: 3),
]

let notesByID = Dictionary(uniqueKeysWithValues: sampleNotes.map { ($0.id, $0) })

// MARK: - Indexing

func makeItem(_ note: TrailNote) -> CSSearchableItem {
    let attributes = CSSearchableItemAttributeSet(contentType: .text)
    attributes.title = note.title
    attributes.contentDescription = note.text
    attributes.textContent = note.text
    attributes.keywords = note.keywords
    attributes.contentCreationDate = Calendar.current.date(
        byAdding: .day, value: -note.daysAgo, to: Date())
    return CSSearchableItem(
        uniqueIdentifier: note.id, domainIdentifier: "trail-notes", attributeSet: attributes)
}

func indexNotes() async throws {
    let index = CSSearchableIndex.default()
    try await index.deleteSearchableItems(withDomainIdentifiers: ["trail-notes"])
    try await index.indexSearchableItems(sampleNotes.map(makeItem))
    print("[index] \(sampleNotes.count) notes indexed under domain trail-notes")
}

/// Indexing is asynchronous; wait until the notes are actually searchable so the first
/// model turn doesn't race an empty index.
func waitUntilSearchable() async throws {
    for attempt in 1...20 {
        let context = CSSearchQueryContext()
        context.fetchAttributes = ["title"]
        let query = CSSearchQuery(queryString: "keywords == \"trail\"cd", queryContext: context)
        var found = 0
        for try await _ in query.results { found += 1 }
        if found >= sampleNotes.count {
            print("[index] all \(found) notes searchable (attempt \(attempt))")
            return
        }
        try await Task.sleep(for: .milliseconds(500))
    }
    print("[index] WARNING: notes not fully searchable after 10 s — continuing anyway")
}

/// The index can ask the app to re-supply items (e.g. after an index loss). Conforming the
/// source to a delegate keeps the index recoverable; the search path itself returns
/// metadata, so the full text is supplied here and by `fetch_note`.
final class NotesIndexDelegate: NSObject, CSSearchableIndexDelegate, @unchecked Sendable {
    func searchableIndex(
        _ index: CSSearchableIndex,
        reindexAllSearchableItemsWithAcknowledgementHandler ack: @escaping () -> Void
    ) {
        ack()  // the demo re-indexes from scratch on every launch
    }

    func searchableIndex(
        _ index: CSSearchableIndex,
        reindexSearchableItemsWithIdentifiers identifiers: [String],
        acknowledgementHandler ack: @escaping () -> Void
    ) {
        ack()
    }

    func searchableItems(
        forIdentifiers identifiers: [String],
        searchableItemsHandler handler: @escaping ([CSSearchableItem]) -> Void
    ) {
        handler(identifiers.compactMap { notesByID[$0].map(makeItem) })
    }
}

// MARK: - Companion hydration tool

// `SpotlightSearchTool` returns lightweight index metadata (title, dates, identifiers), not
// the note body. (A raw `CSSearchQuery` does return `contentDescription`, but the tool does
// not pass it to the model.) That mirrors a real app: the Spotlight index is a finding aid;
// full content lives in your own store. This tool closes the loop — the model finds
// candidates with `spotlight_search`, then reads the bodies it needs with `fetch_note`, and
// grounds its answer in the actual text. Both tools run behind the SAME third-party model.
struct FetchNoteTool: Tool {
    let name = "fetch_note"
    let description = "Read the full saved text of a trail note by its identifier (e.g. note-002)."

    @Generable
    struct Arguments {
        @Guide(description: "The note identifier returned by spotlight_search, like note-002.")
        var id: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [fetch_note] \(arguments.id)")
        guard let note = notesByID[arguments.id] else {
            return "No note found with id \(arguments.id)."
        }
        return "\(note.title): \(note.text)"
    }
}

// MARK: - Search-result stream (live, independent of the model's prose)

func describe(_ reply: SpotlightSearchTool.SearchReply) -> String {
    let label = reply.label.map { " label=\($0)" } ?? ""
    let status = reply.status == .partial ? " (partial)" : ""
    switch reply.content {
    case .items(let items):
        let titles = items.map { $0.attributeSet.title ?? $0.uniqueIdentifier }
        return "items[\(items.count)]\(label)\(status): \(titles.joined(separator: " | "))"
    case .scoredItems(let scored):
        let titles = scored.map { "\($0.item.attributeSet.title ?? $0.item.uniqueIdentifier)" }
        return "scoredItems[\(scored.count)]\(label)\(status): \(titles.joined(separator: " | "))"
    case .groupedItems(let groups):
        return "groupedItems[\(groups.count) groups]\(label)\(status)"
    case .count(let count):
        return "count = \(count.value)\(label)\(status)"
    case .table(let table):
        return "table \(table.columns.map(\.name)) × \(table.rows.count) rows\(label)\(status)"
    case .statistic(let statistic):
        return "statistic \(statistic.name) = \(statistic.value)\(label)\(status)"
    case .text(let text):
        return "text\(label)\(status): \(text.body.prefix(120))"
    @unknown default:
        return "unknown reply\(label)\(status)"
    }
}

// MARK: - Main

@main
enum SpotlightChat {
    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)  // keep prints ordered with stderr in pipes
        var arguments = Array(CommandLine.arguments.dropFirst())
        let useSystem = arguments.contains("--system")
        arguments.removeAll { $0 == "--system" }
        var bundlePath: String?
        if let flag = arguments.firstIndex(of: "--model") {
            guard flag + 1 < arguments.count else {
                FileHandle.standardError.write(
                    Data("usage: --model requires a bundle directory path\n".utf8))
                exit(2)
            }
            bundlePath = arguments[flag + 1]
            arguments.removeSubrange(flag...(flag + 1))
        }
        var question = "What did I write about the night hike?"
        if let flag = arguments.firstIndex(of: "--ask") {
            guard flag + 1 < arguments.count else {
                FileHandle.standardError.write(
                    Data("usage: --ask requires a question string\n".utf8))
                exit(2)
            }
            question = arguments[flag + 1]
            arguments.removeSubrange(flag...(flag + 1))
        }

        guard Bundle.main.bundleIdentifier != nil else {
            fatalError("No bundle identity — the embedded Info.plist did not load.")
        }
        try await indexNotes()
        try await waitUntilSearchable()

        // The tool searches this app's Core Spotlight index. Guidance defaults to .complete,
        // whose instructions are ~13 k tokens — beyond a 4 k-context model. `.focused` +
        // `.compact` keeps the tool's prompt within budget for small local models.
        let tool = SpotlightSearchTool(
            configuration: .init(
                sources: [.coreSpotlight(CoreSpotlightSource(searchableIndexDelegate: NotesIndexDelegate()))],
                guide: SpotlightSearchTool.Guide(level: .focused(.items), format: .compact)))

        // Watch the retriever's results live, separate from the model's answer.
        let observer = Task {
            for await reply in tool.searchResults { print("  [spotlight] \(describe(reply))") }
        }
        defer { observer.cancel() }

        // Two tools, one model: spotlight_search finds notes, fetch_note reads their bodies.
        let tools: [any Tool] = [tool, FetchNoteTool()]
        // `/no_think` disables qwen3's chain-of-thought: with this rich tool schema, thinking
        // turns can run to the token cap (the framework then reports "ended without producing
        // a response"). Disabling it makes the search→fetch chain fast and reliable; harmless
        // to non-qwen models that ignore the directive.
        let instructions = """
            Answer questions about the user's trail notes. First call spotlight_search to find \
            relevant notes, then call fetch_note with a note id to read its full text, then \
            answer briefly quoting only what you read. /no_think
            """

        print("\n> \(question)\n")
        // Build the session per backend, then run the shared respond/print tail once.
        let session: LanguageModelSession
        if useSystem {
            guard case .available = SystemLanguageModel.default.availability else {
                print("SystemLanguageModel unavailable: \(SystemLanguageModel.default.availability)")
                return
            }
            session = LanguageModelSession(tools: tools, instructions: instructions)
        } else {
            let model: KitLanguageModel
            if let bundlePath {
                // Zoo decode-only S=1 bundles need token-by-token prefill chunking; set
                // before the engine is created.
                setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
                model = try await KitLanguageModel(bundleAt: URL(fileURLWithPath: bundlePath))
            } else {
                // qwen3-4B speaks ChatML (so the kit declares .toolCalling) and is reliable
                // at the search→fetch chain. The 0.6B is too small for this tool's rich
                // schema. For the flagship zoo model, pass --model <qwen3.5-0.8B bundle> — a
                // qwen3.5 hybrid needs a coreai-models engine with hybrid KV support; the
                // kit's stock pipelined engine rejects it (see README).
                print("Fetching qwen3 4B (cached afterwards)…")
                model = try await KitLanguageModel(model: .qwen3_4B) { progress in
                    print("  \(Int(progress.fraction * 100))% \(progress.currentFile)")
                }
            }
            session = LanguageModelSession(model: model, tools: tools, instructions: instructions)
        }
        let response = try await session.respond(to: question)
        printTranscript(session.transcript)
        print("\n[answer] \(response.content)")
    }

    static func printTranscript(_ transcript: Transcript) {
        print("\n[transcript]")
        for entry in transcript {
            switch entry {
            case .instructions: print("  instructions")
            case .prompt: print("  prompt")
            case .toolCalls(let calls):
                for call in calls {
                    print("  toolCall \(call.toolName)(\(call.arguments.jsonString.prefix(200)))")
                }
            case .toolOutput(let output):
                let text = output.segments.compactMap { segment -> String? in
                    if case .text(let text) = segment { return text.content }
                    return nil
                }.joined()
                print("  toolOutput \(text.prefix(400))")
            case .response: print("  response")
            default: print("  (other)")
            }
        }
    }
}
