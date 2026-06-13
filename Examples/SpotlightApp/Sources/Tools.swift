// Tools.swift — the companion hydration tool and the helpers that turn the retriever's live
// output into UI-friendly values.
//
// `SpotlightSearchTool` (Apple's WWDC26 tool) hands the model only index METADATA — title,
// ids, dates — never the note body (verified: even with `fetchAttributes: [.contentDescription]`
// the body does not reach the model; a raw `CSSearchQuery` returns it fine). That mirrors a real
// app: the Spotlight index is a finding aid, the full content lives in the app's own store. So a
// model "answering from search" is really answering from titles, and small models then make the
// rest up. `FetchNoteTool` closes the loop: the model finds candidates with `spotlight_search`,
// reads the bodies it needs with `fetch_note`, and grounds its answer in the real text. Both
// tools run behind the SAME third-party model (the zoo bundle via KitLanguageModel).

import CoreSpotlight
import Foundation
import FoundationModels

/// A note surfaced by the Spotlight retriever (what the model sees: identity, not body).
struct FoundNote: Identifiable, Hashable, Sendable {
    let id: String
    let title: String
}

/// Reads a note's full saved text from the app's store by identifier. The injected reporter
/// fires on each call so the UI can show which notes were actually read (the citations).
struct FetchNoteTool: Tool {
    let name = "fetch_note"
    let description = "Read the full saved text of a trail note by its identifier (e.g. note-002)."

    /// Called with the note id the moment the model reads it. `@Sendable` because the framework
    /// invokes `call` off the main actor.
    let onFetch: @Sendable (String) -> Void

    @Generable
    struct Arguments {
        @Guide(description: "The note identifier returned by spotlight_search, like note-002.")
        var id: String
    }

    func call(arguments: Arguments) async throws -> String {
        onFetch(arguments.id)
        guard let note = notesByID[arguments.id] else {
            return "No note found with id \(arguments.id)."
        }
        return "\(note.title): \(note.text)"
    }
}

/// Maps a live `SearchReply` from `tool.searchResults` to the notes it surfaced (title + id),
/// or nil for reply kinds that carry no items.
func foundNotes(from reply: SpotlightSearchTool.SearchReply) -> [FoundNote]? {
    func note(_ item: CSSearchableItem) -> FoundNote {
        FoundNote(id: item.uniqueIdentifier, title: item.attributeSet.title ?? item.uniqueIdentifier)
    }
    switch reply.content {
    case .items(let items):
        return items.map(note)
    case .scoredItems(let scored):
        return scored.map { note($0.item) }
    default:
        return nil
    }
}

/// Recovers the model's own Spotlight queries from a finished transcript — the argument strings
/// it passed to `spotlight_search`. Used for the "How it answered" trace; the live found-notes
/// stream and the fetch reporter cover everything else.
func searchQueries(in transcript: Transcript) -> [String] {
    var queries: [String] = []
    for entry in transcript {
        guard case .toolCalls(let calls) = entry else { continue }
        for call in calls where call.toolName == "spotlight_search" {
            let json = call.arguments.jsonString
            guard
                let data = json.data(using: .utf8),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            // The query schema varies by guidance level; collect any string / [string] values.
            for value in object.values {
                if let string = value as? String {
                    queries.append(string)
                } else if let array = value as? [Any] {
                    queries.append(contentsOf: array.compactMap { $0 as? String })
                }
            }
        }
    }
    return queries
}
