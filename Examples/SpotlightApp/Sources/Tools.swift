// Tools.swift — the companion hydration tool and the helpers that turn the retriever's live output
// into UI-friendly values.
//
// `SpotlightSearchTool` (Apple's WWDC26 tool) hands the model only index METADATA — name, path,
// type, dates — never the file's text (verified for the Core Spotlight source; the `.files` source
// behaves the same way). That mirrors how search really works: the index is a finding aid, the
// content lives in the file. A model "answering from search" is answering from filenames, and a
// small model then makes the body up. `FetchFileTool` closes the loop: the model finds candidate
// files with `spotlight_search`, reads the ones it needs with `fetch_file`, and grounds its answer
// in the real text. Both tools run behind the SAME third-party model (a zoo bundle via
// `KitLanguageModel`).

import CoreSpotlight
import Foundation
import FoundationModels

/// A file surfaced by the Spotlight retriever (what the model sees: identity, not the text). `url`
/// is recovered for the UI so a citation chip can open the real file.
struct FoundFile: Identifiable, Hashable, Sendable {
    let id: String  // the result's uniqueIdentifier (a path, for the `.files` source)
    let name: String
    let url: URL?
}

/// Reads a file's full text from disk by the identifier `spotlight_search` returned. The injected
/// reporter fires with the resolved file the moment the model reads it, so the UI can show which
/// files were actually used (the citations).
struct FetchFileTool: Tool {
    let name = "fetch_file"
    let description =
        "Read the full text of one of the user's files, by the identifier or path that "
        + "spotlight_search returned. Call this before answering so you quote the real content."

    /// The active library (Sendable) — resolves an identifier to a file on disk and reads its text.
    let library: LibrarySnapshot
    /// Fired with the resolved file when it is read. `@Sendable`: the framework calls tools off the
    /// main actor.
    let onFetch: @Sendable (LibraryFile) -> Void

    @Generable
    struct Arguments {
        @Guide(description: "The file identifier or path from spotlight_search, e.g. the file name.")
        var id: String
    }

    func call(arguments: Arguments) async throws -> String {
        guard let file = library.file(forIdentifier: arguments.id) else {
            return "No file found for identifier \(arguments.id)."
        }
        onFetch(file)
        guard let text = FileLibrary.text(of: file.url) else {
            return "\(file.displayName): (could not read this file's text)."
        }
        return "\(file.displayName):\n\(text)"
    }
}

/// Maps a live `SearchReply` from `tool.searchResults` to the files it surfaced, or nil for reply
/// kinds that carry no items. The file URL is recovered from the item's `contentURL`/`path`
/// attributes so a citation chip can open it.
func foundFiles(from reply: SpotlightSearchTool.SearchReply) -> [FoundFile]? {
    func found(_ item: CSSearchableItem) -> FoundFile {
        let attrs = item.attributeSet
        let url = attrs.contentURL ?? attrs.path.map { URL(fileURLWithPath: $0) }
        let name =
            attrs.displayName ?? attrs.title ?? url?.lastPathComponent ?? item.uniqueIdentifier
        return FoundFile(id: item.uniqueIdentifier, name: name, url: url)
    }
    switch reply.content {
    case .items(let items):
        return items.map(found)
    case .scoredItems(let scored):
        return scored.map { found($0.item) }
    default:
        return nil
    }
}

/// Recovers the model's own Spotlight queries from a finished transcript — the argument strings it
/// passed to `spotlight_search`. Used for the "How it answered" trace; the live found-files stream
/// and the fetch reporter cover the rest.
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
