// Notes.swift — the app's own content store, indexed into Core Spotlight. Stands in for a
// real notes app's data. The bodies live here (`notesByID`); the Spotlight index only holds
// lightweight metadata, so `fetch_note` reads the full text from this store (see Tools.swift).
//
// Ported verbatim from Examples/SpotlightChat (the verified CLI): same five trail notes, same
// indexing + delegate. The only difference is an app target gets its bundle identity for free,
// so no embedded Info.plist hack is needed for `CSSearchableIndex.default()`.

import CoreSpotlight
import Foundation

struct TrailNote: Identifiable, Sendable, Hashable {
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

private let domainIdentifier = "trail-notes"

func makeItem(_ note: TrailNote) -> CSSearchableItem {
    let attributes = CSSearchableItemAttributeSet(contentType: .text)
    attributes.title = note.title
    attributes.contentDescription = note.text
    attributes.textContent = note.text
    attributes.keywords = note.keywords
    attributes.contentCreationDate = Calendar.current.date(
        byAdding: .day, value: -note.daysAgo, to: Date())
    return CSSearchableItem(
        uniqueIdentifier: note.id, domainIdentifier: domainIdentifier, attributeSet: attributes)
}

/// Re-indexes the sample notes into the app's Core Spotlight index, then waits until they are
/// actually searchable so the first model turn doesn't race an empty index.
func indexNotes() async throws {
    let index = CSSearchableIndex.default()
    try await index.deleteSearchableItems(withDomainIdentifiers: [domainIdentifier])
    try await index.indexSearchableItems(sampleNotes.map(makeItem))
    try await waitUntilSearchable()
}

private func waitUntilSearchable() async throws {
    for _ in 1...20 {
        let context = CSSearchQueryContext()
        context.fetchAttributes = ["title"]
        let query = CSSearchQuery(queryString: "keywords == \"trail\"cd", queryContext: context)
        var found = 0
        for try await _ in query.results { found += 1 }
        if found >= sampleNotes.count { return }
        try await Task.sleep(for: .milliseconds(500))
    }
    // Best effort: continue even if the index lagged — the model just sees fewer results.
}

/// The index can ask the app to re-supply items (e.g. after an index loss). Conforming the
/// source to a delegate keeps the index recoverable; the search path returns metadata only,
/// so full text is supplied here and by `fetch_note`.
final class NotesIndexDelegate: NSObject, CSSearchableIndexDelegate, @unchecked Sendable {
    func searchableIndex(
        _ index: CSSearchableIndex,
        reindexAllSearchableItemsWithAcknowledgementHandler ack: @escaping () -> Void
    ) {
        ack()
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
