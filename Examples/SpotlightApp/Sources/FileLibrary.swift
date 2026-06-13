// FileLibrary.swift — the corpus the app searches and grounds answers in: the user's OWN real
// files. This replaces the five hardcoded sample notes. The user points the app at a folder (or a
// set of files) with a picker; SpotlightSearchTool's `.files` source searches them in place
// (`FileSource.scopes`), and `fetch_file` (Tools.swift) reads the full text of whatever the model
// chose to read so the answer is grounded in the real document.
//
// We can't read other apps' data (sandbox, by design) — only files the user explicitly grants via
// the picker (security-scoped). To avoid an empty first screen, a few sample documents are written
// to the app's own Application Support directory on first launch and used as the default scope.

import CoreSpotlight
import Foundation
import PDFKit
import UniformTypeIdentifiers

/// One file in the active library — the unit the UI shows and that `fetch_file` reads. Identity is
/// the file's path, which is also what the `.files` search returns as a result identifier.
struct LibraryFile: Identifiable, Hashable, Sendable {
    let url: URL
    var id: String { url.path }
    var displayName: String { url.lastPathComponent }
    var ext: String { url.pathExtension.lowercased() }
}

/// An immutable snapshot of the active library: the scopes handed to `FileSource`, the resolved
/// file list (for the count + the resolver), and lookup maps so `fetch_file` can map whatever
/// identifier the model echoes back (path, file URL, or bare filename) to the file on disk.
/// `Sendable` so the off-actor tool calls can capture it directly.
struct LibrarySnapshot: Sendable {
    /// What the user chose — a folder and/or individual files. Passed straight to `FileSource`.
    let scopes: [URL]
    /// A short human label for the source ("Sample Documents", "Documents", "3 files").
    let displayName: String
    /// Every readable file under the scopes (folders walked recursively, capped), for the count
    /// and the citation list.
    let files: [LibraryFile]
    /// path / filename → url, for the lenient `fetch_file` resolver.
    private let byPath: [String: URL]
    private let byName: [String: URL]

    init(scopes: [URL], displayName: String, files: [LibraryFile]) {
        self.scopes = scopes
        self.displayName = displayName
        self.files = files
        self.byPath = Dictionary(files.map { ($0.url.path, $0.url) }, uniquingKeysWith: { a, _ in a })
        self.byName = Dictionary(
            files.map { ($0.displayName.lowercased(), $0.url) }, uniquingKeysWith: { a, _ in a })
    }

    static let empty = LibrarySnapshot(scopes: [], displayName: "No files", files: [])

    /// Map a result identifier from `spotlight_search` (or an id the model passed to `fetch_file`)
    /// to a file on disk. The `.files` source returns file paths, but be lenient: accept a
    /// `file://` URL, a percent-encoded path, an absolute path, a path suffix, or a bare filename.
    func file(forIdentifier raw: String) -> LibraryFile? {
        var key = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if key.hasPrefix("file://"), let url = URL(string: key) { key = url.path }
        key = key.removingPercentEncoding ?? key

        if let url = byPath[key] { return LibraryFile(url: url) }
        if let url = byName[key.lowercased()] { return LibraryFile(url: url) }
        // Path suffix (the model sometimes quotes a shortened path) or filename without directory.
        let tail = (key as NSString).lastPathComponent.lowercased()
        if let url = byName[tail] { return LibraryFile(url: url) }
        if let match = files.first(where: { $0.url.path.hasSuffix(key) }) { return match }
        return nil
    }
}

/// Builds library snapshots, reads file text, and seeds the sample documents. Stateless helpers
/// (everything it needs is passed in or read from disk), so it is safe to call from any actor.
enum FileLibrary {
    /// File kinds we will index/read. The picker is filtered to these and the folder walk keeps
    /// only these, so we never try to "read" an image or a binary as text.
    static let readableTypes: [UTType] = [.plainText, .text, .pdf, .rtf, .html, .sourceCode, .json, .xml, .commaSeparatedText]

    private static let readableExtensions: Set<String> = [
        "txt", "text", "md", "markdown", "rtf", "rtfd", "html", "htm", "pdf", "csv", "tsv", "log",
        "json", "xml", "yaml", "yml", "swift", "py", "js", "ts", "c", "h", "cpp", "hpp", "m", "mm",
        "java", "kt", "go", "rs", "rb", "sh", "tex", "org",
    ]

    private static let maxFiles = 300
    private static let maxBytesForText = 8 * 1024 * 1024  // skip files too large to read as text
    static let maxCharsPerFetch = 6000  // keep one fetched body inside the model's context window
    private static let maxCharsForIndex = 200_000  // full-text indexed per file (recall, not model)

    /// Our own Core Spotlight domain for the user's files (separate from anything else the app or
    /// system indexes).
    static let domainIdentifier = "user-files"
    /// A constant keyword stamped on every indexed item, used as a reliable "is it searchable yet?"
    /// probe (the same pattern the verified notes example used).
    static let indexTag = "spotlightapp-userfile"

    // MARK: - Snapshot building

    /// Resolve a set of picked roots (folders and/or files) into a searchable snapshot. Folders are
    /// walked recursively (capped); files are taken as-is. `displayName` is derived for the UI.
    static func snapshot(forRoots roots: [URL]) -> LibrarySnapshot {
        var files: [LibraryFile] = []
        var folderCount = 0
        for root in roots {
            if isDirectory(root) {
                folderCount += 1
                files.append(contentsOf: walk(root))
            } else if isReadable(root) {
                files.append(LibraryFile(url: root))
            }
            if files.count >= maxFiles { break }
        }
        files = dedupedSortedCapped(files)

        let label: String
        if roots.count == 1, isDirectory(roots[0]) {
            label = roots[0].lastPathComponent
        } else if folderCount == 0 {
            label = files.count == 1 ? files[0].displayName : "\(files.count) files"
        } else {
            label = roots.count == 1 ? roots[0].lastPathComponent : "\(roots.count) locations"
        }
        return LibrarySnapshot(scopes: roots, displayName: label, files: files)
    }

    private static func walk(_ folder: URL) -> [LibraryFile] {
        var out: [LibraryFile] = []
        let keys: [URLResourceKey] = [.isRegularFileKey, .isDirectoryKey]
        guard
            let walker = FileManager.default.enumerator(
                at: folder, includingPropertiesForKeys: keys,
                options: [.skipsHiddenFiles, .skipsPackageDescendants])
        else { return out }
        for case let url as URL in walker {
            if out.count >= maxFiles { break }
            guard isReadable(url) else { continue }
            out.append(LibraryFile(url: url))
        }
        return out
    }

    private static func dedupedSortedCapped(_ files: [LibraryFile]) -> [LibraryFile] {
        var seen = Set<String>()
        let unique = files.filter { seen.insert($0.url.path).inserted }
        return Array(unique.sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
            .prefix(maxFiles))
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private static func isReadable(_ url: URL) -> Bool {
        guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else {
            return false
        }
        return readableExtensions.contains(url.pathExtension.lowercased())
    }

    // MARK: - Text extraction (what fetch_file hands the model)

    /// Read a file's text. PDFs go through PDFKit, rich text through `NSAttributedString`,
    /// everything else is decoded as text. Truncated to `maxChars` (default = one context-window's
    /// worth, so a long document can't overflow the model). Returns nil if nothing readable came
    /// out. Indexing passes a larger cap for better full-text recall.
    static func text(of url: URL, maxChars: Int = maxCharsPerFetch) -> String? {
        let started = url.startAccessingSecurityScopedResource()
        defer { if started { url.stopAccessingSecurityScopedResource() } }

        let raw: String?
        switch url.pathExtension.lowercased() {
        case "pdf":
            raw = PDFDocument(url: url)?.string
        case "rtf", "rtfd", "html", "htm", "doc":
            raw = attributedText(of: url)
        default:
            raw = decodedText(of: url)
        }
        guard let text = raw?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return nil
        }
        if text.count <= maxChars { return text }
        return String(text.prefix(maxChars)) + "\n…(truncated)"
    }

    private static func decodedText(of url: URL) -> String? {
        guard
            let data = try? Data(contentsOf: url),
            data.count <= maxBytesForText
        else { return nil }
        for encoding in [String.Encoding.utf8, .utf16, .isoLatin1] {
            if let s = String(data: data, encoding: encoding) { return s }
        }
        return nil
    }

    private static func attributedText(of url: URL) -> String? {
        try? NSAttributedString(
            url: url, options: [:], documentAttributes: nil
        ).string
    }

    // MARK: - Core Spotlight indexing of the user's real files

    // We index the files the user picked into the app's OWN Core Spotlight index and search them
    // with the tool's `.coreSpotlight` source. (The `.coreSpotlight` round trip behind our
    // third-party model is the verified path — see SPOT_RAG.) The `.files` source — meant to search
    // file URLs in place — does not honor its `FileSource.scopes`/`maximumResultCount` in the 27.0
    // beta (it returns unrelated files from a broad default scope), so we drive retrieval ourselves
    // and keep full control of the corpus: only the user's chosen files, nothing else.

    /// Build a Core Spotlight item for a file. The identifier is the file path (what `fetch_file`
    /// resolves; the UI opens the file from the library snapshot, not from the item). We deliberately
    /// do NOT set `contentURL`/`path`: with those, `SpotlightSearchTool` treats the entry as a
    /// system file reference and does not surface our app-indexed items (verified — the tool
    /// returned 0 while a raw CSUserQuery found them). Modeling them as plain text items (title +
    /// description + textContent), like the verified notes example, makes the tool surface them.
    static func makeSearchableItem(for file: LibraryFile) -> CSSearchableItem {
        let body = text(of: file.url, maxChars: maxCharsForIndex)
        let attributes = CSSearchableItemAttributeSet(contentType: .text)
        attributes.title = file.displayName
        attributes.displayName = file.displayName
        attributes.keywords = [indexTag]
        attributes.textContent = body
        attributes.contentDescription = body.map { String($0.prefix(240)) }
        attributes.contentCreationDate =
            (try? file.url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate
        return CSSearchableItem(
            uniqueIdentifier: file.url.path, domainIdentifier: domainIdentifier,
            attributeSet: attributes)
    }

    /// Make the given files the entire searchable corpus, then wait until they are actually
    /// searchable so the first question doesn't race an empty index. We wipe the app's whole index
    /// first (not just our domain): `.coreSpotlight` searches every item this bundle has indexed, so
    /// any leftovers — e.g. an earlier build's notes under the same bundle id — would otherwise be
    /// returned alongside the user's real files. A clean slate guarantees the model only ever sees
    /// the current selection.
    static func index(_ files: [LibraryFile]) async throws {
        let index = CSSearchableIndex.default()
        try await index.deleteAllSearchableItems()
        guard !files.isEmpty else { return }
        try await index.indexSearchableItems(files.map(makeSearchableItem))
        await waitUntilSearchable(expected: files.count)
    }

    private static func waitUntilSearchable(expected: Int) async {
        for _ in 1...30 {
            let visible = (try? await searchableCount()) ?? 0
            if visible >= expected { return }
            try? await Task.sleep(for: .milliseconds(400))
        }
        // Best effort: proceed even if the index lagged — the model just sees fewer results.
    }

    /// How many of our freshly-indexed items are query-visible right now (literal index, probe by
    /// the common tag).
    static func searchableCount() async throws -> Int {
        let context = CSSearchQueryContext()
        context.fetchAttributes = ["title"]
        let query = CSSearchQuery(
            queryString: "keywords == \"\(indexTag)\"cd", queryContext: context)
        var found = 0
        for try await _ in query.results { found += 1 }
        return found
    }

    /// How many items a SEMANTIC user query returns right now — this is the index path
    /// `SpotlightSearchTool` actually searches, which lags the literal index after fresh indexing.
    static func userQueryCount(_ queryString: String) async throws -> Int {
        let context = CSUserQueryContext()
        context.maxResultCount = 50
        context.fetchAttributes = ["title"]
        let query = CSUserQuery(userQueryString: queryString, userQueryContext: context)
        var found = 0
        for try await _ in query.results { found += 1 }
        return found
    }

    // MARK: - Sample documents (so the first launch is never empty)

    /// The folder used as the default scope on a fresh install. Lives in the app's own Application
    /// Support — readable without a picker — and is seeded once with the sample documents below.
    static func sampleFolderURL() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("SpotlightApp/Sample Documents", isDirectory: true)
    }

    /// Write the sample documents once (idempotent: skips files that already exist, so the user can
    /// edit or delete them). Returns the sample folder URL.
    @discardableResult
    static func seedSampleDocumentsIfNeeded() -> URL {
        let folder = sampleFolderURL()
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for (name, body) in sampleDocuments {
            let file = folder.appendingPathComponent(name)
            if !FileManager.default.fileExists(atPath: file.path) {
                try? body.data(using: .utf8)?.write(to: file)
            }
        }
        return folder
    }

    /// Five short trail journals, as real Markdown files. Filenames carry the keyword the matching
    /// sample question uses, so retrieval is a clean title/content hit on a small local model.
    private static let sampleDocuments: [(name: String, body: String)] = [
        ("Eagle Ridge loop.md", """
        # Eagle Ridge loop

        Clear morning, a 14 km loop with about 900 m of climb. The summit wind was brutal — I had to
        layer up at the saddle — but the view over the valley was worth every step.

        The highlight: two golden eagles riding the updraft right along the ridge line, close enough
        to see the wing fingers. Trail was dry the whole way; no water after the second creek, so
        carry enough from the start.
        """),
        ("Silver Falls waterfall hike.md", """
        # Silver Falls out-and-back

        A short hike to the waterfall, maybe 5 km round trip. The silver falls were at full flow
        after three days of rain, and the spray soaked the lower viewpoint completely.

        The boardwalk near the base was slippery — bring grippy shoes next time, the smooth planks
        are treacherous when wet. Good half-day outing when the bigger trails are washed out.
        """),
        ("Pine Hollow night hike.md", """
        # Pine Hollow night hike

        First night hike of the season. Headlamp died about halfway in — note to self: pack spare
        batteries, and test the lamp before leaving, not at the trailhead.

        Owls were loud near the hollow, at least three calling back and forth. Cold but dead calm.
        Navigated the last stretch by moonlight, which was slow but unforgettable.
        """),
        ("Granite Pass attempt.md", """
        # Granite Pass attempt

        Turned around about 2 km below the pass — a late-season snowfield was too steep and hard to
        cross safely without an ice axe. Better to bail than to slip here.

        The marmots near the boulder field were completely fearless, posing on the rocks for photos.
        Worth another try in July once the snow has gone; pack the axe and crampons regardless.
        """),
        ("River bend family walk.md", """
        # River bend picnic walk

        Easy flat walk along the water with the family, about 8 km total. Herons fishing patiently at
        the river bend; we counted four before lunch.

        The new boots felt great — no blisters even after the full distance, which is a first. Packed
        a picnic and made an afternoon of it. A keeper for slow days.
        """),
    ]
}

/// Lets Core Spotlight re-request items if its index is lost. Each identifier is a file path, so an
/// item can be rebuilt straight from disk. Wired via `CoreSpotlightSource(searchableIndexDelegate:)`.
final class FileIndexDelegate: NSObject, CSSearchableIndexDelegate, @unchecked Sendable {
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
        let items = identifiers.compactMap { path -> CSSearchableItem? in
            FileManager.default.fileExists(atPath: path)
                ? FileLibrary.makeSearchableItem(for: LibraryFile(url: URL(fileURLWithPath: path)))
                : nil
        }
        handler(items)
    }
}
