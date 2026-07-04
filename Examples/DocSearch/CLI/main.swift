// docsearch-cli — the argument shell over `search(query:pages:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the GUI app's gesture rides, print the ranking. Progress
// goes to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run docsearch-cli --query "monthly revenue trend"
//   swift run docsearch-cli --query "wifi password" --pages ~/Scans

import CoreAIKitEmbeddings
import Foundation

let usage = """
    usage: docsearch-cli --query <text> [--pages <dir>] [--model <catalog-id>]
           docsearch-cli --list-models
    (default --pages: the bundled sample pages in Sources/Pages)
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var query: String?
var pagesPath: String?
var modelID = "colmodernvbert"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--query": query = args.popFirst()
    case "--pages": pagesPath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.retrieval) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let query else { fail(usage) }

// Default corpus: the sample pages bundled with this example (resolved relative to this
// source file, so the command works from any working directory).
let pagesDir = pagesPath.map { URL(fileURLWithPath: $0) }
    ?? URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // CLI
        .deletingLastPathComponent()  // DocSearch
        .appendingPathComponent("Sources/Pages")

let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "heic"]
let pages = ((try? FileManager.default.contentsOfDirectory(
    at: pagesDir, includingPropertiesForKeys: nil)) ?? [])
    .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }
guard !pages.isEmpty else { fail("no page images found in \(pagesDir.path)") }

// Immutable copy: the progress closure is @Sendable and can't read main-actor top-level vars.
let id = modelID

do {
    let ranked = try await search(
        query: query, pages: pages, model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    for (rank, hit) in ranked.enumerated() {
        print(String(format: "%2d. %@  score %.3f", rank + 1, hit.page.lastPathComponent, hit.score))
    }
} catch {
    fail("error: \(error.localizedDescription)")
}
