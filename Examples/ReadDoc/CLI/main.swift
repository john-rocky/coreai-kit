// readdoc-cli — the argument shell over `readDocument(at:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the card's snippet shows, print the markdown. Progress
// goes to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run readdoc-cli --image sample.png

import CoreAIKit
import Foundation

let usage = """
    usage: readdoc-cli --image <file> [--model <catalog-id>]
           readdoc-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var imagePath: String?
var modelID = "unlimited-ocr"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--image": imagePath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.ocr) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let imagePath else { fail(usage) }

let id = modelID
do {
    let markdown = try await readDocument(
        at: URL(fileURLWithPath: imagePath), model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    print(markdown)
} catch {
    fail("error: \(error.localizedDescription)")
}
