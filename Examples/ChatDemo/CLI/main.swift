// chat-cli — the argument shell over `ask(_:model:)` (Sources/QuickStart.swift): parse flags,
// call the same function the card's snippet shows, print the reply. Progress goes to stderr so
// stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run chat-cli --prompt "What can you do, offline?" --model qwen3.5-2b

import CoreAIKit
import Foundation

let usage = """
    usage: chat-cli --prompt <text> [--model <catalog-id>]
           chat-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var promptText: String?
var modelID = "qwen3.5-2b"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--prompt": promptText = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.chat) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let promptText else { fail(usage) }

// Immutable copies: the progress closure is @Sendable and can't read main-actor top-level vars.
let id = modelID

do {
    let reply = try await ask(
        promptText, model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    print(reply)
} catch {
    fail("error: \(error.localizedDescription)")
}
