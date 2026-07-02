// vlchat-cli — the argument shell over `ask(_:aboutImageAt:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the card's snippet shows, print the answer. Progress goes
// to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run vlchat-cli --image photo.jpg --prompt "What's here?" --model qwen3-vl-2b

import CoreAIKit
import Foundation

let usage = """
    usage: vlchat-cli --image <file> --prompt <text> [--model <catalog-id>]
           vlchat-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var promptText: String?
var imagePath: String?
var modelID = "qwen3-vl-2b"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--prompt": promptText = args.popFirst()
    case "--image": imagePath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.vlm) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let promptText, let imagePath else { fail(usage) }

// Immutable copies: the progress closure is @Sendable and can't read main-actor top-level vars.
let id = modelID
let imageURL = URL(fileURLWithPath: imagePath)

do {
    let answer = try await ask(
        promptText, aboutImageAt: imageURL, model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    print(answer)
} catch {
    fail("error: \(error.localizedDescription)")
}
