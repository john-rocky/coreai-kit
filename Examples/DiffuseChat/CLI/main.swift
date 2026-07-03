// diffuse-cli — the argument shell over `denoise(_:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the card's snippet shows, print the reply. The live
// canvas streams to stderr (--quiet suppresses it) so stdout stays machine-checkable
// (agents: assert on stdout).
//
//   swift run diffuse-cli --prompt "What is the capital of France?"

import CoreAIKit
import Foundation

let usage = """
    usage: diffuse-cli --prompt <text> [--model <catalog-id>] [--quiet]
           diffuse-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var promptText: String?
var modelID = "llada-8b"
var quiet = false

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--prompt": promptText = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--quiet": quiet = true
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.dllm) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let promptText else { fail(usage) }

let id = modelID
let showSteps = !quiet
do {
    let reply = try await denoise(
        promptText, model: id,
        onStep: { step in
            guard showSteps else { return }
            let canvas = step.text.replacingOccurrences(of: "\n", with: " ")
            stderrPrint("\u{1B}[2K\r[\(step.forwards)] \(canvas.prefix(120))", terminator: "")
        },
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    if showSteps { stderrPrint("") }
    print(reply)
} catch {
    fail("error: \(error.localizedDescription)")
}
