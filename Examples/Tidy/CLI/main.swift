// tidy-cli — the argument shell over `tidy(transcript:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the GUI app calls, print the rewrite. Progress goes to
// stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run tidy-cli --model s1-mini --text "so um i need to like send the report by friday"
//   swift run tidy-cli --file meeting.txt --styling formal --structure lists

import CoreAIKit
import Foundation

let usage = """
    usage: tidy-cli (--text <transcript> | --file <path>) [--model <catalog-id>]
                    [--styling casual|semi-casual|semi-formal|formal]
                    [--structure prose|lists] [--context general|email]
           tidy-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var transcript: String?
var modelID = "s1-mini"
var styling = TranscriptStyling.semiFormal
var structure = TranscriptStructure.prose
var context = TranscriptContext.general

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--text": transcript = args.popFirst()
    case "--file":
        guard let path = args.popFirst() else { fail(usage) }
        transcript = try String(contentsOf: URL(fileURLWithPath: path), encoding: .utf8)
    case "--model": modelID = args.popFirst() ?? modelID
    case "--styling":
        guard let raw = args.popFirst(), let value = TranscriptStyling(rawValue: raw) else {
            fail("--styling: \(TranscriptStyling.allCases.map(\.rawValue).joined(separator: " | "))")
        }
        styling = value
    case "--structure":
        guard let raw = args.popFirst(), let value = TranscriptStructure(rawValue: raw) else {
            fail("--structure: \(TranscriptStructure.allCases.map(\.rawValue).joined(separator: " | "))")
        }
        structure = value
    case "--context":
        guard let raw = args.popFirst(), let value = TranscriptContext(rawValue: raw) else {
            fail("--context: \(TranscriptContext.allCases.map(\.rawValue).joined(separator: " | "))")
        }
        context = value
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.textNormalizer) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let transcript else { fail(usage) }

// Immutable copy: the progress closure is @Sendable and can't read main-actor top-level vars.
let id = modelID

do {
    let clean = try await tidy(
        transcript: transcript, model: id, styling: styling, structure: structure,
        context: context,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    // Filler-only input normalizes to nothing. Say so on stderr rather than printing a blank
    // line that reads like a failure; stdout stays the empty string it should be.
    if clean.isEmpty { stderrPrint("(empty — the input was nothing but filler)") }
    print(clean)
} catch {
    fail("error: \(error.localizedDescription)")
}
