// transcribe-cli — the argument shell over `transcribe(audio:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the GUI app calls, print the transcript. Progress and the
// detected language go to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run transcribe-cli --audio sample.wav --model whisper-large-v3-turbo

import CoreAIKit
import Foundation

let usage = """
    usage: transcribe-cli --audio <file> [--model <catalog-id>] [--language <code>]
           transcribe-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var audioPath: String?
var modelID = "whisper-large-v3-turbo"
var language: String?

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--audio": audioPath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--language": language = args.popFirst()
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.asr) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let audioPath else { fail(usage) }

// Immutable copies: the progress closure is @Sendable and can't read main-actor top-level vars.
let id = modelID

do {
    let result = try await transcribe(
        audio: URL(fileURLWithPath: audioPath), model: id, language: language,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    if !result.language.isEmpty {
        stderrPrint("detected language: \(result.language)")
    }
    print(result.text)
} catch {
    fail("error: \(error.localizedDescription)")
}
