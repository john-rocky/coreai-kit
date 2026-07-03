// audiochat-cli — the argument shell over `askAboutAudio(_:audioAt:model:)`
// (Sources/QuickStart.swift): parse flags, call the same function the card's snippet shows,
// print the answer. Progress goes to stderr so stdout stays machine-checkable.
//
//   swift run audiochat-cli --audio sample.wav --prompt "What do you hear?"

import CoreAIKit
import Foundation

let usage = """
    usage: audiochat-cli --audio <file> --prompt <text> [--model <catalog-id>]
           audiochat-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var promptText: String?
var audioPath: String?
var modelID = "qwen2.5-omni-3b-audio"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--prompt": promptText = args.popFirst()
    case "--audio": audioPath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.audio) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let promptText, let audioPath else { fail(usage) }

let id = modelID
do {
    let answer = try await askAboutAudio(
        promptText, audioAt: URL(fileURLWithPath: audioPath), model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    print(answer)
} catch {
    fail("error: \(error.localizedDescription)")
}
