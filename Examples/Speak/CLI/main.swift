// speak-cli — the argument shell over `say(_:model:)` (Sources/QuickStart.swift): parse flags,
// call the same function the card's snippet shows, write a WAV. Progress goes to stderr so
// stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run speak-cli --text "Hello from Core AI." --output hello.wav

import CoreAIKit
import Foundation

let usage = """
    usage: speak-cli --text <text> [--output <wav>] [--model <catalog-id>]
           speak-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var text: String?
var outputPath = "speech.wav"
var modelID = "voxcpm-0.5b"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--text": text = args.popFirst()
    case "--output": outputPath = args.popFirst() ?? outputPath
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.tts) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let text else { fail(usage) }

let id = modelID
do {
    let audio = try await say(
        text, model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    try WAVFile.write(
        samples: audio.samples, sampleRate: audio.sampleRate,
        to: URL(fileURLWithPath: outputPath))
    print(String(format: "%@: %.1f s @ %d Hz", outputPath, audio.seconds, audio.sampleRate))
} catch {
    fail("error: \(error.localizedDescription)")
}
