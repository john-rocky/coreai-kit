// music-cli — the argument shell over `compose(_:seconds:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the card's snippet shows, write a WAV. Progress goes to
// stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run music-cli --prompt "128 BPM tech house drum loop" --output loop.wav

import CoreAIKit
import Foundation

let usage = """
    usage: music-cli --prompt <text> [--seconds <n>] [--output <wav>] [--model <catalog-id>]
           music-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var promptText: String?
var outputPath = "music.wav"
var seconds: Float = 11
var modelID = "stable-audio-open-small"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--prompt": promptText = args.popFirst()
    case "--seconds": seconds = Float(args.popFirst() ?? "11") ?? 11
    case "--output": outputPath = args.popFirst() ?? outputPath
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.music) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let promptText else { fail(usage) }

let id = modelID
do {
    let audio = try await compose(
        promptText, seconds: seconds, model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    try WAVFile.write(
        samples: audio.samples, sampleRate: audio.sampleRate, channels: 2,
        to: URL(fileURLWithPath: outputPath))
    print(String(format: "%@: %.1f s @ %d Hz stereo", outputPath,
                 audio.seconds / 2, audio.sampleRate))
} catch {
    fail("error: \(error.localizedDescription)")
}
