// action-cli — the argument shell over `recognizeAction(in:model:)` (Sources/QuickStart.swift):
// parse flags, call the same function the card's snippet shows, print the ranked actions.
// Progress goes to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run action-cli --video clip.mp4 --model vjepa2-vitl-ssv2

import CoreAIKitVision
import Foundation

let usage = """
    usage: action-cli --video <file> [--model <catalog-id>]
           action-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var videoPath: String?
var modelID = "vjepa2-vitl-ssv2"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--video": videoPath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.video) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let videoPath else { fail(usage) }

let id = modelID
do {
    let actions = try await recognizeAction(
        in: URL(fileURLWithPath: videoPath), model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    for action in actions {
        print(String(format: "%5.1f%%  %@", action.probability * 100, action.label))
    }
} catch {
    fail("error: \(error.localizedDescription)")
}
