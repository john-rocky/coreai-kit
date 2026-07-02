// detect-cli — the argument shell over `detect(at:model:)` (Sources/QuickStart.swift): parse
// flags, call the same function the card's snippet shows, print one detection per line.
// Progress goes to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run detect-cli --image photo.jpg --model rf-detr

import CoreAIKitVision
import Foundation

let usage = """
    usage: detect-cli --image <file> [--model <catalog-id>]
           detect-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var imagePath: String?
var modelID = "rf-detr"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--image": imagePath = args.popFirst()
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.detection) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let imagePath else { fail(usage) }

let id = modelID
do {
    let detections = try await detect(
        at: URL(fileURLWithPath: imagePath), model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    for d in detections {
        print(String(
            format: "%@  %.2f  [%.3f, %.3f, %.3f, %.3f]",
            d.label, d.score, d.box.origin.x, d.box.origin.y, d.box.width, d.box.height))
    }
    if detections.isEmpty { print("(no detections above threshold)") }
} catch {
    fail("error: \(error.localizedDescription)")
}
