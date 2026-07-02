// depth-cli — the argument shell over `depthMap(at:model:)` (Sources/QuickStart.swift): parse
// flags, call the same function the card's snippet shows, write the depth map. Progress goes
// to stderr so stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run depth-cli --image photo.jpg --output depth.png --model depth-anything-3-small

import CoreAIKitVision
import Foundation
import ImageIO
import UniformTypeIdentifiers

let usage = """
    usage: depth-cli --image <file> [--output <png>] [--model <catalog-id>]
           depth-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var imagePath: String?
var outputPath = "depth.png"
var modelID = "depth-anything-3-small"

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--image": imagePath = args.popFirst()
    case "--output": outputPath = args.popFirst() ?? outputPath
    case "--model": modelID = args.popFirst() ?? modelID
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.depth) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

guard let imagePath else { fail(usage) }

let id = modelID
do {
    let map = try await depthMap(
        at: URL(fileURLWithPath: imagePath), model: id,
        downloadProgress: { progress in
            stderrPrint(
                String(format: "\rdownloading %@  %3.0f%%", id, progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
        })
    let outURL = URL(fileURLWithPath: outputPath)
    guard
        let dest = CGImageDestinationCreateWithURL(
            outURL as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { fail("cannot write \(outputPath)") }
    CGImageDestinationAddImage(dest, map, nil)
    CGImageDestinationFinalize(dest)
    print("\(outputPath): \(map.width)x\(map.height) depth map")
} catch {
    fail("error: \(error.localizedDescription)")
}
