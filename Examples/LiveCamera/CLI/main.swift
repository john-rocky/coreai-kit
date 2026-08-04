// live-cli — the argument shell over `Sources/QuickStart.swift`. Progress goes to stderr so
// stdout stays machine-checkable (agents: assert on stdout).
//
//   swift run live-cli --video clip.mov --fps 2
//   swift run live-cli --video clip.mov --for person
//   swift run -c release live-cli --bench    # release: a debug build reads ~2x slower

import Accelerate
import CoreAIOps
import CoreImage
import CoreVideo
import Foundation

let usage = """
    usage: live-cli --video <file> [--fps <rate>] [--changes] [--for <label>]
           live-cli --bench
           live-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

var videoPath: String?
var fps = 1.0
var changes = false
var triggerLabel: String?
var runBench = false

var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--video": videoPath = args.popFirst()
    case "--fps": fps = Double(args.popFirst() ?? "") ?? fps
    case "--changes": changes = true
    case "--for": triggerLabel = args.popFirst()
    case "--bench": runBench = true
    case "--list-models":
        for entry in ModelCatalog.builtin.available(.detection) {
            print("\(entry.id)  —  \(entry.name)")
        }
        exit(0)
    default: fail(usage)
    }
}

// MARK: - Benchmark

/// The preprocessing measurement quoted in `PixelBufferPreprocessor`'s documentation, so the
/// number in the comment is reproducible rather than folklore.
func bench() {
    func buffer(width: Int, height: Int) -> CVPixelBuffer {
        var pixels: CVPixelBuffer?
        CVPixelBufferCreate(
            kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
            [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary,
            &pixels)
        let frame = pixels!
        CVPixelBufferLockBaseAddress(frame, [])
        let base = CVPixelBufferGetBaseAddress(frame)!.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(frame)
        for y in 0..<height {
            for x in 0..<width {
                let p = base + y * rowBytes + x * 4
                p[0] = UInt8(x * 255 / max(width - 1, 1))
                p[1] = UInt8(y * 255 / max(height - 1, 1))
                p[2] = UInt8((x + y) * 255 / max(width + height - 2, 1))
                p[3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(frame, [])
        return frame
    }

    let clock = ContinuousClock()
    func measure(_ label: String, _ body: () throws -> Void) rethrows {
        for _ in 0..<30 { try body() }  // warm up
        let start = clock.now
        for _ in 0..<300 { try body() }
        let each = (clock.now - start).seconds / 300
        print(String(format: "%-40@ %7.3f ms", label as NSString, each * 1000))
    }

    let frame = buffer(width: 640, height: 480)
    let viaImage = ImagePreprocessor.imagenet224
    let viaPixels = PixelBufferPreprocessor(viaImage)
    let context = CIContext()

    print("preprocessing one 640x480 frame to 224^2 planar CHW")
    try? measure("CIContext render + CGImage preprocess") {
        let ci = CIImage(cvPixelBuffer: frame)
        guard let cg = context.createCGImage(ci, from: ci.extent) else { return }
        _ = try viaImage.chw(from: cg)
    }
    try? measure("PixelBufferPreprocessor (vImage)") {
        _ = try viaPixels.chw(from: frame)
    }
    measure("CIContext render alone") {
        let ci = CIImage(cvPixelBuffer: frame)
        _ = context.createCGImage(ci, from: ci.extent)
    }
}

if runBench {
    bench()
    exit(0)
}

// MARK: - Video scan

guard let videoPath else { fail(usage) }
let url = URL(fileURLWithPath: videoPath)
guard FileManager.default.fileExists(atPath: url.path) else {
    fail("error: no file at \(url.path)")
}

CoreAI.onDownload { progress in
    stderrPrint(
        String(format: "\rdownloading  %3.0f%%", progress.fraction * 100),
        terminator: progress.fraction < 1 ? "" : "\n")
}

let clock = ContinuousClock()
let started = clock.now
var samples = 0

do {
    if let triggerLabel {
        for try await moment in scanVideo(at: url, for: triggerLabel, framesPerSecond: fps) {
            samples += 1
            let names = Set(moment.value.detections.map(\.label)).sorted()
            print(
                String(
                    format: "%8.2f  %@", moment.time,
                    names.joined(separator: ", ") as NSString))
        }
        if samples == 0 { print("(no moments matched “\(triggerLabel)”)") }
    } else {
        for try await entry in scanVideo(
            at: url, framesPerSecond: fps, minimumChange: changes ? 0.02 : nil)
        {
            samples += 1
            let names = Set(entry.value.map(\.label)).sorted()
            print(
                String(
                    format: "%8.2f  %@", entry.time,
                    (names.isEmpty ? "-" : names.joined(separator: ", ")) as NSString))
        }
    }
} catch {
    fail("error: \(error.localizedDescription)")
}

stderrPrint(
    String(format: "%d samples in %.2f s", samples, (clock.now - started).seconds))
