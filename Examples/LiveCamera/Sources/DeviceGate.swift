// DeviceGate.swift — the device self-test, in the shape `Examples/DetectCamera` established:
// `NSLog("GATE …")` lines a host can read back over `devicectl`, so "it works on a phone" is
// a transcript rather than a claim.
//
// Launch with `-gate` and the app runs every task once — the two live camera ones for a
// couple of seconds each, the video scan over a clip it generates itself — then logs a
// verdict per task and exits. Everything here needs a real device: the CoreAI framework is
// not in the Simulator SDK.
//
//   xcrun devicectl device process launch --device <id> --console \
//       com.coreaikit.livecamera --  -gate

import AVFoundation
import CoreAIOps
import CoreVideo
import Foundation

enum DeviceGate {
    static var isRequested: Bool {
        CommandLine.arguments.contains("-gate")
            || UserDefaults.standard.bool(forKey: "gate")
    }

    static func run() async {
        NSLog("GATE begin device=%@", deviceName)
        await gateDetect(model: "rf-detr")
        // The swap is the claim this layer makes, and YOLOX-S is the model whose 640 input
        // produced an odd 4:3 capture height — the crash this gate exists to catch again.
        await gateDetect(model: "yolox-s")
        await gateDepth()
        await gateScan()
        NSLog("GATE end")
        exit(0)
    }

    private static var deviceName: String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }

    // MARK: - Live camera

    private static func gateDetect(model: String) async {
        do {
            let watch = try await liveDetections(model: model, framesPerSecond: 30)
            var frames = 0
            var lastStats: LiveStats?
            var labels: Set<String> = []
            for try await result in watch {
                frames += 1
                lastStats = result.stats
                labels.formUnion(result.value.map(\.label))
                if frames >= 60 { break }  // ~2 s of a 30 fps request
            }
            watch.stop()
            if let stats = lastStats {
                NSLog(
                    "GATE detect model=%@ frames=%d fps=%.1f median=%.1fms dropped=%d thermal=%d labels=%@",
                    model, frames, stats.framesPerSecond, stats.latency.seconds * 1000,
                    stats.dropped, stats.thermalState.rawValue,
                    labels.sorted().joined(separator: ",") as NSString)
            } else {
                NSLog("GATE detect model=%@ FAIL no frames", model)
            }
        } catch {
            NSLog("GATE detect model=%@ FAIL %@", model, String(describing: error))
        }
    }

    private static func gateDepth() async {
        do {
            let watch = try await liveDepth(framesPerSecond: 15)
            var frames = 0
            var lastStats: LiveStats?
            var extent = "?"
            for try await result in watch {
                frames += 1
                lastStats = result.stats
                extent = "\(result.value.width)x\(result.value.height)"
                if frames >= 30 { break }
            }
            watch.stop()
            if let stats = lastStats {
                NSLog(
                    "GATE depth frames=%d map=%@ fps=%.1f median=%.1fms thermal=%d",
                    frames, extent as NSString, stats.framesPerSecond,
                    stats.latency.seconds * 1000, stats.thermalState.rawValue)
            } else {
                NSLog("GATE depth FAIL no frames")
            }
        } catch {
            NSLog("GATE depth FAIL %@", String(describing: error))
        }
    }

    // MARK: - Video scan

    private static func gateScan() async {
        guard let url = await makeClip() else {
            NSLog("GATE scan FAIL could not write a clip")
            return
        }
        defer { try? FileManager.default.removeItem(at: url) }
        let clock = ContinuousClock()

        // Every sample, then only the ones that changed. The clip is deliberately static
        // after the first frame, so the second number is the sampler working.
        for change in [Float?.none, Float?.some(0.02)] {
            do {
                let started = clock.now
                var samples = 0
                var lastTime: TimeInterval = -1
                var ordered = true
                for try await entry in scanVideo(
                    at: url, framesPerSecond: 4, minimumChange: change)
                {
                    samples += 1
                    if entry.time <= lastTime { ordered = false }
                    lastTime = entry.time
                }
                NSLog(
                    "GATE scan changes=%@ samples=%d ordered=%@ time=%.2fs",
                    (change == nil ? "off" : "on") as NSString, samples,
                    (ordered ? "yes" : "NO") as NSString, (clock.now - started).seconds)
            } catch {
                NSLog("GATE scan FAIL %@", String(describing: error))
            }
        }
    }

    /// A 4 s clip: one moving frame, then a still image. `minimumChange` should collapse it
    /// to roughly one sample while the plain scan returns sixteen.
    private static func makeClip() async -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gate-clip.mov")
        try? FileManager.default.removeItem(at: url)
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else {
            return nil
        }
        let input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: 320, AVVideoHeightKey: 240,
            ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 320,
                kCVPixelBufferHeightKey as String: 240,
            ])
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for i in 0..<120 {  // 4 s at 30 fps
            while !input.isReadyForMoreMediaData {
                try? await Task.sleep(for: .milliseconds(1))
            }
            guard let pool = adaptor.pixelBufferPool else { break }
            var pixels: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixels)
            guard let buffer = pixels else { break }
            CVPixelBufferLockBaseAddress(buffer, [])
            if let base = CVPixelBufferGetBaseAddress(buffer)?
                .assumingMemoryBound(to: UInt8.self)
            {
                let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
                for y in 0..<240 {
                    for x in 0..<320 {
                        let p = base + y * rowBytes + x * 4
                        p[0] = UInt8((x * 3) % 256)
                        p[1] = UInt8((y * 3) % 256)
                        p[2] = UInt8((x + y) % 256)
                        p[3] = 255
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(
                buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 30))
        }
        input.markAsFinished()
        await writer.finishWriting()
        return writer.status == .completed ? url : nil
    }
}
