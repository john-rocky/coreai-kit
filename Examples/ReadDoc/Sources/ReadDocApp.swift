import CoreAIKit
import Foundation
import SwiftUI

@main
struct ReadDocApp: App {
    var body: some Scene {
        WindowGroup {
            ReadDocView()
                .task {
                    // Headless device self-test for MinerU2.5 (env-gated). Loads the sideloaded
                    // h18p bundles from Documents/mineru/{vision,decoder}, OCRs a sideloaded
                    // sample, and writes the markdown + timing to Documents/mineru_result.txt.
                    if ProcessInfo.processInfo.environment["MINERU_SELFTEST"] != nil {
                        await MineruSelfTest.run()
                    }
                    if ProcessInfo.processInfo.environment["GLM_SELFTEST"] != nil {
                        await GlmOcrSelfTest.run()
                    }
                }
        }
    }
}

enum MineruSelfTest {
    static func run() async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let out = docs.appendingPathComponent("mineru_result.txt")
        func log(_ s: String) {
            let line = s + "\n"
            if let handle = try? FileHandle(forWritingTo: out) {
                handle.seekToEndOfFile(); handle.write(Data(line.utf8)); try? handle.close()
            } else {
                try? line.data(using: .utf8)?.write(to: out)
            }
        }
        try? "MinerU self-test\n".data(using: .utf8)?.write(to: out)
        do {
            let base = docs.appendingPathComponent("mineru_pf")
            let layoutVision = base.appendingPathComponent("layout/vision")
            let layoutDecoder = base.appendingPathComponent("layout/decoder")
            let hasLayout = FileManager.default.fileExists(atPath: layoutVision.path)
            let sample = docs.appendingPathComponent("mineru_sample.png")
            log("loading reader… (layout bundle: \(hasLayout))")
            let t0 = Date()
            let reader = try await KitMineruReader(
                visionDir: base.appendingPathComponent("vision"),
                decoderDir: base.appendingPathComponent("decoder"),
                layoutVisionDir: hasLayout ? layoutVision : nil,
                layoutDecoderDir: hasLayout ? layoutDecoder : nil)
            log(String(format: "loaded in %.2fs", Date().timeIntervalSince(t0)))
            let t1 = Date()
            let markdown = reader.supportsStructured
                ? try await reader.readStructured(imageAt: sample)
                : try await reader.read(imageAt: sample, maxTokens: 512)
            log(String(format: "read (%@) in %.2fs",
                reader.supportsStructured ? "2-stage" : "single-pass", Date().timeIntervalSince(t1)))
            log("=== MARKDOWN ===")
            log(markdown)
            log("=== END ===")
        } catch {
            log("ERROR: \(error)")
        }
    }
}

enum GlmOcrSelfTest {
    static func run() async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let out = docs.appendingPathComponent("glm_result.txt")
        func log(_ s: String) {
            let line = s + "\n"
            if let handle = try? FileHandle(forWritingTo: out) {
                handle.seekToEndOfFile(); handle.write(Data(line.utf8)); try? handle.close()
            } else {
                try? line.data(using: .utf8)?.write(to: out)
            }
        }
        try? "GLM-OCR self-test\n".data(using: .utf8)?.write(to: out)
        do {
            let base = docs.appendingPathComponent("glm_ocr_pf")
            let sample = docs.appendingPathComponent("mineru_sample.png")
            log("loading reader…")
            let t0 = Date()
            let reader = try await KitGlmOcrReader(
                visionDir: base.appendingPathComponent("vision"),
                decoderDir: base.appendingPathComponent("decoder"))
            log(String(format: "loaded in %.2fs", Date().timeIntervalSince(t0)))
            let t1 = Date()
            let text = try await reader.read(imageAt: sample, maxTokens: 512)
            log(String(format: "read in %.2fs", Date().timeIntervalSince(t1)))
            log("=== TEXT ===")
            log(text)
            log("=== END ===")
        } catch {
            log("ERROR: \(error)")
        }
    }
}
