// FalconBench.swift — headless on-device benchmark for a local Core AI language bundle.
//
// Enabled by launching the app with the FALCON_BENCH environment variable set. It loads the
// bundle pushed to the app container's Documents/falcon-bench/, runs one greedy generation,
// prints TTFT / decode tok/s / footprint / a coherence sample to stdout (captured by
// `devicectl ... --console`), writes Documents/falcon-bench/result.txt, then exits.

import CoreAIKit
import Foundation
import SwiftUI

enum FalconBench {
    static var isRequested: Bool {
        ProcessInfo.processInfo.environment["FALCON_BENCH"] != nil
    }

    /// Engine variant override via FALCON_ENGINE (auto | static-shape | coreai-pipelined |
    /// coreai-sequential). Defaults to .auto (what the real app uses).
    static var engineVariant: EngineVariant {
        switch ProcessInfo.processInfo.environment["FALCON_ENGINE"] {
        case "static-shape": return .staticShape
        case "coreai-pipelined", "pipelined": return .pipelined
        case "coreai-sequential", "sequential": return .sequential
        default: return .auto
        }
    }

    static var prompt: String {
        ProcessInfo.processInfo.environment["FALCON_PROMPT"]
            ?? "Explain why the sky appears blue during the day, in two or three sentences."
    }

    static func benchDir() -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("falcon-bench", isDirectory: true)
    }

    /// Locates the bundle root (a directory containing metadata.json) under falcon-bench/.
    static func bundleURL() -> URL? {
        let fm = FileManager.default
        let dir = benchDir()
        if fm.fileExists(atPath: dir.appendingPathComponent("metadata.json").path) { return dir }
        guard let entries = try? fm.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return nil }
        for e in entries
        where fm.fileExists(atPath: e.appendingPathComponent("metadata.json").path) {
            return e
        }
        return nil
    }

    static func run() async {
        // Catalog mode never sideloads into falcon-bench/, so the directory may not
        // exist yet — create it or every writeResult silently fails and the results
        // are lost when the devicectl console connection drops.
        try? FileManager.default.createDirectory(
            at: benchDir(), withIntermediateDirectories: true)
        var lines: [String] = []
        func emit(_ s: String) {
            print("FALCONBENCH \(s)")
            FileHandle.standardError.write(Data("FALCONBENCH \(s)\n".utf8))
            lines.append(s)
            writeResult(lines)
        }

        emit("=== Falcon3 iPhone Core AI bench ===")
        emit("device=\(deviceModel()) ios=\(ProcessInfo.processInfo.operatingSystemVersionString)")
        emit("engine_variant=\(engineVariant.rawValue)")

        let env = ProcessInfo.processInfo.environment
        // FALCON_CATALOG=<id>: bench a catalog model already in the app's store — the
        // exact load path and artifacts the interactive chat uses — instead of a
        // sideloaded falcon-bench bundle.
        let catalogID = env["FALCON_CATALOG"]
        var url: URL?
        if let catalogID {
            emit("catalog=\(catalogID)")
        } else {
            guard let bundle = bundleURL() else {
                emit("ERROR: no bundle (metadata.json) found under \(benchDir().path)")
                await Task.yield()
                exit(2)
            }
            emit("bundle=\(bundle.lastPathComponent)")
            emit("bundle_bytes=\(directorySize(bundle))")
            url = bundle
        }

        do {
            var config = ChatSession.Configuration()
            // Greedy by default (reproducible coherence sample); FALCON_TEMP=0.7
            // measures the sampling configuration the interactive app runs with.
            config.temperature = env["FALCON_TEMP"].flatMap(Double.init)
            config.maxResponseTokens = Int(env["FALCON_MAX"] ?? "") ?? 200
            config.engineVariant = engineVariant
            emit("temperature=\(config.temperature.map { "\($0)" } ?? "greedy")")
            emit("max_tokens=\(config.maxResponseTokens)")

            let loadStart = Date()
            let session: ChatSession
            if let catalogID {
                session = try await ChatSession(catalog: catalogID, configuration: config)
            } else {
                session = try await ChatSession(bundleAt: url!, configuration: config)
            }
            emit(String(format: "engine_created_and_loaded_in=%.2fs", Date().timeIntervalSince(loadStart)))
            emit("model_name=\(await session.modelName)")

            try await session.prewarm()
            emit("prewarmed=ok")

            emit("prompt=\(prompt)")

            let reqStart = Date()
            var firstTokenAt: Date?
            var answer = ""
            for try await event in await session.streamResponse(to: prompt) {
                switch event {
                case .response(let delta):
                    if firstTokenAt == nil { firstTokenAt = Date() }
                    answer += delta
                case .stats, .complete, .thinking:
                    break
                }
            }
            let genEnd = Date()
            let stats = await session.stats

            let first = firstTokenAt ?? reqStart
            let decodeSpan = genEnd.timeIntervalSince(first)
            let gen = stats.generatedTokens
            let avgDecode = decodeSpan > 0 ? Double(max(gen - 1, 0)) / decodeSpan : 0

            emit("---- RESULTS ----")
            emit(String(format: "load_seconds=%.2f", stats.loadSeconds ?? -1))
            emit("prompt_tokens=\(stats.promptTokens)")
            emit(String(format: "ttft_seconds=%.3f", stats.ttftSeconds ?? -1))
            emit("generated_tokens=\(gen)")
            emit(String(format: "decode_tok_s_avg=%.2f", avgDecode))
            emit(String(format: "decode_tok_s_rolling32=%.2f", stats.tokensPerSecond ?? -1))
            emit(String(format: "peak_footprint_mb=%.1f", Double(stats.footprintBytes) / 1_048_576.0))
            emit("ANSWER_BEGIN")
            emit(answer)
            emit("ANSWER_END")
            emit("status=SUCCESS")
        } catch {
            emit("status=FAILURE")
            emit("ERROR: \(String(describing: error))")
            emit("ERROR_LOCALIZED: \(error.localizedDescription)")
        }

        emit("=== done ===")
        // Give stdout a moment to flush before tearing down.
        try? await Task.sleep(nanoseconds: 300_000_000)
        exit(0)
    }

    private static func writeResult(_ lines: [String]) {
        let out = benchDir().appendingPathComponent("result.txt")
        try? lines.joined(separator: "\n").write(to: out, atomically: true, encoding: .utf8)
    }

    private static func directorySize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            total += Int64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    private static func deviceModel() -> String {
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        return machine
    }
}

/// Minimal SwiftUI surface shown while the headless bench runs (and so the scene activates).
struct FalconBenchView: View {
    @State private var started = false
    var body: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text("Falcon3 bench running…")
                .font(.headline)
            Text("See console / Documents/falcon-bench/result.txt")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .task {
            guard !started else { return }
            started = true
            await FalconBench.run()
        }
    }
}
