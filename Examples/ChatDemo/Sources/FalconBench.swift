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

        // FALCON_EVICT=<substring>: delete cached model dirs whose repo name contains
        // the substring, plus Library/Caches, before loading — recovers from a
        // poisoned on-device compile cache (re-downloads on next load).
        if let evict = env["FALCON_EVICT"] {
            let fm = FileManager.default
            let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            let modelsDir = support.appendingPathComponent("CoreAIKit/models")
            if let orgs = try? fm.contentsOfDirectory(
                at: modelsDir, includingPropertiesForKeys: nil)
            {
                for org in orgs {
                    guard let repos = try? fm.contentsOfDirectory(
                        at: org, includingPropertiesForKeys: nil) else { continue }
                    for repo in repos
                    where repo.lastPathComponent.localizedCaseInsensitiveContains(evict) {
                        try? fm.removeItem(at: repo)
                        emit("evicted=\(repo.lastPathComponent)")
                    }
                }
            }
            let caches = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            if let entries = try? fm.contentsOfDirectory(
                at: caches, includingPropertiesForKeys: nil)
            {
                for entry in entries { try? fm.removeItem(at: entry) }
            }
            emit("caches_cleared=1")
        }

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

            // FALCON_WARM_PREFILL=<n>: also warm an n-token prefill shape (kills the
            // first-long-turn TTFT spike on dynamic/static-shape engines).
            let warmPrefill = Int(env["FALCON_WARM_PREFILL"] ?? "")
            try await session.prewarm(prefillLength: warmPrefill)
            emit("prewarmed=ok\(warmPrefill.map { " prefill=\($0)" } ?? "")")

            emit("prompt=\(prompt)")

            // FALCON_TURNS=N (>1): multi-turn KV-reuse validation. Turn 2+ TTFT staying
            // flat while prompt_tokens grows is the reuse evidence; cached_prompt_tokens
            // is the engine's own prefix-hit count (0 = engine reuses without reporting,
            // or a full re-prefill happened).
            let turnCount = Int(env["FALCON_TURNS"] ?? "") ?? 1
            if turnCount > 1 {
                let followUps = [
                    "Now compress that explanation into exactly one short sentence.",
                    "Translate that sentence into Japanese.",
                    // Deliberately long turn: pushes this turn's prefill length past any
                    // earlier one, to expose first-time-at-this-length engine costs
                    // (shape specialization / logits growth) as a TTFT spike.
                    "Here is some context you should consider carefully before answering. "
                        + "Light from the sun is composed of many wavelengths, and the "
                        + "atmosphere is composed of nitrogen, oxygen, argon, and trace "
                        + "gases, with molecules far smaller than the wavelength of "
                        + "visible light. Scattering efficiency depends strongly on "
                        + "wavelength for such small particles, and the human eye has "
                        + "three cone types with different spectral sensitivities, so "
                        + "perceived color is not just the physics of scattering but "
                        + "also the biology of vision. Considering all of this context "
                        + "together with everything we discussed earlier in this "
                        + "conversation, explain in two sentences why sunsets appear "
                        + "red and orange rather than blue.",
                    "Thanks. In one word, what is the scattering effect called?",
                ]
                var turnPrompts = [prompt]
                for i in 0..<(turnCount - 1) {
                    turnPrompts.append(followUps[i % followUps.count])
                }
                emit("---- MULTI-TURN ----")
                emit(String(format: "load_seconds=%.2f", await session.stats.loadSeconds ?? -1))
                for (i, p) in turnPrompts.enumerated() {
                    let reqStart = Date()
                    var firstTokenAt: Date?
                    var answer = ""
                    var sawThinking = false
                    for try await event in await session.streamResponse(to: p) {
                        switch event {
                        case .response(let delta):
                            if firstTokenAt == nil { firstTokenAt = Date() }
                            answer += delta
                        case .thinking:
                            if firstTokenAt == nil { firstTokenAt = Date() }
                            sawThinking = true
                        case .stats, .complete:
                            break
                        }
                    }
                    let genEnd = Date()
                    let stats = await session.stats
                    let decodeSpan = genEnd.timeIntervalSince(firstTokenAt ?? reqStart)
                    let gen = stats.generatedTokens
                    let avg = decodeSpan > 0 ? Double(max(gen - 1, 0)) / decodeSpan : 0
                    emit(String(
                        format: "turn=%d prompt_tokens=%d cached_prompt_tokens=%d ttft=%.3f "
                            + "generated=%d decode_tok_s=%.2f thinking=%@",
                        i + 1, stats.promptTokens, stats.cachedPromptTokens,
                        stats.ttftSeconds ?? -1, gen, avg, sawThinking ? "yes" : "no"))
                    emit("turn\(i + 1)_answer=\(String(answer.prefix(160)))")
                }
                emit(String(
                    format: "peak_footprint_mb=%.1f",
                    Double(await session.stats.footprintBytes) / 1_048_576.0))
                emit("status=SUCCESS")
            } else {
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
                emit("cached_prompt_tokens=\(stats.cachedPromptTokens)")
                emit(String(format: "ttft_seconds=%.3f", stats.ttftSeconds ?? -1))
                emit("generated_tokens=\(gen)")
                emit(String(format: "decode_tok_s_avg=%.2f", avgDecode))
                emit(String(format: "decode_tok_s_rolling32=%.2f", stats.tokensPerSecond ?? -1))
                emit(String(format: "peak_footprint_mb=%.1f", Double(stats.footprintBytes) / 1_048_576.0))
                emit("ANSWER_BEGIN")
                emit(answer)
                emit("ANSWER_END")
                emit("status=SUCCESS")
            }

            // FALCON_GUIDED=1: schema-constrained turns (sequential engine only). Two
            // back-to-back guided turns exercise the grammar mask AND the engine's
            // implicit prefix caching through the cumulative per-step feed.
            if env["FALCON_GUIDED"] != nil {
                emit("---- GUIDED ----")
                let schema = """
                    {"type":"object","properties":{"name":{"type":"string"},\
                    "population_millions":{"type":"number"}},\
                    "required":["name","population_millions"]}
                    """
                let json = try await session.respondJSON(
                    to: "Name one big city and its rough population in millions.",
                    schema: schema)
                emit("guided_json=\(json)")
                let valid = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) != nil
                emit("guided_valid_json=\(valid ? "yes" : "no")")
                let json2 = try await session.respondJSON(
                    to: "Now a different city, same JSON shape.", schema: schema)
                emit("guided_json2=\(json2)")
                let valid2 = (try? JSONSerialization.jsonObject(with: Data(json2.utf8))) != nil
                emit("guided_valid_json2=\(valid2 ? "yes" : "no")")
            }
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
