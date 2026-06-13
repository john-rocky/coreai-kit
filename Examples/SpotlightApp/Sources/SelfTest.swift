// SelfTest.swift — headless gate. Seeds the sample documents, points the `.files` search source at
// that folder, loads the default zoo model, and runs one grounded answer through RagEngine (the
// same path the UI uses), printing the retrieval chain and the answer. Exits 0 when the model both
// read a file (a real fetch happened) and produced a non-empty answer — i.e. it grounded in the
// real file text, not hallucinated. Invoked by SPOTLIGHT_SELFTEST=1.

import Foundation

enum SelfTest {
    /// Runs the async test on a detached task while blocking the main thread, then exits with the
    /// result code. (The same block-on-semaphore pattern the kit uses for off-main warm-up; the
    /// test never hops to the main actor, so there is no deadlock.)
    static func runBlocking() -> Never {
        let semaphore = DispatchSemaphore(value: 0)
        let box = CodeBox()
        Task.detached {
            box.code = await run()
            semaphore.signal()
        }
        semaphore.wait()
        exit(box.code)
    }

    static func run() async -> Int32 {
        log("== SpotlightApp self-test — 2-tool RAG over real files (.files source), headless ==")
        do {
            log("seeding sample documents…")
            let folder = FileLibrary.seedSampleDocumentsIfNeeded()
            let library = FileLibrary.snapshot(forRoots: [folder])
            log("library: \(library.displayName) — \(library.files.count) files at \(folder.path)")
            for file in library.files { log("  • \(file.displayName)") }

            log("indexing files into Core Spotlight…")
            try await FileLibrary.index(library.files)
            let visible = (try? await FileLibrary.searchableCount()) ?? -1
            log("→ \(visible) item(s) searchable via Core Spotlight")

            log("loading model (cached after first run)…")
            let engine = try await RagEngine.makeDefault { _ in }
            log("model: \(engine.modelName)")

            let question =
                ProcessInfo.processInfo.environment["SPOTLIGHT_SELFTEST_ASK"]
                ?? "What did I write about the night hike?"
            log("\n> \(question)\n")

            let answer = try await engine.answer(
                to: question,
                library: library,
                onFound: { found in
                    log("  [spotlight] found: \(found.map(\.name).joined(separator: " | "))")
                },
                onReading: { file in log("  [fetch_file] \(file.displayName)") },
                onAnswer: { _ in })

            log("\n[queries] \(answer.queries.joined(separator: " | "))")
            log("[read]    \(answer.readFiles.map(\.displayName).joined(separator: ", "))")
            log("[answer]  \(answer.text)")

            let answered = !answer.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let fetched = !answer.readFiles.isEmpty
            let pass = answered && fetched
            log("\nGATE: \(pass ? "PASS" : "FAIL") — answered=\(answered) fetched=\(fetched)")
            return pass ? 0 : 1
        } catch {
            log("ERROR: \(error)")
            return 2
        }
    }

    // MARK: - Index settle probe (no model, no GPU)

    /// Diagnostic: how long after indexing do the literal index (CSSearchQuery) and the SEMANTIC
    /// index (CSUserQuery — what SpotlightSearchTool actually searches) reflect fresh items.
    /// Invoked by SPOTLIGHT_INDEXPROBE=1.
    static func probeBlocking() -> Never {
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached {
            await probe()
            semaphore.signal()
        }
        semaphore.wait()
        exit(0)
    }

    static func probe() async {
        log("== index settle probe (literal vs semantic), no model ==")
        let folder = FileLibrary.seedSampleDocumentsIfNeeded()
        let library = FileLibrary.snapshot(forRoots: [folder])
        log("indexing \(library.files.count) files…")
        do { try await FileLibrary.index(library.files) } catch { log("index error: \(error)") }
        for i in 0..<40 {
            let lit = (try? await FileLibrary.searchableCount()) ?? -1
            let night = (try? await FileLibrary.userQueryCount("night hike")) ?? -1
            let hike = (try? await FileLibrary.userQueryCount("hike")) ?? -1
            log(
                String(
                    format: "t=%5.1fs  literal=%d  semantic(night hike)=%d  semantic(hike)=%d",
                    Double(i) * 0.5, lit, night, hike))
            try? await Task.sleep(for: .milliseconds(500))
        }
        log("done.")
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}

/// Carries the async result code back across the semaphore wait.
private final class CodeBox: @unchecked Sendable {
    var code: Int32 = 0
}
