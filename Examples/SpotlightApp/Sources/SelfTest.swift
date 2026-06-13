// SelfTest.swift — headless gate. Indexes the notes, loads the default zoo model, and runs one
// grounded answer through RagEngine (the same path the UI uses), printing the retrieval chain and
// the answer. Exits 0 when the model both read a note (a real fetch happened) and produced a
// non-empty answer — i.e. it grounded, not hallucinated. Invoked by SPOTLIGHT_SELFTEST=1.

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
        log("== SpotlightApp self-test — 2-tool RAG, headless ==")
        do {
            log("indexing notes…")
            try await indexNotes()
            log("loading model (cached after first run)…")
            let engine = try await RagEngine.makeDefault { _ in }
            log("model: \(engine.modelName)")

            let question =
                ProcessInfo.processInfo.environment["SPOTLIGHT_SELFTEST_ASK"]
                ?? "What did I write about the night hike?"
            log("\n> \(question)\n")

            let answer = try await engine.answer(
                to: question,
                onFound: { found in
                    log("  [spotlight] found: \(found.map(\.title).joined(separator: " | "))")
                },
                onReading: { id in log("  [fetch_note] \(id)") },
                onAnswer: { _ in })

            log("\n[queries] \(answer.queries.joined(separator: " | "))")
            log("[read]    \(answer.readNoteIDs.joined(separator: ", "))")
            log("[answer]  \(answer.text)")

            let answered = !answer.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            let fetched = !answer.readNoteIDs.isEmpty
            let pass = answered && fetched
            log("\nGATE: \(pass ? "PASS" : "FAIL") — answered=\(answered) fetched=\(fetched)")
            return pass ? 0 : 1
        } catch {
            log("ERROR: \(error)")
            return 2
        }
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }
}

/// Carries the async result code back across the semaphore wait.
private final class CodeBox: @unchecked Sendable {
    var code: Int32 = 0
}
