// SiriAskGate — headless Mac sanity for the SIRI-GEMMA ask-anything rework. It exercises the EXACT
// code path the `AskGemmaIntent.perform()` uses (`ModelHost.shared.ask(question:)`) so the model
// story is provable on a Mac before the USER does the make-or-break iPhone device gate (G1).
//
// This is the OPTIONAL Mac sanity from the kickoff — Gemma 4 is already Mac-validated by GEMMA-KIT
// ("capital of France" → Paris through `LanguageModelSession(model: KitGemmaModel)`); this just
// confirms the reworked ask-anything host produces sane free-text answers end to end.
//
// Usage:
//   swift run SiriAskGate                                  # downloads the macOS Gemma 4 E2B bundle
//   swift run SiriAskGate --decoder <dir> --tables <dir>   # local bundle dirs (skip the download)
//
// GPU: this loads a model onto the GPU — hold the repo `_GPU_LOCK` while running.
// COREAI_CHUNK_THRESHOLD=1 is set defensively (the `…_tbl` decode graph is S=1).

import CoreAIKit
import Foundation
import SiriAskCore

setvbuf(stdout, nil, _IONBF, 0)
setenv("COREAI_CHUNK_THRESHOLD", "1", 1)

// ---- arguments -----------------------------------------------------------------------------
// A closure (not a global func) so it stays within main.swift's @MainActor top-level context and
// can read `argv`; top-level bindings are main-actor-isolated under Swift 6.
let argv = Array(CommandLine.arguments.dropFirst())
let flagValue: (String) -> String? = { name in
    guard let i = argv.firstIndex(of: name), i + 1 < argv.count else { return nil }
    return argv[i + 1]
}
let decoderDir = flagValue("--decoder")
let tablesDir = flagValue("--tables")

let host: ModelHost
switch (decoderDir, tablesDir) {
case let (decoder?, tables?):
    host = ModelHost(
        localDecoderAt: URL(fileURLWithPath: decoder),
        tablesAt: URL(fileURLWithPath: tables))
    print("[gate] model = local Gemma 4 E2B\n  decoder: \(decoder)\n  tables:  \(tables)")
case (nil, nil):
    host = ModelHost.shared
    print("[gate] model = Gemma 4 E2B (macOS bundle, hub download)")
default:
    FileHandle.standardError.write(Data("usage: pass BOTH --decoder and --tables, or neither\n".utf8))
    exit(2)
}

print("[gate] loading + warming model…")
try await host.ensureReady { progress in
    if progress.fraction < 1 {
        print("  \(Int(progress.fraction * 100))% \(progress.currentFile)")
    }
}
print("[gate] model ready\n")

// ---- ask-anything: free-text questions answered from the model's own knowledge --------------
// Pick questions with an unambiguous factual answer so the gate is deterministic-ish (greedy
// decode); assert the answer contains the expected fact.
struct AskCase { let q: String; let mustMention: [String] }
let cases = [
    AskCase(q: "What is the capital of France?", mustMention: ["paris"]),
    AskCase(q: "What planet is known as the Red Planet?", mustMention: ["mars"]),
    AskCase(q: "Reply with one word: the opposite of hot.", mustMention: ["cold"]),
]

var failures = 0
print("=== ask-anything via ModelHost.ask (the AskGemmaIntent path) ===")
for c in cases {
    let answer = try await host.ask(question: c.q)
    let lower = answer.lowercased()
    let ok = c.mustMention.contains { lower.contains($0) }
    print("Q: \(c.q)")
    print("  A: \(answer)")
    print("  -> \(ok ? "PASS" : "FAIL") (any of \(c.mustMention) present: \(ok))\n")
    if !ok { failures += 1 }
}

// ---- verdict -------------------------------------------------------------------------------
if failures == 0 {
    print("GATE PASS — Gemma 4 E2B answers free-text questions via the intent's exact code path.")
} else {
    print("GATE FAIL — \(failures) case(s) failed.")
    exit(1)
}
