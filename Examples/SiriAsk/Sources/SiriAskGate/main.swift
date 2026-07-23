// SiriAskGate — headless Mac sanity for the ask-anything path. Exercises the EXACT code path the
// `AskGemmaIntent.perform()` uses (`ModelHost.shared.ask(question:)`) so the model story is provable
// on a Mac before the device gate.
//
// Usage:  swift run -c release SiriAskGate
// GPU: this loads a model onto the GPU — hold the repo `_GPU_LOCK` while running.

import CoreAIKit
import Foundation
import SiriAskCore

setvbuf(stdout, nil, _IONBF, 0)
setenv("COREAI_CHUNK_THRESHOLD", "1", 1)

let host = ModelHost.shared
print("[gate] loading + warming model…")
try await host.ensureReady { progress in
    if progress.fraction < 1 {
        print("  \(Int(progress.fraction * 100))% \(progress.currentFile)")
    }
}
print("[gate] model ready\n")

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
    print("  -> \(ok ? "PASS" : "FAIL")\n")
    if !ok { failures += 1 }
}

if failures == 0 {
    print("GATE PASS — the model answers free-text questions via the intent's exact code path.")
} else {
    print("GATE FAIL — \(failures) case(s) failed.")
    exit(1)
}
