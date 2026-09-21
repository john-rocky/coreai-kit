// QuickStart.swift — the take-home core of this runner: a state and typed questions in,
// answers with probabilities out, one typed function, no UI. Both the GUI app and the CLI
// call exactly this function (the view models are display shells, `CLI/main.swift` an
// argument shell). Want typed decisions in your own app? This file is the part you copy; the
// model card's 💻 snippet is the marked block below.

import CoreAIOps
import Foundation

/// Score typed questions about one state with a catalog chat model
/// (`ModelCatalog.builtin.available(.chat)`; MiniCPM5 2B is the default). Nothing is
/// generated: each question is one prompt read at its answer slot, and the answers are the
/// probabilities over the listed options. First use downloads the model (progress via
/// `downloadProgress`), later runs load from the local cache.
///
/// The op form (`CoreAI.decide`) is the same call with the model resolved and cached behind
/// it; this function holds the model-level `TypedDecisions` so an app that decides
/// continuously — a speech gate, a clipboard watcher — keeps it warm.
func decide(
    state: String,
    questions: [String: Decision.Question],
    model id: String = "minicpm5-2b",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> [String: Decision.Answer] {
    // CARD-SNIPPET-BEGIN
    let decider = try await TypedDecisions(catalog: id, downloadProgress: downloadProgress)
    // One prefill of the state, then one scored prompt per question (the engine rewinds to
    // the shared prefix between them). `answer.noul` / `.choice` / `.score` read the value;
    // `answer.timing` says how many tokens were reused and how long it took.
    return try await decider.decide(state, questions)
    // CARD-SNIPPET-END
}
