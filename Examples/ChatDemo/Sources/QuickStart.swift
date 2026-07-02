// QuickStart.swift — the take-home core of this runner: prompt in → reply out, one typed
// function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this function;
// the GUI drives the same `ChatSession(catalog:)` gesture, held across turns for its
// transcript. Want an on-device LLM in your own app? This file is the part you copy; the
// model card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation

/// Ask any chat model in the catalog one question
/// (`ModelCatalog.builtin.available(.chat)`: Qwen3.5/3.6, Gemma 4, LFM2.5, Granite 4, …).
/// First use downloads the model (progress via `downloadProgress`), later runs load from the
/// local cache. Multi-turn? Hold the `ChatSession` and call `respond(to:)` per turn — it keeps
/// the conversation history; `streamResponse(to:)` yields tokens as they decode.
func ask(
    _ prompt: String,
    model id: String = "qwen3.5-2b",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    // CARD-SNIPPET-BEGIN
    let chat = try await ChatSession(catalog: id, downloadProgress: downloadProgress)
    let reply = try await chat.respond(to: prompt)
    // CARD-SNIPPET-END
    return reply
}
