// QuickStart.swift — the take-home core of this runner: prompt in → denoised reply out, one
// typed function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly this
// function; the GUI drives the same `KitDiffusionLM(catalog:)` and renders the live canvas
// (tokens pop in IN PARALLEL, out of reading order — the fill-in UX a diffusion LM actually
// has). Want an on-device diffusion LM in your own app? This file is the part you copy; the
// model card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation

/// Ask any diffusion LM in the catalog (`ModelCatalog.builtin.available(.dllm)`). First use
/// downloads the bundle (progress via `downloadProgress`), later runs load from the local
/// cache. Live canvas? Pass `onStep` — each denoising step hands you the whole canvas with
/// still-masked positions as ░.
func denoise(
    _ prompt: String,
    model id: String = "llada-8b",
    onStep: (@Sendable (KitDiffusionLM.Step) -> Void)? = nil,
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    // CARD-SNIPPET-BEGIN
    let dlm = try await KitDiffusionLM(catalog: id, downloadProgress: downloadProgress)
    let reply = try await dlm.reply(to: prompt, onStep: onStep)
    // CARD-SNIPPET-END
    return reply
}
