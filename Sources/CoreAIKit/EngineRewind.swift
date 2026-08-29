// EngineRewind.swift — the kit's cross-turn rewind contract over any InferenceEngine.
//
// Every turn the kit rewinds the engine to the longest prefix its KV mirror shares with the
// newly rendered prompt, then feeds the FULL token list (the engine's implicit prefix caching
// prefills only the tail). Hybrid GDN/SSM models carry a recurrent scan that cannot be rewound
// mid-sequence: coreai-models 0.2.3-zoo throws `invalidState` for a partial reset on them
// (upstream #132), where the fork tags before it degraded to a full reset internally. The kit
// keeps the older contract here, because a full reset costs a re-prefill of the prompt and
// nothing else — the caller feeds the full sequence either way. Reproduced before this landed:
// the second turn of every Qwen3.5 chat failed with that error on the 0.2.3-zoo pin.

import CoreAILanguageModels

extension InferenceEngine {
    /// Rewinds the KV cache to `tokenIndex`, falling back to a full reset on engines that
    /// cannot rewind mid-sequence. Returns the index the engine actually kept — `tokenIndex`,
    /// or 0 after the fallback — so callers report cached tokens truthfully. Any other error
    /// from the engine propagates unchanged.
    func rewind(to tokenIndex: Int) async throws -> Int {
        guard tokenIndex > 0 else {
            try await reset(to: 0)
            return 0
        }
        do {
            try await reset(to: tokenIndex)
            return tokenIndex
        } catch InferenceRuntimeError.invalidState {
            try await reset(to: 0)
            return 0
        }
    }
}
