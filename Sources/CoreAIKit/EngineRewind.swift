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
// An engine holding a checkpoint (`InferenceEngine.checkpoint()`: the sequential and, from
// 0.2.9-zoo, the pipelined engine on a hybrid) returns to it for a rewind at or past its
// position, so it can keep fewer tokens than asked for: the count reported is the engine's own
// after the reset. ChatSession and KitExecutor have the engine checkpoint each turn's prompt at
// the end of its last finished message (`InferenceOptions.checkpointAfter` at `lastTurnEnd`), so
// the next turn's rewind lands there instead of on 0.

import CoreAILanguageModels
import Tokenizers

/// The tokens chat templates end a message with: the tokenizer's eos and those of these markers
/// its vocabulary has (a round trip guards the unknown-token false positive).
let turnEndMarkers = ["<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<turn|>"]

func turnEndTokenIDs(_ tokenizer: any Tokenizer) -> Set<Int32> {
    var ids = Set(turnEndMarkers.compactMap { marker -> Int32? in
        guard let id = tokenizer.convertTokenToId(marker), tokenizer.convertIdToToken(id) == marker
        else { return nil }
        return Int32(id)
    })
    if let eos = tokenizer.eosTokenId { ids.insert(Int32(eos)) }
    return ids
}

/// Where the checkpoint for the next turn goes in `prompt`: just past the last of `turnEnds` before
/// its final token, the end of its last finished message. The next turn re-renders that history as
/// it was and changes what follows it: the generation prompt becomes the reply's own turn, which
/// templates render otherwise (Qwen3.5 drops the empty thinking block its generation prompt
/// carries), so its rewind lands on the checkpoint; a template that rewrites the earlier history
/// still falls back to a full reset. Without such a token, before the final token; nil for a
/// one-token prompt.
func lastTurnEnd(of prompt: [Int32], turnEnds: Set<Int32>) -> Int? {
    guard prompt.count > 1 else { return nil }
    return prompt.dropLast().lastIndex(where: { turnEnds.contains($0) }).map { $0 + 1 } ?? prompt.count - 1
}

extension InferenceEngine {
    /// Rewinds the KV cache to `tokenIndex`, falling back to a full reset on engines that
    /// cannot rewind mid-sequence. Returns the index the engine actually kept — `tokenIndex`,
    /// a checkpoint's position below it, or 0 after the fallback — so callers report cached
    /// tokens truthfully and feed from there. Any other error from the engine propagates
    /// unchanged.
    func rewind(to tokenIndex: Int) async throws -> Int {
        guard tokenIndex > 0 else {
            try await reset(to: 0)
            return 0
        }
        do {
            try await reset(to: tokenIndex)
            return processedTokenCount
        } catch InferenceRuntimeError.invalidState {
            try await reset(to: 0)
            return 0
        }
    }
}
