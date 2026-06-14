// GemmaArchitecture.swift — the family constants a host must know to drive a Gemma 4
// (E2B/E4B) decode bundle behind FoundationModels: the per-layer-embedding (PLE) static
// graph inputs, and the Gemma 4 turn/channel chat-format tokens.
//
// Gemma 4 small models carry a non-standard extra input: a giant per-layer-embedding table
// the decoder gathers from per token (the analogue of a VL decoder's `image_embeds`). The
// published `…_tbl` pipelined bundle moves that gather IN-GRAPH and exposes the table as two
// STATIC graph inputs — `ple_table` (int8 [vocab, ld·layers]) and `ple_scale` (f32 [vocab]) —
// so the host only has to bind the two table files once as constant buffers (see GemmaRuntime).
// Both files live in the bundle's paired `gemma4_gather_raw` directory.
//
// Gemma 4 also uses a new chat format (not Gemma 3's `<start_of_turn>`): turns are framed by
// `<|turn>` … `<turn|>`, and an optional reasoning span rides a `<|channel>thought … <channel|>`
// channel. The published E2B bundle ships the stock `google/gemma-4-E2B-it` tokenizer with NO
// embedded chat template, so the prompt is rendered explicitly from these tokens
// (GemmaPromptRenderer). E2B and E4B share the format and the static-input file names — only the
// download repo/path differs (see GemmaModelID).

import Foundation

/// Static-input file map + chat-format tokens for a Gemma 4 decode bundle.
public struct GemmaArchitecture: Sendable, Hashable {
    /// Beginning-of-sequence token. Gemma's tokenizer post-processor does NOT add one, so the
    /// renderer emits it explicitly.
    public let bos: String
    /// Start-of-turn marker (`<|turn>`), followed by the role and a newline.
    public let startOfTurn: String
    /// End-of-turn marker (`<turn|>`). Also the primary stop token — the model ends every
    /// turn with it, while the tokenizer's `eos` (`<eos>`) is a separate, rarer id.
    public let endOfTurn: String
    /// Opening of the thinking channel (`<|channel>thought\n`). With thinking disabled the
    /// renderer emits an immediately-closed empty channel here so the model answers directly.
    public let thoughtOpen: String
    /// Closing of the thinking channel (`<channel|>`).
    public let thoughtClose: String

    /// Graph input name → file name inside the paired tables directory. These two constant
    /// buffers are bound once via `EngineOptions.staticInputBuffers`; the decoder graph does
    /// the PLE gather in-graph from `input_ids`. Identical across E2B and E4B (only the file
    /// byte sizes differ: E2B per-layer table [vocab, 8960], E4B [vocab, 10752]).
    public let staticInputFiles: [String: String]

    public init(
        bos: String = "<bos>",
        startOfTurn: String = "<|turn>",
        endOfTurn: String = "<turn|>",
        thoughtOpen: String = "<|channel>thought\n",
        thoughtClose: String = "<channel|>",
        staticInputFiles: [String: String] = [
            "ple_table": "embed_per_layer.i8",
            "ple_scale": "embed_per_layer.scale.f32",
        ]
    ) {
        self.bos = bos
        self.startOfTurn = startOfTurn
        self.endOfTurn = endOfTurn
        self.thoughtOpen = thoughtOpen
        self.thoughtClose = thoughtClose
        self.staticInputFiles = staticInputFiles
    }

    /// The Gemma 4 family format, shared by every published size (E2B / E4B).
    public static let gemma4 = GemmaArchitecture()
}
