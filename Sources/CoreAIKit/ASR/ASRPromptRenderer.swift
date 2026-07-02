// ASRPromptRenderer.swift — builds the Qwen3-ASR prompt token ids for a clip of N audio tokens.
//
// Template (confirmed from chat_template.json + processing_qwen3_asr.py; {system} is EMPTY for ASR):
//   <|im_start|>system\n<|im_end|>\n
//   <|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n[language {Language}<asr_text>]?
//
// The N `<|audio_pad|>` ids are rewritten to extension ids `vocab + slot`, which the decoder graph
// gathers from the static `audio_embeds`. With a forced `language`, the assistant turn is primed
// with `language {Language}<asr_text>` so the model emits transcript text only; nil = auto-detect
// (the model emits `{Language}<asr_text>{text}` and the runtime splits it).

import Foundation
import Tokenizers

enum ASRPromptRenderer {
    static func render(
        tokenizer: any Tokenizer, arch: ASRArchitecture, audioTokenCount n: Int,
        language: String? = nil
    ) -> [Int32] {
        let audioBlock = arch.audioStart + String(repeating: arch.audioPad, count: n) + arch.audioEnd
        var text =
            "\(arch.imStart)system\n\(arch.imEnd)\n"
            + "\(arch.imStart)user\n\(audioBlock)\(arch.imEnd)\n"
            + "\(arch.imStart)assistant\n"
        if let language, !language.isEmpty {
            text += "language \(language)\(arch.asrText)"
        }

        var tokens = tokenizer.encode(text: text).map(Int32.init)
        guard n > 0, let padID = singleTokenID(arch.audioPad, tokenizer: tokenizer) else {
            return tokens
        }
        var slot: Int32 = 0
        for i in tokens.indices where tokens[i] == padID && slot < Int32(n) {
            tokens[i] = arch.vocab + slot
            slot += 1
        }
        return tokens
    }

    private static func singleTokenID(_ token: String, tokenizer: any Tokenizer) -> Int32? {
        if let id = tokenizer.convertTokenToId(token) { return Int32(id) }
        let ids = tokenizer.encode(text: token)
        return ids.count == 1 ? Int32(ids[0]) : nil
    }
}
