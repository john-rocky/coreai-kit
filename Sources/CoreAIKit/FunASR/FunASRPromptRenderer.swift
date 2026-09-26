// FunASRPromptRenderer.swift — the Fun-ASR-Nano prompt token ids for a clip of N audio slots.
//
// funasr 1.4.16 (`FunASRNano.get_prompt` / `generate_chatml` / `data_load_speech`) builds
//   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user text}
//   [N audio rows]
//   <|im_end|>\n<|im_start|>assistant\n
// and tokenizes the text before and after the audio in one call each; no marker token surrounds
// the audio. The user text is `get_prompt(hotwords, language, itn)`:
//   hotwords (if any): "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。
//                       \n\n\n**上下文信息：**\n\n\n热词列表：[h1, h2]\n"
//   then "语音转写" (no language) or "语音转写成{language}", then "，不进行文本规整" when itn is off,
//   then "：". As in funasr's vLLM prompt builder (`inference_vllm._build_prompt_text`), the codes
//   zh / en / ja / ko become 中文 / 英文 / 日文 / 韩文; any other language string is written as given.
// The audio slots are written as extension ids `vocab + slot` (slot 0..<N), which the decoder graph
// gathers from its static `audio_embeds` input. The default prompt (no hotwords, no language, itn
// on) is 18 + N + 5 ids — the id list the port was gated on (the zoo's logs/r2_prompt_ids.json).

import Foundation
import Tokenizers

@available(macOS 27, iOS 27, *)
enum FunASRPromptRenderer {
    static let systemTurn =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    static let hotwordPreamble =
        "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
    static let assistantTurn = "<|im_end|>\n<|im_start|>assistant\n"
    static let languageAliases = ["zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文"]

    /// funasr's `get_prompt`: the user text that precedes the audio. An empty `language` counts as
    /// none, as in the other kit transcribers.
    static func userText(hotwords: [String], language: String?, itn: Bool) -> String {
        var text = hotwords.isEmpty
            ? "" : hotwordPreamble + "热词列表：[\(hotwords.joined(separator: ", "))]\n"
        if let language, !language.isEmpty {
            text += "语音转写成\(languageAliases[language.lowercased()] ?? language)"
        } else {
            text += "语音转写"
        }
        if !itn { text += "，不进行文本规整" }
        return text + "："
    }

    /// The text before and after the audio slots, as token ids (no special tokens added).
    static func segments(
        tokenizer: any Tokenizer, hotwords: [String] = [], language: String? = nil, itn: Bool = true
    ) -> (prefix: [Int32], suffix: [Int32]) {
        let prefix = tokenizer.encode(
            text: systemTurn + userText(hotwords: hotwords, language: language, itn: itn),
            addSpecialTokens: false)
        let suffix = tokenizer.encode(text: assistantTurn, addSpecialTokens: false)
        return (prefix.map(Int32.init), suffix.map(Int32.init))
    }

    /// The full prompt: prefix + `vocab + slot` for each of the `n` audio slots + suffix.
    static func render(
        tokenizer: any Tokenizer, arch: FunASRArchitecture, audioTokenCount n: Int,
        hotwords: [String] = [], language: String? = nil, itn: Bool = true
    ) -> [Int32] {
        let (prefix, suffix) = segments(
            tokenizer: tokenizer, hotwords: hotwords, language: language, itn: itn)
        return prefix + (0..<Int32(n)).map { arch.vocab + $0 } + suffix
    }
}
