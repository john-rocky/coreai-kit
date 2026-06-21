// AudioPromptRenderer.swift — renders a FoundationModels `Transcript` into audio-decoder tokens.
//
// FoundationModels has no audio attachment segment (unlike images), so the audio is attached to
// the runtime out-of-band by the app; this renderer only needs the attached audio-token count `N`.
// It splices the audio block — `<|audio_bos|>` + N×`<|AUDIO|>` + `<|audio_eos|>` — at the START of
// the LATEST user turn (v1: one clip per session), then rewrites the N `<|AUDIO|>` ids to extension
// ids `vocab + slot`, which the decoder graph gathers from `audio_embeds`. With `N == 0` it is a
// plain Qwen2.5 text prompt (the same bundle answers text-only turns).

import Foundation
import FoundationModels
import Tokenizers

enum AudioPromptRenderer {
    struct Rendered {
        let tokens: [Int32]
    }

    static func render(
        transcript: Transcript, tokenizer: any Tokenizer, arch: AudioArchitecture,
        audioTokenCount n: Int
    ) throws -> Rendered {
        let entries = Array(transcript)
        var lastPromptIndex = -1
        for (i, entry) in entries.enumerated() { if case .prompt = entry { lastPromptIndex = i } }

        // Qwen2.5-Omni's chat template defaults to this system prompt (the oracle used it).
        var system = "You are a helpful assistant."
        var body = ""
        for (i, entry) in entries.enumerated() {
            switch entry {
            case .instructions(let instructions):
                let text = plainText(instructions.segments)
                if !text.isEmpty { system = text }
            case .prompt(let prompt):
                let block =
                    (i == lastPromptIndex && n > 0)
                    ? arch.audioBos + String(repeating: arch.audioPad, count: n) + arch.audioEos
                    : ""
                body += "\(arch.imStart)user\n\(block)\(plainText(prompt.segments))\(arch.imEnd)\n"
            case .response(let response):
                body += "\(arch.imStart)assistant\n\(plainText(response.segments))\(arch.imEnd)\n"
            default:
                continue  // reasoning / tool entries are not part of the audio chat path
            }
        }
        let text =
            "\(arch.imStart)system\n\(system)\(arch.imEnd)\n" + body + "\(arch.imStart)assistant\n"

        var tokens = tokenizer.encode(text: text).map(Int32.init)
        guard n > 0 else { return Rendered(tokens: tokens) }

        guard let padID = singleTokenID(arch.audioPad, tokenizer: tokenizer) else {
            throw KitAudioError.renderFailed("\(arch.audioPad) is not a single token")
        }
        var slot: Int32 = 0
        for i in tokens.indices where tokens[i] == padID && slot < Int32(n) {
            tokens[i] = arch.vocab + slot
            slot += 1
        }
        guard slot == Int32(n) else {
            throw KitAudioError.renderFailed("\(arch.audioPad)×\(n) expected, rewrote \(slot)")
        }
        return Rendered(tokens: tokens)
    }

    // MARK: - Helpers

    private static func plainText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }

    private static func singleTokenID(_ token: String, tokenizer: any Tokenizer) -> Int32? {
        if let id = tokenizer.convertTokenToId(token) { return Int32(id) }
        let ids = tokenizer.encode(text: token)
        return ids.count == 1 ? Int32(ids[0]) : nil
    }
}
