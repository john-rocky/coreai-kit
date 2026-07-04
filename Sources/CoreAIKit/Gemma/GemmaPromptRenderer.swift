// GemmaPromptRenderer.swift — renders a FoundationModels `Transcript` into Gemma 4 decoder
// tokens, using Gemma 4's turn/channel chat format directly.
//
// The published E2B bundle ships the stock `google/gemma-4-E2B-it` tokenizer with NO embedded
// chat template, so `applyChatTemplate` has nothing to apply — the format is emitted here from
// the architecture's special tokens. Gemma 4 frames each turn as `<|turn>ROLE\n … <turn|>`
// (roles: system / user / model), with an explicit `<bos>` at the very start (the tokenizer's
// post-processor does not add one).
//
// The generation prompt is just `<|turn>model\n`: on this gemma4-it-QAT bundle that yields a
// direct answer for simple queries, and the model opens its own `<|channel>thought\n … <channel|>`
// reasoning channel when it chooses to think (routed by the model's OutputProfile). Pre-injecting
// an empty thought channel was measured to do the opposite — it makes the model emit a verbose
// reasoning block — so it is deliberately NOT emitted here.

import Foundation
import FoundationModels
import Tokenizers

enum GemmaPromptRenderer {
    struct Rendered {
        let tokens: [Int32]
    }

    static func render(
        transcript: Transcript,
        tokenizer: any Tokenizer,
        arch: GemmaArchitecture
    ) throws -> Rendered {
        var system = ""
        var body = ""

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                // Gemma has no standalone system role mid-conversation; instructions fold into
                // a leading system turn.
                let text = plainText(instructions.segments)
                if !text.isEmpty { system = text }
            case .prompt(let prompt):
                body += turn(role: "user", plainText(prompt.segments), arch: arch)
            case .response(let response):
                body += turn(role: "model", plainText(response.segments), arch: arch)
            default:
                // Reasoning is not replayed; tool entries are not part of the v1 chat path.
                continue
            }
        }

        var text = arch.bos
        if !system.isEmpty {
            text += turn(role: "system", system, arch: arch)
        }
        text += body
        // Generation prompt: open the model turn and let the model answer (or open its own
        // reasoning channel).
        text += "\(arch.startOfTurn)model\n"

        let tokens = tokenizer.encode(text: text).map(Int32.init)
        return Rendered(tokens: tokens)
    }

    /// Renders a `ChatSession` history in the same turn format (the ChatSession face of the
    /// Transcript renderer above). Assistant turns replay only the final answer — thinking is
    /// surfaced live but never re-prompted — and the generation prompt opens a model turn.
    static func render(
        system: String?,
        history: [Message],
        tokenizer: any Tokenizer,
        arch: GemmaArchitecture
    ) -> [Int32] {
        var text = arch.bos
        if let system, !system.isEmpty {
            text += turn(role: "system", system, arch: arch)
        }
        for message in history {
            switch message.role {
            case .system:
                text += turn(role: "system", message.content, arch: arch)
            case .user:
                text += turn(role: "user", message.content, arch: arch)
            case .assistant:
                text += turn(role: "model", message.content, arch: arch)
            }
        }
        text += "\(arch.startOfTurn)model\n"
        return tokenizer.encode(text: text).map(Int32.init)
    }

    // MARK: - Helpers

    private static func turn(role: String, _ content: String, arch: GemmaArchitecture) -> String {
        "\(arch.startOfTurn)\(role)\n\(content)\(arch.endOfTurn)\n"
    }

    private static func plainText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }
}
