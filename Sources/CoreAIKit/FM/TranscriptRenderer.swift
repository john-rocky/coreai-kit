// TranscriptRenderer.swift — renders a FoundationModels `Transcript` into prompt tokens.
//
// Two paths, picked per turn:
// * No tools anywhere: the bundle tokenizer's own chat template
//   (`applyChatTemplate(messages:)`) — correct for every model family.
// * Tools in play (definitions enabled, or tool turns in history): the Qwen/Hermes
//   tool-calling ChatML dialect — `<tools>` advertised in the system message, calls
//   replayed as `<tool_call>` turns, results as `<tool_response>` — which requires a
//   ChatML-speaking model. Adapted from the coreai-model-zoo ZooFMProvider.
//
// Reasoning entries are never replayed (matches upstream chat templates, which strip
// historic thinking).

import Foundation
import FoundationModels
import Tokenizers

enum TranscriptRenderer {
    struct Rendered {
        let tokens: [Int32]
        let usedToolDialect: Bool
    }

    static func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool,
        tokenizer: any Tokenizer,
        supportsHermesTools: Bool
    ) throws -> Rendered {
        let hasToolHistory = transcript.contains { entry in
            switch entry {
            case .toolCalls, .toolOutput: return true
            default: return false
            }
        }

        guard !tools.isEmpty || hasToolHistory else {
            let messages = templateMessages(from: transcript)
            let tokens = try tokenizer.applyChatTemplate(messages: messages).map(Int32.init)
            return Rendered(tokens: tokens, usedToolDialect: false)
        }

        guard supportsHermesTools else {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .toolCalling,
                    debugDescription:
                        "Tool calling uses the Hermes ChatML dialect, which needs a ChatML "
                        + "model that is not LFM-family (LFM bundles emit a pythonic call "
                        + "syntax this path does not parse). This bundle does not qualify."))
        }

        let text = renderHermesChatML(
            transcript: transcript, tools: tools, requireToolCall: requireToolCall)
        let tokens = tokenizer.encode(text: text).map(Int32.init)
        return Rendered(tokens: tokens, usedToolDialect: true)
    }

    // MARK: - Template path (no tools)

    private static func templateMessages(
        from transcript: Transcript
    ) -> [[String: any Sendable]] {
        var messages: [[String: any Sendable]] = []
        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                messages.append(
                    ["role": "system", "content": segmentsText(instructions.segments)])
            case .prompt(let prompt):
                messages.append(["role": "user", "content": segmentsText(prompt.segments)])
            case .response(let response):
                messages.append(
                    ["role": "assistant", "content": segmentsText(response.segments)])
            default:
                continue  // reasoning is not replayed; tool entries never reach this path
            }
        }
        return messages
    }

    // MARK: - Hermes ChatML path (tools)

    private static func renderHermesChatML(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String {
        var system = "You are a helpful assistant."
        var body = ""

        func append(role: String, _ content: String) {
            body += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                system = segmentsText(instructions.segments)
            case .prompt(let prompt):
                append(role: "user", segmentsText(prompt.segments))
            case .response(let response):
                append(role: "assistant", segmentsText(response.segments))
            case .toolCalls(let calls):
                let rendered = calls.map { call in
                    "<tool_call>\n{\"name\": \"\(jsonEscaped(call.toolName))\", "
                        + "\"arguments\": \(call.arguments.jsonString)}\n</tool_call>"
                }.joined(separator: "\n")
                append(role: "assistant", rendered)
            case .toolOutput(let output):
                append(
                    role: "user",
                    "<tool_response>\n\(segmentsText(output.segments))\n</tool_response>")
            default:
                continue
            }
        }

        let head =
            "<|im_start|>system\n"
            + systemContent(base: system, tools: tools, requireToolCall: requireToolCall)
            + "<|im_end|>\n"
        return head + body + "<|im_start|>assistant\n"
    }

    /// Appends the `<tools>` block to the system message. No tools — no block: a
    /// plain-chat prompt is byte-identical to one rendered without tool support.
    private static func systemContent(
        base: String,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String {
        guard !tools.isEmpty else { return base }
        var lines: [String] = []
        for tool in tools {
            let schemaJSON: String
            if let data = try? JSONEncoder().encode(tool.parameters),
                let s = String(data: data, encoding: .utf8)
            {
                schemaJSON = s
            } else {
                schemaJSON = "{}"
            }
            lines.append(
                #"{"type": "function", "function": {"name": "\#(jsonEscaped(tool.name))", "description": "\#(jsonEscaped(tool.description))", "parameters": \#(schemaJSON)}}"#
            )
        }
        let requirement =
            requireToolCall
            ? "\nYou MUST respond with a function call; do not answer directly.\n"
            : ""
        return base + """


            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            \(lines.joined(separator: "\n"))
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": <function-name>, "arguments": <args-json-object>}
            </tool_call>
            \(requirement)
            """
    }

    private static func segmentsText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }

    /// Escapes a string for embedding inside a JSON string literal. Tool names and
    /// (natural-language) descriptions are interpolated into the `<tools>` JSON and the
    /// replayed `<tool_call>` JSON, so a `"` or newline must be escaped or the block
    /// becomes invalid JSON.
    private static func jsonEscaped(_ s: String) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    out += String(format: "\\u%04x", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out
    }
}
