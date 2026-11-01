// LFM3ToolDialect.swift — LFM3's tool-calling prompt and call syntax (LiquidAI/LFM3-1.7B), rendered
// string-for-string like the model's chat_template.jinja: transformers' `apply_chat_template` and this render
// agree on the 1-turn, 3-turn and tool round-trip conversations, thinking on and off. Adapted from the
// coreai-model-zoo ZooFMProvider's LFM3Dialect (see NOTICE.txt).
//
// * tools advertised in the system message inside `<tools>…</tools>`, one
//   `{"name": …, "description": …, "parameters": …}` object per line in Python `json.dumps` form, after the
//   instructions and a blank line
// * calls as XML, one `<tool_call>` block per call:
//   `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`, blocks
//   joined by a blank line; a string VALUE is raw, any other value JSON
// * results replayed as `tool`-role turns, `<tool_response>\n…\n</tool_response>`
// * the generation prompt ends with `<think>` (thinking on, the template's default) or `<think></think>`;
//   an assistant turn after the last user turn (the tool loop in progress) replays its reasoning, an earlier
//   one renders `<think></think>` and its answer
// * the BOS `<|startoftext|>` is part of the rendered text (the tokenizer's post-processor adds none)
//
// FoundationModels encodes a `GenerationSchema` and a `GeneratedContent` with a different key order on each
// encode, so the render fixes one: schema keys `type`, `title`, `description`, `properties`, `items`,
// `required`, then the rest sorted; `properties` and call arguments in the schema's declaration order
// (`x-order`). A re-render is byte-stable, which the KV prefix reuse depends on.

import Foundation
import FoundationModels
import Tokenizers

@available(macOS 27, iOS 27, *)
enum LFM3ToolDialect {
    /// LFM3 carries its call syntax as single `<function=` / `<parameter=` tokens (it also keeps LFM2.5's
    /// `<|tool_call_start|>` pair, which is why the Hermes path refuses it). Round trip guards the UNK-vocab
    /// false positive.
    static func probe(_ tokenizer: any Tokenizer) -> Bool {
        func has(_ token: String) -> Bool {
            guard let id = tokenizer.convertTokenToId(token) else { return false }
            return tokenizer.convertIdToToken(id) == token
        }
        return has("<function=") && has("<parameter=")
    }

    /// One template message: consecutive reasoning / response / tool-call entries form a single assistant
    /// turn, as the template renders reasoning, answer and calls in one `<|im_start|>assistant` block.
    private struct Turn {
        let role: String
        var content = ""
        var reasoning: String? = nil
        var calls: [Transcript.ToolCall] = []
    }

    static func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool,
        thinking: Bool = true
    ) -> String {
        var system = ""
        var turns: [Turn] = []
        func assistantTurn() -> Int {
            if let last = turns.indices.last, turns[last].role == "assistant" { return last }
            turns.append(Turn(role: "assistant"))
            return turns.count - 1
        }
        for (index, entry) in transcript.enumerated() {
            switch entry {
            case .instructions(let instructions):
                // The template lifts a leading system message only; a later one stays a turn in place.
                if index == 0 {
                    system = segmentsText(instructions.segments)
                } else {
                    turns.append(Turn(role: "system", content: segmentsText(instructions.segments)))
                }
            case .prompt(let prompt):
                turns.append(Turn(role: "user", content: segmentsText(prompt.segments)))
            case .toolOutput(let output):
                turns.append(Turn(role: "tool", content: segmentsText(output.segments)))
            case .reasoning(let reasoning):
                let i = assistantTurn()
                turns[i].reasoning = (turns[i].reasoning ?? "") + segmentsText(reasoning.segments)
            case .response(let response):
                turns[assistantTurn()].content += segmentsText(response.segments)
            case .toolCalls(let calls):
                turns[assistantTurn()].calls += calls
            default:
                continue
            }
        }

        var systemPrompt = system
        if !tools.isEmpty {
            let lines = tools.map(toolLine)
            systemPrompt += (systemPrompt.isEmpty ? "" : "\n\n") + "<tools>\n" + lines.joined(separator: "\n")
                + "\n</tools>"
            if requireToolCall {
                systemPrompt += "\nYou MUST call a tool to answer; do not answer directly."
            }
        }
        var out = "<|startoftext|>"
        if !systemPrompt.isEmpty {
            out += "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n"
        }
        let lastUser = turns.lastIndex { $0.role == "user" } ?? -1
        for (index, turn) in turns.enumerated() {
            out += "<|im_start|>" + turn.role + "\n"
            switch turn.role {
            case "assistant":
                out += assistantBody(turn, keepThinking: index > lastUser, tools: tools)
            case "tool":
                out += turn.content.contains("<tool_response>")
                    ? turn.content : "<tool_response>\n" + turn.content + "\n</tool_response>"
            default:
                out += turn.content
            }
            out += "<|im_end|>\n"
        }
        return out + "<|im_start|>assistant\n" + (thinking ? "<think>" : "<think></think>")
    }

    /// The template's assistant block: reasoning kept only after the last user turn; before it, the answer
    /// after any inline `</think>`, prefixed by an empty think block.
    private static func assistantBody(
        _ turn: Turn, keepThinking: Bool, tools: [Transcript.ToolDefinition]
    ) -> String {
        var out = ""
        var content = turn.content
        if keepThinking {
            if let reasoning = turn.reasoning {
                out += "<think>" + reasoning + "</think>"
            } else if !content.contains("</think>") {
                out += "<think></think>"
            }
        } else {
            if content.contains("</think>") {
                content = content.components(separatedBy: "</think>").last!
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            content = "<think></think>" + content
        }
        out += content
        out += turn.calls.map { callXML($0, tools: tools) }.joined(separator: "\n\n")
        return out
    }

    /// One replayed call, arguments in the tool's declaration order (sorted when the tool is unknown).
    static func callXML(_ call: Transcript.ToolCall, tools: [Transcript.ToolDefinition]) -> String {
        var out = "<tool_call>\n<function=" + call.toolName + ">\n"
        if let arguments = jsonValue(call.arguments.jsonString) as? [String: Any] {
            let declared = schemaObject(tools.first { $0.name == call.toolName })?["x-order"] as? [String] ?? []
            for key in ordered(arguments.keys, declared: declared) {
                out += "<parameter=" + key + ">\n" + parameterText(arguments[key]!) + "\n</parameter>\n"
            }
        }
        return out + "</function>\n</tool_call>"
    }

    /// The template's `render_value`: a string raw, anything else as `json.dumps` writes it.
    private static func parameterText(_ value: Any) -> String {
        if let s = value as? String { return s }
        return pythonJSON(value)
    }

    /// `{"name": …, "description": …, "parameters": …}` — the template's `tool_to_json`.
    private static func toolLine(_ tool: Transcript.ToolDefinition) -> String {
        #"{"name": "# + pythonJSONString(tool.name) + #", "description": "#
            + pythonJSONString(tool.description) + #", "parameters": "#
            + schemaText(schemaObject(tool) ?? [:]) + "}"
    }

    private static func schemaObject(_ tool: Transcript.ToolDefinition?) -> [String: Any]? {
        guard let tool, let data = try? JSONEncoder().encode(tool.parameters) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static let schemaKeyOrder = ["type", "title", "description", "properties", "items", "required"]

    /// A JSON-schema value in `json.dumps` form with the key order described at the top of the file.
    private static func schemaText(_ value: Any) -> String {
        switch value {
        case let object as [String: Any]:
            let first = schemaKeyOrder.filter { object[$0] != nil }
            let keys = first + object.keys.filter { !schemaKeyOrder.contains($0) }.sorted()
            return "{" + keys.map { key -> String in
                let child = object[key]!
                if key == "properties", let properties = child as? [String: Any] {
                    let declared = object["x-order"] as? [String] ?? []
                    let members = ordered(properties.keys, declared: declared).map {
                        pythonJSONString($0) + ": " + schemaText(properties[$0]!)
                    }
                    return pythonJSONString(key) + ": {" + members.joined(separator: ", ") + "}"
                }
                return pythonJSONString(key) + ": " + schemaText(child)
            }.joined(separator: ", ") + "}"
        case let array as [Any]:
            return "[" + array.map(schemaText).joined(separator: ", ") + "]"
        default:
            return pythonJSON(value)
        }
    }

    /// The declared keys that are present, then the rest sorted.
    private static func ordered(_ keys: Dictionary<String, Any>.Keys, declared: [String]) -> [String] {
        let present = Set(keys)
        let first = declared.filter { present.contains($0) }
        return first + present.subtracting(first).sorted()
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

    // MARK: - Parsing

    /// Parses the body of ONE `<tool_call>` block (markers stripped by the stream parser): `<function=NAME>`
    /// then `<parameter=KEY>` elements, as the tokenizer config's `response_template` reads them — one newline
    /// after the opening tag and one before `</parameter>` are layout, the rest of the value is kept. A
    /// string-typed parameter (per the tool's schema) stays text; any other declared type is read as JSON (a
    /// value that does not parse stays text, and the framework's schema decode reports the mismatch); without
    /// a schema hint, a JSON literal other than a string is that value. Several `<function=` elements in one
    /// block become several calls; a block without one throws.
    static func parseToolCalls(
        _ payload: String, tools: [Transcript.ToolDefinition]
    ) throws -> [KitExecutor.ParsedToolCall] {
        let openers = payload.matches(of: /<function=([^\n>]+)>/)
        guard !openers.isEmpty else {
            throw KitFMError.malformedToolCall(payload: payload.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        var calls: [KitExecutor.ParsedToolCall] = []
        for (i, opener) in openers.enumerated() {
            let name = String(opener.1)
            let limit = i + 1 < openers.count ? openers[i + 1].range.lowerBound : payload.endIndex
            var body = payload[opener.range.upperBound..<limit]
            if let close = body.range(of: "</function>", options: .backwards) {
                body = body[..<close.lowerBound]
            }
            let properties =
                schemaObject(tools.first { $0.name == name })?["properties"] as? [String: Any] ?? [:]
            var arguments: [String: Any] = [:]
            let params = body.matches(of: /<parameter=([^\n>]+)>/)
            for (j, param) in params.enumerated() {
                let end = j + 1 < params.count ? params[j + 1].range.lowerBound : body.endIndex
                var raw = body[param.range.upperBound..<end]
                if let close = raw.range(of: "</parameter>", options: .backwards) {
                    raw = raw[..<close.lowerBound]
                }
                var text = String(raw)
                if text.hasPrefix("\n") { text.removeFirst() }
                if text.hasSuffix("\n") { text.removeLast() }
                let key = String(param.1)
                let declared = (properties[key] as? [String: Any])?["type"] as? String
                arguments[key] = typedValue(text, declared: declared)
            }
            guard
                let data = try? JSONSerialization.data(
                    withJSONObject: arguments, options: [.sortedKeys, .withoutEscapingSlashes]),
                let json = String(data: data, encoding: .utf8)
            else {
                throw KitFMError.malformedToolCall(payload: String(body))
            }
            calls.append(KitExecutor.ParsedToolCall(name: name, argumentsJSON: json))
        }
        return calls
    }

    private static func typedValue(_ text: String, declared: String?) -> Any {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        switch declared {
        case "string":
            return text
        case "integer":
            if let n = Int(trimmed) { return n }
            if let d = Double(trimmed), d.rounded() == d, abs(d) < 9.0e15 { return Int(d) }
            return text
        case "boolean":
            switch trimmed.lowercased() {
            case "true": return true
            case "false": return false
            default: return text
            }
        case "number", "array", "object", "null":
            return jsonValue(trimmed) ?? text
        default:
            if let value = jsonValue(trimmed), !(value is String) { return value }
            return text
        }
    }
}

// MARK: - JSON in Python's `json.dumps(ensure_ascii=False)` form (transformers' `tojson`)

/// Parses JSON text (fragments allowed); nil when it does not parse.
private func jsonValue(_ text: String) -> Any? {
    guard let data = text.data(using: .utf8) else { return nil }
    return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
}

/// `json.dumps(value, ensure_ascii=False)` for a JSONSerialization value; object keys sorted (the value
/// carries no order of its own).
private func pythonJSON(_ value: Any) -> String {
    switch value {
    case let s as String:
        return pythonJSONString(s)
    case is NSNull:
        return "null"
    case let n as NSNumber:
        // JSONSerialization booleans are CFBoolean-backed (objCType "c" is shared with Int8).
        if CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue ? "true" : "false" }
        // Swift's shortest round-trip double text is Python's repr for finite values (0.5, 2.0, 1e-07).
        return CFNumberIsFloatType(n as CFNumber) ? "\(n.doubleValue)" : n.stringValue
    case let array as [Any]:
        return "[" + array.map(pythonJSON).joined(separator: ", ") + "]"
    case let object as [String: Any]:
        return "{" + object.keys.sorted().map { pythonJSONString($0) + ": " + pythonJSON(object[$0]!) }
            .joined(separator: ", ") + "}"
    default:
        return pythonJSONString("\(value)")
    }
}

/// A JSON string literal as `json.dumps(ensure_ascii=False)` writes it: `"` `\` and the C0 controls
/// escaped (`\n \r \t \b \f`, else `\u00xx`), everything else — `/` and non-ASCII included — as is.
private func pythonJSONString(_ s: String) -> String {
    var out = "\""
    for scalar in s.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        case "\u{8}": out += "\\b"
        case "\u{C}": out += "\\f"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out + "\""
}
