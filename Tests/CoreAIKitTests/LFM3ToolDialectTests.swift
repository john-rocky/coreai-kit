// LFM3ToolDialectTests.swift — LFM3's tool dialect against the model's own chat template. The goldens are
// transformers 5.20.2 `apply_chat_template` of LiquidAI/LFM3-1.7B's chat_template.jinja (sha256 f45005c5…)
// on the same conversations, split at turn boundaries; the parse cases follow the tokenizer config's
// `response_template` (one newline of layout on each side of a value, strings kept as written).

import Foundation
import FoundationModels
import Testing

@testable import CoreAIKit

@available(macOS 27, iOS 27, *)
private struct LookupSecretTool: Tool {
    let name = "lookup_secret"
    let description = "Returns the secret value stored under a key in the user's vault."

    @Generable
    struct Arguments {
        @Guide(description: "Vault key")
        var key: String
    }

    func call(arguments: Arguments) async throws -> String { "48213" }
}

@available(macOS 27, iOS 27, *)
private struct CreateEventTool: Tool {
    let name = "create_event"
    let description = "Creates a calendar event."

    @Generable
    struct Arguments {
        @Guide(description: "Event title")
        var title: String
        @Guide(description: "Start hour, 0-23")
        var start_hour: Int
        @Guide(description: "Length in hours")
        var duration_hours: Double
        @Guide(description: "Whole-day event")
        var all_day: Bool
        @Guide(description: "Attendee names")
        var attendees: [String]
    }

    func call(arguments: Arguments) async throws -> String { "created" }
}

// The system turn both tool goldens share: the instructions, a blank line, one json.dumps line per tool.
private let toolsSystem =
    "<|im_start|>system\nYou are a helpful assistant.\n\n<tools>\n"
    + "{\"name\": \"lookup_secret\", \"description\": \"Returns the secret value stored under a key in the user's vault.\", \"parameters\": {\"type\": \"object\", \"title\": \"Arguments\", \"properties\": {\"key\": {\"type\": \"string\", \"description\": \"Vault key\"}}, \"required\": [\"key\"], \"additionalProperties\": false, \"x-order\": [\"key\"]}}\n"
    + "{\"name\": \"create_event\", \"description\": \"Creates a calendar event.\", \"parameters\": {\"type\": \"object\", \"title\": \"Arguments\", \"properties\": {\"title\": {\"type\": \"string\", \"description\": \"Event title\"}, \"start_hour\": {\"type\": \"integer\", \"description\": \"Start hour, 0-23\"}, \"duration_hours\": {\"type\": \"number\", \"description\": \"Length in hours\"}, \"all_day\": {\"type\": \"boolean\", \"description\": \"Whole-day event\"}, \"attendees\": {\"type\": \"array\", \"description\": \"Attendee names\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"title\", \"start_hour\", \"duration_hours\", \"all_day\", \"attendees\"], \"additionalProperties\": false, \"x-order\": [\"title\", \"start_hour\", \"duration_hours\", \"all_day\", \"attendees\"]}}\n"
    + "</tools><|im_end|>\n"

private let toolQuestion =
    "Use the lookup_secret tool to read the value stored under the key \"alpha\", then tell me that value."
private let twoCallQuestion =
    "Add a half-hour standup at 9 with Ann and Bo, then read the secret under the key \"beta\"."

/// transformers' renders (ids: 359 / 367 / 429 / 105 / 22).
private let goldens: [String: String] = [
    "tool_roundtrip_off":
        "<|startoftext|>" + toolsSystem
        + "<|im_start|>user\n\(toolQuestion)<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think><tool_call>\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\n48213\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>",
    "tool_roundtrip_on":
        "<|startoftext|>" + toolsSystem
        + "<|im_start|>user\n\(toolQuestion)<|im_end|>\n"
        + "<|im_start|>assistant\n<think>I should call the tool with key alpha.</think><tool_call>\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\n48213\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n<think>",
    "tool_twocalls_off":
        "<|startoftext|>" + toolsSystem
        + "<|im_start|>user\n\(twoCallQuestion)<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think><tool_call>\n<function=create_event>\n<parameter=title>\nStandup\n</parameter>\n<parameter=start_hour>\n9\n</parameter>\n<parameter=duration_hours>\n0.5\n</parameter>\n<parameter=all_day>\nfalse\n</parameter>\n<parameter=attendees>\n[\"Ann\", \"Bo\"]\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=lookup_secret>\n<parameter=key>\nbeta\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\ncreated\n</tool_response><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\n77120\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>",
    "multiturn3_on":
        "<|startoftext|>"
        + "<|im_start|>user\nRemember this code word for later: TANGERINE-7319. Reply with one short sentence.<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>Nice to meet you. I will remember it.<|im_end|>\n"
        + "<|im_start|>user\nWhat is 12 + 30? Answer briefly.<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>12 + 30 = 42.<|im_end|>\n"
        + "<|im_start|>user\nWhat was the code word I gave you at the start? Answer with the code word only.<|im_end|>\n"
        + "<|im_start|>assistant\n<think>",
    "q1_off":
        "<|startoftext|>"
        + "<|im_start|>user\nWhat is 17 + 25? Answer briefly.<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>",
]

// Availability goes on each test, not on the suite: Swift Testing refuses @Test inside an @available type.
struct LFM3ToolDialectTests {
    @available(macOS 27, iOS 27, *)
    private var tools: [Transcript.ToolDefinition] {
        [Transcript.ToolDefinition(tool: LookupSecretTool()), Transcript.ToolDefinition(tool: CreateEventTool())]
    }

    @available(macOS 27, iOS 27, *)
    private func text(_ s: String) -> [Transcript.Segment] { [.text(.init(content: s))] }

    @available(macOS 27, iOS 27, *)
    private func call(_ id: String, _ name: String, _ json: String) throws -> Transcript.ToolCall {
        Transcript.ToolCall(id: id, toolName: name, arguments: try GeneratedContent(json: json))
    }

    @available(macOS 27, iOS 27, *)
    private func system() -> Transcript.Entry {
        .instructions(.init(segments: text("You are a helpful assistant."), toolDefinitions: tools))
    }

    @available(macOS 27, iOS 27, *)
    private func roundTrip(thinking: Bool) throws -> Transcript {
        var entries: [Transcript.Entry] = [system(), .prompt(.init(segments: text(toolQuestion)))]
        if thinking { entries.append(.reasoning(.init(segments: text("I should call the tool with key alpha.")))) }
        entries.append(.toolCalls(.init([try call("c1", "lookup_secret", #"{"key":"alpha"}"#)])))
        entries.append(.toolOutput(.init(id: "c1", toolName: "lookup_secret", segments: text("48213"))))
        return Transcript(entries: entries)
    }

    @available(macOS 27, iOS 27, *)
    private func twoCalls() throws -> Transcript {
        Transcript(entries: [
            system(),
            .prompt(.init(segments: text(twoCallQuestion))),
            .toolCalls(.init([
                // Keys out of declaration order on purpose: the replay follows the schema's x-order.
                try call(
                    "c1", "create_event",
                    #"{"attendees":["Ann","Bo"],"all_day":false,"title":"Standup","duration_hours":0.5,"start_hour":9}"#),
                try call("c2", "lookup_secret", #"{"key":"beta"}"#),
            ])),
            .toolOutput(.init(id: "c1", toolName: "create_event", segments: text("created"))),
            .toolOutput(.init(id: "c2", toolName: "lookup_secret", segments: text("77120"))),
        ])
    }

    @available(macOS 27, iOS 27, *)
    @Test func rendersLikeTransformers() throws {
        let multiTurn = Transcript(entries: [
            .prompt(.init(segments: text(
                "Remember this code word for later: TANGERINE-7319. Reply with one short sentence."))),
            .reasoning(.init(segments: text("The user gave a code word to remember."))),
            .response(.init(segments: text("Nice to meet you. I will remember it."))),
            .prompt(.init(segments: text("What is 12 + 30? Answer briefly."))),
            .reasoning(.init(segments: text("Simple addition: 12 plus 30 is 42."))),
            .response(.init(segments: text("12 + 30 = 42."))),
            .prompt(.init(segments: text(
                "What was the code word I gave you at the start? Answer with the code word only."))),
        ])
        let oneTurn = Transcript(entries: [.prompt(.init(segments: text("What is 17 + 25? Answer briefly.")))])
        let renders: [(String, String)] = [
            ("tool_roundtrip_off", LFM3ToolDialect.render(
                transcript: try roundTrip(thinking: false), tools: tools, requireToolCall: false, thinking: false)),
            ("tool_roundtrip_on", LFM3ToolDialect.render(
                transcript: try roundTrip(thinking: true), tools: tools, requireToolCall: false)),
            ("tool_twocalls_off", LFM3ToolDialect.render(
                transcript: try twoCalls(), tools: tools, requireToolCall: false, thinking: false)),
            ("multiturn3_on", LFM3ToolDialect.render(transcript: multiTurn, tools: [], requireToolCall: false)),
            ("q1_off", LFM3ToolDialect.render(
                transcript: oneTurn, tools: [], requireToolCall: false, thinking: false)),
        ]
        for (name, rendered) in renders {
            #expect(rendered == goldens[name], "\(name)")
        }
    }

    /// FoundationModels' own schema / arguments encoding changes key order per encode; the render must not.
    @available(macOS 27, iOS 27, *)
    @Test func reRenderIsByteStable() throws {
        let transcript = try twoCalls()
        let first = LFM3ToolDialect.render(transcript: transcript, tools: tools, requireToolCall: false)
        for _ in 0..<5 {
            #expect(LFM3ToolDialect.render(transcript: transcript, tools: tools, requireToolCall: false) == first)
        }
    }

    @available(macOS 27, iOS 27, *)
    private func parse(_ payload: String, tools: [Transcript.ToolDefinition]? = nil) throws -> [String] {
        try LFM3ToolDialect.parseToolCalls(payload, tools: tools ?? self.tools).map { "\($0.name) \($0.argumentsJSON)" }
    }

    @available(macOS 27, iOS 27, *)
    @Test func parsesOneArgument() throws {
        #expect(
            try parse("\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n")
                == [#"lookup_secret {"key":"alpha"}"#])
    }

    @available(macOS 27, iOS 27, *)
    @Test func parsesArgumentsTypedBySchema() throws {
        let payload =
            "\n<function=create_event>\n<parameter=title>\nStandup\n</parameter>\n<parameter=start_hour>\n9\n</parameter>\n"
            + "<parameter=duration_hours>\n0.5\n</parameter>\n<parameter=all_day>\nfalse\n</parameter>\n"
            + "<parameter=attendees>\n[\"Ann\", \"Bo\"]\n</parameter>\n</function>\n"
        #expect(
            try parse(payload) == [
                #"create_event {"all_day":false,"attendees":["Ann","Bo"],"duration_hours":0.5,"start_hour":9,"title":"Standup"}"#
            ])
    }

    @available(macOS 27, iOS 27, *)
    @Test func parsesTwoCallsInOneBlock() throws {
        let payload =
            "\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n"
            + "<function=lookup_secret>\n<parameter=key>\nbeta\n</parameter>\n</function>\n"
        #expect(try parse(payload) == [#"lookup_secret {"key":"alpha"}"#, #"lookup_secret {"key":"beta"}"#])
    }

    /// No schema for the tool: a JSON literal is its value; anything else, or a JSON string, is text.
    @available(macOS 27, iOS 27, *)
    @Test func parsesUntypedJSONValues() throws {
        let payload =
            "<function=configure>\n<parameter=n>\n3\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n"
            + "<parameter=flag>\ntrue\n</parameter>\n<parameter=none>\nnull\n</parameter>\n"
            + "<parameter=list>\n[1, 2]\n</parameter>\n<parameter=opts>\n{\"a\": {\"b\": [true]}}\n</parameter>\n"
            + "<parameter=word>\nTokyo Station\n</parameter>\n<parameter=quoted>\n\"x\"\n</parameter>\n</function>"
        #expect(
            try parse(payload, tools: []) == [
                #"configure {"flag":true,"list":[1,2],"n":3,"none":null,"opts":{"a":{"b":[true]}},"quoted":"\"x\"","ratio":1.5,"word":"Tokyo Station"}"#
            ])
    }

    /// Only one newline after the opening tag and one before `</parameter>` are layout: a string keeps the
    /// rest, a typed value is trimmed; whitespace between elements is ignored.
    @available(macOS 27, iOS 27, *)
    @Test func whitespaceAroundValues() throws {
        let payload =
            "\n\n  <function=create_event>\n\n<parameter=title>\n\n Two words \n\n</parameter>\n"
            + "<parameter=start_hour> 7 </parameter>\n\n</function>\n\n"
        #expect(try parse(payload) == [#"create_event {"start_hour":7,"title":"\n Two words \n"}"#])
    }

    @available(macOS 27, iOS 27, *)
    @Test func stringTypedValueStaysString() throws {
        #expect(
            try parse("<function=create_event>\n<parameter=title>\n123\n</parameter>\n</function>")
                == [#"create_event {"title":"123"}"#])
    }

    @available(macOS 27, iOS 27, *)
    @Test func truncatedBlockKeepsWhatArrived() throws {
        #expect(try parse("\n<function=lookup_secret>\n<parameter=key>\nalp") == [#"lookup_secret {"key":"alp"}"#])
    }

    @available(macOS 27, iOS 27, *)
    @Test func blockWithoutFunctionThrows() {
        #expect(throws: KitFMError.self) { try parse("{\"name\": \"lookup_secret\"}") }
    }
}
