// OutputProfile.swift — describes how a model family frames its raw output stream
// (thinking blocks, tool calls, channels), so one streaming parser serves every family.

import Tokenizers

public struct OutputRule: Sendable, Equatable {
    public enum Kind: Sendable, Equatable {
        case thinking
        case toolCall
        case response
    }

    public let open: String
    /// Any of these markers closes the section; empty = the section runs to end of stream.
    public let close: [String]
    public let kind: Kind
    /// Streaming sections emit body deltas as they arrive (thinking, response). Buffering
    /// sections accumulate and emit once at close (tool calls — the caller needs one
    /// complete JSON payload per call).
    public let streamsBody: Bool

    public init(open: String, close: [String], kind: Kind, streamsBody: Bool) {
        self.open = open
        self.close = close
        self.kind = kind
        self.streamsBody = streamsBody
    }
}

public struct OutputProfile: Sendable, Equatable {
    public enum DefaultText: Sendable, Equatable {
        /// Text outside any section streams out as response (qwen3, plain models).
        case stream
        /// Text outside any section is dropped (gpt-oss harmony: only channel bodies count).
        case suppress
    }

    public let rules: [OutputRule]
    public let defaultText: DefaultText

    public init(rules: [OutputRule], defaultText: DefaultText) {
        self.rules = rules
        self.defaultText = defaultText
    }

    /// No markup: every delta is response text.
    public static let plain = OutputProfile(rules: [], defaultText: .stream)

    /// Think-tag models (qwen3 family): `<think>…</think>` streams as thinking;
    /// `<tool_call>…</tool_call>` buffers to one payload per call.
    public static func thinkTags(
        open: String = "<think>", close: String = "</think>"
    ) -> OutputProfile {
        OutputProfile(
            rules: [
                OutputRule(open: open, close: [close], kind: .thinking, streamsBody: true),
                OutputRule(
                    open: "<tool_call>", close: ["</tool_call>"],
                    kind: .toolCall, streamsBody: false),
            ],
            defaultText: .stream)
    }

    /// gpt-oss "harmony" channels: the analysis channel streams as thinking, the final
    /// channel as response; everything between channel markers is dropped.
    public static let harmony = OutputProfile(
        rules: [
            OutputRule(
                open: "<|channel|>analysis<|message|>", close: ["<|end|>"],
                kind: .thinking, streamsBody: true),
            OutputRule(
                open: "<|channel|>final<|message|>",
                close: ["<|return|>", "<|end|>", "<|endoftext|>"],
                kind: .response, streamsBody: true),
        ],
        defaultText: .suppress)

    /// Picks the profile by probing the tokenizer vocab for marker tokens (markers only
    /// exist as dedicated tokens on models trained to emit them).
    public static func detect(probing tokenizer: any Tokenizer) -> OutputProfile {
        // On vocabs that define an UNK token (granite / llama family), convertTokenToId
        // maps ANY unknown string to the UNK id — a bare nil-check makes every probe hit,
        // misdetects harmony, and its .suppress default then swallows the whole reply.
        // Require the round trip: the id must map back to the probed marker.
        func has(_ token: String) -> Bool {
            guard let id = tokenizer.convertTokenToId(token) else { return false }
            return tokenizer.convertIdToToken(id) == token
        }
        if has("<|channel|>") { return .harmony }
        if has("<think>"), has("</think>") { return .thinkTags() }
        if has("<|reasoning_start|>"), has("<|reasoning_end|>") {
            return .thinkTags(open: "<|reasoning_start|>", close: "<|reasoning_end|>")
        }
        return .plain
    }
}
