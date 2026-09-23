// SystemOneMCP.swift — typed decisions as tools of a Model Context Protocol server: the
// JSON-RPC messages an MCP client (Claude Code, Codex, Cursor, an inspector) sends over stdio
// and the replies, so a coding agent calls the model on this machine — `decide` takes the
// `/v1/systemone` request form as its arguments and answers in the response form; `models`
// lists the catalog ids that can answer. `SystemOneMCPServer` puts this on stdin/stdout
// (`systemone mcp`); this file only reads and writes the messages.
//
// Two revisions of the protocol are spoken. Clients up to revision 2025-11-25 open with an
// `initialize` handshake and then send `tools/list` / `tools/call` (Claude Code 2.1 and
// Codex 0.154 do); revision 2026-07-28 removed the handshake — every request carries its
// protocol version in `params._meta`, and `server/discover` replaces `initialize`. A request
// is served in the form it arrived in (the spec's "dual-era" server).
//
//   {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "decide",
//    "arguments": {"state": "Help! My payouts have been failing for 3 days.",
//                  "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}}}
//   → {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "{…}"}],
//      "structuredContent": {"model": "minicpm5-2b", "answers": {"urgent": {"type": "noul", "noul": 0.9274, …}}, …}}}

import Foundation

public struct SystemOneMCP: Sendable {
    /// Revisions that open with `initialize`, newest first — the reply names the client's own
    /// revision when it is one of these, else the first.
    public static let legacyVersions = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
    /// Revisions that carry the version on every request (`server/discover`, no handshake).
    public static let modernVersions = ["2026-07-28"]

    /// What the client shows its model about this server.
    public static let instructions = """
        The tools call a typed-decision model on this machine: a state (text, or JSON) and \
        questions with fixed answers in — choice (which of these), score (where on this ordered \
        scale), noul (does this hold) — and each answer's probability out. Nothing is generated \
        and nothing leaves the machine. Use `decide` when you need a label from a fixed set, a \
        level on a scale, or a calibrated yes/no for one text or many; put everything the model \
        must read in `state`, name each option in `criteria` with what it means, and ask several \
        questions of one state in one call.
        """

    public static let toolNames = ["decide", "models"]

    /// `serverInfo`: what the client records about this server.
    public let serverInfo: JSONValue

    /// - Parameters:
    ///   - name: the server's name in `serverInfo`.
    ///   - version: the binary's version (`systemone --version`); "dev" from a source build.
    public init(name: String = "systemone", version: String = "dev", title: String = "System One on device (CoreAIKit)") {
        serverInfo = .object([
            .init("name", .string(name)),
            .init("title", .string(title)),
            .init("version", .string(version)),
        ])
    }

    // MARK: - Messages

    /// One JSON-RPC request, with what the routing needs.
    public struct Request: Sendable, Equatable {
        public let id: JSONValue
        public let method: String
        public let params: JSONValue?
        /// `params._meta["io.modelcontextprotocol/protocolVersion"]` — set on a 2026-07-28 request.
        public let protocolVersion: String?
        /// True for the 2026-07-28 form; its results carry `resultType` and the server's `_meta`.
        public var modern: Bool { protocolVersion != nil }

        public init(id: JSONValue, method: String, params: JSONValue?, protocolVersion: String?) {
            self.id = id
            self.method = method
            self.params = params
            self.protocolVersion = protocolVersion
        }
    }

    public enum Incoming: Sendable, Equatable {
        case request(Request)
        case notification(method: String, params: JSONValue?)
        /// A reply to a request — this server sends none, so nothing is owed.
        case response
        /// Not a JSON-RPC 2.0 message; the value is the error to write back (id null).
        case malformed(JSONValue)
    }

    /// One line of stdin.
    public static func incoming(_ line: String) -> Incoming {
        let root: JSONValue
        do {
            root = try JSONValue.parse(line)
        } catch let parseError {
            return .malformed(error(id: .null, code: -32700, message: "Parse error: \(parseError.localizedDescription)"))
        }
        guard root.members != nil, root["jsonrpc"]?.stringValue == "2.0" else {
            return .malformed(error(id: root["id"] ?? .null, code: -32600, message: "Invalid Request: not a JSON-RPC 2.0 message"))
        }
        let params = root["params"]
        if let method = root["method"]?.stringValue {
            guard let id = root["id"], id != .null else {
                return .notification(method: method, params: params)
            }
            guard params == nil || params?.members != nil else {
                return .malformed(error(id: id, code: -32600, message: "Invalid Request: 'params' must be an object"))
            }
            let version = params?["_meta"]?["io.modelcontextprotocol/protocolVersion"]?.stringValue
            return .request(Request(id: id, method: method, params: params, protocolVersion: version))
        }
        if root["result"] != nil || root["error"] != nil { return .response }
        return .malformed(error(id: root["id"] ?? .null, code: -32600, message: "Invalid Request: no 'method'"))
    }

    // MARK: - Routing

    public enum Route: Sendable, Equatable {
        /// Write this message.
        case reply(JSONValue)
        /// Run the tool, then write `callResult` or `callError` for `request`.
        case call(tool: String, arguments: JSONValue, request: Request)
        /// `notifications/cancelled` for this request id: if it has not started, drop it; if it
        /// has, write nothing for it.
        case cancelled(id: JSONValue)
        /// Nothing to write.
        case none
    }

    public func route(_ incoming: Incoming) -> Route {
        switch incoming {
        case .malformed(let reply): return .reply(reply)
        case .response: return .none
        case .notification(let method, let params):
            if method == "notifications/cancelled", let id = params?["requestId"] { return .cancelled(id: id) }
            return .none
        case .request(let request):
            if let version = request.protocolVersion {
                guard Self.modernVersions.contains(version) || Self.legacyVersions.contains(version) else {
                    return .reply(Self.error(
                        id: request.id, code: -32022, message: "Unsupported protocol version",
                        data: .object([
                            .init("supported", .array((Self.modernVersions + Self.legacyVersions).map { .string($0) })),
                            .init("requested", .string(version)),
                        ])))
                }
                guard request.params?["_meta"]?["io.modelcontextprotocol/clientCapabilities"] != nil else {
                    return .reply(Self.error(
                        id: request.id, code: -32602,
                        message: "Invalid params: _meta['io.modelcontextprotocol/clientCapabilities'] is required"))
                }
            }
            switch request.method {
            case "initialize":
                return .reply(initializeResult(id: request.id, requested: request.params?["protocolVersion"]?.stringValue))
            case "server/discover":
                return .reply(discoverResult(id: request.id))
            case "ping":
                return .reply(result(id: request.id, [], modern: request.modern))
            case "tools/list":
                return .reply(toolsListResult(id: request.id, modern: request.modern))
            case "tools/call":
                guard let name = request.params?["name"]?.stringValue else {
                    return .reply(Self.error(id: request.id, code: -32602, message: "Invalid params: 'name' is required"))
                }
                guard Self.toolNames.contains(name) else {
                    return .reply(Self.error(id: request.id, code: -32602, message: "Unknown tool: \(name)"))
                }
                let arguments = request.params?["arguments"] ?? .object([])
                guard arguments.members != nil else {
                    return .reply(Self.error(id: request.id, code: -32602, message: "Invalid params: 'arguments' must be an object"))
                }
                return .call(tool: name, arguments: arguments, request: request)
            default:
                return .reply(Self.error(id: request.id, code: -32601, message: "Method not found: \(request.method)"))
            }
        }
    }

    // MARK: - Replies

    /// A result. The 2026-07-28 form adds `resultType` and the server's identity in `_meta`.
    public func result(id: JSONValue, _ members: [JSONValue.Member], modern: Bool) -> JSONValue {
        var body = members
        if modern {
            body.insert(.init("resultType", .string("complete")), at: 0)
            body.append(.init("_meta", .object([.init("io.modelcontextprotocol/serverInfo", serverInfo)])))
        }
        return .object([.init("jsonrpc", .string("2.0")), .init("id", id), .init("result", .object(body))])
    }

    public static func error(id: JSONValue, code: Int, message: String, data: JSONValue? = nil) -> JSONValue {
        var body: [JSONValue.Member] = [.init("code", .int(code)), .init("message", .string(message))]
        if let data { body.append(.init("data", data)) }
        return .object([.init("jsonrpc", .string("2.0")), .init("id", id), .init("error", .object(body))])
    }

    /// The `initialize` reply: the client's revision when this server speaks it, else the
    /// newest legacy one (the client disconnects if it cannot follow).
    public func initializeResult(id: JSONValue, requested: String?) -> JSONValue {
        let version = requested.flatMap { Self.legacyVersions.contains($0) ? $0 : nil } ?? Self.legacyVersions[0]
        return result(id: id, [
            .init("protocolVersion", .string(version)),
            .init("capabilities", .object([.init("tools", .object([.init("listChanged", .bool(false))]))])),
            .init("serverInfo", serverInfo),
            .init("instructions", .string(Self.instructions)),
        ], modern: false)
    }

    /// The `server/discover` reply (2026-07-28): every revision spoken, newest first.
    public func discoverResult(id: JSONValue) -> JSONValue {
        result(id: id, [
            .init("supportedVersions", .array((Self.modernVersions + Self.legacyVersions).map { .string($0) })),
            .init("capabilities", .object([.init("tools", .object([]))])),
            .init("instructions", .string(Self.instructions)),
            .init("ttlMs", .int(3_600_000)),
            .init("cacheScope", .string("public")),
        ], modern: true)
    }

    public func toolsListResult(id: JSONValue, modern: Bool) -> JSONValue {
        var members: [JSONValue.Member] = [.init("tools", Self.tools)]
        if modern {
            members.append(.init("ttlMs", .int(3_600_000)))
            members.append(.init("cacheScope", .string("public")))
        }
        return result(id: id, members, modern: modern)
    }

    /// A tool's answer: the JSON as `structuredContent` and, for clients that read only text,
    /// serialized in a text block.
    public func callResult(id: JSONValue, structured: JSONValue, modern: Bool) -> JSONValue {
        result(id: id, [
            .init("content", .array([.object([.init("type", .string("text")), .init("text", .string(structured.dumps()))])])),
            .init("structuredContent", structured),
        ], modern: modern)
    }

    /// A tool that could not answer (a bad question, a model that would not load): the message
    /// goes to the model as text so it can correct the call. Not a protocol error.
    public func callError(id: JSONValue, message: String, modern: Bool) -> JSONValue {
        result(id: id, [
            .init("content", .array([.object([.init("type", .string("text")), .init("text", .string(message))])])),
            .init("isError", .bool(true)),
        ], modern: modern)
    }

    // MARK: - The tools

    /// `decide`'s arguments are the `/v1/systemone` request.
    public static func decideRequest(_ arguments: JSONValue) throws -> SystemOne.Request {
        try SystemOne.request(from: arguments)
    }

    /// `models`: the catalog ids `TypedDecisions` loads on this platform.
    public static func modelsResult(catalog: ModelCatalog = .builtin, default defaultID: String, loaded: String?) -> JSONValue {
        let entries = catalog.available().filter { TypedDecisions.supports($0) }
        return .object([
            .init("models", .array(entries.map { entry in
                var members: [JSONValue.Member] = [
                    .init("id", .string(entry.id)),
                    .init("name", .string(entry.name)),
                    .init("kind", .string(entry.kind.rawValue)),
                ]
                if let size = entry.variant?.sizeMB { members.append(.init("sizeMB", .int(size))) }
                if entry.id == defaultID { members.append(.init("default", .bool(true))) }
                if entry.id == loaded { members.append(.init("loaded", .bool(true))) }
                return .object(members)
            })),
            .init("default", .string(defaultID)),
            .init("loaded", loaded.map { .string($0) } ?? .null),
        ])
    }

    /// The `tools/list` entries. Schemas are JSON Schema 2020-12 (the protocol's default).
    public static let tools: JSONValue = try! JSONValue.parse(toolsJSON)  // swiftlint:disable:this force_try — checked by SystemOneMCPTests

    static let toolsJSON = """
        [
          {
            "name": "decide",
            "title": "Typed decisions on this machine",
            "description": "Ask typed questions about a text and get each answer's probability from a model on this machine. Nothing is generated and nothing leaves the machine. state: the text the questions are about (a ticket, a transcript, a document, a tool result), or a JSON object/array. questions: an object keyed by ids of your choosing, each {type, instructions, criteria}: choice (which of these; criteria = an object of option → what it means, 2–255 options (some models take fewer and say so), or an array of option names), score (where on this ordered scale; criteria = 2–10 level descriptions, lowest first), noul (does this hold, as a probability; criteria = optional {\\"true\\": …, \\"false\\": …}). Ask several questions of one state in one call. Returns answers under your ids — choice with every option's probability, score with its legend, noul as P(true) — with confidence, usage and timing_ms.",
            "inputSchema": {
              "type": "object",
              "properties": {
                "state": {
                  "description": "What the questions are about: text, or a JSON object/array (passed to the model as JSON, key order kept).",
                  "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}]
                },
                "questions": {
                  "type": "object",
                  "description": "The questions, keyed by ids of your choosing; the answers come back under the same ids.",
                  "minProperties": 1,
                  "additionalProperties": {
                    "type": "object",
                    "properties": {
                      "type": {"type": "string", "enum": ["choice", "score", "noul"]},
                      "instructions": {
                        "description": "The question, as you would put it to a careful reader.",
                        "anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}]
                      },
                      "criteria": {
                        "description": "choice: {option: what it means, …} (2–255) or [option, …]. score: [lowest level, …, highest] (2–10). noul: optional {\\"true\\": what makes it true, \\"false\\": what makes it false}.",
                        "anyOf": [{"type": "object"}, {"type": "array"}]
                      }
                    },
                    "required": ["type", "instructions"]
                  }
                },
                "model": {
                  "type": "string",
                  "description": "A catalog id from the models tool. Omit for the server's default; naming another id loads it (a few seconds)."
                }
              },
              "required": ["state", "questions"],
              "additionalProperties": false
            },
            "outputSchema": {
              "type": "object",
              "properties": {
                "model": {"type": "string"},
                "answers": {
                  "type": "object",
                  "additionalProperties": {
                    "type": "object",
                    "properties": {
                      "type": {"type": "string", "enum": ["choice", "score", "noul"]},
                      "choice": {"type": "string", "description": "the option with the highest probability"},
                      "probabilities": {"type": "object", "description": "choice: option → probability; score: level index → probability"},
                      "abstain": {"type": "number", "description": "a slot-head model's none-of-these mass (choice only, some models)"},
                      "score": {"type": "number", "description": "the expected level, 0 … levels − 1"},
                      "legend": {"type": "object", "description": "level index → its description"},
                      "noul": {"type": "number", "description": "P(the statement holds)"},
                      "confidence": {"type": "number", "description": "1 − normalised entropy (choice, score); max(p, 1 − p) (noul)"}
                    },
                    "required": ["type", "confidence"]
                  }
                },
                "usage": {
                  "type": "object",
                  "properties": {"input_tokens": {"type": "integer"}, "output_tokens": {"type": "integer"}}
                },
                "timing_ms": {"type": "number", "description": "this machine's wall clock for the decisions"}
              },
              "required": ["model", "answers", "usage", "timing_ms"]
            },
            "annotations": {"readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
          },
          {
            "name": "models",
            "title": "Models that can decide",
            "description": "The catalog ids the decide tool accepts on this machine, which one is the default and which one is loaded. Weights download on first use.",
            "inputSchema": {"type": "object", "additionalProperties": false},
            "outputSchema": {
              "type": "object",
              "properties": {
                "models": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "id": {"type": "string"}, "name": {"type": "string"},
                      "kind": {"type": "string", "description": "chat: a chat model read at its answer slot; decision: a model trained for these questions"},
                      "sizeMB": {"type": "integer"}, "default": {"type": "boolean"}, "loaded": {"type": "boolean"}
                    },
                    "required": ["id", "name", "kind"]
                  }
                },
                "default": {"type": "string"},
                "loaded": {"anyOf": [{"type": "string"}, {"type": "null"}]}
              },
              "required": ["models", "default"]
            },
            "annotations": {"readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
          }
        ]
        """
}
