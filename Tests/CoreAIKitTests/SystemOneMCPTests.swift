// SystemOneMCPTests.swift — the MCP messages without a model or a pipe: a stdin line parses
// into a request or a notification, the handshake and discovery replies have the spec's shape
// in both revisions, `tools/list` names the two tools with schemas, `tools/call` routes to the
// tool or refuses, and a tool's reply carries the JSON twice (structured, and as text).
// `SystemOneMCPServerTests` drives the server itself over pipes.

import Foundation
import Testing

@testable import CoreAIKit

struct SystemOneMCPTests {
    let codec = SystemOneMCP(version: "test")

    func line(_ json: String) -> SystemOneMCP.Route {
        codec.route(SystemOneMCP.incoming(json))
    }

    func reply(_ json: String) -> JSONValue? {
        if case .reply(let value) = line(json) { return value }
        return nil
    }

    // MARK: - Lines

    @Test func linesParseIntoRequestsNotificationsAndResponses() {
        // What Claude Code 2.1 sends first, id 0 and all.
        let first = SystemOneMCP.incoming(
            "{\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{}},\"jsonrpc\":\"2.0\",\"id\":0}")
        guard case .request(let request) = first else { Issue.record("not a request: \(first)"); return }
        #expect(request.id == .number("0"))
        #expect(request.method == "initialize")
        #expect(request.modern == false)

        #expect(SystemOneMCP.incoming("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")
            == .notification(method: "notifications/initialized", params: nil))
        #expect(SystemOneMCP.incoming("{\"jsonrpc\":\"2.0\",\"id\":\"abc\",\"result\":{}}") == .response)

        if case .request(let named) = SystemOneMCP.incoming("{\"jsonrpc\":\"2.0\",\"id\":\"r-1\",\"method\":\"ping\"}") {
            #expect(named.id == .string("r-1"))
        } else { Issue.record("string ids are ids") }
    }

    @Test func malformedLinesGetAJSONRPCError() {
        guard case .malformed(let parseError) = SystemOneMCP.incoming("{not json") else { Issue.record("no error"); return }
        #expect(parseError["error"]?["code"]?.doubleValue == -32700)
        #expect(parseError["id"] == .null)

        guard case .malformed(let notRPC) = SystemOneMCP.incoming("{\"id\": 7, \"method\": \"ping\"}") else { Issue.record("no error"); return }
        #expect(notRPC["error"]?["code"]?.doubleValue == -32600)
        #expect(notRPC["id"] == .number("7"))

        guard case .malformed(let noMethod) = SystemOneMCP.incoming("{\"jsonrpc\": \"2.0\", \"id\": 1}") else { Issue.record("no error"); return }
        #expect(noMethod["error"]?["code"]?.doubleValue == -32600)
    }

    // MARK: - Handshake (2025-11-25 and earlier)

    @Test func initializeAnswersInTheClientsRevision() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}"))
        #expect(value["id"] == .number("0"))
        let result = try #require(value["result"])
        #expect(result["protocolVersion"]?.stringValue == "2025-06-18")
        #expect(result["capabilities"]?["tools"]?["listChanged"] == .bool(false))
        #expect(result["serverInfo"]?["name"]?.stringValue == "systemone")
        #expect(result["serverInfo"]?["version"]?.stringValue == "test")
        #expect(result["instructions"]?.stringValue?.isEmpty == false)
        #expect(result["resultType"] == nil)  // not the 2026-07-28 form
    }

    @Test func initializeWithAnUnknownRevisionOffersTheNewestLegacyOne() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"1.0.0\",\"capabilities\":{}}}"))
        #expect(value["result"]?["protocolVersion"]?.stringValue == "2025-11-25")
    }

    @Test func notificationsAndPingsAreHandled() throws {
        #expect(line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}") == .none)
        #expect(line("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":5,\"reason\":\"user\"}}") == .cancelled(id: .number("5")))
        let pong = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":\"p\",\"method\":\"ping\"}"))
        #expect(pong["result"] == .object([]))
        #expect(pong["id"] == .string("p"))
    }

    @Test func unknownMethodsAreMethodNotFound() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"resources/list\"}"))
        #expect(value["error"]?["code"]?.doubleValue == -32601)
    }

    // MARK: - Tools

    @Test func toolsListNamesTheTwoToolsWithSchemas() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}"))
        let tools = try #require(value["result"]?["tools"]?.elements)
        #expect(tools.map { $0["name"]?.stringValue } == ["decide", "models"])
        let decide = tools[0]
        #expect(decide["inputSchema"]?["type"]?.stringValue == "object")
        #expect(decide["inputSchema"]?["required"]?.elements == [.string("state"), .string("questions")])
        #expect(decide["inputSchema"]?["properties"]?["questions"]?["additionalProperties"]?["required"]?.elements
            == [.string("type"), .string("instructions")])
        #expect(decide["outputSchema"]?["required"]?.elements?.contains(.string("answers")) == true)
        #expect(decide["annotations"]?["readOnlyHint"] == .bool(true))
        #expect(decide["description"]?.stringValue?.contains("noul") == true)
        let models = tools[1]
        #expect(models["inputSchema"] == .object([.init("type", .string("object")), .init("additionalProperties", .bool(false))]))
        #expect(value["result"]?["ttlMs"] == nil)  // legacy form: no cache fields
    }

    @Test func toolsCallRoutesToTheToolOrRefuses() throws {
        let call = line("{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"decide\",\"arguments\":{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"noul\",\"instructions\":\"Urgent?\"}}}}}")
        guard case .call(let tool, let arguments, let request) = call else { Issue.record("not routed: \(call)"); return }
        #expect(tool == "decide")
        #expect(request.id == .number("3"))
        let parsed = try SystemOneMCP.decideRequest(arguments)
        #expect(parsed.state == "s")
        #expect(parsed.questions.map(\.id) == ["q"])
        #expect(parsed.questions[0].question.kind == .noul(yes: nil, no: nil))

        let unknown = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"summarize\",\"arguments\":{}}}"))
        #expect(unknown["error"]?["code"]?.doubleValue == -32602)
        #expect(unknown["error"]?["message"]?.stringValue == "Unknown tool: summarize")

        let noArguments = line("{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"models\"}}")
        guard case .call("models", let empty, _) = noArguments else { Issue.record("models takes no arguments: \(noArguments)"); return }
        #expect(empty == .object([]))
    }

    @Test func badDecideArgumentsAreToolErrorsNotProtocolErrors() {
        // The transport maps a WireError to `isError: true` so the model can fix the call.
        #expect(throws: SystemOne.WireError.self) {
            try SystemOneMCP.decideRequest(try JSONValue.parse("{\"state\": \"s\"}"))
        }
        let value = codec.callError(id: .number("6"), message: "'questions' is empty", modern: false)
        #expect(value["result"]?["isError"] == .bool(true))
        #expect(value["result"]?["content"]?.elements?.first?["text"]?.stringValue == "'questions' is empty")
        #expect(value["error"] == nil)
    }

    @Test func aToolReplyCarriesTheJSONTwiceAndStaysOnOneLine() throws {
        let structured = try JSONValue.parse("{\"model\": \"m\", \"answers\": {\"q\": {\"type\": \"noul\", \"noul\": 0.9, \"confidence\": 0.9}}, \"note\": \"two\\nlines\"}")
        let value = codec.callResult(id: .number("7"), structured: structured, modern: false)
        let result = try #require(value["result"])
        #expect(result["structuredContent"] == structured)
        let text = try #require(result["content"]?.elements?.first?["text"]?.stringValue)
        #expect(try JSONValue.parse(text) == structured)
        #expect(result["isError"] == nil)
        #expect(!value.dumps().contains("\n"))
    }

    @available(macOS 27, iOS 27, *)
    @Test func modelsListsWhatTypedDecisionsLoads() throws {
        let value = SystemOneMCP.modelsResult(default: "minicpm5-2b", loaded: "decider-0.8b")
        let models = try #require(value["models"]?.elements)
        let ids = models.compactMap { $0["id"]?.stringValue }
        #expect(ids.contains("minicpm5-2b"))
        #expect(ids.contains("decider-0.8b"))
        #expect(!ids.contains("gemma-4-e2b-metal"))
        #expect(!ids.contains("gemma-4-e2b"))
        #expect(!ids.contains("whisper-v3-turbo"))
        #expect(models.first { $0["id"]?.stringValue == "minicpm5-2b" }?["default"] == .bool(true))
        #expect(models.first { $0["id"]?.stringValue == "decider-0.8b" }?["loaded"] == .bool(true))
        #expect(models.first { $0["id"]?.stringValue == "decider-0.8b" }?["kind"]?.stringValue == "decision")
        #expect(value["loaded"]?.stringValue == "decider-0.8b")
        #expect(SystemOneMCP.modelsResult(default: "minicpm5-2b", loaded: nil)["loaded"] == .null)
    }

    // MARK: - 2026-07-28 (no handshake; the version rides on every request)

    let meta = "\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\",\"io.modelcontextprotocol/clientCapabilities\":{}}"

    @Test func discoverListsEveryRevisionSpoken() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":\"d\",\"method\":\"server/discover\",\"params\":{\(meta)}}"))
        let result = try #require(value["result"])
        #expect(result["resultType"]?.stringValue == "complete")
        #expect(result["supportedVersions"]?.elements?.first == .string("2026-07-28"))
        #expect(result["supportedVersions"]?.elements?.contains(.string("2025-11-25")) == true)
        #expect(result["capabilities"]?["tools"] == .object([]))
        #expect(result["_meta"]?["io.modelcontextprotocol/serverInfo"]?["name"]?.stringValue == "systemone")
        #expect(result["cacheScope"]?.stringValue == "public")
        #expect(result["ttlMs"]?.doubleValue == 3_600_000)
    }

    @Test func modernRequestsGetModernResults() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"tools/list\",\"params\":{\(meta)}}"))
        let result = try #require(value["result"])
        #expect(result["resultType"]?.stringValue == "complete")
        #expect(result["tools"]?.elements?.count == 2)
        #expect(result["ttlMs"] != nil)
        #expect(result["cacheScope"]?.stringValue == "public")
        #expect(result["_meta"]?["io.modelcontextprotocol/serverInfo"]?["version"]?.stringValue == "test")

        let call = line("{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/call\",\"params\":{\"name\":\"models\",\(meta)}}")
        guard case .call(_, _, let request) = call else { Issue.record("not routed: \(call)"); return }
        #expect(request.modern)
        let answer = codec.callResult(id: request.id, structured: .object([]), modern: request.modern)
        #expect(answer["result"]?["resultType"]?.stringValue == "complete")
    }

    @Test func aRevisionNotSpokenIsRefusedWithTheList() throws {
        let value = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/list\",\"params\":{\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2031-01-01\",\"io.modelcontextprotocol/clientCapabilities\":{}}}}"))
        #expect(value["error"]?["code"]?.doubleValue == -32022)
        #expect(value["error"]?["data"]?["requested"]?.stringValue == "2031-01-01")
        #expect(value["error"]?["data"]?["supported"]?.elements?.first == .string("2026-07-28"))

        let noCapabilities = try #require(reply("{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/list\",\"params\":{\"_meta\":{\"io.modelcontextprotocol/protocolVersion\":\"2026-07-28\"}}}"))
        #expect(noCapabilities["error"]?["code"]?.doubleValue == -32602)
    }
}
