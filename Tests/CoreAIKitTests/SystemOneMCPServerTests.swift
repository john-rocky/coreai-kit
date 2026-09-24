// SystemOneMCPServerTests.swift — `SystemOneMCPServer` over two pipes and no model: the
// handshake, a ping, the tool list and a `models` call are answered one line each, a bad
// line gets its JSON-RPC error, the log never touches the output, and closing the input ends
// `run()` — after the call still in flight has answered (a one-shot pipe works).

import Foundation
import Testing

@testable import CoreAIKit

struct SystemOneMCPServerTests {
    /// Reads `count` lines from `handle` on a thread of its own; closes `writer` after
    /// `timeout` so a server that never answers ends the read instead of hanging the test.
    func collect(_ count: Int, from handle: FileHandle, closing writer: FileHandle, timeout: Duration = .seconds(20)) async -> [String] {
        let reader = Task.detached { () -> [String] in
            var buffer: [UInt8] = []
            var lines: [String] = []
            while lines.count < count {
                let chunk = handle.availableData
                if chunk.isEmpty { break }
                buffer.append(contentsOf: chunk)
                while let newline = buffer.firstIndex(of: 0x0A) {
                    lines.append(String(decoding: buffer[..<newline], as: UTF8.self))
                    buffer.removeSubrange(...newline)
                }
            }
            return lines
        }
        let guardTask = Task {
            try await Task.sleep(for: timeout)
            try? writer.close()
        }
        let lines = await reader.value
        guardTask.cancel()
        return lines
    }

    @available(macOS 27, iOS 27, *)
    @Test func answersOverPipesAndStopsWhenTheInputCloses() async throws {
        let input = Pipe()
        let output = Pipe()
        let logged = Logged()
        let server = SystemOneMCPServer(
            defaultModel: "minicpm5-2b", codec: SystemOneMCP(version: "test"),
            input: input.fileHandleForReading, output: output.fileHandleForWriting,
            log: { line in Task { await logged.append(line) } })
        let running = Task { try await server.run() }

        let messages = [
            "{\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}},\"jsonrpc\":\"2.0\",\"id\":0}",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}",
            "",  // a blank line is skipped, not answered
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}",
            "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"models\",\"arguments\":{}}}",
            "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"nope\"}}",
            "not json at all",
        ]
        try input.fileHandleForWriting.write(contentsOf: Data((messages.joined(separator: "\n") + "\n").utf8))
        try input.fileHandleForWriting.close()  // end of file right behind the last message, as a one-shot pipe does
        let replies = await collect(6, from: output.fileHandleForReading, closing: output.fileHandleForWriting)
        try await running.value  // returns once the input is at end of file and the `models` call has answered

        #expect(replies.count == 6)
        // A tool call is answered from a task of its own, so its reply may follow the errors
        // written straight from the loop: match replies by id, as a client does.
        var byID: [String: JSONValue] = [:]
        for reply in replies {
            #expect(!reply.contains("\n"))
            let value = try JSONValue.parse(reply)
            byID[(value["id"] ?? .null).dumps()] = value
        }
        let initialized = try #require(byID["0"])
        #expect(initialized["result"]?["protocolVersion"]?.stringValue == "2025-11-25")
        #expect(initialized["result"]?["serverInfo"]?["version"]?.stringValue == "test")
        #expect(byID["1"]?["result"] == .object([]))
        #expect(byID["2"]?["result"]?["tools"]?.elements?.map { $0["name"]?.stringValue } == ["decide", "models"])
        let models = try #require(byID["3"]?["result"]?["structuredContent"])
        #expect(models["default"]?.stringValue == "minicpm5-2b")
        #expect(models["loaded"] == .null)  // nothing was loaded: `models` needs no weights
        #expect(byID["3"]?["result"]?["content"]?.elements?.first?["type"]?.stringValue == "text")
        #expect(byID["4"]?["error"]?["code"]?.doubleValue == -32602)
        #expect(byID["4"]?["error"]?["message"]?.stringValue == "Unknown tool: nope")
        #expect(byID["null"]?["error"]?["code"]?.doubleValue == -32700)

        let lines = await logged.lines
        #expect(lines.contains("input closed"))
        #expect(await server.loadedModel == nil)
    }
}

private actor Logged {
    private(set) var lines: [String] = []
    func append(_ line: String) { lines.append(line) }
}
