import XCTest

@testable import CoreAIKit

final class StreamingTagParserTests: XCTestCase {
    private func collect(
        _ chunks: [String], profile: OutputProfile
    ) -> [StreamingTagParser.Event] {
        var parser = StreamingTagParser(profile: profile)
        var events: [StreamingTagParser.Event] = []
        for chunk in chunks { events += parser.consume(chunk) }
        events += parser.flush()
        return events
    }

    private func joined(
        _ events: [StreamingTagParser.Event]
    ) -> (response: String, thinking: String, tools: [String]) {
        var response = "", thinking = ""
        var tools: [String] = []
        for event in events {
            switch event {
            case .response(let s): response += s
            case .thinking(let s): thinking += s
            case .toolCallPayload(let s): tools.append(s)
            }
        }
        return (response, thinking, tools)
    }

    // MARK: - plain

    func testPlainPassthrough() {
        let events = collect(["Hello", " <world>", "!"], profile: .plain)
        XCTAssertEqual(joined(events).response, "Hello <world>!")
    }

    func testPlainEmitsImmediately() {
        var parser = StreamingTagParser(profile: .plain)
        XCTAssertEqual(parser.consume("abc"), [.response("abc")])
    }

    // MARK: - think tags

    func testThinkBlockSeparates() {
        let events = collect(
            ["<think>plan</think>", "answer"], profile: .thinkTags())
        let j = joined(events)
        XCTAssertEqual(j.thinking, "plan")
        XCTAssertEqual(j.response, "answer")
    }

    func testMarkerStraddlesChunkBoundaries() {
        let events = collect(
            ["Hello <th", "ink>secret</th", "ink> world"], profile: .thinkTags())
        let j = joined(events)
        XCTAssertEqual(j.response, "Hello  world")
        XCTAssertEqual(j.thinking, "secret")
    }

    func testTextBeforePossibleMarkerStreamsEarly() {
        var parser = StreamingTagParser(profile: .thinkTags())
        // "Hello " is provably not a marker prefix and must stream right away.
        XCTAssertEqual(parser.consume("Hello <th"), [.response("Hello ")])
    }

    func testThinkingStreamsDeltaByDelta() {
        var parser = StreamingTagParser(profile: .thinkTags())
        _ = parser.consume("<think>")
        XCTAssertEqual(parser.consume("step one, "), [.thinking("step one, ")])
        XCTAssertEqual(parser.consume("step two"), [.thinking("step two")])
    }

    func testUnclosedThinkFlushes() {
        let events = collect(["<think>half a thought"], profile: .thinkTags())
        XCTAssertEqual(joined(events).thinking, "half a thought")
    }

    // MARK: - tool calls

    func testToolCallBuffersToOnePayload() {
        let payload = #"{"name":"get_weather","arguments":{"city":"Sapporo"}}"#
        let chunks = ["<tool_call>", String(payload.prefix(20)),
                      String(payload.dropFirst(20)), "</tool_call>"]
        let events = collect(chunks, profile: .thinkTags())
        XCTAssertEqual(joined(events).tools, [payload])
    }

    func testToolCallBodyNotStreamedAsText() {
        var parser = StreamingTagParser(profile: .thinkTags())
        _ = parser.consume("<tool_call>")
        XCTAssertEqual(parser.consume(#"{"name":"#), [])
    }

    func testUnterminatedToolCallFlushesPartialPayload() {
        let events = collect(["<tool_call>", #"{"na"#], profile: .thinkTags())
        XCTAssertEqual(joined(events).tools, [#"{"na"#])
    }

    func testTwoToolCallsInOneTurn() {
        let events = collect(
            ["<tool_call>{\"a\":1}</tool_call>between<tool_call>{\"b\":2}</tool_call>"],
            profile: .thinkTags())
        let j = joined(events)
        XCTAssertEqual(j.tools, ["{\"a\":1}", "{\"b\":2}"])
        XCTAssertEqual(j.response, "between")
    }

    // MARK: - harmony

    func testHarmonyChannels() {
        let raw = "<|channel|>analysis<|message|>I think<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Answer<|return|>"
        // Stress the hold-back with awkward chunk boundaries.
        let chunks = stride(from: 0, to: raw.count, by: 7).map { offset -> String in
            let s = raw.index(raw.startIndex, offsetBy: offset)
            let e = raw.index(s, offsetBy: 7, limitedBy: raw.endIndex) ?? raw.endIndex
            return String(raw[s..<e])
        }
        let j = joined(collect(chunks, profile: .harmony))
        XCTAssertEqual(j.thinking, "I think")
        XCTAssertEqual(j.response, "Answer")
    }

    func testHarmonySuppressesInterChannelText() {
        let j = joined(collect(
            ["<|start|>assistant<|channel|>final<|message|>Hi<|end|>"], profile: .harmony))
        XCTAssertEqual(j.response, "Hi")
        XCTAssertEqual(j.thinking, "")
    }

    func testHarmonyFinalRunsToStreamEndWithoutCloser() {
        let j = joined(collect(
            ["<|channel|>final<|message|>open ended"], profile: .harmony))
        XCTAssertEqual(j.response, "open ended")
    }

    // MARK: - edges

    func testEmptyFlush() {
        var parser = StreamingTagParser(profile: .thinkTags())
        XCTAssertEqual(parser.flush(), [])
    }
}
