// SystemOneServerTests.swift — the HTTP framing under `SystemOneServer` without a socket or a
// model: a request is complete only when its whole body has arrived, an oversized head or
// body is refused, and a response carries the headers a browser client needs.

import Foundation
import Testing

@testable import CoreAIKit

struct HTTPFramingTests {
    @Test func requestIsCompleteOnlyWithItsWholeBody() throws {
        let head = "POST /v1/systemone HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 12\r\n\r\n"
        #expect(try HTTPRequest.parse(Data("POST /v1/sys".utf8)) == nil)
        #expect(try HTTPRequest.parse(Data((head + "{\"state\":").utf8)) == nil)
        let request = try #require(try HTTPRequest.parse(Data((head + "{\"state\": 1}").utf8)))
        #expect(request.method == "POST")
        #expect(request.path == "/v1/systemone")
        #expect(request.headers["content-type"] == "application/json")
        #expect(String(decoding: request.body, as: UTF8.self) == "{\"state\": 1}")
    }

    @Test func getWithoutBodyIsComplete() throws {
        let request = try #require(try HTTPRequest.parse(Data("GET /health HTTP/1.1\r\nHost: x\r\n\r\n".utf8)))
        #expect(request.method == "GET")
        #expect(request.body.isEmpty)
    }

    @Test func oversizedHeadOrBodyIsRefused() {
        let hugeHead = Data(repeating: UInt8(ascii: "a"), count: 64 * 1024 + 1)
        #expect(throws: HTTPError.self) { try HTTPRequest.parse(hugeHead) }
        let hugeBody = Data("POST /v1/systemone HTTP/1.1\r\nContent-Length: 9000000\r\n\r\n".utf8)
        #expect(throws: HTTPError.self) { try HTTPRequest.parse(hugeBody) }
        #expect(throws: HTTPError.self) { try HTTPRequest.parse(Data("nonsense\r\n\r\n".utf8)) }
    }

    @Test func responseCarriesLengthCloseAndCORS() {
        let response = HTTPResponse.json(422, SystemOne.errorValue(type: "invalid_request_error", message: "x"))
        let text = String(decoding: response.bytes, as: UTF8.self)
        #expect(text.hasPrefix("HTTP/1.1 422 Unprocessable Entity\r\n"))
        #expect(text.contains("Content-Length: \(response.body.count)\r\n"))
        #expect(text.contains("Connection: close\r\n"))
        #expect(text.contains("Access-Control-Allow-Origin: *\r\n"))
        #expect(text.hasSuffix("\r\n\r\n{\"error\": {\"type\": \"invalid_request_error\", \"message\": \"x\"}}"))
    }
}
