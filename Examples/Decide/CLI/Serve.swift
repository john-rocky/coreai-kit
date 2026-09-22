// Serve.swift — `decide-cli serve`: a `/v1/systemone` endpoint on this machine over one loaded
// model, in the request and answer forms a hosted System One client already speaks. Point the
// client's base URL at it and the decisions happen here:
//
//   swift run -c release decide-cli serve --model minicpm5-2b --port 8090
//   curl -s http://127.0.0.1:8090/v1/systemone -H 'Content-Type: application/json' \
//     -d '{"state": "Help! My payouts have been failing for 3 days.", "model": "minicpm5-2b",
//          "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'
//
// Routes: POST /v1/systemone (the decisions), GET /v1/models (what is loaded), GET /health.
// One connection per request (Connection: close), CORS open so a page or a browser extension
// can call it. The model answers one request at a time; the HTTP side accepts concurrently.
// HTTP/1.1 over Network.framework, no dependency added to the kit.

import CoreAIOps
import Foundation
import Network

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    /// A complete request from the bytes received so far, or nil while more is needed.
    /// Throws when the head is not HTTP.
    static func parse(_ buffer: Data) throws -> HTTPRequest? {
        guard let headEnd = buffer.range(of: Data("\r\n\r\n".utf8)) else {
            if buffer.count > 64 * 1024 { throw ServeError.badRequest("request head too large") }
            return nil
        }
        let head = String(decoding: buffer[..<headEnd.lowerBound], as: UTF8.self)
        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ")
        guard requestLine.count >= 2 else { throw ServeError.badRequest("malformed request line") }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()] =
                line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        guard length <= 8 * 1024 * 1024 else { throw ServeError.badRequest("body too large") }
        let bodyStart = headEnd.upperBound
        guard buffer.count >= bodyStart + length else { return nil }
        return HTTPRequest(
            method: String(requestLine[0]), path: String(requestLine[1]), headers: headers,
            body: buffer[bodyStart..<bodyStart + length])
    }
}

struct HTTPResponse {
    let status: Int
    let body: Data
    let contentType: String

    static func json(_ status: Int, _ value: JSONValue) -> HTTPResponse {
        HTTPResponse(status: status, body: Data(value.dumps().utf8), contentType: "application/json; charset=utf-8")
    }

    var bytes: Data {
        let reason = [200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
                      422: "Unprocessable Entity", 500: "Internal Server Error"][status] ?? "OK"
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        head += "Content-Type: \(contentType)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n"
        head += "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        head += "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n"
        return Data(head.utf8) + body
    }
}

enum ServeError: Error {
    case badRequest(String)
}

/// NWConnection is not Sendable; the box lets the receive loop hand it to a task.
final class ConnectionBox: @unchecked Sendable {
    let connection: NWConnection
    init(_ connection: NWConnection) { self.connection = connection }
}

final class SystemOneServer: @unchecked Sendable {
    let host: String
    let port: UInt16
    let modelID: String
    let decider: TypedDecisions
    private let queue = DispatchQueue(label: "decide-cli.serve")
    private var listener: NWListener?

    init(host: String, port: UInt16, modelID: String, decider: TypedDecisions) {
        self.host = host
        self.port = port
        self.modelID = modelID
        self.decider = decider
    }

    /// Listens until the process ends.
    func run() async throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        if let address = IPv4Address(host) {
            parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv4(address), port: NWEndpoint.Port(rawValue: port)!)
        } else if let address = IPv6Address(host) {
            parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv6(address), port: NWEndpoint.Port(rawValue: port)!)
        } else {
            throw ServeError.badRequest("--host must be an IP address (127.0.0.1 for this machine, 0.0.0.0 for the network)")
        }
        let listener = try NWListener(using: parameters)
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            let box = ConnectionBox(connection)
            connection.start(queue: self.queue)
            self.receive(box, buffer: Data())
        }
        listener.stateUpdateHandler = { [modelID, host, port] state in
            switch state {
            case .ready:
                stderrPrint("decide-cli serve: \(modelID) at http://\(host):\(port)\(SystemOne.path)  (GET /v1/models, GET /health; Ctrl-C stops)")
            case .failed(let error):
                stderrPrint("decide-cli serve: listener failed: \(error)")
                exit(1)
            default: break
            }
        }
        listener.start(queue: queue)
        while true { try await Task.sleep(for: .seconds(3600)) }
    }

    private func receive(_ box: ConnectionBox, buffer: Data) {
        box.connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            do {
                if let request = try HTTPRequest.parse(buffer) {
                    Task {
                        let response = await self.route(request)
                        self.send(box, response)
                    }
                    return
                }
            } catch {
                self.send(box, .json(400, SystemOne.errorValue(type: "invalid_request_error", message: "\(error)")))
                return
            }
            if isComplete || error != nil {
                box.connection.cancel()
            } else {
                self.receive(box, buffer: buffer)
            }
        }
    }

    private func send(_ box: ConnectionBox, _ response: HTTPResponse) {
        box.connection.send(content: response.bytes, completion: .contentProcessed { _ in
            box.connection.cancel()
        })
    }

    // MARK: - Routes

    func route(_ request: HTTPRequest) async -> HTTPResponse {
        if request.method == "OPTIONS" {
            return HTTPResponse(status: 204, body: Data(), contentType: "text/plain")
        }
        let path = request.path.split(separator: "?").first.map(String.init) ?? request.path
        switch (request.method, path) {
        case ("GET", "/health"), ("GET", "/"):
            return .json(200, .object([.init("status", .string("ok")), .init("model", .string(modelID))]))
        case ("GET", "/v1/models"):
            return .json(200, .object([
                .init("object", .string("list")),
                .init("data", .array([.object([
                    .init("id", .string(modelID)), .init("object", .string("model")),
                    .init("owned_by", .string("local")),
                ])])),
            ]))
        case ("POST", SystemOne.path):
            return await decide(request)
        case (_, SystemOne.path), (_, "/v1/models"), (_, "/health"):
            return .json(405, SystemOne.errorValue(type: "invalid_request_error", message: "method not allowed"))
        default:
            return .json(404, SystemOne.errorValue(type: "not_found_error", message: "no route \(request.method) \(path); POST \(SystemOne.path)"))
        }
    }

    private func decide(_ request: HTTPRequest) async -> HTTPResponse {
        let parsed: SystemOne.Request
        do {
            parsed = try SystemOne.request(from: request.body)
        } catch let error as SystemOne.WireError {
            return .json(422, SystemOne.errorValue(type: "invalid_request_error", message: error.message))
        } catch {
            return .json(422, SystemOne.errorValue(type: "invalid_request_error", message: "\(error)"))
        }
        do {
            let prefilled = try await decider.prefill(parsed.state)
            var answers: [(id: String, question: Decision.Question, answer: Decision.Answer)] = []
            for (id, question) in parsed.questions {
                answers.append((id, question, try await prefilled.decide(question)))
            }
            let response = SystemOne.response(model: modelID, answers: answers)
            let milliseconds = answers.map(\.answer.timing.milliseconds).reduce(0, +) + prefilled.timing.milliseconds
            stderrPrint("POST \(SystemOne.path)  \(parsed.questions.count) question(s), state \(prefilled.tokens) tokens, \(fmt(milliseconds, 0)) ms")
            return .json(200, response)
        } catch let error as DecisionError {
            return .json(422, SystemOne.errorValue(type: "invalid_request_error", message: error.localizedDescription))
        } catch {
            return .json(500, SystemOne.errorValue(type: "server_error", message: error.localizedDescription))
        }
    }
}
