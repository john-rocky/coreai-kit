// SystemOneServer.swift — a `/v1/systemone` endpoint over one loaded model, in the request and
// answer forms a hosted System One client already speaks. Point the client's base URL at it
// and the decisions happen on this machine:
//
//   let decider = try await TypedDecisions(catalog: "minicpm5-2b")
//   try await SystemOneServer(modelID: "minicpm5-2b", decider: decider).run()
//
//   curl -s http://127.0.0.1:8090/v1/systemone -H 'Content-Type: application/json' \
//     -d '{"state": "Help! My payouts have been failing for 3 days.", "model": "minicpm5-2b",
//          "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}'
//
// Routes: POST /v1/systemone (the decisions), GET /v1/models (what is loaded, in the hosted
// list form plus the OpenAI-style keys — `SystemOne.modelsValue`), GET /health.
// One connection per request (Connection: close), CORS open so a page or a browser extension
// can call it. The model answers one request at a time (`DecisionQueue`); the HTTP side
// accepts concurrently. HTTP/1.1 over Network.framework — no dependency added, and the same
// code listens on an iPhone (`host: "0.0.0.0"` serves the local network).
//
// The model is any `DecisionBackend`: a `TypedDecisions` (a catalog or local bundle, probabilities
// from the logits) or a `FoundationModelDecisions` (Apple's on-device foundation model, one-hot
// answers under a schema, `metadata` on every response saying so — `decide-cli serve --backend fm`).
//
// `systemone serve` (the Homebrew-installed CLI) and `decide-cli serve` (Examples/Decide) are
// argument shells over this type.

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
            if buffer.count > 64 * 1024 { throw HTTPError.badRequest("request head too large") }
            return nil
        }
        let head = String(decoding: buffer[..<headEnd.lowerBound], as: UTF8.self)
        var lines = head.components(separatedBy: "\r\n")
        let requestLine = lines.removeFirst().split(separator: " ")
        guard requestLine.count >= 2 else { throw HTTPError.badRequest("malformed request line") }
        var headers: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()] =
                line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        guard length <= 8 * 1024 * 1024 else { throw HTTPError.badRequest("body too large") }
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

enum HTTPError: Error {
    case badRequest(String)
}

/// NWConnection is not Sendable; the box lets the receive loop hand it to a task.
private final class ConnectionBox: @unchecked Sendable {
    let connection: NWConnection
    init(_ connection: NWConnection) { self.connection = connection }
}

/// Resumes a continuation once, whichever listener state arrives first.
private final class Once<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<T, any Error>?
    init(_ continuation: CheckedContinuation<T, any Error>) { self.continuation = continuation }
    func resume(with result: Result<T, any Error>) {
        lock.lock()
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.resume(with: result)
    }
}

public final class SystemOneServer: @unchecked Sendable {
    public enum Failure: Error, LocalizedError, Equatable {
        /// `host` was not an IP address.
        case invalidHost(String)
        /// The listener could not bind or stopped (the port is taken, the interface went away).
        case listenerFailed(String)

        public var errorDescription: String? {
            switch self {
            case .invalidHost(let host):
                return "host must be an IP address (127.0.0.1 for this machine, 0.0.0.0 for the network), not \(host)"
            case .listenerFailed(let reason):
                return "listener failed: \(reason)"
            }
        }
    }

    public let host: String
    public let port: UInt16
    /// What `GET /v1/models` and every response name as the model.
    public let modelID: String
    /// What `GET /v1/models` says about the loaded model (`SystemOne.modelsValue`).
    public let models: JSONValue
    /// What answers: a `TypedDecisions` or a `FoundationModelDecisions`.
    public let backend: any DecisionBackend
    /// The backend when it is a `TypedDecisions`; nil over the system model.
    public var decider: TypedDecisions? { backend as? TypedDecisions }
    /// One line per event (listening, each request served); stderr by default.
    public let log: @Sendable (String) -> Void
    private let decisions = DecisionQueue()
    private let queue = DispatchQueue(label: "coreai-kit.systemone.serve")
    private let lock = NSLock()
    private var listener: NWListener?

    /// - Parameters:
    ///   - host: an IP address; `127.0.0.1` answers this machine only, `0.0.0.0` the network.
    ///   - port: `8090` is what the reference implementations and the clients default to.
    ///   - models: the `GET /v1/models` body; `SystemOne.modelsValue(id:description:revision:)`
    ///     with the catalog entry's name and pin says what a hosted client expects. Left nil,
    ///     the id stands in for the description and the revision is empty.
    public convenience init(
        host: String = "127.0.0.1", port: UInt16 = 8090, modelID: String, models: JSONValue? = nil,
        decider: TypedDecisions,
        log: @escaping @Sendable (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) {
        self.init(host: host, port: port, modelID: modelID, models: models, backend: decider, log: log)
    }

    /// The same over any backend — `FoundationModelDecisions` for the system model.
    public init(
        host: String = "127.0.0.1", port: UInt16 = 8090, modelID: String, models: JSONValue? = nil,
        backend: any DecisionBackend,
        log: @escaping @Sendable (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) {
        self.host = host
        self.port = port
        self.modelID = modelID
        self.models = models ?? SystemOne.modelsValue(id: modelID, description: modelID, revision: nil)
        self.backend = backend
        self.log = log
    }

    /// Listens until `stop()` is called or the task is cancelled; throws when the listener
    /// cannot bind or fails later.
    public func run() async throws {
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true
        guard let nwPort = NWEndpoint.Port(rawValue: port) else { throw Failure.listenerFailed("port \(port)") }
        if let address = IPv4Address(host) {
            parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv4(address), port: nwPort)
        } else if let address = IPv6Address(host) {
            parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv6(address), port: nwPort)
        } else {
            throw Failure.invalidHost(host)
        }
        let listener: NWListener
        do {
            listener = try NWListener(using: parameters)
        } catch {
            throw Failure.listenerFailed("\(error)")
        }
        lock.withLock { self.listener = listener }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            let box = ConnectionBox(connection)
            connection.start(queue: self.queue)
            self.receive(box, buffer: Data())
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let once = Once(continuation)
                listener.stateUpdateHandler = { [modelID, host, port, log] state in
                    switch state {
                    case .ready:
                        log("listening: \(modelID) at http://\(host):\(port)\(SystemOne.path)  (GET /v1/models, GET /health)")
                    case .failed(let error):
                        once.resume(with: .failure(Failure.listenerFailed("\(error)")))
                    case .cancelled:
                        once.resume(with: .success(()))
                    default:
                        break
                    }
                }
                listener.start(queue: queue)
            }
        } onCancel: {
            stop()
        }
    }

    /// Stops listening; `run()` returns.
    public func stop() {
        let listener = lock.withLock { () -> NWListener? in
            let current = self.listener
            self.listener = nil
            return current
        }
        listener?.cancel()
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
            return .json(200, models)
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
            // The loaded model's own option count, so a list it cannot read is a 422 that says so.
            parsed = try SystemOne.request(from: request.body, maxOptions: backend.maxOptions)
        } catch let error as SystemOne.WireError {
            return .json(422, SystemOne.errorValue(type: "invalid_request_error", message: error.message))
        } catch {
            return .json(422, SystemOne.errorValue(type: "invalid_request_error", message: "\(error)"))
        }
        do {
            let response = try await decisions.run { [backend] in try await backend.systemOne(parsed) }
            // The system model counts the state inside each answer, not as a prefix.
            let state = response.stateTokens > 0 ? "state \(response.stateTokens) tokens, " : ""
            log("POST \(SystemOne.path)  \(parsed.questions.count) question(s), \(state)\(Int(response.milliseconds.rounded())) ms")
            return .json(200, SystemOne.response(
                model: modelID, answers: response.answers.map { ($0.id, $0.question, $0.answer) }, metadata: response.metadata))
        } catch let error as DecisionError {
            return .json(422, SystemOne.errorValue(type: "invalid_request_error", message: error.localizedDescription))
        } catch {
            return .json(500, SystemOne.errorValue(type: "server_error", message: error.localizedDescription))
        }
    }
}
