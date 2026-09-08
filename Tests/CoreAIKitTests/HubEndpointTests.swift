import Foundation
import Network
import XCTest

@testable import CoreAIKitCore

final class HubEndpointTests: XCTestCase {
    private let revision = String(repeating: "a", count: 40)

    func testDefaultEndpointsPreserveRevisionAndRootLayout() throws {
        let hub = HubClient()
        XCTAssertEqual(
            try hub.listingURL(repo: "org/model", revision: revision, path: "").absoluteString,
            "https://huggingface.co/api/models/org/model/tree/\(revision)?recursive=true")
        XCTAssertEqual(
            try hub.downloadURL(repo: "org/model", revision: revision, path: "macos/model.bin").absoluteString,
            "https://huggingface.co/org/model/resolve/\(revision)/macos/model.bin")
    }

    func testCustomEndpointPreservesPrefixAndEscapesPaths() throws {
        for base in ["https://mirror.example/hf", "https://mirror.example/hf/"] {
            let hub = HubClient(baseURL: URL(string: base)!)
            XCTAssertEqual(
                try hub.listingURL(repo: "org/model", revision: revision, path: "macos").absoluteString,
                "https://mirror.example/hf/api/models/org/model/tree/\(revision)/macos?recursive=true")
            XCTAssertEqual(
                try hub.downloadURL(repo: "org/model", revision: revision, path: "macos/a b.bin").absoluteString,
                "https://mirror.example/hf/org/model/resolve/\(revision)/macos/a%20b.bin")
        }
    }

    func testInvalidBaseURLsFailBeforeNetworking() throws {
        for base in ["file:///tmp/hub", "ftp://mirror.example", "/relative",
                     "https://user:secret@mirror.example", "https://mirror.example?token=secret",
                     "https://mirror.example#fragment"] {
            let hub = HubClient(baseURL: URL(string: base)!)
            XCTAssertThrowsError(try hub.listingURL(repo: "org/model", revision: revision, path: "macos")) {
                guard case CoreAIKitError.invalidHubBaseURL = $0 else {
                    return XCTFail("Expected a base URL error, received \($0)")
                }
            }
        }
    }

    func testMirrorListingDownloadAndCacheWithoutHFCredentials() async throws {
        let metadata = Data("{}".utf8)
        let weights = Data([0, 3, 255, 7])
        let tree = try JSONSerialization.data(withJSONObject: [
            ["type": "file", "path": "macos/metadata.json", "size": metadata.count],
            ["type": "file", "path": "macos/weights test.bin", "size": weights.count],
        ])
        let listingPath = "/hf/api/models/org/model/tree/\(revision)/macos?recursive=true"
        let metadataPath = "/hf/org/model/resolve/\(revision)/macos/metadata.json"
        let weightsPath = "/hf/org/model/resolve/\(revision)/macos/weights%20test.bin"
        let server = try HubFixtureServer(responses: [
            listingPath: tree, metadataPath: metadata, weightsPath: weights,
        ])
        defer { server.stop() }
        let baseURL = try await server.start()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }

        // A cookie scoped to HF must never appear on the explicitly selected mirror host.
        let cookieName = "CoreAIKitEndpointTest-\(UUID().uuidString)"
        let cookie = HTTPCookie(properties: [
            .domain: "huggingface.co", .path: "/", .name: cookieName, .value: "test-only",
        ])!
        HTTPCookieStorage.shared.setCookie(cookie)
        defer { HTTPCookieStorage.shared.deleteCookie(cookie) }

        let model = ModelID("org/model", path: "macos", revision: revision)
        let store = ModelStore(directory: root, hubBaseURL: baseURL)
        XCTAssertNil(store.localURL(for: model))
        let bundle = try await store.download(model)
        XCTAssertEqual(bundle.path, root.appendingPathComponent("org/model/\(revision)/macos").path)
        XCTAssertEqual(try Data(contentsOf: bundle.appendingPathComponent("metadata.json")), metadata)
        XCTAssertEqual(try Data(contentsOf: bundle.appendingPathComponent("weights test.bin")), weights)

        let repeated = try await store.download(model)
        // Changing the endpoint still finds the same pinned, complete cached bundle.
        let defaultStore = ModelStore(directory: root)
        let throughDefault = try await defaultStore.download(model)
        XCTAssertEqual(repeated, bundle)
        XCTAssertEqual(throughDefault, bundle)
        let requests = server.receivedRequests
        XCTAssertEqual(requests.map { $0.components(separatedBy: " ")[1] },
                       [listingPath, metadataPath, weightsPath])
        XCTAssertTrue(requests.allSatisfy { !$0.lowercased().contains("authorization:") })
        XCTAssertTrue(requests.allSatisfy { !$0.contains(cookieName) })
    }

    func testTransientListingFailuresRecoverThroughMirrorAndPinnedRevision() async throws {
        let listingPath = "/hf/api/models/org/model/tree/\(revision)/macos?recursive=true"
        let metadataPath = "/hf/org/model/resolve/\(revision)/macos/metadata.json"
        let metadata = Data("{}".utf8)
        let tree = try JSONSerialization.data(withJSONObject: [
            ["type": "directory", "path": "macos/tokenizer"],
            ["type": "file", "path": "macos/metadata.json", "size": 0,
             "lfs": ["size": metadata.count]],
        ])
        let server = try HubFixtureServer(responseSequences: [
            listingPath: [.init(status: 429), .init(status: 500), .init(status: 502),
                          .init(status: 599), .init(body: tree)],
            metadataPath: [.init(body: metadata)],
        ])
        defer { server.stop() }
        let baseURL = try await server.start()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(directory: root, hubBaseURL: baseURL)
        let model = ModelID("org/model", path: "macos", revision: revision)

        let bundle = try await store.download(model)
        XCTAssertEqual(try Data(contentsOf: bundle.appendingPathComponent("metadata.json")), metadata)
        XCTAssertEqual(bundle.path, root.appendingPathComponent("org/model/\(revision)/macos").path)
        let cached = try await store.download(model)
        XCTAssertEqual(cached, bundle)
        XCTAssertEqual(server.receivedTargets, Array(repeating: listingPath, count: 5) + [metadataPath])

        // Exercise the production sleeps, including the longest wait before the final attempt.
        let times = Array(server.receivedTimes.prefix(5))
        XCTAssertEqual(times.count, 5)
        for (interval, minimum) in zip(zip(times, times.dropFirst()), [2.0, 4.0, 8.0, 16.0]) {
            XCTAssertGreaterThanOrEqual(interval.0.duration(to: interval.1), .seconds(minimum - 0.1))
        }
    }

    func testListingRetryLimitPreservesHTTPStatus() async throws {
        for status in [429, 503] {
            // The fixture repeats its final response, so an unbounded retry cannot pass.
            let listingPath = "/hf/api/models/org/model/tree/\(revision)?recursive=true"
            let server = try HubFixtureServer(responseSequences: [listingPath: [.init(status: status)]])
            defer { server.stop() }
            let baseURL = try await server.start()
            let hub = HubClient(baseURL: baseURL)
            let task = Task { [revision] in
                try await hub.listFiles(repo: "org/model", revision: revision, path: "")
            }
            let deadline = Task {
                try await Task.sleep(for: .seconds(45))
                task.cancel()
            }
            defer { deadline.cancel(); task.cancel() }
            do {
                _ = try await task.value
                XCTFail("Expected retry exhaustion for HTTP \(status)")
            } catch CoreAIKitError.httpError(let receivedStatus, let file) {
                XCTAssertEqual(receivedStatus, status)
                XCTAssertTrue(file.contains("org/model"))
                XCTAssertTrue(file.contains(revision))
            }
            XCTAssertEqual(server.receivedTargets, Array(repeating: listingPath, count: 5))
        }
    }

    func testMissingVariantAndOtherPermanentErrorsDoNotRetry() async throws {
        for status in [404, 400, 401, 403, 418] {
            let listingPath = "/hf/api/models/org/model/tree/\(revision)/missing?recursive=true"
            let server = try HubFixtureServer(responseSequences: [
                listingPath: [.init(status: status), .init(body: Data("[]".utf8))],
            ])
            defer { server.stop() }
            let baseURL = try await server.start()
            do {
                _ = try await HubClient(baseURL: baseURL).listFiles(
                    repo: "org/model", revision: revision, path: "missing")
                XCTFail("Expected HTTP \(status) to fail immediately")
            } catch CoreAIKitError.variantNotFound(let repo, let path, let receivedRevision) {
                XCTAssertEqual(status, 404, "Authentication and other HTTP errors are not missing variants")
                XCTAssertEqual(repo, "org/model")
                XCTAssertEqual(path, "missing")
                XCTAssertEqual(receivedRevision, revision)
            } catch CoreAIKitError.httpError(let receivedStatus, _) {
                XCTAssertNotEqual(status, 404)
                XCTAssertEqual(receivedStatus, status)
            }
            XCTAssertEqual(server.receivedTargets, [listingPath])
        }
    }

    func testCancellationDuringListingBackoffStopsBeforeNextRequest() async throws {
        let listingPath = "/hf/api/models/org/model/tree/\(revision)/macos?recursive=true"
        let server = try HubFixtureServer(responseSequences: [
            listingPath: [.init(status: 429), .init(body: Data("[]".utf8))],
        ])
        defer { server.stop() }
        let baseURL = try await server.start()
        let task = Task { [revision] in
            try await HubClient(baseURL: baseURL).listFiles(repo: "org/model", revision: revision, path: "macos")
        }
        defer { task.cancel() }
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while server.receivedTargets.isEmpty, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(server.receivedTargets, [listingPath])
        // Give the loopback response time to complete and enter the two-second backoff.
        try await Task.sleep(for: .milliseconds(200))
        let cancelledAt = ContinuousClock.now
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation during backoff")
        } catch is CancellationError {
            // Task.sleep must propagate cancellation, not consume it as another retry.
        }
        XCTAssertLessThan(cancelledAt.duration(to: .now), .seconds(1))
        XCTAssertEqual(server.receivedTargets, [listingPath])
    }

    func testAlreadyCancelledListingDoesNotMakeARequest() async throws {
        let server = try HubFixtureServer(responses: [:])
        defer { server.stop() }
        let baseURL = try await server.start()
        let task = Task { [revision] in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await HubClient(baseURL: baseURL).listFiles(
                repo: "org/model", revision: revision, path: "macos")
        }
        do {
            _ = try await task.value
            XCTFail("Expected cancellation before networking")
        } catch is CancellationError {}
        XCTAssertTrue(server.receivedTargets.isEmpty)
    }

    func testMalformedSuccessfulListingIsNotRetried() async throws {
        let listingPath = "/hf/api/models/org/model/tree/\(revision)?recursive=true"
        let server = try HubFixtureServer(responseSequences: [
            listingPath: [.init(body: Data("not JSON".utf8)), .init(body: Data("[]".utf8))],
        ])
        defer { server.stop() }
        let baseURL = try await server.start()
        do {
            _ = try await HubClient(baseURL: baseURL).listFiles(repo: "org/model", revision: revision, path: "")
            XCTFail("Expected the decoding error to propagate")
        } catch is DecodingError {}
        XCTAssertEqual(server.receivedTargets, [listingPath])
    }
}

/// Loopback HTTP fixture: exercises URLSession's real listing and download delegates,
/// staging and cache reuse, without reaching a public server or loading Core AI.
private final class HubFixtureServer: @unchecked Sendable {
    struct Response {
        var status = 200
        var body = Data()
    }

    private let listener: NWListener
    private let responses: [String: [Response]]
    // Accessed only on the serial listener queue. The last response is repeated.
    private var responseCounts: [String: Int] = [:]
    private let queue = DispatchQueue(label: "CoreAIKit.HubFixtureServer")
    private let lock = NSLock()
    private var requests: [String] = []
    private var times: [ContinuousClock.Instant] = []
    private var ready: CheckedContinuation<URL, Error>?

    var receivedRequests: [String] { lock.withLock { requests } }
    var receivedTargets: [String] { receivedRequests.map { $0.components(separatedBy: " ")[1] } }
    var receivedTimes: [ContinuousClock.Instant] { lock.withLock { times } }

    convenience init(responses: [String: Data]) throws {
        try self.init(responseSequences: responses.mapValues { [Response(body: $0)] })
    }

    init(responseSequences: [String: [Response]]) throws {
        precondition(responseSequences.values.allSatisfy { !$0.isEmpty })
        self.responses = responseSequences
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        self.listener = try NWListener(using: parameters)
    }

    func start() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock { ready = continuation }
            listener.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .ready:
                    finishStarting(.success(URL(string: "http://127.0.0.1:\(listener.port!.rawValue)/hf")!))
                case .failed(let error): finishStarting(.failure(error))
                case .cancelled: finishStarting(.failure(CancellationError()))
                default: break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                guard let self else { connection.cancel(); return }
                connection.start(queue: queue)
                receive(connection)
            }
            listener.start(queue: queue)
        }
    }

    func stop() { listener.cancel() }

    private func finishStarting(_ result: Result<URL, Error>) {
        let continuation = lock.withLock {
            defer { ready = nil }
            return ready
        }
        continuation?.resume(with: result)
    }

    private func receive(_ connection: NWConnection, partial: Data = Data()) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, done, error in
            guard let self else { connection.cancel(); return }
            var bytes = partial
            if let data { bytes.append(data) }
            guard bytes.range(of: Data("\r\n\r\n".utf8)) != nil else {
                if done || error != nil { connection.cancel() }
                else { receive(connection, partial: bytes) }
                return
            }
            let request = String(decoding: bytes, as: UTF8.self)
            lock.withLock {
                requests.append(request)
                times.append(.now)
            }
            let target = request.components(separatedBy: " ").dropFirst().first ?? ""
            let index = responseCounts[target, default: 0]
            responseCounts[target] = index + 1
            let planned = responses[target].map { $0[min(index, $0.count - 1)] } ?? Response(status: 404)
            let payload = planned.body
            var response = Data("HTTP/1.1 \(planned.status) Fixture\r\nContent-Length: \(payload.count)\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n".utf8)
            response.append(payload)
            connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}
