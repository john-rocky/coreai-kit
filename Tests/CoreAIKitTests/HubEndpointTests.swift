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
}

/// Loopback HTTP fixture: exercises URLSession's real listing and download delegates,
/// staging and cache reuse, without reaching a public server or loading Core AI.
private final class HubFixtureServer: @unchecked Sendable {
    private let listener: NWListener
    private let responses: [String: Data]
    private let queue = DispatchQueue(label: "CoreAIKit.HubFixtureServer")
    private let lock = NSLock()
    private var requests: [String] = []
    private var ready: CheckedContinuation<URL, Error>?

    var receivedRequests: [String] { lock.withLock { requests } }

    init(responses: [String: Data]) throws {
        self.responses = responses
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
            lock.withLock { requests.append(request) }
            let target = request.components(separatedBy: " ").dropFirst().first ?? ""
            let payload = responses[target] ?? Data("not found".utf8)
            let status = responses[target] == nil ? "404 Not Found" : "200 OK"
            var response = Data("HTTP/1.1 \(status)\r\nContent-Length: \(payload.count)\r\nConnection: close\r\n\r\n".utf8)
            response.append(payload)
            connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}
