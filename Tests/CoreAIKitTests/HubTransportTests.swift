import Foundation
import HuggingFace
import XCTest

@testable import CoreAIKitCore

final class HubTransportTests: XCTestCase {
    private let revision = String(repeating: "a", count: 40)

    func testListingFailsFastWhileTransfersWaitForConnectivity() {
        // A listing that waits for connectivity never reaches the offline fallback.
        XCTAssertFalse(CoreAIKitCore.HubClient.listingConfiguration.waitsForConnectivity)
        XCTAssertTrue(CoreAIKitCore.HubClient.transferConfiguration.waitsForConnectivity)
    }

    func testAuthenticatedListingAndDownloadProduceBundle() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [AuthenticatedTreeProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let hub = CoreAIKitCore.HubClient(
            session: session, tokenProvider: .fixed(token: "fixture-token"))
        let store = ModelStore(directory: root.appendingPathComponent("models"), hub: hub)
        let model = ModelID("org/model", path: "macos", revision: revision)
        let bundle = try await store.download(model)
        // The fixture rejects every request that lacks the token.
        XCTAssertEqual(try Data(contentsOf: bundle.appendingPathComponent("metadata.json")), Data("{}".utf8))
        XCTAssertEqual(try bundle.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup, true)
        let cached = try await store.download(model)
        XCTAssertEqual(cached, bundle)
    }

    func testTransportFailureReturnsCompleteSiblingRevision() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let oldBundle = root.appendingPathComponent("org/model/old/macos")
        try FileManager.default.createDirectory(at: oldBundle, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: oldBundle.appendingPathComponent("metadata.json"))
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [OfflineHubProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let store = ModelStore(directory: root, hub: CoreAIKitCore.HubClient(session: session, tokenProvider: .none))
        let model = ModelID("org/model", path: "macos", revision: revision)
        let bundle = try await store.download(model)
        XCTAssertEqual(bundle.resolvingSymlinksInPath().path, oldBundle.resolvingSymlinksInPath().path)
        XCTAssertNil(store.localURL(for: model))
    }

    func testCancellationDoesNotReturnSiblingRevision() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("org/model/old/macos"), withIntermediateDirectories: true)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [CancelledHubProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let store = ModelStore(directory: root, hub: CoreAIKitCore.HubClient(session: session, tokenProvider: .none))
        do {
            try await store.download(ModelID("org/model", path: "macos", revision: revision))
            XCTFail("Expected cancellation")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .cancelled)
        }
    }
}

private final class AuthenticatedTreeProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let revision = String(repeating: "a", count: 40)
        let bodies = [
            "/api/models/org/model/tree/\(revision)/macos":
                Data("[{\"type\":\"file\",\"path\":\"macos/metadata.json\",\"size\":2}]".utf8),
            "/org/model/resolve/\(revision)/macos/metadata.json": Data("{}".utf8),
        ]
        guard request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-token",
              let body = request.url.flatMap({ bodies[$0.path] })
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.userAuthenticationRequired))
            return
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if request.httpMethod != "HEAD" { client?.urlProtocol(self, didLoad: body) }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

private final class OfflineHubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() { client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet)) }
    override func stopLoading() {}
}

private final class CancelledHubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() { client?.urlProtocol(self, didFailWithError: URLError(.cancelled)) }
    override func stopLoading() {}
}
