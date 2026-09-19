import Foundation
import HuggingFace
import XCTest

@testable import CoreAIKitCore

final class HubTransportTests: XCTestCase {
    private let revision = String(repeating: "a", count: 40)

    func testAuthenticatedListingAndSharedCacheProduceIndependentBundle() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheDirectory = root.appendingPathComponent("hub-cache")
        let cache = HubCache(cacheDirectory: cacheDirectory)
        try await cache.storeData(
            Data("{}".utf8), repo: "org/model", kind: .model, revision: revision,
            filename: "macos/metadata.json", etag: String(repeating: "b", count: 40))
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [AuthenticatedTreeProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let hub = CoreAIKitCore.HubClient(
            session: session, tokenProvider: .fixed(token: "fixture-token"), cache: cache)
        let store = ModelStore(directory: root.appendingPathComponent("models"), hub: hub)
        let model = ModelID("org/model", path: "macos", revision: revision)
        let bundle = try await store.download(model)
        // The fixture rejects every file request, so this must come from HubCache.
        XCTAssertEqual(try Data(contentsOf: bundle.appendingPathComponent("metadata.json")), Data("{}".utf8))
        try FileManager.default.removeItem(at: cacheDirectory)
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
        guard request.value(forHTTPHeaderField: "Authorization") == "Bearer fixture-token",
              request.url?.path == "/api/models/org/model/tree/\(String(repeating: "a", count: 40))/macos"
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.userAuthenticationRequired))
            return
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("[{\"type\":\"file\",\"path\":\"macos/metadata.json\",\"size\":2}]".utf8))
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
