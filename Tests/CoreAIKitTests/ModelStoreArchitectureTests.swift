// ModelStoreArchitectureTests.swift — an iPhone's `ios-<arch>/`: the store takes the subtree a repo
// compiled for this device when the Hub has it and `ios/` when it does not, keeps whichever copy is
// on disk, and leaves every other path, and the Mac, as they were. A stub Hub stands in for the
// network and the device architecture is injected, so all of it runs on the Mac.

import Foundation
import Testing

@testable import CoreAIKitCore

struct ModelStoreArchitectureTests {
    private let rev = StubHub.revision

    // MARK: The rule

    @Test func onlyTheIOSPathTriesTheDeviceSubtree() {
        #expect(ModelID("org/model", path: "ios").subtrees(deviceArchitecture: "h19p")
            == ["ios-h19p", "ios"])
        for path in ["ios-h18p", "ios-h19p", "gpu-pipelined/x_decode", "x.aimodel", "ios/wfp16-s256",
                     "macos", "model", ""] {
            #expect(ModelID("org/model", path: path).subtrees(deviceArchitecture: "h19p") == [path])
        }
        for arch in [nil, "", "h19p/x"] {
            #expect(ModelID("org/model", path: "ios").subtrees(deviceArchitecture: arch) == ["ios"])
        }
        // The architecture is read only when the path is `ios`.
        let reads = Reads()
        _ = ModelID("org/model", path: "gpu-pipelined/x").subtrees(deviceArchitecture: reads.next())
        #expect(reads.count == 0)
        _ = ModelID("org/model", path: "ios").subtrees(deviceArchitecture: reads.next())
        #expect(reads.count == 1)
    }

    // MARK: (a) The repo has this device's subtree

    @Test func takesTheSubtreeCompiledForThisDevice() async throws {
        let hub = StubHub(files: [
            "ios-h19p/metadata.json", "ios-h19p/x.h19p.aimodelc/main-h19p.mlirb",
            "ios/metadata.json", "ios/x.aimodel/main.mlirb",
        ])
        defer { hub.remove() }
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        let bundle = try await store.download(model)
        #expect(bundle.path == hub.bundle("ios-h19p").path)
        #expect(FileManager.default.fileExists(
            atPath: bundle.appendingPathComponent("x.h19p.aimodelc/main-h19p.mlirb").path))
        #expect(hub.listings == [hub.tree("ios-h19p")])
        #expect(hub.requests.count == 3)
        #expect(!FileManager.default.fileExists(atPath: hub.bundle("ios").path))
        #expect(store.localURL(for: model)?.path == bundle.path)

        // On disk: no Hub call at all.
        #expect(try await store.download(model).path == bundle.path)
        #expect(hub.requests.count == 3)
    }

    @Test func sizeProbesFollowTheSubtreeADownloadTakes() async throws {
        let hub = StubHub(files: ["ios-h19p/metadata.json", "ios-h19p/x.h19p.aimodelc/weights.bin",
                                  "ios/metadata.json"])
        defer { hub.remove() }
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        let plan = try await store.downloadPlan(for: model)
        #expect(plan.map(\.destination.path).sorted() == [
            hub.bundle("ios-h19p").appendingPathComponent("metadata.json").path,
            hub.bundle("ios-h19p").appendingPathComponent("x.h19p.aimodelc/weights.bin").path,
        ])
        #expect(try await store.remoteSize(of: model) == hub.bytes(under: "ios-h19p"))
        #expect(hub.listings == [hub.tree("ios-h19p"), hub.tree("ios-h19p")])
    }

    // MARK: (b) It does not

    @Test(arguments: [false, true])
    func fallsBackToIOSWithOneMoreListing(emptyTree: Bool) async throws {
        // The Hub answers 404 for a subtree a repo does not have; an empty tree counts the same.
        let hub = StubHub(
            files: ["ios/metadata.json", "ios/x.aimodel/main.mlirb"],
            emptyTrees: emptyTree ? ["ios-h19p"] : [])
        defer { hub.remove() }
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        let bundle = try await store.download(model)
        #expect(bundle.path == hub.bundle("ios").path)
        #expect(hub.listings == [hub.tree("ios-h19p"), hub.tree("ios")])
        #expect(hub.requests.count == 4)
        #expect(!FileManager.default.fileExists(atPath: hub.bundle("ios-h19p").path))

        #expect(try await store.download(model).path == bundle.path)
        #expect(hub.requests.count == 4)
    }

    @Test func aHubErrorIsNotReadAsAbsence() async throws {
        // Falling back on an error would leave this device on `ios/` for good.
        let hub = StubHub(files: ["ios/metadata.json"], statuses: ["ios-h19p": 403])
        defer { hub.remove() }
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        await #expect(throws: CoreAIKitError.self) { try await store.download(model) }
        #expect(hub.listings == [hub.tree("ios-h19p")])
        #expect(store.localURL(for: model) == nil)
    }

    @Test func neitherSubtreeIsAMissingVariant() async throws {
        let hub = StubHub(files: ["macos/metadata.json"])
        defer { hub.remove() }
        let store = hub.store(deviceArchitecture: "h19p")
        do {
            try await store.download(ModelID("org/model", path: "ios", revision: rev))
            Issue.record("Expected a missing variant")
        } catch CoreAIKitError.variantNotFound(_, let path, _) {
            #expect(path == "ios")
        }
        #expect(hub.listings == [hub.tree("ios-h19p"), hub.tree("ios")])
    }

    // MARK: (c) The copy on disk

    @Test func aCachedIOSIsKeptAndTheHubIsNotAsked() async throws {
        // The repo has the device's subtree, but `ios/` was downloaded first: it stays.
        let hub = StubHub(files: ["ios-h19p/metadata.json", "ios/metadata.json"])
        defer { hub.remove() }
        try hub.plant("\(rev)/ios")
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        #expect(try await store.download(model).path == hub.bundle("ios").path)
        #expect(hub.requests.isEmpty)
    }

    @Test func offlineACachedIOSUnderAnotherRevisionIsReturned() async throws {
        let hub = StubHub(files: [], offline: true)
        defer { hub.remove() }
        try hub.plant("old/ios")
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        let bundle = try await store.download(model)
        #expect(bundle.resolvingSymlinksInPath().path
            == hub.bundle("ios", revision: "old").resolvingSymlinksInPath().path)
        #expect(hub.listings == [hub.tree("ios-h19p")])
        #expect(store.localURL(for: model) == nil)
    }

    @Test func theDeviceSubtreeWinsWhenBothAreOnDisk() async throws {
        let hub = StubHub(files: [])
        defer { hub.remove() }
        for subpath in ["\(rev)/ios", "\(rev)/ios-h19p", "old/ios", "old/ios-h19p"] {
            try hub.plant(subpath)
        }
        let store = hub.store(deviceArchitecture: "h19p")
        let model = ModelID("org/model", path: "ios", revision: rev)

        #expect(store.localURL(for: model)?.path == hub.bundle("ios-h19p").path)
        #expect(store.siblingRevisionURL(for: model)?.resolvingSymlinksInPath().path
            == hub.bundle("ios-h19p", revision: "old").resolvingSymlinksInPath().path)

        // Delete takes every copy of the revision, and nothing else.
        try await store.delete(model)
        #expect(store.localURL(for: model) == nil)
        #expect(FileManager.default.fileExists(atPath: hub.bundle("ios", revision: "old").path))
        await #expect(throws: (any Error).self) { try await store.delete(model) }
        #expect(hub.requests.isEmpty)
    }

    // MARK: (d) Any other path

    @Test func anExplicitPathIsTheCallersChoice() async throws {
        let hub = StubHub(files: ["ios-h18p/metadata.json", "ios-h19p/metadata.json",
                                  "ios/metadata.json", "gpu-pipelined/x/metadata.json"])
        defer { hub.remove() }
        let store = hub.store(deviceArchitecture: "h19p")

        let h18p = try await store.download(ModelID("org/model", path: "ios-h18p", revision: rev))
        #expect(h18p.path == hub.bundle("ios-h18p").path)
        let pipelined = try await store.download(
            ModelID("org/model", path: "gpu-pipelined/x", revision: rev))
        #expect(pipelined.path == hub.bundle("gpu-pipelined/x").path)
        #expect(hub.listings == [hub.tree("ios-h18p"), hub.tree("gpu-pipelined/x")])
    }

    // MARK: (e) The Mac

    #if os(macOS)
    @Test func theMacIsUnchanged() async throws {
        #expect(ModelStore.deviceArchitecture == nil)
        let hub = StubHub(files: ["macos/metadata.json", "ios/metadata.json",
                                  "ios-h16c/metadata.json"])
        defer { hub.remove() }
        let store = hub.store()  // the production architecture

        let mac = try await store.download(ModelID("org/model", revision: rev))
        #expect(mac.path == hub.bundle("macos").path)
        let ios = try await store.download(ModelID("org/model", path: "ios", revision: rev))
        #expect(ios.path == hub.bundle("ios").path)
        #expect(hub.listings == [hub.tree("macos"), hub.tree("ios")])
    }
    #endif
}

/// Counts reads of an injected architecture.
private final class Reads: @unchecked Sendable {
    private(set) var count = 0
    func next() -> String? {
        count += 1
        return "h19p"
    }
}

/// A Hub for one test behind a URLProtocol. The tree API lists a subtree's files, or answers 404
/// when there are none, as the Hub does; `resolve` returns a file's bytes (its path, as UTF-8).
/// Keyed by host, so tests running in parallel keep apart; every request is recorded.
private final class StubHub: @unchecked Sendable {
    static let revision = String(repeating: "b", count: 40)

    private static let registry = NSLock()
    nonisolated(unsafe) private static var hubs: [String: StubHub] = [:]

    let root: URL
    private let baseURL: URL
    private let session: URLSession
    private let files: [String: Data]
    private let emptyTrees: Set<String>
    private let statuses: [String: Int]
    private let offline: Bool
    private let lock = NSLock()
    private var log: [String] = []

    init(
        files: [String], emptyTrees: Set<String> = [], statuses: [String: Int] = [:],
        offline: Bool = false
    ) {
        let host = "\(UUID().uuidString.lowercased()).hub.test"
        self.root = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-arch-\(UUID().uuidString)")
        self.baseURL = URL(string: "https://\(host)")!
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubHubProtocol.self]
        self.session = URLSession(configuration: config)
        self.files = Dictionary(uniqueKeysWithValues: files.map { ($0, Data($0.utf8)) })
        self.emptyTrees = emptyTrees
        self.statuses = statuses
        self.offline = offline
        Self.registry.withLock { Self.hubs[host] = self }
    }

    static func hub(for host: String?) -> StubHub? {
        registry.withLock { host.flatMap { hubs[$0] } }
    }

    func remove() {
        Self.registry.withLock { Self.hubs[baseURL.host!] = nil }
        session.invalidateAndCancel()
        try? FileManager.default.removeItem(at: root)
    }

    /// A store on this Hub, on a device of architecture `arch`.
    func store(deviceArchitecture arch: String?) -> ModelStore {
        ModelStore(directory: root, hub: HubClient(baseURL: baseURL, session: session),
                   deviceArchitecture: { arch })
    }

    /// A store on this Hub that reads this machine's architecture, as the public stores do.
    func store() -> ModelStore {
        ModelStore(directory: root, hub: HubClient(baseURL: baseURL, session: session))
    }

    func tree(_ subtree: String) -> String {
        "/api/models/org/model/tree/\(Self.revision)/\(subtree)?recursive=true"
    }

    func bundle(_ subtree: String, revision: String = StubHub.revision) -> URL {
        root.appendingPathComponent("org/model/\(revision)/\(subtree)", isDirectory: true)
    }

    func bytes(under subtree: String) -> Int64 {
        files.filter { $0.key.hasPrefix(subtree + "/") }.reduce(0) { $0 + Int64($1.value.count) }
    }

    /// A complete cached bundle at `org/model/<subpath>`.
    func plant(_ subpath: String) throws {
        let dir = root.appendingPathComponent("org/model/\(subpath)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("metadata.json"))
    }

    var requests: [String] { lock.withLock { log } }
    var listings: [String] { requests.filter { $0.hasPrefix("/api/") } }

    /// The status (-1: the network is down) and body for a request.
    func answer(_ request: URLRequest) -> (status: Int, body: Data) {
        let url = request.url!
        if request.httpMethod != "HEAD" {
            lock.withLock { log.append(url.query.map { "\(url.path)?\($0)" } ?? url.path) }
        }
        if offline { return (-1, Data()) }
        let tree = "/api/models/org/model/tree/\(Self.revision)/"
        let resolve = "/org/model/resolve/\(Self.revision)/"
        if url.path.hasPrefix(tree) {
            let subtree = String(url.path.dropFirst(tree.count))
            if let status = statuses[subtree] { return (status, Data()) }
            if emptyTrees.contains(subtree) { return (200, Data("[]".utf8)) }
            let paths = files.keys.filter { $0.hasPrefix(subtree + "/") }.sorted()
            guard !paths.isEmpty else {
                return (404, Data("{\"error\":\"\(subtree) does not exist\"}".utf8))
            }
            let entries = paths.map { ["type": "file", "path": $0, "size": files[$0]!.count] as [String: Any] }
            return (200, try! JSONSerialization.data(withJSONObject: entries))
        }
        if url.path.hasPrefix(resolve), let body = files[String(url.path.dropFirst(resolve.count))] {
            return (200, body)
        }
        return (404, Data())
    }
}

private final class StubHubProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url, let hub = StubHub.hub(for: url.host) else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotFindHost))
            return
        }
        let (status, body) = hub.answer(request)
        guard status >= 0 else {
            client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
            return
        }
        let response = HTTPURLResponse(
            url: url, statusCode: status, httpVersion: "HTTP/1.1",
            headerFields: ["Content-Length": "\(body.count)"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if request.httpMethod != "HEAD" { client?.urlProtocol(self, didLoad: body) }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
