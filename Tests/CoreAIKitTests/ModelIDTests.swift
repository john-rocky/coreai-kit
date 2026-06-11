import XCTest

@testable import CoreAIKitCore

final class ModelIDTests: XCTestCase {
    func testPlatformDefaultPath() {
        let id = ModelID("org/name")
        #if os(iOS)
        XCTAssertEqual(id.resolvedPath, "ios")
        #else
        XCTAssertEqual(id.resolvedPath, "macos")
        #endif
    }

    func testExplicitPathWins() {
        XCTAssertEqual(ModelID("org/name", path: "model").resolvedPath, "model")
    }

    func testCacheSubpathLayout() {
        let id = ModelID("org/name", path: "macos", revision: "main")
        XCTAssertEqual(id.cacheSubpath, "org/name/main/macos")
    }

    func testRepoIdFromURL() {
        XCTAssertEqual(
            HubClient.repoId(from: "https://huggingface.co/org/name"), "org/name")
        XCTAssertEqual(
            HubClient.repoId(from: "https://huggingface.co/org/name/tree/main"), "org/name")
    }

    func testRepoIdFromBareId() {
        XCTAssertEqual(HubClient.repoId(from: " org/name "), "org/name")
    }

    func testRepoIdRejectsGarbage() {
        XCTAssertNil(HubClient.repoId(from: "not-a-repo"))
        XCTAssertNil(HubClient.repoId(from: "https://example.com/org/name"))
    }
}
