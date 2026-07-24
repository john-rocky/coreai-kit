// ModelStoreOfflineFallbackTests.swift — the sibling-revision scan behind the offline
// fallback: a catalog pin moves, the pinned revision isn't cached, the Hub is
// unreachable — the store must find the complete copy cached under the old revision.

import Foundation
import Testing

@testable import CoreAIKitCore

struct ModelStoreOfflineFallbackTests {
    private func makeStore() throws -> (ModelStore, URL) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-fallback-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        return (ModelStore(directory: tmp), tmp)
    }

    private func plant(_ subpath: String, under root: URL) throws {
        let dir = root.appendingPathComponent(subpath, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("metadata.json"))
    }

    @Test func findsCopyUnderAnotherRevision() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try plant("org/model/main/macos", under: root)

        let pinned = ModelID("org/model", path: "macos", revision: "abc123")
        #expect(store.localURL(for: pinned) == nil)
        let sibling = store.siblingRevisionURL(for: pinned)
        #expect(sibling?.path.hasSuffix("org/model/main/macos") == true)
    }

    @Test func ignoresOtherVariantsAndStaging() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try plant("org/model/main/ios", under: root)  // wrong variant
        try plant("org/model/.staging-xyz/macos", under: root)  // hidden staging

        let pinned = ModelID("org/model", path: "macos", revision: "abc123")
        #expect(store.siblingRevisionURL(for: pinned) == nil)
    }

    @Test func exactRevisionStillWinsViaLocalURL() throws {
        let (store, root) = try makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        try plant("org/model/abc123/macos", under: root)

        let pinned = ModelID("org/model", path: "macos", revision: "abc123")
        #expect(store.localURL(for: pinned) != nil)
    }
}
