import XCTest

@testable import CoreAIKitCore

/// The compiled-in catalog is what a shipped app falls back to when the network fetch of
/// `catalog.json` fails — which is exactly the moment it must not silently resolve `main`
/// instead of the revision that was gated. `CatalogEntry.modelID(path:)` resolves
/// `revision ?? "main"`, so an unpinned built-in entry is an unpinned download.
final class BuiltinPinsTests: XCTestCase {

    func testBuiltinIsFullyPinned() {
        let unpinned = ModelCatalog.builtin.models
            .filter { $0.revision == nil }
            .map(\.id)
        XCTAssertTrue(
            unpinned.isEmpty,
            """
            built-in catalog entries with no revision pin: \(unpinned.sorted())
            A fallback would download these from the model repo's main branch instead of the
            reviewed bytes. Fix by pinning them in catalog.json, then:
                python3 scripts/gen-builtin-pins.py
            """)
    }

    func testBuiltinPinsMatchCatalogFile() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // CoreAIKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
            .appendingPathComponent("catalog.json")
        let shipped = try JSONDecoder().decode(ModelCatalog.self, from: Data(contentsOf: url))
        let shippedPins = Dictionary(
            uniqueKeysWithValues: shipped.models.compactMap { e in
                e.revision.map { (e.id, $0) }
            })

        for entry in ModelCatalog.builtin.models {
            guard let expected = shippedPins[entry.id] else { continue }
            XCTAssertEqual(
                entry.revision, expected,
                "built-in pin for \(entry.id) is stale — run scripts/gen-builtin-pins.py")
        }
    }

    /// Pins fill only what is missing: an entry that states its own revision keeps it.
    func testExplicitRevisionWins() {
        let catalog = ModelCatalog(
            version: 1,
            models: [
                CatalogEntry(
                    id: "explicit", name: "E", repo: "org/repo", revision: "kept",
                    kind: .chat, variants: ["macos": .init(path: "macos", sizeMB: 1)]),
                CatalogEntry(
                    id: "bare", name: "B", repo: "org/repo", kind: .chat,
                    variants: ["macos": .init(path: "macos", sizeMB: 1)]),
            ])
        let pinned = catalog.pinned(with: ["explicit": "overwritten", "bare": "filled"])
        XCTAssertEqual(pinned.models[0].revision, "kept")
        XCTAssertEqual(pinned.models[1].revision, "filled")
    }
}
