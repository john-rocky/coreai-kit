import XCTest

@testable import CoreAIKitCore

final class ModelCatalogTests: XCTestCase {
    func testDecodesCatalogJSON() throws {
        let json = """
            {"version": 1, "models": [
              {"id": "x", "name": "X", "repo": "org/repo", "revision": "abc123",
               "kind": "chat",
               "variants": {"macos": {"path": "macos", "sizeMB": 10}},
               "thinking": true},
              {"id": "y", "name": "Y", "repo": "org/legacy", "kind": "chat",
               "variants": {"macos": {"path": "macos"}}}
            ]}
            """
        let catalog = try JSONDecoder().decode(ModelCatalog.self, from: Data(json.utf8))
        XCTAssertEqual(catalog.version, 1)
        XCTAssertEqual(catalog.models.count, 2)
        let entry = catalog.models[0]
        XCTAssertEqual(entry.kind, .chat)
        XCTAssertEqual(entry.thinking, true)
        XCTAssertEqual(entry.revision, "abc123")
        #if os(macOS)
        // The pin rides into every ModelID the entry resolves; an unpinned (older)
        // catalog falls back to `main`.
        XCTAssertEqual(entry.modelID, ModelID("org/repo", path: "macos", revision: "abc123"))
        XCTAssertEqual(entry.modelID(path: "extra"), ModelID("org/repo", path: "extra", revision: "abc123"))
        XCTAssertEqual(entry.variant?.sizeMB, 10)
        XCTAssertEqual(catalog.models[1].modelID, ModelID("org/legacy", path: "macos"))
        #else
        XCTAssertNil(entry.modelID)  // no ios variant published
        #endif
    }

    func testDecodesKnownAndUnknownKinds() throws {
        let json = """
            {"version": 1, "models": [
              {"id": "a", "name": "A", "repo": "org/a", "kind": "asr",
               "variants": {"macos": {"path": "macos"}, "ios": {"path": "ios"}}},
              {"id": "d", "name": "D", "repo": "org/d", "kind": "detection",
               "variants": {"macos": {"path": "m.aimodel"}, "ios": {"path": "m.aimodel"}}},
              {"id": "f", "name": "F", "repo": "org/f", "kind": "hologram",
               "variants": {"macos": {"path": "macos"}, "ios": {"path": "ios"}}}
            ]}
            """
        let catalog = try JSONDecoder().decode(ModelCatalog.self, from: Data(json.utf8))
        XCTAssertEqual(catalog.models.map(\.kind), [.asr, .detection, .unknown])
        // Forward-compat: the unknown kind decodes but never surfaces in available().
        XCTAssertEqual(catalog.available().map(\.id), ["a", "d"])
        XCTAssertEqual(catalog.available(.asr).map(\.id), ["a"])
    }

    func testBuiltinAvailableFiltering() {
        let chat = ModelCatalog.builtin.available(.chat)
        XCTAssertFalse(chat.isEmpty)
        XCTAssertTrue(chat.allSatisfy { $0.kind == .chat && $0.modelID != nil })

        let asr = ModelCatalog.builtin.available(.asr)
        // Whisper + Nemotron publish both platform variants; the JIT-only ASR bundles are
        // macOS-only.
        #if os(macOS)
        XCTAssertEqual(
            asr.map(\.id),
            [
                "whisper-large-v3-turbo", "qwen3-asr-1.7b", "parakeet-tdt-0.6b-v3",
                "nemotron-3.5-asr-streaming-0.6b",
            ])
        #else
        XCTAssertEqual(
            asr.map(\.id), ["whisper-large-v3-turbo", "nemotron-3.5-asr-streaming-0.6b"])
        #endif

        // rf-detr regression: "detection" used to decode to .unknown, hiding the entry.
        XCTAssertEqual(
            ModelCatalog.builtin.available(.detection).map(\.id), ["yolox-s", "rf-detr"])
    }

    func testEntryLookup() {
        XCTAssertEqual(ModelCatalog.builtin.entry(id: "whisper-large-v3-turbo")?.kind, .asr)
        XCTAssertNil(ModelCatalog.builtin.entry(id: "nope"))
    }

    func testBuiltinMatchesShippedCatalogFile() throws {
        // Keep the builtin snapshot in sync with catalog.json at the repo root. The shipped
        // file carries revision pins (scripts/pin-catalog.py); the builtin snapshot stays
        // unpinned — it is the offline fallback, and the live catalog is where pins are
        // maintained — so compare modulo `revision`.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // CoreAIKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let data = try Data(contentsOf: root.appendingPathComponent("catalog.json"))
        let shipped = try JSONDecoder().decode(ModelCatalog.self, from: data)
        XCTAssertEqual(shipped.version, ModelCatalog.builtin.version)
        // Every shipped entry must be pinned — an unpinned entry means a repo was added
        // without running scripts/pin-catalog.py.
        for entry in shipped.models {
            XCTAssertNotNil(entry.revision, "catalog.json entry '\(entry.id)' has no revision pin")
        }
        let unpinned = shipped.models.map { entry in
            CatalogEntry(
                id: entry.id, name: entry.name, repo: entry.repo, kind: entry.kind,
                variants: entry.variants, thinking: entry.thinking, engine: entry.engine)
        }
        XCTAssertEqual(unpinned.map(\.id), ModelCatalog.builtin.models.map(\.id))
        for (shippedEntry, builtinEntry) in zip(unpinned, ModelCatalog.builtin.models) {
            XCTAssertEqual(
                shippedEntry, builtinEntry,
                "catalog.json/builtin drift at '\(shippedEntry.id)'")
        }
    }
}
