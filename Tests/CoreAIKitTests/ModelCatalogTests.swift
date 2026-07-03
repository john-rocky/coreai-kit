import XCTest

@testable import CoreAIKitCore

final class ModelCatalogTests: XCTestCase {
    func testDecodesCatalogJSON() throws {
        let json = """
            {"version": 1, "models": [
              {"id": "x", "name": "X", "repo": "org/repo", "kind": "chat",
               "variants": {"macos": {"path": "macos", "sizeMB": 10}},
               "thinking": true}
            ]}
            """
        let catalog = try JSONDecoder().decode(ModelCatalog.self, from: Data(json.utf8))
        XCTAssertEqual(catalog.version, 1)
        XCTAssertEqual(catalog.models.count, 1)
        let entry = catalog.models[0]
        XCTAssertEqual(entry.kind, .chat)
        XCTAssertEqual(entry.thinking, true)
        #if os(macOS)
        XCTAssertEqual(entry.modelID, ModelID("org/repo", path: "macos"))
        XCTAssertEqual(entry.variant?.sizeMB, 10)
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
        // Whisper publishes both platform variants; the JIT-only ASR bundles are macOS-only.
        #if os(macOS)
        XCTAssertEqual(
            asr.map(\.id), ["whisper-large-v3-turbo", "qwen3-asr-1.7b", "parakeet-tdt-0.6b-v3"])
        #else
        XCTAssertEqual(asr.map(\.id), ["whisper-large-v3-turbo"])
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
        // Keep the builtin snapshot in sync with catalog.json at the repo root.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // CoreAIKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
        let data = try Data(contentsOf: root.appendingPathComponent("catalog.json"))
        let shipped = try JSONDecoder().decode(ModelCatalog.self, from: data)
        XCTAssertEqual(shipped.version, ModelCatalog.builtin.version)
        XCTAssertEqual(shipped.models, ModelCatalog.builtin.models)
    }
}
