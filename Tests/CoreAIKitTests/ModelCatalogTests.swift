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

    func testBuiltinAvailableFiltering() {
        let chat = ModelCatalog.builtin.available(.chat)
        XCTAssertFalse(chat.isEmpty)
        XCTAssertTrue(chat.allSatisfy { $0.kind == .chat && $0.modelID != nil })
        #if os(macOS)
        XCTAssertEqual(chat.count, 4)
        #else
        XCTAssertEqual(chat.count, 2)  // qwen3 only on iOS
        #endif
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
