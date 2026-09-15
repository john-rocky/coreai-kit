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

    func testVariantEngineHintAndDeviceKeyResolution() throws {
        let json = """
            {"version": 1, "models": [
              {"id": "m", "name": "M", "repo": "org/m", "revision": "abc", "kind": "chat",
               "variants": {"macos": {"path": "int8", "sizeMB": 1},
                            "ios": {"path": "int8", "sizeMB": 1},
                            "ios-ane-h18p": {"path": "ios-ane-h18p", "sizeMB": 2,
                                             "engine": "static-shape"}},
               "engine": "pipelined"}
            ]}
            """
        let entry = try JSONDecoder().decode(ModelCatalog.self, from: Data(json.utf8)).models[0]
        XCTAssertEqual(entry.variants["ios-ane-h18p"]?.engine, "static-shape")
        XCTAssertNil(entry.variants["ios"]?.engine)

        // The key order is a pure function of platform + architecture: a device-specific
        // key first when the architecture is known, then the portable key; macOS has no
        // device-specific keys.
        XCTAssertEqual(
            CatalogEntry.variantKeys(platform: "ios", architecture: "h18p"), ["ios-ane-h18p", "ios"])
        XCTAssertEqual(CatalogEntry.variantKeys(platform: "ios", architecture: nil), ["ios"])
        XCTAssertEqual(CatalogEntry.variantKeys(platform: "macos", architecture: "h18p"), ["macos"])

        #if os(iOS)
        XCTAssertEqual(entry.resolvedVariantKey(architecture: "h18p"), "ios-ane-h18p")
        // An architecture with no bundle of its own rides the portable variant.
        XCTAssertEqual(entry.resolvedVariantKey(architecture: "h17p"), "ios")
        XCTAssertEqual(entry.resolvedVariantKey(architecture: nil), "ios")
        XCTAssertEqual(entry.portableVariant?.path, "int8")
        XCTAssertEqual(entry.portableModelID, ModelID("org/m", path: "int8", revision: "abc"))
        #else
        XCTAssertEqual(entry.resolvedVariantKey(architecture: "h18p"), "macos")
        XCTAssertEqual(entry.variantKey, "macos")
        XCTAssertEqual(entry.resolvedEngine, "pipelined")
        XCTAssertEqual(entry.portableModelID, entry.modelID)
        #endif
    }

    func testMiniCPM5CarriesTheNeuralEngineVariantBesideThePortableOne() {
        for id in ["minicpm5-1b", "minicpm5-2b"] {
            let entry = try! XCTUnwrap(ModelCatalog.builtin.entry(id: id))
            let ane = entry.variants["ios-ane-h18p"]
            XCTAssertEqual(ane?.path, "ios-ane-h18p", id)
            XCTAssertEqual(ane?.engine, "static-shape", id)
            XCTAssertNotNil(ane?.sizeMB, id)
            // The portable iOS variant stays: kit builds older than the key, and every
            // device the architecture table does not know, ride it.
            XCTAssertEqual(entry.variants["ios"]?.path, "int8", id)
            XCTAssertNil(entry.variants["ios"]?.engine, id)
            XCTAssertEqual(entry.engine, "pipelined", id)
        }
    }

    func testEntryLookup() {
        XCTAssertEqual(ModelCatalog.builtin.entry(id: "whisper-large-v3-turbo")?.kind, .asr)
        XCTAssertNil(ModelCatalog.builtin.entry(id: "nope"))
    }

    func testBuiltinMatchesShippedCatalogFile() throws {
        // Keep the builtin snapshot in sync with catalog.json at the repo root, in every field
        // except the pins: the hand-written literal carries paths, sizes, kinds and engine
        // hints, while its revisions are overlaid from BuiltinPins (generated from the same
        // catalog.json). So this compares the *literal* modulo `revision`, and
        // `BuiltinPinsTests` covers the overlay — including that no entry reaches
        // `ModelCatalog.builtin` unpinned, which would make an offline fallback resolve `main`.
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
        XCTAssertEqual(unpinned.map(\.id), ModelCatalog.builtinLiteral.models.map(\.id))
        for (shippedEntry, builtinEntry) in zip(unpinned, ModelCatalog.builtinLiteral.models) {
            XCTAssertEqual(
                shippedEntry, builtinEntry,
                "catalog.json/builtin drift at '\(shippedEntry.id)'")
        }
    }
}
