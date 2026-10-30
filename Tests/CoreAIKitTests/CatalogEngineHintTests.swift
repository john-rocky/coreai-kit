// CatalogEngineHintTests.swift — a variant's own engine hint (`CatalogEntry.Variant.engine`) over the
// entry's, checkable for either platform without weights: LFM3 1.7B's iPhone variant is the static
// Neural Engine form while its Mac bundle runs on the pipelined engine, so `ChatSession(catalog:)` has
// to take the variant's hint on an iPhone, and a kit built before the key has to keep decoding the file.

import Foundation
import Testing

@testable import CoreAIKit
@testable import CoreAIKitCore

struct CatalogEngineHintTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // CoreAIKitTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repo root

    @Test func variantDecodesWithAndWithoutTheKey() throws {
        let decoder = JSONDecoder()
        let with = try decoder.decode(
            CatalogEntry.Variant.self, from: Data(#"{"path": "ios", "engine": "static-shape"}"#.utf8))
        #expect(with == CatalogEntry.Variant(path: "ios", engine: "static-shape"))
        let without = try decoder.decode(
            CatalogEntry.Variant.self, from: Data(#"{"path": "macos", "sizeMB": 10}"#.utf8))
        #expect(without.engine == nil && without.sizeMB == 10)
    }

    /// A catalog written before the key (every engine on the entry) reads as before: no variant
    /// hint, so each platform gets the entry's.
    @Test func catalogWithoutTheKeyFallsBackToTheEntry() throws {
        let json = """
            {"version": 1, "models": [
              {"id": "p", "name": "P", "repo": "org/p", "kind": "chat",
               "variants": {"macos": {"path": "gpu-pipelined/p_decode", "sizeMB": 10},
                            "ios": {"path": "gpu-pipelined/p_decode", "sizeMB": 10}},
               "engine": "pipelined"},
              {"id": "q", "name": "Q", "repo": "org/q", "kind": "chat",
               "variants": {"macos": {"path": "macos"}, "ios": {"path": "ios"}}}
            ]}
            """
        let catalog = try JSONDecoder().decode(ModelCatalog.self, from: Data(json.utf8))
        for entry in catalog.models {
            #expect(entry.variants.values.allSatisfy { $0.engine == nil })
            #expect(entry.resolvedEngine(for: "macos") == entry.engine)
            #expect(entry.resolvedEngine(for: "ios") == entry.engine)
            #expect(entry.resolvedEngine == entry.engine)
        }
    }

    /// The variant type of a kit without the key (0.7.3: `path` and `sizeMB`) reads the shipped
    /// catalog.json, which every shipped kit fetches live: the unknown key is skipped, not an error.
    @Test func kitWithoutTheKeyDecodesTheShippedCatalog() throws {
        struct OldVariant: Decodable { let path: String; let sizeMB: Int? }
        struct OldEntry: Decodable { let id: String; let variants: [String: OldVariant]; let engine: String? }
        struct OldCatalog: Decodable { let version: Int; let models: [OldEntry] }
        let data = try Data(contentsOf: Self.repoRoot.appendingPathComponent("catalog.json"))
        let old = try JSONDecoder().decode(OldCatalog.self, from: data)
        let lfm3 = try #require(old.models.first { $0.id == "lfm3-1.7b" })
        #expect(lfm3.variants["ios"]?.path == "ios")
        #expect(lfm3.engine == "pipelined")
    }

    @Test func lfm3TakesTheStaticShapeEngineOnIPhoneOnly() throws {
        let data = try Data(contentsOf: Self.repoRoot.appendingPathComponent("catalog.json"))
        let shipped = try JSONDecoder().decode(ModelCatalog.self, from: data)
        for catalog in [shipped, ModelCatalog.builtin] {
            let lfm3 = try #require(catalog.entry(id: "lfm3-1.7b"))
            #expect(lfm3.engine == "pipelined")
            #expect(lfm3.variants["ios"] == CatalogEntry.Variant(path: "ios", engine: "static-shape"))
            #expect(lfm3.resolvedEngine(for: "ios") == "static-shape")
            #expect(lfm3.resolvedEngine(for: "macos") == "pipelined")
            // An entry whose variants name no engine: the entry's on both platforms.
            let qwen = try #require(catalog.entry(id: "qwen3.5-2b"))
            #expect(qwen.resolvedEngine(for: "ios") == "pipelined" && qwen.resolvedEngine(for: "macos") == "pipelined")
        }
        #if os(iOS)
        #expect(ModelCatalog.builtin.entry(id: "lfm3-1.7b")?.resolvedEngine == "static-shape")
        #else
        #expect(ModelCatalog.builtin.entry(id: "lfm3-1.7b")?.resolvedEngine == "pipelined")
        #endif
    }

    /// The engine `ChatSession(catalog:)` loads with — the consumer, not only the catalog: reading
    /// the entry's `engine` there would hand the iPhone's static bundle to the pipelined engine,
    /// which refuses a chunked static model at load, after the download.
    @available(macOS 27, iOS 27, *)
    @Test func chatSessionLoadsTheVariantsEngine() throws {
        let lfm3 = try #require(ModelCatalog.builtin.entry(id: "lfm3-1.7b"))
        #expect(ChatSession.engineVariant(for: lfm3, requested: .auto, platform: "ios") == .staticShape)
        #expect(ChatSession.engineVariant(for: lfm3, requested: .auto, platform: "macos") == .pipelined)
        // An explicit choice still wins.
        #expect(ChatSession.engineVariant(for: lfm3, requested: .sequential, platform: "ios") == .sequential)
        // No hint anywhere (the official Qwen3 repos): auto-detection on both platforms.
        let qwen = try #require(ModelCatalog.builtin.entry(id: "qwen3-0.6b"))
        #expect(ChatSession.engineVariant(for: qwen, requested: .auto, platform: "ios") == .auto)
        #expect(ChatSession.engineVariant(for: qwen, requested: .auto, platform: "macos") == .auto)
    }
}
