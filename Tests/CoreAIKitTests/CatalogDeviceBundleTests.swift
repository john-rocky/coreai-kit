// CatalogDeviceBundleTests.swift — whether an iPhone has a bundle of a catalog entry, read from the
// catalog alone (`CatalogEntry.unavailableReason`), so a picker marks the model before any download:
// LFM3 1.7B's repo has `ios-h19p/` and no `ios/`, and its `ios` variant says so in `architectures`.
// The device architecture is injected, so every row runs on the Mac; the kits already shipped have
// to keep decoding the file with the new key in it.

import Foundation
import Testing

@testable import CoreAIKitCore

struct CatalogDeviceBundleTests {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // CoreAIKitTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // repo root

    private static func shipped() throws -> ModelCatalog {
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("catalog.json"))
        return try JSONDecoder().decode(ModelCatalog.self, from: data)
    }

    private static func entry(_ variants: [String: CatalogEntry.Variant], name: String = "M") -> CatalogEntry {
        CatalogEntry(id: "m", name: name, repo: "org/m", kind: .chat, variants: variants)
    }

    @Test func variantDecodesWithAndWithoutTheKey() throws {
        let decoder = JSONDecoder()
        let with = try decoder.decode(
            CatalogEntry.Variant.self,
            from: Data(#"{"path": "ios", "sizeMB": 2164, "architectures": ["h19p"]}"#.utf8))
        #expect(with == CatalogEntry.Variant(path: "ios", sizeMB: 2164, architectures: ["h19p"]))
        let without = try decoder.decode(
            CatalogEntry.Variant.self, from: Data(#"{"path": "ios", "sizeMB": 10}"#.utf8))
        #expect(without.architectures == nil)
    }

    @Test func lfm3HasABundleOnTheIPhone18ProOnly() throws {
        for catalog in [try Self.shipped(), ModelCatalog.builtin] {
            let lfm3 = try #require(catalog.entry(id: "lfm3-1.7b"))
            #expect(lfm3.variants["ios"]?.architectures == ["h19p"])
            #expect(lfm3.unavailableReason(platform: "ios", deviceArchitecture: "h19p") == nil)
            #expect(lfm3.unavailableReason(platform: "ios", deviceArchitecture: "h18p")
                == "LFM3 1.7B's iPhone version is built for the iPhone 18 Pro only.")
            #expect(lfm3.unavailableReason(platform: "ios", deviceArchitecture: "h20p") != nil)
            // A phone Core AI names no architecture for, and the Mac, are not refused here.
            #expect(lfm3.unavailableReason(platform: "ios", deviceArchitecture: nil) == nil)
            #expect(lfm3.unavailableReason(platform: "macos", deviceArchitecture: "h18p") == nil)
        }
    }

    /// A path inside an `ios-<arch>/` folder is that architecture's alone, the way the store
    /// downloads it (`ModelID.subtrees` never swaps a named folder).
    @Test func aPathInsideAnArchitectureFolderIsThatArchitecturesOnly() throws {
        let nemotron = try #require(try Self.shipped().entry(id: "nemotron-3-nano-4b"))
        #expect(nemotron.variants["ios"]?.path.hasPrefix("ios-h18p/") == true)
        #expect(nemotron.unavailableReason(platform: "ios", deviceArchitecture: "h18p") == nil)
        #expect(nemotron.unavailableReason(platform: "ios", deviceArchitecture: "h19p")
            == "Nemotron-3-Nano 4B's iPhone version is built for the iPhone 17 Pro only.")
        for (path, arch) in [("ios-h18p/x_decode", "h18p"), ("ios-h19p", "h19p"), ("ios-h20/x", "h20")] {
            #expect(CatalogEntry.architecture(ofFolderIn: path) == arch)
        }
        for path in ["ios", "ios/wfp16-s256", "ios-frontend", "ios-static/x", "ios-ane-h18p", "ios-hx",
                     "ios-h", "ios-h18p.bak/x", "gpu-pipelined/x_decode", "macos", ""] {
            #expect(CatalogEntry.architecture(ofFolderIn: path) == nil, "\(path)")
        }
    }

    @Test func aVariantThatDeclaresNothingIsOfferedToEveryIPhone() {
        for variant in [CatalogEntry.Variant(path: "ios"), .init(path: "ios-frontend"),
                        .init(path: "gpu-pipelined/x_decode"), .init(path: "ios/wfp16-s256")] {
            let e = Self.entry(["ios": variant, "macos": .init(path: "macos")])
            for arch in ["h18p", "h19p", nil] {
                #expect(e.unavailableReason(platform: "ios", deviceArchitecture: arch) == nil)
            }
        }
        // The list is read only for the `ios` path; a named folder answers for itself.
        let named = Self.entry(["ios": .init(path: "ios-h18p/x", architectures: ["h19p"])])
        #expect(named.unavailableReason(platform: "ios", deviceArchitecture: "h18p") == nil)
        // An architecture with no name yet still gets a sentence.
        let future = Self.entry(["ios": .init(path: "ios", architectures: ["h18p", "h21p"])], name: "F")
        #expect(future.unavailableReason(platform: "ios", deviceArchitecture: "h19p")
            == "F's iPhone version is built for other iPhones only.")
        let both = Self.entry(["ios": .init(path: "ios", architectures: ["h18p", "h19p"])], name: "B")
        #expect(both.unavailableReason(platform: "ios", deviceArchitecture: "h19p") == nil)
        #expect(both.unavailableReason(platform: "ios", deviceArchitecture: "h20p")
            == "B's iPhone version is built for the iPhone 17 Pro and the iPhone 18 Pro only.")
    }

    /// Every arch-specific entry of the shipped file gets a sentence on the other generation, and
    /// none of the rest is refused anywhere: the catalog itself decides, not a table in the kit.
    @Test func theShippedCatalogRefusesOnlyItsArchitectureSpecificBundles() throws {
        let catalog = try Self.shipped()
        var refused: [String: Set<String>] = [:]
        for entry in catalog.models {
            for arch in ["h18p", "h19p"]
            where entry.unavailableReason(platform: "ios", deviceArchitecture: arch) != nil {
                refused[entry.id, default: []].insert(arch)
            }
        }
        #expect(refused == [
            "lfm3-1.7b": ["h18p"],
            "nemotron-3-nano-4b": ["h19p"],
            "lfm2.5-vl-450m": ["h19p"],
            "lfm2.5-vl-3b": ["h19p"],
            "north-micro-vision": ["h19p"],
        ])
    }

    /// The variant types of the kits already shipped read the file with the key in it: 0.7.3
    /// (`path`, `sizeMB`) and the 2.0.2 app's kit (`path`, `sizeMB`, `engine`). Synthesized
    /// `Decodable` skips a key it does not know.
    @Test func shippedKitsDecodeTheFileWithTheKey() throws {
        struct V073: Decodable { let path: String; let sizeMB: Int? }
        struct V202: Decodable { let path: String; let sizeMB: Int?; let engine: String? }
        struct Entry<V: Decodable>: Decodable { let id: String; let variants: [String: V] }
        struct Catalog<V: Decodable>: Decodable { let version: Int; let models: [Entry<V>] }
        let data = try Data(contentsOf: Self.repoRoot.appendingPathComponent("catalog.json"))
        let old = try JSONDecoder().decode(Catalog<V073>.self, from: data)
        let app202 = try JSONDecoder().decode(Catalog<V202>.self, from: data)
        let current = try Self.shipped()
        #expect(old.models.count == current.models.count && app202.models.count == current.models.count)
        let lfm3 = try #require(app202.models.first { $0.id == "lfm3-1.7b" }?.variants["ios"])
        #expect(lfm3.path == "ios" && lfm3.sizeMB == 2164 && lfm3.engine == "static-shape")
        #expect(old.models.first { $0.id == "lfm3-1.7b" }?.variants["ios"]?.path == "ios")
    }
}
