// GraphBundleTests.swift — which graph a bundle directory resolves to.
//
// The case that matters is the iOS one: a bundle published from an AOT build holds a
// `*.aimodelc` and nothing else. A resolver that looks for `.aimodel` alone finds nothing and
// falls back to a conventional name, handing the runtime the path of a file that was never
// published — which is how every catalog VLM failed to load on iOS with "Missing hash file".
// Empty directories stand in for the graphs; resolution is a filesystem question.

import Foundation
import Testing

@testable import CoreAIKit

struct GraphBundleTests {
    private func bundle(holding names: [String]) throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "graph-bundle-\(UUID().uuidString)")
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        for name in names {
            try fm.createDirectory(
                at: root.appending(path: name), withIntermediateDirectories: true)
        }
        return root
    }

    @Test func anAOTOnlyBundleResolvesToItsCompiledGraph() throws {
        let root = try bundle(holding: ["tower.h18p.aimodelc"])
        #expect(try GraphBundle.resolve(in: root).lastPathComponent == "tower.h18p.aimodelc")
    }

    @Test func aJITOnlyBundleResolvesToItsGraph() throws {
        let root = try bundle(holding: ["tower.aimodel"])
        #expect(try GraphBundle.resolve(in: root).lastPathComponent == "tower.aimodel")
    }

    @Test func theCompiledGraphWinsWhenABundleCarriesBoth() throws {
        // Picking the JIT form here would pay on-device specialization every cold start —
        // the cost the AOT build exists to remove.
        let root = try bundle(holding: ["tower.aimodel", "tower.h18p.aimodelc"])
        #expect(try GraphBundle.resolve(in: root).lastPathComponent == "tower.h18p.aimodelc")
    }

    @Test func aGraphPassedDirectlyIsUsedAsIs() throws {
        let root = try bundle(holding: ["tower.h18p.aimodelc"])
        let graph = root.appending(path: "tower.h18p.aimodelc")
        #expect(try GraphBundle.resolve(in: graph) == graph)
    }

    @Test func aBundleHoldingNoGraphSaysSo() throws {
        let root = try bundle(holding: ["tokenizer"])
        #expect(throws: KitBundleError.self) { try GraphBundle.resolve(in: root) }
    }
}
