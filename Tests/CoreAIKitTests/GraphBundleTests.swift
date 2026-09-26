// GraphBundleTests.swift — which graph a bundle directory resolves to, per device architecture.
//
// A compiled graph loads on the one architecture it was compiled for: the iPhone 18 Pro (`h19p`)
// refuses the 17 Pro's `.h18p.aimodelc` and falls back to nothing. So the resolver takes the AOT
// graph only when it was compiled for this device, and the JIT `.aimodel` otherwise; with neither,
// it says which architectures the bundle was built for instead of handing the runtime a graph it
// will refuse. The architecture is injected, so every device's answer is checked on one machine.
// Empty directories stand in for the graphs, and for the compiled files a graph holds.

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

    private let everyForm = ["x.aimodel", "x.h18p.aimodelc", "x.h19p.aimodelc"]

    @available(macOS 27, iOS 27, *)
    @Test func theDevicesOwnCompiledGraphWins() throws {
        let root = try bundle(holding: everyForm)
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h19p").lastPathComponent == "x.h19p.aimodelc")
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h18p").lastPathComponent == "x.h18p.aimodelc")
    }

    @available(macOS 27, iOS 27, *)
    @Test func aDeviceNoGraphWasCompiledForTakesTheJITGraph() throws {
        // An M4 Mac, or an iPhone generation after the ones the bundle was compiled for.
        let root = try bundle(holding: everyForm)
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h16c").lastPathComponent == "x.aimodel")
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h20p").lastPathComponent == "x.aimodel")
    }

    @available(macOS 27, iOS 27, *)
    @Test func anotherDevicesCompiledGraphIsNeverPicked() throws {
        let root = try bundle(holding: ["x.aimodel", "x.h18p.aimodelc"])
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h19p").lastPathComponent == "x.aimodel")
    }

    @available(macOS 27, iOS 27, *)
    @Test func compiledGraphsForOtherDevicesAloneSayWhichDevicesTheyAreFor() throws {
        let root = try bundle(holding: ["x.h18p.aimodelc", "tokenizer"])
        #expect(throws: KitBundleError.self) { try GraphBundle.resolve(in: root, deviceArchitecture: "h19p") }
        do {
            _ = try GraphBundle.resolve(in: root, deviceArchitecture: "h19p")
        } catch {
            #expect(error.localizedDescription == "Bundle \(root.lastPathComponent) holds AOT graphs for h18p; "
                + "this device is h19p; no .aimodel to specialize.")
        }
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h18p").lastPathComponent == "x.h18p.aimodelc")
    }

    @available(macOS 27, iOS 27, *)
    @Test func aJITOnlyBundleResolvesToItsGraphOnEveryDevice() throws {
        let root = try bundle(holding: ["tower.aimodel"])
        for arch in ["h18p", "h19p", "h16c"] {
            #expect(try GraphBundle.resolve(in: root, deviceArchitecture: arch).lastPathComponent == "tower.aimodel")
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func theArchitectureInsideTheGraphIsReadWhenTheNameCarriesNone() throws {
        // `coreai-build compile` output keeps the function's compiled files under the arch name;
        // a bundle can name its compiled graph `x.aimodelc` alone.
        let root = try bundle(holding: ["x.aimodel", "x.aimodelc/main-h19p-delegates"])
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h19p").lastPathComponent == "x.aimodelc")
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h18p").lastPathComponent == "x.aimodel")
    }

    @available(macOS 27, iOS 27, *)
    @Test func aJITDirectoryCarryingAnotherDevicesCompiledFilesIsNoFallback() throws {
        // The shipped GLiNER2-PII `ios/` layout: the runtime checks the compiled files first and
        // refuses the directory on an 18 Pro, IR and all.
        let root = try bundle(holding: ["x.aimodel/main-h18p-delegates"])
        #expect(throws: KitBundleError.self) { try GraphBundle.resolve(in: root, deviceArchitecture: "h19p") }
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h18p").lastPathComponent == "x.aimodel")
    }

    @available(macOS 27, iOS 27, *)
    @Test func aGraphPassedDirectlyIsUsedAsIs() throws {
        let root = try bundle(holding: ["tower.h18p.aimodelc"])
        let graph = root.appending(path: "tower.h18p.aimodelc")
        #expect(try GraphBundle.resolve(in: graph, deviceArchitecture: "h19p") == graph)
    }

    @available(macOS 27, iOS 27, *)
    @Test func aBundleHoldingNoGraphSaysSo() throws {
        let root = try bundle(holding: ["tokenizer"])
        #expect(throws: KitBundleError.self) { try GraphBundle.resolve(in: root, deviceArchitecture: "h19p") }
    }

    @available(macOS 27, iOS 27, *)
    @Test func aNamedGraphIsPickedAmongItsOwnForms() throws {
        let root = try bundle(holding: everyForm + ["y.aimodel", "y.h19p.aimodelc"])
        #expect(try GraphBundle.graph(named: "x", in: root, deviceArchitecture: "h19p")?.lastPathComponent == "x.h19p.aimodelc")
        #expect(try GraphBundle.graph(named: "y", in: root, deviceArchitecture: "h18p")?.lastPathComponent == "y.aimodel")
        #expect(try GraphBundle.graph(named: "z", in: root, deviceArchitecture: "h19p") == nil)
    }

    @available(macOS 27, iOS 27, *)
    @Test func graphsBelowTheBundleAreFoundWhenAsked() throws {
        let root = try bundle(holding: ["ios/x.aimodel", "ios/x.h19p.aimodelc"])
        #expect(throws: KitBundleError.self) { try GraphBundle.resolve(in: root, deviceArchitecture: "h19p") }
        #expect(try GraphBundle.resolve(in: root, deviceArchitecture: "h19p", searchSubdirectories: true)
            .lastPathComponent == "x.h19p.aimodelc")
    }
}
