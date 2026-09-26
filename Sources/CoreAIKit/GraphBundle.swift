// GraphBundle.swift — the one rule for which graph of a bundle directory a device loads, shared by
// every loader that finds its own graph. Each of them used to carry a copy of this, and the copies
// drifted twice: the VL one looked for `.aimodel` alone, so an iOS bundle that ships only the AOT
// form was handed the path of a file that was never published ("Missing hash file"); later most of
// them took "AOT on iOS" to mean the iPhone 17 Pro's `.h18p.aimodelc`, which every other iPhone
// refuses.
//
// A compiled (AOT) graph loads on the one architecture it was compiled for (`h18p` is the iPhone
// 17 Pro, `h19p` the 18 Pro, `h16c` an M4 Mac). Anywhere else the runtime refuses it
// (`incompatibleCompiledAssetArchitecture`) and falls back to nothing. A JIT graph (`.aimodel`)
// loads everywhere: the device specializes it on the first load and caches the result. So: the
// compiled graph when it was compiled for this device, else the JIT one.

import CoreAI
import Foundation

/// A bundle directory that does not hold a graph this device can load.
public enum KitBundleError: Error, LocalizedError {
    case graphMissing(URL)
    /// Every graph in the directory was compiled for another architecture, and there is no JIT
    /// `.aimodel` for this device to specialize.
    case noGraphForDevice(URL, compiledFor: [String], device: String)

    public var errorDescription: String? {
        switch self {
        case .graphMissing(let url):
            return "Bundle \(url.lastPathComponent) holds no .aimodel or .aimodelc graph."
        case .noGraphForDevice(let url, let archs, let device):
            return "Bundle \(url.lastPathComponent) holds AOT graphs for "
                + "\(archs.joined(separator: ", ")); this device is \(device); "
                + "no .aimodel to specialize."
        }
    }
}

@available(macOS 27, iOS 27, *)
enum GraphBundle {
    /// The architecture Core AI compiles for on this device: `h19p` on an iPhone 18 Pro, `h18p` on
    /// a 17 Pro, `h16c` on an M4 Mac.
    static var deviceArchitecture: String { AIModel.deviceArchitectureName }

    /// The graph at `url`, or the one inside it this device loads: a compiled graph when it was
    /// compiled for `deviceArchitecture`, else the JIT `.aimodel`. A `url` that is itself a graph
    /// is the caller's choice and comes back as is. `searchSubdirectories` also looks below the
    /// directories that are not graphs.
    static func resolve(
        in url: URL, deviceArchitecture: String = deviceArchitecture,
        searchSubdirectories: Bool = false
    ) throws -> URL {
        if isGraph(url) { return url }
        let found = graphs(in: url, searchSubdirectories: searchSubdirectories)
        guard let graph = pick(found, deviceArchitecture: deviceArchitecture) else {
            throw unloadable(found, in: url, deviceArchitecture: deviceArchitecture)
        }
        return graph
    }

    /// The graph called `name` in `dir` — `<name>.aimodel`, `<name>.aimodelc` or
    /// `<name>.<arch>.aimodelc` — by the same rule; nil when `dir` holds no graph of that name.
    static func graph(
        named name: String, in dir: URL, deviceArchitecture: String = deviceArchitecture
    ) throws -> URL? {
        let found = graphs(in: dir, searchSubdirectories: false).filter { stem(of: $0) == name }
        guard !found.isEmpty else { return nil }
        guard let graph = pick(found, deviceArchitecture: deviceArchitecture) else {
            throw unloadable(found, in: dir, deviceArchitecture: deviceArchitecture)
        }
        return graph
    }

    /// The one of `graphs` this device loads: one compiled for it; else a JIT `.aimodel`; else a
    /// compiled one that names no architecture, for the runtime to judge. nil when every graph was
    /// compiled for another architecture, or there is none.
    static func pick(_ graphs: [URL], deviceArchitecture: String = deviceArchitecture) -> URL? {
        let sorted = graphs.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let archs = sorted.map(compiledArchitectures(of:))
        if let i = archs.firstIndex(where: { $0.contains(deviceArchitecture) }) { return sorted[i] }
        let unlabeled = sorted.indices.filter { archs[$0].isEmpty }
        let jit = unlabeled.first { sorted[$0].pathExtension == "aimodel" }
        return (jit ?? unlabeled.first).map { sorted[$0] }
    }

    /// Why no graph of `graphs` loads here: they were all compiled for other architectures
    /// (`noGraphForDevice`), or there is none (`graphMissing`).
    static func unloadable(
        _ graphs: [URL], in dir: URL, deviceArchitecture: String = deviceArchitecture
    ) -> KitBundleError {
        let archs = Set(graphs.flatMap(compiledArchitectures(of:)))
        guard !archs.isEmpty else { return .graphMissing(dir) }
        return .noGraphForDevice(dir, compiledFor: archs.sorted(), device: deviceArchitecture)
    }

    /// Why `dir` holds nothing this device loads, for a loader that needs several graphs laid out
    /// by name and found none of them: see `unloadable(_:in:deviceArchitecture:)`.
    static func unloadable(
        _ dir: URL, deviceArchitecture: String = deviceArchitecture
    ) -> KitBundleError {
        unloadable(
            graphs(in: dir, searchSubdirectories: true), in: dir,
            deviceArchitecture: deviceArchitecture)
    }

    /// The architectures `graph` was compiled for. What the runtime checks is inside it — the
    /// `<arch>` of each `<function>-<arch>.mlirb` and `<function>-<arch>-delegates/` — so that wins;
    /// otherwise the one a `<name>.<arch>.aimodelc` carries in its name. Empty for a JIT graph. A
    /// `.aimodel` directory holding compiled files (the layout `coreai-build compile` leaves when it
    /// writes into one) counts as compiled for their architecture: the runtime checks them first
    /// and refuses the directory on any other device, IR and all.
    static func compiledArchitectures(of graph: URL) -> [String] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: graph.path)) ?? []
        let inside = Set(names.compactMap(architecture(ofCompiledFile:)))
        if !inside.isEmpty { return inside.sorted() }
        return architecture(inName: graph).map { [$0] } ?? []
    }

    // MARK: - Names

    static func isGraph(_ url: URL) -> Bool {
        url.pathExtension == "aimodel" || url.pathExtension == "aimodelc"
    }

    /// Every graph in `dir`, not looking inside the graphs themselves.
    static func graphs(in dir: URL, searchSubdirectories: Bool) -> [URL] {
        let fm = FileManager.default
        guard searchSubdirectories else {
            let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
            return items.filter(isGraph)
        }
        var found: [URL] = []
        if let it = fm.enumerator(at: dir, includingPropertiesForKeys: nil) {
            for case let url as URL in it where isGraph(url) {
                found.append(url)
                it.skipDescendants()
            }
        }
        return found
    }

    /// A graph's name without its extension and architecture: `x` for `x.aimodel`, `x.aimodelc`
    /// and `x.h19p.aimodelc`.
    static func stem(of graph: URL) -> String {
        let base = graph.deletingPathExtension()
        return architecture(inName: graph) == nil
            ? base.lastPathComponent : base.deletingPathExtension().lastPathComponent
    }

    /// `h19p` for `x.h19p.aimodelc`; nil for a JIT graph and a compiled one without it.
    private static func architecture(inName graph: URL) -> String? {
        guard graph.pathExtension == "aimodelc" else { return nil }
        let arch = graph.deletingPathExtension().pathExtension
        return isArchitecture(arch[...]) ? arch : nil
    }

    /// `h19p` for `main-h19p.mlirb` and `main-h19p-delegates`; nil for `main.mlirb`.
    private static func architecture(ofCompiledFile name: String) -> String? {
        let body: Substring
        if name.hasSuffix(".mlirb") {
            body = name.dropLast(".mlirb".count)
        } else if name.hasSuffix("-delegates") {
            body = name.dropLast("-delegates".count)
        } else {
            return nil
        }
        guard let dash = body.lastIndex(of: "-"), dash > body.startIndex else { return nil }
        let arch = body[body.index(after: dash)...]
        return isArchitecture(arch) ? String(arch) : nil
    }

    /// Core AI's architecture names are `h`, the device generation and a letter for the family
    /// (`h18p`, `h19p`, `h16c`). A name outside that shape is not read as one, so an unknown one
    /// only leaves a compiled graph unlabeled, which the rule tries after the JIT graph.
    private static func isArchitecture(_ s: Substring) -> Bool {
        let u = Array(s.utf8)
        guard u.count >= 3, u.first == UInt8(ascii: "h"), let last = u.last,
            (UInt8(ascii: "a")...UInt8(ascii: "z")).contains(last)
        else { return false }
        return u.dropFirst().dropLast().allSatisfy { (UInt8(ascii: "0")...UInt8(ascii: "9")).contains($0) }
    }
}
