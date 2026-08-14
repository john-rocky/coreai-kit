// coreai-doctor — what will this app download, and can the device do it.
//
//     swift run coreai-doctor path/to/MyApp     # scan an app's sources
//     swift run coreai-doctor --all             # every op
//     swift run coreai-doctor --ops transcribe,caption
//
// The number an app engineer is asked for before shipping is "how big does this make the
// first run", and answering it meant reading `catalog.json` and knowing which op resolves to
// which model. That mapping lives in `CoreAI.Op` — so this asks the kit rather than
// reimplementing it, which is the difference between a tool that stays right and one that
// drifts the first time a default changes.
//
// It is deliberately a *scanner*, not a linker: it looks for `CoreAI.<op>` in Swift sources.
// That over-reports an op mentioned in a comment and misses one reached through a variable,
// so it prints what it found and where, and `--ops` is there for when you would rather say.

import CoreAIOps
import Foundation

func stderrPrint(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func formatted(_ bytes: Int64) -> String {
    bytes >= 1_000_000_000
        ? String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
        : String(format: "%.0f MB", Double(bytes) / 1_000_000)
}

/// Ops named in the Swift sources under `root`, with the files that named them.
func scan(_ root: URL) -> [CoreAI.Op: [String]] {
    let names = Dictionary(uniqueKeysWithValues: CoreAI.Op.allCases.map { ("\($0)", $0) })
    var found: [CoreAI.Op: [String]] = [:]
    guard let walker = FileManager.default.enumerator(
        at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    else { return [:] }
    for case let url as URL in walker {
        // Anything under a build directory is a copy of source that is already being scanned,
        // or a dependency's — either way it is not this app.
        if url.pathComponents.contains(where: { $0.hasPrefix(".build") || $0 == "build" }) {
            walker.skipDescendants()
            continue
        }
        guard url.pathExtension == "swift",
            let text = try? String(contentsOf: url, encoding: .utf8)
        else { continue }
        for (name, op) in names where text.contains("CoreAI.\(name)(") {
            found[op, default: []].append(
                url.path.replacingOccurrences(of: root.path + "/", with: ""))
        }
    }
    return found
}

func describe(_ capability: Capability) -> String {
    switch capability {
    case .ready: return "ready — nothing to download"
    case .needsDownload(let bytes): return formatted(bytes)
    case .needsSystemAssets(let what): return "OS assets (\(what)) — not the app's bytes"
    case .insufficientStorage(let needs, let free):
        return "NO ROOM — needs \(formatted(needs)), \(formatted(free)) free"
    case .unsupportedDevice(let reason): return "UNSUPPORTED — \(reason)"
    }
}

var target: URL?
var explicit: [CoreAI.Op] = []
var all = false
var arguments = CommandLine.arguments.dropFirst()
while let argument = arguments.popFirst() {
    switch argument {
    case "--all": all = true
    case "--ops":
        let names = (arguments.popFirst() ?? "").split(separator: ",").map(String.init)
        for name in names {
            guard let op = CoreAI.Op.allCases.first(where: { "\($0)" == name }) else {
                stderrPrint("unknown op '\(name)' — one of: "
                    + CoreAI.Op.allCases.map { "\($0)" }.joined(separator: ", "))
                exit(2)
            }
            explicit.append(op)
        }
    case "-h", "--help":
        print("""
            usage: coreai-doctor <app-source-dir>
                   coreai-doctor --ops transcribe,caption
                   coreai-doctor --all
            """)
        exit(0)
    default: target = URL(fileURLWithPath: argument)
    }
}

var ops: [CoreAI.Op]
var sources: [CoreAI.Op: [String]] = [:]
if all {
    ops = CoreAI.Op.allCases.sorted { "\($0)" < "\($1)" }
} else if !explicit.isEmpty {
    ops = explicit
} else if let target {
    guard FileManager.default.fileExists(atPath: target.path) else {
        stderrPrint("no such directory: \(target.path)")
        exit(1)
    }
    sources = scan(target)
    ops = sources.keys.sorted { "\($0)" < "\($1)" }
    if ops.isEmpty {
        print("No CoreAI ops found under \(target.lastPathComponent).")
        print("If the app reaches them through a variable, name them with --ops.")
        exit(0)
    }
} else {
    stderrPrint("usage: coreai-doctor <app-source-dir> | --ops a,b | --all")
    exit(2)
}

// The app pays for each distinct model once, however many ops share it — reporting the sum
// of the ops would triple-count the 2.3 GB chat model behind summarize, translate and
// proofread, which is the number an engineer would then take to a meeting.
var total: Int64 = 0
var counted: Set<String> = []
print(String(format: "%-20@ %@", "op" as NSString, "first-run cost" as NSString))
for op in ops {
    let capability = await CoreAI.capability(op)
    print(String(format: "%-20@ %@", "\(op)" as NSString, describe(capability) as NSString))
    if let files = sources[op], !files.isEmpty {
        print("                     used in \(files.prefix(3).joined(separator: ", "))"
            + (files.count > 3 ? " (+\(files.count - 3) more)" : ""))
    }
    if case .needsDownload(let bytes) = capability,
        let id = op.defaultModelID, counted.insert(id).inserted
    {
        total += bytes
    }
}
print(String(format: "\n%-20@ %@ across %d distinct model(s)",
             "total" as NSString, formatted(total) as NSString, counted.count))
if total == 0 {
    print("Nothing to download: every op here runs on what the device already has.")
}
