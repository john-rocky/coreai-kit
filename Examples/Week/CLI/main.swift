// week-cli — the week demo headless: plans a synthetic calendar week, or a file of events, with
// decider-0.8b through the app's WeekPlanner — no EventKit, no UI. Progress and every answer go to
// stderr; stdout gets one JSON line (the summary), `--out` the whole result, the same fields the
// app writes to Documents/week-result-<epoch>.json.
//
//   swift run -c release week-cli run --count 20 --seed 7 --bundle <dir> --out results/mac-week-20.json
//   swift run -c release week-cli run --events week.json --out result.json
//   swift run week-cli dump --count 20 --seed 7 [--out week.json]     # the events only, no model
//
// Without --bundle the catalog's decider-0.8b is downloaded once (1.34 GB) and cached. The
// bundle is loaded, then one throwaway decision warms the engine, then the clock starts. Exit 0 =
// planned, 1 = usage or load error.

import CoreAIKit
import Foundation
import WeekCore

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func fail(_ s: String) -> Never { err(s); exit(1) }

let usage = """
    usage: week-cli run [--count 20] [--seed 7] [--events <json>] [--bundle <dir>] [--out <json>]
           week-cli dump [--count 20] [--seed 7] [--out <json>]
    (no --bundle: the catalog's decider-0.8b, downloaded once and cached)
    """

var args = CommandLine.arguments.dropFirst()
guard let command = args.popFirst(), ["run", "dump"].contains(command) else { fail(usage) }
var count = 20
var seed: UInt64 = 7
var eventsPath: String?
var bundlePath: String?
var outPath: String?
while let a = args.popFirst() {
    switch a {
    case "--count":
        guard let n = args.popFirst().flatMap(Int.init), n > 0 else { fail(usage) }
        count = n
    case "--seed":
        guard let n = args.popFirst().flatMap(UInt64.init) else { fail(usage) }
        seed = n
    case "--events": eventsPath = args.popFirst()
    case "--bundle": bundlePath = args.popFirst()
    case "--out": outPath = args.popFirst()
    default: fail("unknown argument \(a)\n\(usage)")
    }
}

func expand(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }

func write(_ text: String, to path: String) {
    let url = expand(path)
    do {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data((text + "\n").utf8).write(to: url)
        err("[week] wrote \(url.path)")
    } catch {
        fail("[week] cannot write \(url.path): \(error.localizedDescription)")
    }
}

let events: [WeekEvent]
if let eventsPath {
    do {
        let data = try Data(contentsOf: expand(eventsPath))
        events = try JSONDecoder().decode([WeekEvent].self, from: data).enumerated().map { $0.element.with(id: $0.offset) }
    } catch {
        fail("[week] cannot read \(eventsPath): \(error)")
    }
} else {
    events = Week.generate(count: count, seed: seed)
}

if command == "dump" {
    for e in events { print(e.state) }
    if let outPath {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        write(String(decoding: try enc.encode(events), as: UTF8.self), to: outPath)
    }
    exit(0)
}

// MARK: - run

err("[week] loading \(bundlePath ?? "\(WeekQuestion.catalogID) (catalog)") …")
let t0 = ContinuousClock.now
let decider: TypedDecisions
do {
    decider = try await WeekPlanner.load(bundle: bundlePath.map(expand)) { p in
        err(String(format: "[week] downloading %.0f%%", p.fraction * 100))
    }
} catch {
    fail("[week] load failed: \(error.localizedDescription)")
}
let loadSeconds = t0.duration(to: .now).inSeconds
let planner = WeekPlanner(decider: decider)
let warmSeconds: Double
do { warmSeconds = try await planner.warm() } catch { fail("[week] warm-up failed: \(error.localizedDescription)") }
let modelName = await decider.modelName
err(String(format: "[week] %@ loaded in %.2f s, format %@, T %.2f; warm-up decision %.2f s",
           modelName, loadSeconds, decider.format.rawValue, decider.temperature, warmSeconds))

err("[week] planning \(events.count) events (\(eventsPath.map { "from \($0)" } ?? "seed \(seed)")) …")
let thermalBefore = ProcessInfo.processInfo.thermalState
let run: PlanRun
do {
    let byIndex = Dictionary(uniqueKeysWithValues: events.map { ($0.id, $0) })
    let total = events.count
    run = try await planner.plan(events) { r in
        let e = byIndex[r.index]!
        err(String(format: "[week] %2d/%d %@ %@ %@ → %@ %.2f · %.0f ms · %d tokens", r.index + 1, total,
                   e.dayName, e.time, e.title, r.bin.rawValue, r.confidence, r.seconds * 1000, r.promptTokens))
    }
} catch {
    fail("[week] \(error.localizedDescription)")
}
let thermalAfter = ProcessInfo.processInfo.thermalState

var out = run.summary(events: events)
out["seed"] = eventsPath == nil ? .int(Int(seed)) : .null
out["events_file"] = eventsPath.map { .string($0) } ?? .null
out["model"] = .string(WeekQuestion.catalogID)
out["bundle"] = .string(bundlePath.map { expand($0).path } ?? "catalog")
out["source"] = .string(bundlePath == nil ? "catalog" : "bundle")
if bundlePath == nil, let entry = try? await ModelCatalog.entry(forID: WeekQuestion.catalogID) {
    out["bundle_revision"] = entry.revision.map { .string($0) } ?? .null
} else {
    out["bundle_revision"] = .null
}
out["bundle_compiled"] = bundlePath.flatMap { Device.compiled(expand($0)) }.map { .string($0) } ?? .null
out["format"] = .string(decider.format.rawValue)
out["temperature"] = .rounded(decider.temperature, 3)
out["device"] = .string(Device.model)
out["machine"] = .string(Device.machine)
out["os"] = .string(Device.os)
out["os_build"] = .string(ProcessInfo.processInfo.operatingSystemVersionString)
out["compute_units"] = .string("gpu")
out["load_s"] = .rounded(loadSeconds, 3)
out["warm_s"] = .rounded(warmSeconds, 3)
out["store"] = .string("none (week-cli: the generator's week, no EventKit)")
out["reminders_added"] = .int(0)
out["thermal_before"] = .string(Device.name(thermalBefore))
out["thermal_after"] = .string(Device.name(thermalAfter))
out["timestamp"] = .string(ISO8601DateFormatter().string(from: Date()))

err(String(format: "[week] DONE %d events · %.2f s · median %.0f ms · p90 %.0f ms · %.2f events/s", run.count,
           run.totalSeconds, Stats.percentile(run.milliseconds, 0.5), Stats.percentile(run.milliseconds, 0.9),
           run.eventsPerSecond))
let bins = run.binCounts
err("[week] " + WeekBin.allCases.map { "\($0.label) \(bins[$0] ?? 0)" }.joined(separator: " · "))
if let agree = run.agreement(with: events) {
    err(String(format: "[week] agrees with the generator's own label on %.0f%%", agree * 100))
}

var summary = out
summary["series"] = nil
summary["answers"] = nil
print(JSONValue.object(summary).json())
if let outPath { write(JSONValue.object(out).json(pretty: true), to: outPath) }
