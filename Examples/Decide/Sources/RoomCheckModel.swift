// RoomCheckModel — the Room check screen: one photo per room after cleaning, five checks per photo, every check of
// a photo read in one pass by decider-2b-vision (Mapika, Apache-2.0), the image decider (`KitVisionDecider`). One
// press checks every room in the folder and gives each a verdict a host can act on.
//
// The five questions, their options (no, then yes) and the context are fixed word for word: they are the ones the
// model was measured on (30 generated rooms, 0.958 correct at g448), and a changed word needs a new measurement.
// The verdict is a rule fixed before any recording. A clean room has the bed made, towels on the bed, nothing on the
// floor, the lamp off and no suitcase. An answer at p ≥ 0.8 against that makes the room RECLEAN; an answer below 0.8
// is a row to check by hand, and its room is CHECK unless it is already RECLEAN; a room with every answer in place at
// p ≥ 0.8 is READY.
//
// The image model loads here and nowhere else — never through `DecideRuntime`, so an iPhone never holds it beside
// the text model (3.3 GB with one tower). The screen holds its own `KitVisionDecider` rather than calling
// `CoreAI.decide(image:…)`: `CoreAI.prepare` loads the 256 tower only, and this screen's grid (448 unless
// `-grid 256`) is loaded, and warmed with one decision on a blank image, before READY. Each room is timed here: the
// wall clock from reading its PNG to the answers. Every number on the screen is computed from the run it records
// (`-log 1`: Documents/room-result-<epoch>.json).
//
// Rooms are the PNGs in Documents/rooms (`-rooms <dir>` for another folder), in file-name order. The kit applies no
// EXIF orientation, so a camera photo reads as stored.

import CoreAIOps
import CoreGraphics
import Foundation
import ImageIO
import Observation

@MainActor
@Observable
final class RoomCheckModel {
    enum Phase: Equatable { case loading, ready, checking, done, failed }

    /// One check: the key the record uses, the question as the model reads it, the short label the screen shows,
    /// and the answer a clean room gives.
    struct Check: Sendable {
        let key: String
        let question: String
        let label: String
        let expected: String
    }

    nonisolated static let context = "This is a visual question about the image."
    nonisolated static let options = ["no", "yes"]
    nonisolated static let checks: [Check] = [
        Check(key: "bed", question: "Is the bed made neatly?", label: "Bed made?", expected: "yes"),
        Check(key: "towels", question: "Are there folded towels on the bed?", label: "Towels on the bed?", expected: "yes"),
        Check(
            key: "floor", question: "Is there clothing or trash on the floor?", label: "Clothes or trash on the floor?",
            expected: "no"),
        Check(key: "lamp", question: "Is the lamp turned on?", label: "Lamp on?", expected: "no"),
        Check(key: "suitcase", question: "Is there a suitcase left in the room?", label: "Suitcase left?", expected: "no"),
    ]
    /// In the order above: the array form keeps it (a keyed dictionary would be laid out in key order).
    nonisolated static let questions: [Decision.Question] = checks.map { .choice($0.question, options) }
    /// An answer below this probability goes to a person.
    nonisolated static let handCheckBelow = 0.8
    nonisolated static let modelID = "decider-2b-vision"

    enum Flag: String, Sendable { case ok, miss, check }

    /// One answered check.
    struct Row: Sendable, Identifiable {
        let check: Check
        /// In option order: no, yes.
        let probabilities: [Double]
        let answer: String
        /// Probability of `answer`.
        let p: Double

        var id: String { check.key }
        var flag: Flag {
            if p < RoomCheckModel.handCheckBelow { return .check }
            return answer == check.expected ? .ok : .miss
        }
    }

    enum Verdict: String, Sendable { case ready = "READY", reclean = "RECLEAN", check = "CHECK" }

    /// One room's pass: the rows, this screen's wall clock, and what the kit reports about the pass.
    struct Result: Sendable {
        let rows: [Row]
        /// Seconds from reading the PNG to the answers.
        let seconds: Double
        let tokens: Int
        /// The kit's seconds per stage ("decode", "resize", "patches", "tower", "decoder", "readout", "wall", …).
        let stages: [String: Double]
        let prefillCalls: Int
        let mainCalls: Int

        var verdict: Verdict {
            if rows.contains(where: { $0.flag == .miss }) { return .reclean }
            if rows.contains(where: { $0.flag == .check }) { return .check }
            return .ready
        }
    }

    struct Room: Identifiable {
        let id: Int
        let name: String
        let url: URL
        let image: CGImage?
    }

    /// The run's counts and times, computed once at DONE: the screen reads these, and the record writes them.
    struct Totals {
        let rooms: Int
        let checks: Int
        let ready: Int
        let reclean: Int
        let checkRooms: Int
        let checkRows: Int
        let totalSeconds: Double
        let medianRoomSeconds: Double
        let p90RoomSeconds: Double
    }

    let grid: KitVisionDecider.Grid
    let roomsDirectory: URL
    /// `-log 1`: write the result file at DONE.
    let record: Bool
    /// `-kit <revision>`: the kit revision the app was built from, for the record (the build does not know it).
    let kitVersion: String

    private(set) var phase = Phase.loading
    /// LOADING: the download percent or the step; FAILED: the error.
    private(set) var detail = ""
    private(set) var rooms: [Room] = []
    private(set) var results: [Int: Result] = [:]
    /// The room whose pass is running.
    private(set) var checking: Int?
    /// The room the large view shows: the last one answered (the first, before any is).
    private(set) var shown: Int?
    private(set) var startedAt: Date?
    private(set) var totals: Totals?
    private(set) var loadSeconds: Double?
    private(set) var warmSeconds: Double?
    private(set) var resultPath: String?
    /// A room opened by a tap.
    var selected: Int?

    private var decider: KitVisionDecider?
    private var modelName = ""
    private var modelRevision: String?
    private var loading: Task<Void, Never>?
    private var thermalBefore: ProcessInfo.ThermalState?
    private var thermalAfter: ProcessInfo.ThermalState?

    init() {
        let defaults = UserDefaults.standard  // -key value command-line pairs land here
        grid = KitVisionDecider.Grid(tile: defaults.integer(forKey: "grid")) ?? .g448
        roomsDirectory = defaults.string(forKey: "rooms").map {
            URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath, isDirectory: true)
        } ?? URL.documentsDirectory.appending(path: "rooms", directoryHint: .isDirectory)
        record = defaults.bool(forKey: "log")
        kitVersion = defaults.string(forKey: "kit") ?? "unknown"
    }

    var count: Int { rooms.count }
    var doneCount: Int { results.count }

    // MARK: - loading

    /// Reads the rooms, loads the model with this screen's tower, and warms it with one decision on a blank image
    /// (the row has the same length whatever the image, so every shape the rooms need is specialized here). Every
    /// caller awaits the same load, which a view's task being cancelled does not stop.
    func load() async {
        if loading == nil { loading = Task { await performLoad() } }
        await loading?.value
    }

    private func performLoad() async {
        let t0 = Date()
        do {
            rooms = try Self.rooms(in: roomsDirectory)
            detail = "0%"
            let decider = try await KitVisionDecider(catalog: Self.modelID, grids: [grid]) { progress in
                Task { @MainActor in
                    if self.phase == .loading, progress.fraction < 1 { self.detail = "\(Int(progress.fraction * 100))%" }
                }
            }
            loadSeconds = Date().timeIntervalSince(t0)
            modelName = decider.modelName
            modelRevision = try? await ModelCatalog.entry(forID: Self.modelID, expecting: .visionDecision).revision
            detail = "warming"
            let tWarm = Date()
            _ = try await decider.decide(
                image: Self.blank(side: 1024), state: Self.context, questions: Self.questions, grid: grid)
            warmSeconds = Date().timeIntervalSince(tWarm)
            self.decider = decider
            detail = ""
            phase = .ready
        } catch {
            detail = error.localizedDescription
            phase = .failed
        }
    }

    /// The PNGs in `directory`, in file-name order, each decoded once for the screen (the pass reads the file again,
    /// inside its clock).
    private static func rooms(in directory: URL) throws -> [Room] {
        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else { throw RoomCheckError.noRooms(directory.path) }
        return urls.enumerated().map { index, url in
            let source = CGImageSourceCreateWithURL(url as CFURL, nil)
            let image = source.flatMap {
                CGImageSourceCreateImageAtIndex($0, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
            }
            return Room(id: index, name: url.deletingPathExtension().lastPathComponent, url: url, image: image)
        }
    }

    /// A mid-grey square, for the warm-up decision.
    private static func blank(side: Int) -> CGImage {
        let context = CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(srgbRed: 0.5, green: 0.5, blue: 0.5, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        return context.makeImage()!
    }

    // MARK: - checking

    /// Checks every room from the first; DONE starts it over.
    func checkAll() {
        guard let decider, phase == .ready || phase == .done else { return }
        results = [:]
        totals = nil
        resultPath = nil
        selected = nil
        checking = 0
        shown = 0
        thermalBefore = ProcessInfo.processInfo.thermalState
        let start = Date()
        startedAt = start
        phase = .checking
        let rooms = self.rooms.map { ($0.id, $0.url) }
        let grid = self.grid
        Task {
            do {
                for (index, url) in rooms {
                    checking = index
                    results[index] = try await Self.check(url, decider: decider, grid: grid)
                    shown = index
                }
                finish(start: start)
            } catch {
                checking = nil
                detail = error.localizedDescription
                phase = .failed
            }
        }
    }

    /// One room: the PNG read from disk, the five questions in one pass, the answers — timed from the file to the
    /// read-out, off the main actor.
    @concurrent
    nonisolated private static func check(
        _ url: URL, decider: KitVisionDecider, grid: KitVisionDecider.Grid
    ) async throws -> Result {
        let t0 = Date()
        let image = try ImageFile.load(url).cgImage
        let readout = try await decider.readout(image: image, state: context, questions: questions, grid: grid)
        let seconds = Date().timeIntervalSince(t0)
        let rows = zip(checks, readout.answers).map { check, answer in
            let probabilities = answer.probabilities
            let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
            return Row(
                check: check, probabilities: probabilities, answer: answer.choice ?? options[best],
                p: probabilities[best])
        }
        return Result(
            rows: rows, seconds: seconds, tokens: readout.row.ids.count, stages: readout.seconds,
            prefillCalls: readout.prefillCalls, mainCalls: readout.mainCalls)
    }

    private func finish(start: Date) {
        let end = Date()
        thermalAfter = ProcessInfo.processInfo.thermalState
        let done = rooms.compactMap { results[$0.id] }
        let seconds = done.map(\.seconds).sorted()
        totals = Totals(
            rooms: done.count, checks: done.reduce(0) { $0 + $1.rows.count },
            ready: done.filter { $0.verdict == .ready }.count,
            reclean: done.filter { $0.verdict == .reclean }.count,
            checkRooms: done.filter { $0.verdict == .check }.count,
            checkRows: done.reduce(0) { $0 + $1.rows.filter { $0.flag == .check }.count },
            totalSeconds: end.timeIntervalSince(start), medianRoomSeconds: median(seconds),
            // the round-1 tools' p90: the sorted value at index ⌊0.9 (n − 1)⌋
            p90RoomSeconds: seconds.isEmpty ? 0 : seconds[Int(0.9 * Double(seconds.count - 1))])
        checking = nil
        if record { writeResult() }
        phase = .done
    }

    // MARK: - what the screen and the log read

    /// Seconds on the clock: from the press to now while checking, the run's total once done.
    func elapsed(at now: Date) -> Double? {
        switch phase {
        case .checking: return startedAt.map { now.timeIntervalSince($0) }
        case .done: return totals?.totalSeconds
        default: return nil
        }
    }

    /// Median seconds per room over the rooms answered so far.
    var medianSoFar: Double? {
        let seconds = results.values.map(\.seconds)
        return seconds.isEmpty ? nil : median(seconds)
    }

    /// The screen's state as one line, for the autoplay log.
    var statusLine: String {
        switch phase {
        case .loading: return "LOADING \(detail)"
        case .ready:
            return String(
                format: "READY %d rooms · %@ · load %.2f s · warm %.2f s", count, grid.description, loadSeconds ?? 0,
                warmSeconds ?? 0)
        case .checking:
            guard let last = shown, let result = results[last] else { return "CHECKING 0/\(count)" }
            return String(
                format: "CHECKING %d/%d · %@ %@ %.2f s", doneCount, count, rooms[last].name, result.verdict.rawValue,
                result.seconds)
        case .done:
            guard let t = totals else { return "DONE" }
            return String(
                format: "DONE %d/%d · %.2f s · median %.2f s · p90 %.2f s · %d ready · %d reclean · %d check · %d to check by hand%@",
                t.rooms, count, t.totalSeconds, t.medianRoomSeconds, t.p90RoomSeconds, t.ready, t.reclean,
                t.checkRooms, t.checkRows, resultPath.map { " · result \($0)" } ?? "")
        case .failed: return "FAILED \(detail)"
        }
    }

    var isFinished: Bool { phase == .done || phase == .failed }

    /// "Apple M4 Max" on a Mac, the model identifier on an iPhone.
    nonisolated static let deviceName: String = {
        #if os(macOS)
        let brand = sysctlString("machdep.cpu.brand_string")
        return brand.isEmpty ? sysctlString("hw.model") : brand
        #else
        return sysctlString("hw.machine")
        #endif
    }()

    nonisolated static let osName: String = {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        #if os(iOS)
        let name = "iOS"
        #else
        let name = "macOS"
        #endif
        return v.patchVersion > 0
            ? "\(name) \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)" : "\(name) \(v.majorVersion).\(v.minorVersion)"
    }()

    nonisolated private static func sysctlString(_ key: String) -> String {
        var size = 0
        sysctlbyname(key, nil, &size, nil, 0)
        var bytes = [UInt8](repeating: 0, count: max(size, 1))
        sysctlbyname(key, &bytes, &size, nil, 0)
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }

    nonisolated private static func thermal(_ state: ProcessInfo.ThermalState?) -> String {
        switch state {
        case .nominal?: return "nominal"
        case .fair?: return "fair"
        case .serious?: return "serious"
        case .critical?: return "critical"
        default: return "unknown"
        }
    }

    /// The finished run as the result file: every room's answers and seconds, the totals the screen shows, where it
    /// ran.
    func resultJSON() -> [String: Any] {
        let roomRecords: [[String: Any]] = rooms.compactMap { room in
            guard let r = results[room.id] else { return nil }
            return [
                "name": room.name,
                "verdict": r.verdict.rawValue,
                "seconds": r.seconds,
                "questions": r.rows.map { row -> [String: Any] in
                    [
                        "key": row.check.key, "text": row.check.question, "options": Self.options,
                        "probabilities": row.probabilities, "answer": row.answer, "p": row.p,
                        "flag": row.flag.rawValue, "expected": row.check.expected,
                    ]
                },
                "tokens": r.tokens,
                "stages_s": r.stages,
                "prefill_calls": r.prefillCalls,
                "main_calls": r.mainCalls,
            ]
        }
        var out: [String: Any] = [
            "schema": "decide-room-check/1",
            "grid": grid.description,
            "model": Self.modelID,
            "model_name": modelName,
            "model_revision": modelRevision ?? "unknown",
            "context": Self.context,
            "hand_check_below": Self.handCheckBelow,
            "rooms_dir": roomsDirectory.path,
            "rooms": roomRecords,
            "load_s": loadSeconds ?? 0,
            "warm_s": warmSeconds ?? 0,
            "device": [
                "name": Self.deviceName, "model": Self.sysctlString("hw.model"),
                "machine": Self.sysctlString("hw.machine"), "os": Self.osName,
                "os_build": ProcessInfo.processInfo.operatingSystemVersionString,
            ],
            "thermal_before": Self.thermal(thermalBefore),
            "thermal_after": Self.thermal(thermalAfter),
            "kit_version": kitVersion,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
        ]
        if let t = totals {
            out["totals"] = [
                "rooms": t.rooms, "checks": t.checks, "ready": t.ready, "reclean": t.reclean,
                "check_rooms": t.checkRooms, "check_rows": t.checkRows, "total_s": t.totalSeconds,
                "median_room_s": t.medianRoomSeconds, "p90_room_s": t.p90RoomSeconds,
            ]
        }
        return out
    }

    private func writeResult() {
        let url = URL.documentsDirectory.appending(path: "room-result-\(Int(Date().timeIntervalSince1970)).json")
        do {
            let data = try JSONSerialization.data(
                withJSONObject: resultJSON(), options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
            try data.write(to: url)
            resultPath = url.path
        } catch {
            resultPath = "not written: \(error.localizedDescription)"
        }
    }
}

enum RoomCheckError: LocalizedError {
    case noRooms(String)

    var errorDescription: String? {
        if case .noRooms(let path) = self { return "No PNG rooms in \(path)" }
        return nil
    }
}
