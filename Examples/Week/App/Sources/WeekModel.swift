// WeekModel — the screen's state: the week, the planner, and the run as its answers arrive, then
// the reminders. The week is read from the app's own Demo week calendar (written there from the
// generator when it holds no event this week), or with `-store 0` taken straight from the
// generator. The planning loop runs on WeekPlanner's actor; every number on the screen is computed
// here from the run's own measurements.

import CoreAIKit
import Foundation
import Observation

@MainActor
@Observable
final class WeekModel {
    enum Phase: Equatable { case loading, ready, planning, done, failed }

    let seed: UInt64
    /// The week on screen, in time order: the calendar's once it is read, the generator's before.
    private(set) var events: [WeekEvent]
    private(set) var phase = Phase.loading
    /// LOADING: what is loading ("calendar", the download percent, "model"); FAILED: the error.
    private(set) var detail = ""
    /// By event index; nil until the event is planned.
    private(set) var results: [PlannedEvent?]
    private(set) var done = 0
    /// Median per-event time (ms) and events per second, over the events planned so far.
    private(set) var medianMs: Double?
    private(set) var rate: Double?
    private(set) var counts: [WeekBin: Int] = [:]
    /// When Plan was pressed; the clock runs from here.
    private(set) var startedAt: ContinuousClock.Instant?
    private(set) var run: PlanRun?
    private(set) var loadSeconds: Double?
    private(set) var warmSeconds: Double?
    private(set) var bundlePath = ""
    /// "bundle" (-bundle), "sideload" (Documents/decider-0.8b) or "catalog" (downloaded).
    private(set) var source = ""
    private(set) var bundleRevision: String?
    private(set) var bundleCompiled: String?
    private(set) var thermalBefore: ProcessInfo.ThermalState?
    private(set) var thermalAfter: ProcessInfo.ThermalState?
    /// True when the week came from the Demo week calendar; false for the in-app sample week.
    private(set) var fromCalendar = false
    /// Why the week is the in-app one ("-store 0", "calendar access denied", an error).
    private(set) var storeNote = ""
    /// Events written into the Demo week calendar at this launch.
    private(set) var inserted = 0
    /// Where the Demo week calendar and the Before your week list live (`CalendarStore.describe`),
    /// or "none (<why>; in-app sample week)" when the week is the in-app one.
    private(set) var calendarSource = ""
    private(set) var remindersSource = ""
    private(set) var access: (events: String, reminders: String) = ("", "")
    /// nil until Add reminders was pressed; then how many were added and how many were there already.
    private(set) var remindersAdded: Int?
    private(set) var remindersExisting = 0
    private(set) var addingReminders = false
    private(set) var remindersError: String?

    private let useStore: Bool
    /// `-syncedStore 1`: the calendar and the list may go to a synced source (iCloud, CalDAV,
    /// Exchange) when there is no local one.
    let syncedStore: Bool
    private var calendar: CalendarStore?
    private var planner: WeekPlanner?
    private var loadStarted = false
    private var milliseconds: [Double] = []
    private var runTask: Task<Void, Never>?

    init(count: Int, seed: UInt64, store: Bool, synced: Bool) {
        self.seed = seed
        useStore = store
        syncedStore = synced
        let week = Week.generate(count: count, seed: seed)
        events = week
        results = Array(repeating: nil, count: week.count)
        storeNote = store ? "" : "-store 0"
        calendarSource = store ? "" : "none (-store 0; in-app sample week)"
    }

    var count: Int { events.count }

    /// The planned events that need something before them, in time order.
    var plan: [(event: WeekEvent, bin: WeekBin)] {
        events.compactMap { e in
            guard let r = results[e.id], r.bin != .nothing else { return nil }
            return (e, r.bin)
        }
    }

    // MARK: - loading

    /// Asks for Calendar and Reminders access and reads the week, then finds the bundle, loads it
    /// and runs one throwaway decision.
    func load(bundle: String?) async {
        guard !loadStarted else { return }  // once, whichever window asks
        loadStarted = true
        if useStore { await readWeek() }
        let t0 = ContinuousClock.now
        do {
            let url = try await resolveBundle(bundle)
            detail = "model"
            // The catalog path reports its download; a cached or local bundle goes straight to loading.
            let decider = try await WeekPlanner.load(bundle: url) { progress in
                Task { @MainActor in
                    if self.phase == .loading {
                        self.detail = progress.fraction < 1 ? "\(Int(progress.fraction * 100))%" : "model"
                    }
                }
            }
            let planner = WeekPlanner(decider: decider)
            warmSeconds = try await planner.warm()
            self.planner = planner
            loadSeconds = t0.duration(to: .now).inSeconds
            detail = ""
            phase = .ready
        } catch {
            detail = error.localizedDescription
            phase = .failed
        }
    }

    /// The Demo week calendar's week; the generator's when access is refused, when there is no
    /// local source to keep the calendar in (and no `-syncedStore 1`), or when EventKit fails.
    private func readWeek() async {
        detail = "calendar"
        let store = CalendarStore(synced: syncedStore)
        _ = await store.requestAccess()
        access = (CalendarStore.status(.event), CalendarStore.status(.reminder))
        guard access.events == "fullAccess" else {
            storeNote = "calendar access \(access.events)"
            calendarSource = "none (calendar access \(access.events); in-app sample week)"
            return
        }
        do {
            let (week, inserted) = try store.prepareWeek(events)
            guard !week.isEmpty else {
                storeNote = "the Demo week calendar has no event this week"
                calendarSource = "none (\(storeNote); in-app sample week)"
                return
            }
            events = week
            results = Array(repeating: nil, count: week.count)
            self.inserted = inserted
            calendarSource = store.calendarSource
            calendar = store
            fromCalendar = true
        } catch CalendarStoreError.noLocalSource {
            storeNote = "no local source"
            calendarSource = CalendarStore.noLocalSource
        } catch {
            storeNote = "calendar: \(error.localizedDescription)"
            calendarSource = "none (\(error.localizedDescription); in-app sample week)"
        }
    }

    /// `-bundle <dir>`; else, on an iPhone, a copy sideloaded into Documents/decider-0.8b; else nil:
    /// the catalog's decider-0.8b at its pinned revision, downloaded once and cached.
    private func resolveBundle(_ explicit: String?) async throws -> URL? {
        if let explicit {
            source = "bundle"
            let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath, isDirectory: true)
            bundlePath = url.path
            bundleCompiled = Device.compiled(url)
            return url
        }
        #if os(iOS)
        let sideload = URL.documentsDirectory.appending(path: "decider-0.8b", directoryHint: .isDirectory)
        if FileManager.default.fileExists(atPath: sideload.appending(path: "metadata.json").path) {
            source = "sideload"
            bundlePath = sideload.path
            bundleCompiled = Device.compiled(sideload)
            return sideload
        }
        #endif
        source = "catalog"
        bundlePath = "catalog"
        bundleRevision = try? await ModelCatalog.entry(forID: WeekQuestion.catalogID).revision
        return nil
    }

    // MARK: - planning

    /// Plans the whole week in time order; DONE starts it over with the answers cleared.
    func planWeek() {
        guard let planner, phase == .ready || phase == .done else { return }
        reset()
        let start = ContinuousClock.now
        startedAt = start
        thermalBefore = ProcessInfo.processInfo.thermalState
        phase = .planning
        let events = self.events
        runTask = Task {
            let (answers, sink) = AsyncStream.makeStream(of: PlannedEvent.self)
            let loop = Task.detached(priority: .userInitiated) { () async throws -> PlanRun in
                defer { sink.finish() }
                return try await planner.plan(events, start: start) { sink.yield($0) }
            }
            for await answer in answers { apply(answer) }
            do {
                finish(try await loop.value)
            } catch {
                detail = error.localizedDescription
                phase = .failed
            }
        }
    }

    private func reset() {
        results = Array(repeating: nil, count: events.count)
        done = 0
        milliseconds = []
        medianMs = nil
        rate = nil
        counts = [:]
        run = nil
        thermalAfter = nil
        remindersAdded = nil
        remindersExisting = 0
        remindersError = nil
    }

    private func apply(_ r: PlannedEvent) {
        results[r.index] = r
        counts[r.bin, default: 0] += 1
        done += 1
        milliseconds.append(r.seconds * 1000)
        medianMs = Stats.percentile(milliseconds.sorted(), 0.5)
        if r.at > 0 { rate = Double(done) / r.at }
    }

    private func finish(_ run: PlanRun) {
        self.run = run
        medianMs = Stats.percentile(run.milliseconds, 0.5)
        rate = run.eventsPerSecond
        thermalAfter = ProcessInfo.processInfo.thermalState
        phase = .done
    }

    // MARK: - reminders

    /// Whether the DONE screen offers Add reminders: the week came from the calendar, Reminders
    /// access is full, something needs doing, and the reminders have not been added yet.
    var canAddReminders: Bool {
        phase == .done && fromCalendar && access.reminders == "fullAccess" && !plan.isEmpty
            && remindersAdded == nil && !addingReminders
    }

    /// One reminder per event that needs something, in the Before your week list.
    func addReminders() async {
        guard canAddReminders, let calendar else { return }
        addingReminders = true
        defer { addingReminders = false }
        do {
            let (added, existing) = try await calendar.addReminders(plan)
            remindersSource = calendar.listSource
            remindersExisting = existing
            remindersAdded = added
        } catch {
            remindersError = error.localizedDescription
            remindersSource = "none (\(error.localizedDescription))"
            remindersAdded = 0
        }
    }

    // MARK: - what the screen and the log read

    /// Seconds on the clock: from Plan to now while planning, the run's total once done.
    func elapsed(at now: ContinuousClock.Instant) -> Double? {
        switch phase {
        case .planning: return startedAt.map { $0.duration(to: now).inSeconds }
        case .done: return run?.totalSeconds
        default: return nil
        }
    }

    /// "Demo week calendar" or "in-app sample week".
    var weekSource: String { fromCalendar ? "Demo week calendar" : "in-app sample week" }

    /// The screen's state as one line, for the autoplay log.
    var statusLine: String {
        switch phase {
        case .loading: return "LOADING \(detail)"
        case .ready:
            return String(format: "READY %d events · %@ · calendar_source %@ · load %.2f s", count, weekSource,
                          calendarSource, loadSeconds ?? 0)
        case .planning:
            return "PLANNING \(done)/\(count) · " + Self.ms(medianMs) + " per event · " + Self.rate(rate)
        case .done:
            let ms = run?.milliseconds ?? []
            var line = String(format: "DONE %d/%d · %.2f s · median %@ · p90 %@ · %@", done, count, run?.totalSeconds ?? 0,
                              Self.ms(Stats.percentile(ms, 0.5)), Self.ms(Stats.percentile(ms, 0.9)), Self.rate(rate))
            if let added = remindersAdded {
                line += " · reminders added \(added)" + (remindersExisting > 0 ? ", \(remindersExisting) already there" : "")
                if let remindersError { line += " · \(remindersError)" }
            }
            return line
        case .failed: return "FAILED \(detail)"
        }
    }

    static func ms(_ v: Double?) -> String {
        guard let v else { return "– ms" }
        return v < 10 ? String(format: "%.1f ms", v) : String(format: "%.0f ms", v)
    }

    static func rate(_ v: Double?) -> String {
        guard let v else { return "– events/s" }
        return String(format: "%.2f events/s", v)
    }

    /// The finished run as the result file's JSON: the CLI's fields plus where it ran.
    func resultJSON() -> JSONValue? {
        guard let run else { return nil }
        var out = run.summary(events: events)
        out["seed"] = .int(Int(seed))
        out["model"] = .string(WeekQuestion.catalogID)
        out["bundle"] = .string(bundlePath)
        out["source"] = .string(source)
        out["bundle_revision"] = bundleRevision.map { .string($0) } ?? .null
        out["bundle_compiled"] = bundleCompiled.map { .string($0) } ?? .null
        out["format"] = .string(planner?.decider.format.rawValue ?? "")
        out["temperature"] = .rounded(planner?.decider.temperature ?? 0, 3)
        out["device"] = .string(Device.model)
        out["machine"] = .string(Device.machine)
        out["os"] = .string(Device.os)
        out["os_build"] = .string(ProcessInfo.processInfo.operatingSystemVersionString)
        out["compute_units"] = .string("gpu")
        out["load_s"] = .rounded(loadSeconds ?? 0, 3)
        out["warm_s"] = .rounded(warmSeconds ?? 0, 3)
        out["store"] = .string(weekSource + (storeNote.isEmpty ? "" : " (\(storeNote))"))
        out["calendar_events_inserted"] = .int(inserted)
        out["calendar_source"] = .string(calendarSource)
        out["reminders_source"] = .string(remindersSource)
        out["synced_store"] = .bool(syncedStore)
        out["access"] = .object(["events": .string(access.events), "reminders": .string(access.reminders)])
        out["reminders_added"] = .int(remindersAdded ?? 0)
        out["reminders_existing"] = .int(remindersExisting)
        if let remindersError { out["reminders_error"] = .string(remindersError) }
        out["thermal_before"] = .string(Device.name(thermalBefore))
        out["thermal_after"] = .string(Device.name(thermalAfter))
        out["timestamp"] = .string(ISO8601DateFormatter().string(from: Date()))
        return .object(out)
    }
}
