// WeekModel — the screen's state: where the week comes from (your calendar, pasted lines or a
// file, the sample week), the planner, the run as its answers arrive, then the reminders. The
// model loads once at launch whatever the source; the week can be switched before Plan and again
// after DONE. The planning loop runs on WeekPlanner's actor; every number on the screen is
// computed here from the run's own measurements.

import CoreAIKit
import Foundation
import Observation

@MainActor
@Observable
final class WeekModel {
    enum Phase: Equatable { case loading, ready, planning, done, failed }

    /// Where the week comes from.
    enum Source: String, CaseIterable, Sendable {
        case calendar, paste, sample

        /// The picker's words.
        var title: String {
            switch self {
            case .calendar: return "Your calendar"
            case .paste: return "Paste"
            case .sample: return "Sample week"
            }
        }

        /// What the footer and the DONE line call the week.
        var weekName: String {
            switch self {
            case .calendar: return "your calendar"
            case .paste: return "pasted week"
            case .sample: return "sample week"
            }
        }
    }

    /// The sample week: the generator's 20 events from seed 7, the week `week-cli run` plans
    /// when it is given no file.
    static let sampleCount = 20
    static let sampleSeed: UInt64 = 7

    private(set) var source: Source
    /// The week on screen, in time order.
    private(set) var events: [WeekEvent] = []
    private(set) var phase = Phase.loading
    /// LOADING: what is loading (the download percent, "model"); FAILED: the error.
    private(set) var detail = ""
    /// One line about the week: why there is nothing to plan, or how many pasted lines were skipped.
    private(set) var note: String?
    /// Pasted lines that were not read as events, for the week on screen.
    private(set) var skippedLines = 0
    /// True from launch until the first week is read, and while the calendar is asked or read.
    private(set) var readingWeek = true
    /// Calendar and Reminders access as EventKit reports it ("fullAccess", "denied", …), once asked.
    private(set) var access: (events: String, reminders: String) = ("", "")

    // The Paste sheet: its text is kept between openings.
    var showingPaste = false
    var pasteText = ""
    /// Why the sheet's last Plan or file was refused, until the text changes.
    private(set) var pasteRefusal: String?

    /// By event index; nil until the event is planned.
    private(set) var results: [PlannedEvent?] = []
    private(set) var done = 0
    /// Median per-event time (ms) and events per second, over the events planned so far.
    private(set) var medianMs: Double?
    private(set) var rate: Double?
    private(set) var counts: [WeekBin: Int] = [:]
    /// When Plan was pressed; the clock runs from here.
    private(set) var startedAt: ContinuousClock.Instant?
    private(set) var run: PlanRun?
    /// Loading the bundle (a first launch includes the download), then the throwaway decision.
    private(set) var loadSeconds: Double?
    private(set) var warmSeconds: Double?
    private(set) var bundlePath = ""
    /// "bundle" (-bundle), "sideload" (Documents/decider-0.8b) or "catalog" (downloaded).
    private(set) var modelSource = ""
    private(set) var bundleRevision: String?
    private(set) var bundleCompiled: String?
    private(set) var thermalBefore: ProcessInfo.ThermalState?
    private(set) var thermalAfter: ProcessInfo.ThermalState?
    /// nil until Add reminders was pressed; then how many were added and how many were there already.
    private(set) var remindersAdded: Int?
    private(set) var remindersExisting = 0
    private(set) var addingReminders = false
    private(set) var remindersError: String?
    /// The account of the list the reminders went to.
    private(set) var remindersSource = ""

    private let store = CalendarStore()
    private var planner: WeekPlanner?
    private var loadStarted = false
    private var begun = false
    /// The last week a paste or a file gave, shown again when Paste is picked.
    private var pasted: (events: [WeekEvent], skipped: Int)?
    private var milliseconds: [Double] = []
    private var runTask: Task<Void, Never>?

    init(source: Source) {
        self.source = source
    }

    var count: Int { events.count }

    /// The planned events that need something before them, in time order.
    var plan: [(event: WeekEvent, bin: WeekBin)] {
        events.compactMap { e in
            guard e.id < results.count, let r = results[e.id], r.bin != .nothing else { return nil }
            return (e, r.bin)
        }
    }

    /// What Add reminders adds: the planned events that need something and have not started yet.
    var upcoming: [(event: WeekEvent, bin: WeekBin)] {
        let now = Date()
        return plan.filter { store.startDate(of: $0.event) > now }
    }

    /// The event the spotlight card shows: the one whose answer arrived last, with that answer;
    /// the week's first event, unanswered, before any answer (READY, and a run's first second).
    var spotlight: (event: WeekEvent, result: PlannedEvent?)? {
        if let last = results.compactMap({ $0 }).max(by: { $0.at < $1.at }),
           let event = events.first(where: { $0.id == last.index }) {
            return (event, last)
        }
        return events.first.map { ($0, nil) }
    }

    // MARK: - the week

    /// The week the app opens on: the calendar's (asking for access first), the sample week, or
    /// for Paste the sheet.
    func begin() async {
        guard !begun else { return }  // once, whichever window asks
        begun = true
        await select(source)
    }

    /// Switches where the week comes from (Paste opens the sheet, even when already picked). Not
    /// while planning.
    func select(_ source: Source) async {
        guard phase != .planning else { return }
        self.source = source
        switch source {
        case .calendar:
            await readCalendar()
        case .paste:
            if let pasted {
                setWeek(pasted.events, skipped: pasted.skipped)
            } else {
                setWeek([], note: "Nothing pasted yet")
            }
            showingPaste = true
            readingWeek = false
        case .sample:
            setWeek(Week.generate(count: Self.sampleCount, seed: Self.sampleSeed))
            readingWeek = false
        }
    }

    /// Asks for Calendar access (the system asks once) and reads this week from every calendar.
    private func readCalendar() async {
        readingWeek = true
        defer { readingWeek = false }
        setWeek([])
        _ = await store.requestCalendarAccess()
        access.events = CalendarStore.status(.event)
        guard source == .calendar else { return }  // the source changed while the alert was up
        guard access.events == "fullAccess" else {
            setWeek([], note: Self.calendarRefusal(access.events))
            return
        }
        let reading = WeekInput.checked(store.readThisWeek())
        if let problem = reading.problem {
            setWeek([], note: problem.message)
        } else {
            setWeek(reading.events, note: reading.events.isEmpty ? "No events this week" : nil)
        }
    }

    /// The one line shown when the calendar cannot be read.
    static func calendarRefusal(_ status: String) -> String {
        #if os(macOS)
        let fix = "Turn it on in System Settings › Privacy & Security › Calendars, or paste your week."
        #else
        let fix = "Turn it on in Settings, or paste your week."
        #endif
        switch status {
        case "writeOnly": return "Week can only add to your calendar, not read it. " + fix
        case "restricted": return "Calendar access is restricted on this device. Paste your week instead."
        default: return "Calendar access is off. " + fix
        }
    }

    private func setWeek(_ week: [WeekEvent], skipped: Int = 0, note: String? = nil) {
        events = week
        skippedLines = skipped
        self.note = note ?? (skipped > 0 ? WeekInput.skippedNote(skipped) : nil)
        reset()
        if phase == .done { phase = .ready }
    }

    // MARK: - the Paste sheet

    /// The sheet's text as it would be read.
    var pasteReading: WeekInput.Reading { WeekInput.read(lines: pasteText) }

    /// The sheet's line under the text: "20 events", "8 events · 2 lines skipped", or why nothing
    /// can be planned.
    var pasteStatus: String {
        if let pasteRefusal { return pasteRefusal }
        let reading = pasteReading
        if let problem = reading.problem { return problem.message }
        let n = reading.events.count
        var line = "\(n) event\(n == 1 ? "" : "s")"
        if reading.skipped > 0 { line += " · " + WeekInput.skippedNote(reading.skipped) }
        return line
    }

    /// The sheet's Plan button: a readable week goes on screen and, once the model is ready, is
    /// planned at once; a refusal stays in the sheet. Returns whether the sheet closed.
    @discardableResult
    func submitPaste() -> Bool {
        guard phase != .planning else { return false }
        let reading = pasteReading
        guard reading.problem == nil else {
            pasteRefusal = reading.note
            return false
        }
        pasteRefusal = nil
        pasted = (reading.events, reading.skipped)
        source = .paste
        setWeek(reading.events, skipped: reading.skipped)
        showingPaste = false
        if phase == .ready { planWeek() }
        return true
    }

    /// Import .json (or a text file): the file's events become the sheet's lines, read as pasted
    /// lines are.
    func importFile(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            pasteText = try WeekInput.text(contentsOf: url)
            pasteRefusal = nil
        } catch {
            pasteRefusal = WeekInput.Problem.unreadable(error.localizedDescription).message
        }
    }

    /// The text changed: an earlier refusal no longer applies.
    func pasteEdited() { pasteRefusal = nil }

    // MARK: - loading

    /// Finds the bundle, loads it and runs one throwaway decision.
    func load(bundle: String?) async {
        guard !loadStarted else { return }  // once, whichever window asks
        loadStarted = true
        // The download, the load and the warm-up are work a person asked for. Without this, macOS App Naps a
        // window that is hidden or behind others: a first download fell to 0.13 MB/s on a Mac whose curl got 5 MB/s.
        let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Loading the model")
        defer { ProcessInfo.processInfo.endActivity(activity) }
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
            loadSeconds = t0.duration(to: .now).inSeconds
            let planner = WeekPlanner(decider: decider)
            warmSeconds = try await planner.warm()
            self.planner = planner
            detail = ""
            phase = .ready
        } catch {
            detail = error.localizedDescription
            phase = .failed
        }
    }

    /// `-bundle <dir>`; else, on an iPhone, a copy sideloaded into Documents/decider-0.8b; else nil:
    /// the catalog's decider-0.8b at its pinned revision, downloaded once and cached.
    private func resolveBundle(_ explicit: String?) async throws -> URL? {
        if let explicit {
            modelSource = "bundle"
            let url = URL(fileURLWithPath: (explicit as NSString).expandingTildeInPath, isDirectory: true)
            bundlePath = url.path
            bundleCompiled = Device.compiled(url)
            return url
        }
        #if os(iOS)
        let sideload = URL.documentsDirectory.appending(path: "decider-0.8b", directoryHint: .isDirectory)
        if FileManager.default.fileExists(atPath: sideload.appending(path: "metadata.json").path) {
            modelSource = "sideload"
            bundlePath = sideload.path
            bundleCompiled = Device.compiled(sideload)
            return sideload
        }
        #endif
        modelSource = "catalog"
        bundlePath = "catalog"
        bundleRevision = try? await ModelCatalog.entry(forID: WeekQuestion.catalogID).revision
        return nil
    }

    // MARK: - planning

    /// Plans the whole week in time order; DONE starts it over with the answers cleared.
    func planWeek() {
        guard let planner, phase == .ready || phase == .done, !events.isEmpty else { return }
        reset()
        let start = ContinuousClock.now
        startedAt = start
        thermalBefore = ProcessInfo.processInfo.thermalState
        phase = .planning
        let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Planning the week")
        let events = self.events
        runTask = Task {
            defer { ProcessInfo.processInfo.endActivity(activity) }
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
        startedAt = nil
        thermalAfter = nil
        remindersAdded = nil
        remindersExisting = 0
        remindersError = nil
        remindersSource = ""
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

    /// Whether the DONE screen offers Add reminders: a week of your own (not the sample), something
    /// still ahead that needs doing, and the reminders not added yet.
    var canAddReminders: Bool {
        phase == .done && source != .sample && remindersAdded == nil && !addingReminders && !upcoming.isEmpty
    }

    /// Asks for Reminders access (only now, when pressed), then adds one reminder per upcoming
    /// event that needs something to the default Reminders list.
    func addReminders() async {
        guard canAddReminders else { return }
        addingReminders = true
        defer { addingReminders = false }
        let items = upcoming
        _ = await store.requestRemindersAccess()
        access.reminders = CalendarStore.status(.reminder)
        guard access.reminders == "fullAccess" else {
            remindersError = "Reminders access is off, so nothing was added"
            remindersAdded = 0
            return
        }
        do {
            let (added, existing) = try await store.addReminders(items)
            remindersSource = store.listSource
            remindersExisting = existing
            remindersAdded = added
        } catch {
            remindersError = error.localizedDescription
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

    /// The screen's state as one line, for the autoplay log. It never holds an event's text.
    var statusLine: String {
        var line: String
        switch phase {
        case .loading: return "LOADING \(detail)"
        case .ready:
            line = String(format: "READY %d events · source %@ · skipped_lines %d · load %.2f s · warm %.2f s",
                          count, source.rawValue, skippedLines, loadSeconds ?? 0, warmSeconds ?? 0)
            if readingWeek { line += " · reading the week" }
        case .planning:
            return "PLANNING \(done)/\(count) · " + Self.ms(medianMs) + " per event · " + Self.rate(rate)
        case .done:
            let ms = run?.milliseconds ?? []
            line = String(format: "DONE %d/%d · %.2f s · median %@ · p90 %@ · %@ · source %@", done, count,
                          run?.totalSeconds ?? 0, Self.ms(Stats.percentile(ms, 0.5)), Self.ms(Stats.percentile(ms, 0.9)),
                          Self.rate(rate), source.rawValue)
            if let added = remindersAdded {
                line += " · reminders added \(added)" + (remindersExisting > 0 ? ", \(remindersExisting) already there" : "")
                if let remindersError { line += " · \(remindersError)" }
            }
        case .failed: return "FAILED \(detail)"
        }
        if let note { line += " · note: \(note)" }
        if showingPaste { line += " · paste sheet: " + pasteStatus }
        return line
    }

    static func ms(_ v: Double?) -> String {
        guard let v else { return "– ms" }
        return v < 10 ? String(format: "%.1f ms", v) : String(format: "%.0f ms", v)
    }

    static func rate(_ v: Double?) -> String {
        guard let v else { return "– events/s" }
        return String(format: "%.2f events/s", v)
    }

    /// The finished run as the result file's JSON: the CLI's fields plus where the week came from
    /// and where it ran. For your calendar's week each answer's event is "calendar event <n>": the
    /// file keeps the answers and the times, never an event's text.
    func resultJSON() -> JSONValue? {
        guard let run else { return nil }
        var out = run.summary(events: events)
        if source == .calendar, case .array(let rows)? = out["answers"] {
            out["answers"] = .array(rows.enumerated().map { i, row in
                guard case .object(var fields) = row else { return row }
                fields["event"] = .string("calendar event \(i + 1)")
                return .object(fields)
            })
        }
        out["source"] = .string(source.rawValue)
        out["skipped_lines"] = .int(skippedLines)
        out["seed"] = source == .sample ? .int(Int(Self.sampleSeed)) : .null
        out["model"] = .string(WeekQuestion.catalogID)
        out["bundle"] = .string(bundlePath)
        out["model_source"] = .string(modelSource)
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
        out["access"] = .object(["events": .string(access.events), "reminders": .string(access.reminders)])
        out["reminders_added"] = .int(remindersAdded ?? 0)
        out["reminders_existing"] = .int(remindersExisting)
        out["reminders_source"] = .string(remindersSource)
        if let remindersError { out["reminders_error"] = .string(remindersError) }
        out["thermal_before"] = .string(Device.name(thermalBefore))
        out["thermal_after"] = .string(Device.name(thermalAfter))
        out["timestamp"] = .string(ISO8601DateFormatter().string(from: Date()))
        return .object(out)
    }
}
