// CalendarStore — the app's only EventKit code: the Calendar and Reminders permissions, the
// app's own `Demo week` calendar (created once, filled with the synthetic week when it holds no
// event this week, and the only calendar ever read), and the `Before your week` reminders list.
//
// Both live in the device's local source ("On My iPhone") and nowhere else: a calendar or a list
// in a synced source (iCloud, CalDAV, Exchange) reaches every device on that account. Without a
// local source the app does not write at all and plans its in-app week; `-syncedStore 1` is the
// only way to let it write into the default source instead.
//
// The completion-handler forms of EventKit are wrapped in continuations, so no event store
// object crosses an actor boundary.

import EventKit
import Foundation

@MainActor
final class CalendarStore {
    static let calendarTitle = "Demo week"
    static let listTitle = "Before your week"
    /// `calendar_source` when there is no local source to write into.
    static let noLocalSource = "none (no local source; in-app sample week)"

    private let store = EKEventStore()
    /// Whether a synced source may be written into when the local one cannot (`-syncedStore 1`).
    let synced: Bool
    /// Where the Demo week calendar and the Before your week list live ("On My iPhone (local)").
    private(set) var calendarSource = ""
    private(set) var listSource = ""
    /// Weeks start on Monday, whatever the locale says.
    private let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.firstWeekday = 2
        c.timeZone = .current
        return c
    }()

    init(synced: Bool = false) {
        self.synced = synced
    }

    // MARK: - access

    /// "fullAccess", "writeOnly", "denied", "restricted" or "notDetermined".
    static func status(_ type: EKEntityType) -> String {
        switch EKEventStore.authorizationStatus(for: type) {
        case .fullAccess: return "fullAccess"
        case .writeOnly: return "writeOnly"
        case .denied: return "denied"
        case .restricted: return "restricted"
        case .notDetermined: return "notDetermined"
        @unknown default: return "unknown"
        }
    }

    /// Asks for full Calendar access, then full Reminders access. The system shows each alert once;
    /// after that, the calls return the answer at once.
    func requestAccess() async -> (events: Bool, reminders: Bool) {
        let events = await withCheckedContinuation { (done: CheckedContinuation<Bool, Never>) in
            store.requestFullAccessToEvents { granted, _ in done.resume(returning: granted) }
        }
        let reminders = await withCheckedContinuation { (done: CheckedContinuation<Bool, Never>) in
            store.requestFullAccessToReminders { granted, _ in done.resume(returning: granted) }
        }
        if events || reminders { store.refreshSourcesIfNecessary() }
        return (events, reminders)
    }

    // MARK: - the week

    /// Monday 00:00 of the current week, and the Monday after.
    var currentWeek: (start: Date, end: Date) {
        let interval = calendar.dateInterval(of: .weekOfYear, for: Date())!
        return (interval.start, interval.end)
    }

    /// The start of `event` in the current week.
    func startDate(of event: WeekEvent) -> Date {
        let day = calendar.date(byAdding: .day, value: event.day, to: currentWeek.start)!
        return calendar.date(bySettingHour: event.start / 60, minute: event.start % 60, second: 0, of: day)!
    }

    /// Finds or creates the Demo week calendar and, when it holds no event this week, writes
    /// `generated` into it. Returns this week's events read back from that calendar alone, in
    /// time order, with how many were written now. An event the generator wrote keeps its
    /// written label (matched by day, time and title) for the result file.
    func prepareWeek(_ generated: [WeekEvent]) throws -> (events: [WeekEvent], inserted: Int) {
        let demo = try demoCalendar()
        let week = currentWeek
        var inserted = 0
        if events(in: demo, week: week).isEmpty {
            for event in generated {
                let e = EKEvent(eventStore: store)
                e.calendar = demo
                e.title = event.title
                e.startDate = startDate(of: event)
                e.endDate = e.startDate.addingTimeInterval(TimeInterval(event.minutes * 60))
                e.location = event.location
                e.notes = event.notes
                try store.save(e, span: .thisEvent, commit: false)
                inserted += 1
            }
            try store.commit()
        }
        let written = Dictionary(
            generated.map { ("\($0.day) \($0.start) \($0.title)", $0.written) }, uniquingKeysWith: { a, _ in a })
        let read = events(in: demo, week: week).compactMap { e -> WeekEvent? in
            guard !e.isAllDay, let start = e.startDate, let end = e.endDate else { return nil }
            let dayStart = calendar.startOfDay(for: start)
            guard let day = calendar.dateComponents([.day], from: week.start, to: dayStart).day, (0..<7).contains(day)
            else { return nil }
            let time = calendar.dateComponents([.hour, .minute], from: start)
            let minuteOfDay = (time.hour ?? 0) * 60 + (time.minute ?? 0)
            let title = e.title ?? ""
            return WeekEvent(
                id: 0, day: day, start: minuteOfDay, minutes: max(1, Int(end.timeIntervalSince(start) / 60)),
                title: title, location: e.location.flatMap { $0.isEmpty ? nil : $0 },
                notes: e.notes.flatMap { $0.isEmpty ? nil : $0 },
                written: written["\(day) \(minuteOfDay) \(title)"] ?? nil)
        }
        let ordered = read.sorted { ($0.day, $0.start, $0.title) < ($1.day, $1.start, $1.title) }
            .enumerated().map { $0.element.with(id: $0.offset) }
        return (ordered, inserted)
    }

    private func events(in demo: EKCalendar, week: (start: Date, end: Date)) -> [EKEvent] {
        store.events(matching: store.predicateForEvents(withStart: week.start, end: week.end, calendars: [demo]))
    }

    /// The app's own calendar: one titled Demo week in a source it may write into, else a new one
    /// in the local source (then, with `-syncedStore 1` only, where new events go by default).
    private func demoCalendar() throws -> EKCalendar {
        let demo = try store.calendars(for: .event).first(where: { $0.title == Self.calendarTitle && writable($0.source) })
            ?? create(.event, title: Self.calendarTitle, in: sources(default: store.defaultCalendarForNewEvents?.source))
        calendarSource = Self.describe(demo.source)
        return demo
    }

    /// The sources the app may write into, in order: the local one, and the default one only with
    /// `-syncedStore 1`.
    private func sources(default fallback: EKSource?) -> [EKSource] {
        var out: [EKSource] = []
        if let local = store.sources.first(where: { $0.sourceType == .local }) { out.append(local) }
        if synced, let fallback, !out.contains(where: { $0.sourceIdentifier == fallback.sourceIdentifier }) {
            out.append(fallback)
        }
        return out
    }

    private func writable(_ source: EKSource?) -> Bool {
        guard let source else { return false }
        return source.sourceType == .local || synced
    }

    /// What `calendar_source` will say, before anything is written: the local source, else (with
    /// `-syncedStore 1`) the default one, else `noLocalSource`. For access.json.
    var plannedCalendarSource: String {
        sources(default: store.defaultCalendarForNewEvents?.source).first.map { Self.describe($0) } ?? Self.noLocalSource
    }

    /// A new calendar or list, in the given sources in order until one accepts it.
    private func create(_ type: EKEntityType, title: String, in sources: [EKSource]) throws -> EKCalendar {
        guard !sources.isEmpty else { throw CalendarStoreError.noLocalSource }
        var reason = ""
        for source in sources {
            let c = EKCalendar(for: type, eventStore: store)
            c.title = title
            c.source = source
            do {
                try store.saveCalendar(c, commit: true)
                return c
            } catch {
                reason = error.localizedDescription
            }
        }
        throw CalendarStoreError.saveFailed(title: title, reason: reason)
    }

    // MARK: - reminders

    /// One reminder per planned event in the Before your week list: title `<bin>: <event title>`,
    /// due the day before the event at 9:00, the event's notes as its notes. A reminder already
    /// there with the same title and due day is left alone, so pressing twice adds nothing twice.
    func addReminders(_ plan: [(event: WeekEvent, bin: WeekBin)]) async throws -> (added: Int, existing: Int) {
        let list = try remindersList()
        let existing = await openReminders(in: list)
        var added = 0, skipped = 0
        for (event, bin) in plan {
            let title = Self.reminderTitle(bin, event)
            let due = dueDate(for: event)
            if existing.contains(where: { $0.title == title && $0.due == due }) {
                skipped += 1
                continue
            }
            let r = EKReminder(eventStore: store)
            r.calendar = list
            r.title = title
            r.notes = event.notes
            r.dueDateComponents = due
            try store.save(r, commit: false)
            added += 1
        }
        if added > 0 { try store.commit() }
        return (added, skipped)
    }

    /// "Documents: Lease renewal signing".
    static func reminderTitle(_ bin: WeekBin, _ event: WeekEvent) -> String {
        bin.label.prefix(1).uppercased() + bin.label.dropFirst() + ": " + event.title
    }

    /// The day before the event, 9:00.
    func dueDate(for event: WeekEvent) -> DateComponents {
        let day = calendar.date(byAdding: .day, value: event.day - 1, to: currentWeek.start)!
        var c = calendar.dateComponents([.year, .month, .day], from: day)
        c.hour = 9
        c.minute = 0
        return c
    }

    /// The Before your week list, under the same rule as the calendar: local only, unless
    /// `-syncedStore 1`.
    private func remindersList() throws -> EKCalendar {
        let list = try store.calendars(for: .reminder).first(where: { $0.title == Self.listTitle && writable($0.source) })
            ?? create(.reminder, title: Self.listTitle, in: sources(default: store.defaultCalendarForNewReminders()?.source))
        listSource = Self.describe(list.source)
        return list
    }

    /// "iCloud (calDAV)".
    static func describe(_ source: EKSource?) -> String {
        guard let source else { return "no source" }
        let type: String
        switch source.sourceType {
        case .local: type = "local"
        case .exchange: type = "exchange"
        case .calDAV: type = "calDAV"
        case .mobileMe: type = "mobileMe"
        case .subscribed: type = "subscribed"
        case .birthdays: type = "birthdays"
        @unknown default: type = "unknown"
        }
        return "\(source.title) (\(type))"
    }

    /// The list's incomplete reminders as (title, due day and hour).
    private func openReminders(in list: EKCalendar) async -> [(title: String, due: DateComponents?)] {
        let predicate = store.predicateForIncompleteReminders(withDueDateStarting: nil, ending: nil, calendars: [list])
        return await withCheckedContinuation { (done: CheckedContinuation<[(title: String, due: DateComponents?)], Never>) in
            store.fetchReminders(matching: predicate) { reminders in
                let rows = (reminders ?? []).map { r -> (title: String, due: DateComponents?) in
                    let d = r.dueDateComponents
                    var due = DateComponents()
                    due.year = d?.year
                    due.month = d?.month
                    due.day = d?.day
                    due.hour = d?.hour
                    due.minute = d?.minute
                    return (r.title ?? "", d == nil ? nil : due)
                }
                done.resume(returning: rows)
            }
        }
    }
}

enum CalendarStoreError: LocalizedError {
    /// The device has no local source, and `-syncedStore 1` was not given.
    case noLocalSource
    /// Every source the app may write into refused the new calendar or list.
    case saveFailed(title: String, reason: String)

    var errorDescription: String? {
        switch self {
        case .noLocalSource: return "no local source"
        case .saveFailed(let title, let reason): return "\(title) not saved: \(reason)"
        }
    }
}
