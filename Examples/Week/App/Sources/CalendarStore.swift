// CalendarStore — the app's only EventKit code. It reads this week's events, Monday to Sunday, from
// every calendar on the device, after asking for full Calendar access; it never writes to a
// calendar and never creates one. The one write is Add reminders: pressed, it asks for Reminders
// access and adds one reminder per listed event to the Reminders list new reminders go to.
//
// The completion-handler forms of EventKit are wrapped in continuations, so no event store
// object crosses an actor boundary.

import EventKit
import Foundation

@MainActor
final class CalendarStore {
    /// The calendar an earlier version of this app wrote a synthetic week into; never read.
    static let oldDemoCalendar = "Demo week"

    private let store = EKEventStore()
    /// The account of the list the reminders went to ("iCloud (calDAV)"), once they were added.
    private(set) var listSource = ""
    /// Weeks start on Monday, whatever the locale says.
    private let calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.firstWeekday = 2
        c.timeZone = .current
        return c
    }()

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

    /// Asks for full Calendar access. The system shows its alert once; after that the call returns
    /// the answer at once.
    func requestCalendarAccess() async -> Bool {
        let granted = await withCheckedContinuation { (done: CheckedContinuation<Bool, Never>) in
            store.requestFullAccessToEvents { granted, _ in done.resume(returning: granted) }
        }
        if granted { store.refreshSourcesIfNecessary() }
        return granted
    }

    /// Asks for full Reminders access, the same way.
    func requestRemindersAccess() async -> Bool {
        let granted = await withCheckedContinuation { (done: CheckedContinuation<Bool, Never>) in
            store.requestFullAccessToReminders { granted, _ in done.resume(returning: granted) }
        }
        if granted { store.refreshSourcesIfNecessary() }
        return granted
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

    /// This week's events from every event calendar (the old Demo week calendar excepted): those
    /// that start between Monday 00:00 and Sunday midnight, all-day, cancelled and declined events
    /// left out, their text cleaned as a pasted line's is (`WeekInput`). Unordered; `WeekInput.checked`
    /// orders them.
    func readThisWeek() -> [WeekEvent] {
        let week = currentWeek
        let calendars = store.calendars(for: .event).filter { $0.title != Self.oldDemoCalendar }
        guard !calendars.isEmpty else { return [] }
        let found = store.events(matching: store.predicateForEvents(withStart: week.start, end: week.end, calendars: calendars))
        return found.compactMap { e -> WeekEvent? in
            guard !e.isAllDay, e.status != .canceled, !Self.declined(e),
                  let start = e.startDate, let end = e.endDate, start >= week.start, start < week.end
            else { return nil }
            guard let day = calendar.dateComponents([.day], from: week.start, to: calendar.startOfDay(for: start)).day,
                  (0..<7).contains(day)
            else { return nil }
            let time = calendar.dateComponents([.hour, .minute], from: start)
            let title = (e.title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            return WeekInput.event(
                day: day, start: (time.hour ?? 0) * 60 + (time.minute ?? 0),
                minutes: Int(end.timeIntervalSince(start) / 60), title: title.isEmpty ? "Untitled" : title,
                location: e.location, notes: e.notes)
        }
    }

    /// An invitation this device's user said no to.
    private static func declined(_ e: EKEvent) -> Bool {
        e.attendees?.contains { $0.isCurrentUser && $0.participantStatus == .declined } ?? false
    }

    // MARK: - reminders

    /// One reminder per planned event in the default Reminders list: title `<what>: <event title>`,
    /// due the day before the event at 9:00, the event's notes as its notes. A reminder already
    /// there with the same title and due day is left alone, so pressing twice adds nothing twice.
    func addReminders(_ plan: [(event: WeekEvent, bin: WeekBin)]) async throws -> (added: Int, existing: Int) {
        guard let list = store.defaultCalendarForNewReminders() else { throw CalendarStoreError.noList }
        listSource = Self.describe(list.source)
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
    /// No Reminders list to add to (no Reminders account on the device).
    case noList

    var errorDescription: String? {
        switch self {
        case .noList: return "no Reminders list to add to"
        }
    }
}
