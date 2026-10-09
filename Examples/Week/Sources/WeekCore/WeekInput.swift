// WeekInput.swift — a week you hand over instead of the generator's: lines pasted into the app,
// one event a line in the form the model reads, or a JSON file in `week-cli dump`'s form. The
// app's Paste sheet, its Import .json button and the autoplay harness (`-events`) all read through
// `read(lines:)` (a JSON file becomes lines first), and the calendar path builds its events with
// the same `event(day:start:minutes:title:location:notes:)`, so every source reaches the model in
// one shape.
//
// A line: `Mon 11:00 Lease renewal signing · the property office · Notes: bring two forms of ID`.
// A day (Mon or Monday, any case), a 24-hour time (H:MM or HH:MM), the title, then, after ` · `,
// the place and `Notes:`, both optional. Everything after `Notes:` is the notes, separators
// included, and `Notes: none` is no notes, as `WeekEvent.state` writes it: the lines
// `week-cli dump` prints read back as the same states. A blank line is passed over; any other line
// that does not start with a day, a time and a title is skipped and counted.

import Foundation

public enum WeekInput {
    /// The most events one week may hold. A longer week is refused whole, not cut.
    public static let limit = 200
    /// Characters of an event's text the model reads; a longer text is cut and ends in "…". Every
    /// character is part of the prompt and a prompt costs its length in steps (WeekPlanner.swift),
    /// and a meeting invite's notes run to thousands of characters with the link near the top.
    public static let titleLimit = 120
    public static let placeLimit = 120
    public static let notesLimit = 240
    /// Between the title, the place and the notes.
    public static let separator = " · "
    /// The form of a line, for a hint or a refusal.
    public static let form = "Mon 11:00 Title · Place · Notes: …"

    /// What reading a paste or a file gave.
    public struct Reading: Sendable, Equatable {
        /// The week in time order, ids from 0; empty when `problem` is set.
        public let events: [WeekEvent]
        /// Lines that held text but did not start with a day, a time and a title.
        public let skipped: Int
        /// Why there is nothing to plan; nil when `events` is not empty.
        public let problem: Problem?

        /// One line for the screen: the problem, else how many lines were skipped, else nil.
        public var note: String? {
            if let problem { return problem.message }
            return skipped == 0 ? nil : WeekInput.skippedNote(skipped)
        }
    }

    public enum Problem: Sendable, Equatable {
        /// Nothing but blank lines.
        case empty
        /// Text, but not one line with a day, a time and a title.
        case noEvents
        /// More events than `limit`.
        case tooMany(Int)
        /// A file that could not be read as a week.
        case unreadable(String)

        public var message: String {
            switch self {
            case .empty: return "Nothing to plan"
            case .noEvents: return "No events found. One per line: \(WeekInput.form)"
            case .tooMany(let n): return "\(n) events: Week plans at most \(WeekInput.limit) at a time"
            case .unreadable(let why): return "Could not read the file: \(why)"
            }
        }
    }

    /// "1 line skipped" / "2 lines skipped".
    public static func skippedNote(_ n: Int) -> String {
        "\(n) line\(n == 1 ? "" : "s") skipped"
    }

    // MARK: - reading

    /// Pasted text, one event a line.
    public static func read(lines text: String) -> Reading {
        var events: [WeekEvent] = []
        var skipped = 0
        for line in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            if line.allSatisfy(\.isWhitespace) { continue }
            if let event = event(line: String(line)) {
                events.append(event)
            } else {
                skipped += 1
            }
        }
        if events.isEmpty { return Reading(events: [], skipped: skipped, problem: skipped == 0 ? .empty : .noEvents) }
        return checked(events, skipped: skipped)
    }

    /// Events from elsewhere (the calendar): the limit, then time order.
    public static func checked(_ events: [WeekEvent], skipped: Int = 0) -> Reading {
        guard events.count <= limit else { return Reading(events: [], skipped: skipped, problem: .tooMany(events.count)) }
        return Reading(events: ordered(events), skipped: skipped, problem: nil)
    }

    /// The file at `url` as lines to paste: a `.json` file in `week-cli dump --out`'s form becomes
    /// one line per event, any other file is read as text.
    public static func text(contentsOf url: URL) throws -> String {
        let data = try Data(contentsOf: url)
        if url.pathExtension.lowercased() == "json" { return try lines(json: data) }
        guard let text = String(data: data, encoding: .utf8) else { throw InputError.notText }
        return text
    }

    /// A JSON array of events (`week-cli dump --out`) as the lines it would paste.
    public static func lines(json data: Data) throws -> String {
        let decoded: [WeekEvent]
        do {
            decoded = try JSONDecoder().decode([WeekEvent].self, from: data)
        } catch DecodingError.dataCorrupted(let context) where !context.codingPath.isEmpty {
            throw InputError.notAWeek(context.debugDescription)
        } catch {
            throw InputError.notAWeek("not a list of events in week-cli dump's form")
        }
        return decoded.compactMap {
            event(day: $0.day, start: $0.start, minutes: $0.minutes, title: $0.title, location: $0.location, notes: $0.notes)?
                .state
        }.joined(separator: "\n")
    }

    public enum InputError: Error, LocalizedError {
        case notText
        case notAWeek(String)

        public var errorDescription: String? {
            switch self {
            case .notText: return "not UTF-8 text"
            case .notAWeek(let why): return why
            }
        }
    }

    // MARK: - one event

    /// One line as an event (id 0), or nil when it does not start with a day, a time and a title.
    public static func event(line: String) -> WeekEvent? {
        let parts = line.components(separatedBy: separator)
        let head = parts[0].split(maxSplits: 2, omittingEmptySubsequences: true, whereSeparator: { $0 == " " || $0 == "\t" })
        guard head.count == 3, let day = day(head[0]), let start = time(head[1]) else { return nil }
        var place: [String] = []
        var notes: String?
        for (i, part) in parts.enumerated().dropFirst() {
            let trimmed = part.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("notes:") {
                notes = ([String(trimmed.dropFirst(6))] + parts[(i + 1)...]).joined(separator: separator)
                break
            }
            place.append(part)
        }
        return event(
            day: day, start: start, minutes: 60, title: String(head[2]), location: place.joined(separator: separator),
            notes: notes)
    }

    /// An event with its text cleaned the one way every source is: runs of white space (line breaks
    /// included) become one space, the ends are trimmed, a text past its limit is cut. An empty
    /// place is none, and empty notes or "none" are none. Nil when the title is empty.
    public static func event(
        day: Int, start: Int, minutes: Int, title: String, location: String?, notes: String?
    ) -> WeekEvent? {
        let title = clean(title, titleLimit)
        guard !title.isEmpty, (0..<7).contains(day) else { return nil }
        let place = location.map { clean($0, placeLimit) }.flatMap { $0.isEmpty ? nil : $0 }
        let notes = notes.map { clean($0, notesLimit) }.flatMap { $0.isEmpty || $0.lowercased() == "none" ? nil : $0 }
        return WeekEvent(
            id: 0, day: day, start: start, minutes: max(1, minutes), title: title, location: place, notes: notes,
            written: nil)
    }

    /// Time order (the day, the start, then the title, as the generator sorts), ids from 0.
    public static func ordered(_ events: [WeekEvent]) -> [WeekEvent] {
        events.sorted { ($0.day, $0.start, $0.title) < ($1.day, $1.start, $1.title) }
            .enumerated().map { $0.element.with(id: $0.offset) }
    }

    /// 0 = Monday … 6 = Sunday, from "Mon" or "Monday" in any case (a trailing "." or "," allowed).
    static func day<S: StringProtocol>(_ word: S) -> Int? {
        var w = word.lowercased()
        if w.hasSuffix(".") || w.hasSuffix(",") { w.removeLast() }
        return Week.dayNames.firstIndex { $0.lowercased() == w } ?? Week.fullDayNames.firstIndex { $0.lowercased() == w }
    }

    /// Minutes after midnight from H:MM or HH:MM, 24-hour; nil for anything else.
    static func time<S: StringProtocol>(_ word: S) -> Int? {
        let parts = word.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 2, (1...2).contains(parts[0].count), parts[1].count == 2,
              parts.allSatisfy({ $0.allSatisfy { $0.isASCII && $0.isNumber } }),
              let h = Int(parts[0]), let m = Int(parts[1]), h < 24, m < 60
        else { return nil }
        return h * 60 + m
    }

    /// White space collapsed to single spaces, trimmed, and cut to `limit` characters.
    static func clean(_ text: String, _ limit: Int) -> String {
        let out = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard out.count > limit else { return out }
        return String(out.prefix(limit - 1)).trimmingCharacters(in: .whitespaces) + "…"
    }
}
