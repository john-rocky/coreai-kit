// Week.swift — a synthetic week of calendar events for the demo: Monday to Sunday, each event one
// template with its slots filled (a title, a place, and the notes someone would jot in the event)
// at a start time from the template's hours that is still free that day.
//
// Nothing in it names a real company, brand or person: people appear only as a given name, places
// are generic ("the property office", "Room 4B"), phone numbers are 555 numbers and links are on
// example domains. Every proper name the templates use is listed in ../../NAMES.txt.
//
// A seed gives the same week on every platform and toolchain: every draw is SplitMix64 through
// `below(_:)`, never the standard library's random algorithms.

import Foundation

/// One calendar event.
public struct WeekEvent: Sendable, Identifiable, Hashable {
    /// Position in the week, in time order, from 0.
    public let id: Int
    /// 0 = Monday … 6 = Sunday.
    public let day: Int
    /// Start, in minutes after midnight.
    public let start: Int
    /// Length in minutes.
    public let minutes: Int
    public let title: String
    public let location: String?
    public let notes: String?
    /// What the generator wrote the event as. Only for reading the model's answers against: it is
    /// never shown and never given to the model. nil for an event it did not write.
    public let written: WeekBin?

    public init(
        id: Int, day: Int, start: Int, minutes: Int, title: String, location: String?, notes: String?,
        written: WeekBin?
    ) {
        self.id = id
        self.day = day
        self.start = start
        self.minutes = minutes
        self.title = title
        self.location = location
        self.notes = notes
        self.written = written
    }

    /// "Mon".
    public var dayName: String { Week.dayNames[day] }
    /// "09:30".
    public var time: String { Week.clock(start) }

    /// What the model reads: `Mon 11:00 Lease renewal signing · the property office · Notes: …`.
    /// An event without a place leaves it out; one without notes says `Notes: none`.
    public var state: String {
        var s = "\(dayName) \(time) \(title)"
        if let location, !location.isEmpty { s += " · \(location)" }
        let notes = self.notes.flatMap { $0.isEmpty ? nil : $0 } ?? "none"
        return s + " · Notes: \(notes)"
    }

    /// The same event at another position (the order changes when events come from a file).
    public func with(id: Int) -> WeekEvent {
        WeekEvent(
            id: id, day: day, start: start, minutes: minutes, title: title, location: location, notes: notes,
            written: written)
    }
}

/// The file form of an event (`week-cli dump --out`, `week-cli run --events`):
/// `{"day": "Mon", "time": "09:00", "minutes": 15, "title": …, "location": …, "notes": …, "written": …}`.
extension WeekEvent: Codable {
    enum CodingKeys: String, CodingKey { case day, time, minutes, title, location, notes, written }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let dayName = try c.decode(String.self, forKey: .day)
        guard let day = Week.dayNames.firstIndex(where: { $0.lowercased() == dayName.prefix(3).lowercased() }) else {
            throw DecodingError.dataCorruptedError(forKey: .day, in: c, debugDescription: "day must be Mon … Sun")
        }
        let time = try c.decode(String.self, forKey: .time)
        guard let start = Week.minutes(time) else {
            throw DecodingError.dataCorruptedError(forKey: .time, in: c, debugDescription: "time must be HH:MM")
        }
        self.init(
            id: 0, day: day, start: start, minutes: try c.decodeIfPresent(Int.self, forKey: .minutes) ?? 60,
            title: try c.decode(String.self, forKey: .title),
            location: try c.decodeIfPresent(String.self, forKey: .location),
            notes: try c.decodeIfPresent(String.self, forKey: .notes),
            written: try c.decodeIfPresent(String.self, forKey: .written).flatMap(WeekBin.init(rawValue:)))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(dayName, forKey: .day)
        try c.encode(time, forKey: .time)
        try c.encode(minutes, forKey: .minutes)
        try c.encode(title, forKey: .title)
        try c.encodeIfPresent(location, forKey: .location)
        try c.encodeIfPresent(notes, forKey: .notes)
        try c.encodeIfPresent(written?.rawValue, forKey: .written)
    }
}

public enum Week {
    public static let dayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    static let fullDayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    /// `count` events from `seed`, Monday to Sunday in time order. With seven or more, every day
    /// has at least one. The kinds of event are mixed in fixed shares (20 events: 4 with nothing
    /// to prepare, 3 each of documents, preparing, travel and joining online, 2 each of buying
    /// or bringing and confirming).
    public static func generate(count: Int = 20, seed: UInt64 = 7) -> [WeekEvent] {
        let count = max(0, count)
        var rng = SplitMix64(seed: seed)
        var bins = apportion(count)
        rng.shuffle(&bins)
        let days = self.days(count, rng: &rng)
        var names = NameDeck(rng: &rng)
        var decks: [WeekBin: [Int]] = [:]
        var taken: [[ClosedRange<Int>]] = Array(repeating: [], count: 7)

        var drafts: [WeekEvent] = []
        for (bin, day) in zip(bins, days) {
            let template = pick(bin, day: day, decks: &decks, rng: &rng)
            let start = slot(template, day: day, taken: &taken, rng: &rng)
            var filler = Filler(day: day, start: start, names: names)
            let title = filler.fill(template.title)
            let location = template.location.map { filler.fill($0) }
            let notes = template.notes.map { filler.fill($0) }
            names.advance(past: filler.used, rng: &rng)
            drafts.append(WeekEvent(
                id: 0, day: day, start: start, minutes: template.minutes, title: title, location: location,
                notes: notes, written: bin))
        }
        return drafts
            .sorted { ($0.day, $0.start, $0.title) < ($1.day, $1.start, $1.title) }
            .enumerated().map { $0.element.with(id: $0.offset) }
    }

    /// "HH:MM" from minutes after midnight.
    public static func clock(_ minutes: Int) -> String {
        let m = ((minutes % 1440) + 1440) % 1440
        return String(format: "%02d:%02d", m / 60, m % 60)
    }

    /// Minutes after midnight from "HH:MM"; nil when it is not a time.
    public static func minutes(_ clock: String) -> Int? {
        let parts = clock.split(separator: ":")
        guard parts.count == 2, let h = Int(parts[0]), let m = Int(parts[1]), (0..<24).contains(h), (0..<60).contains(m)
        else { return nil }
        return h * 60 + m
    }

    // MARK: - the mix

    /// Shares per kind of event, in `WeekBin` order; 20 events split exactly this way.
    static let shares: [(WeekBin, Int)] = [
        (.nothing, 4), (.document, 3), (.prepare, 3), (.travel, 3), (.online, 3), (.bring, 2), (.confirm, 2),
    ]

    /// `count` kinds in the fixed shares: the whole parts, then the rest by largest remainder (ties
    /// to the earlier kind).
    static func apportion(_ count: Int) -> [WeekBin] {
        let total = shares.reduce(0) { $0 + $1.1 }
        var parts = shares.map { count * $0.1 / total }
        let order = shares.indices.sorted {
            let a = count * shares[$0].1 % total, b = count * shares[$1].1 % total
            return a != b ? a > b : $0 < $1
        }
        var left = count - parts.reduce(0, +)
        for i in order where left > 0 {
            parts[i] += 1
            left -= 1
        }
        return zip(shares, parts).flatMap { Array(repeating: $0.0.0, count: $0.1) }
    }

    /// The day of each event: every day once (while there are events for it), then the rest drawn
    /// with the weekdays three times as likely as the weekend days' two, up to a cap per day.
    static func days(_ count: Int, rng: inout SplitMix64) -> [Int] {
        var week = Array(0..<7)
        rng.shuffle(&week)
        var out = Array(week.prefix(count))
        var load = Array(repeating: 0, count: 7)
        for d in out { load[d] += 1 }
        let weights = [3, 3, 3, 3, 3, 2, 2]
        // 5 a weekday and 3 a weekend day fit 20 events comfortably; a longer week raises both.
        let scale = max(1, Int((Double(count) / 20).rounded(.up)))
        let cap = [5, 5, 5, 5, 5, 3, 3].map { $0 * scale }
        while out.count < count {
            let open = (0..<7).filter { load[$0] < cap[$0] }
            let pool = open.isEmpty ? Array(0..<7) : open
            let total = pool.reduce(0) { $0 + weights[$1] }
            var x = rng.below(total)
            var day = pool[0]
            for d in pool {
                if x < weights[d] {
                    day = d
                    break
                }
                x -= weights[d]
            }
            out.append(day)
            load[day] += 1
        }
        return out
    }

    // MARK: - one event

    enum Days: Sendable { case weekday, weekend, any }

    /// One kind of event. `{name}` is a given name (the same one each time it appears in an event),
    /// `{t-N}` the start time minus N minutes, `{prevDay}` the day before the event, `{slug}` a
    /// meeting-link path.
    struct Template: Sendable {
        let bin: WeekBin
        let title: String
        let location: String?
        let notes: String?
        /// The start times it may have, "HH:MM".
        let times: [String]
        let minutes: Int
        let days: Days

        init(
            _ bin: WeekBin, _ title: String, at location: String? = nil, notes: String? = nil, times: [String],
            minutes: Int, _ days: Days
        ) {
            self.bin = bin
            self.title = title
            self.location = location
            self.notes = notes
            self.times = times
            self.minutes = minutes
            self.days = days
        }

        func allowed(on day: Int) -> Bool {
            switch days {
            case .any: return true
            case .weekday: return day < 5
            case .weekend: return day >= 5
            }
        }
    }

    /// A template of `bin` that may fall on `day`: each kind is dealt from its own shuffled deck, so
    /// a week repeats a template only after using every other one of its kind.
    static func pick(
        _ bin: WeekBin, day: Int, decks: inout [WeekBin: [Int]], rng: inout SplitMix64
    ) -> Template {
        let pool = templates.indices.filter { templates[$0].bin == bin }
        if decks[bin, default: []].allSatisfy({ !templates[$0].allowed(on: day) }) {
            var fresh = pool
            rng.shuffle(&fresh)
            decks[bin] = fresh
        }
        var deck = decks[bin]!
        let at = deck.firstIndex { templates[$0].allowed(on: day) }!
        let index = deck.remove(at: at)
        decks[bin] = deck
        return templates[index]
    }

    /// A start time for `template` on `day` that overlaps nothing already there: its own times in a
    /// drawn order, then every half hour from 07:00 to 21:00; its own earliest-listed time if the
    /// day is full.
    static func slot(_ template: Template, day: Int, taken: inout [[ClosedRange<Int>]], rng: inout SplitMix64) -> Int {
        var own = template.times.compactMap(minutes)
        rng.shuffle(&own)
        let halfHours = stride(from: 7 * 60, through: 21 * 60, by: 30).map { $0 }
        for start in own + halfHours {
            // an end that touches the next start is not an overlap
            let span = start...(start + template.minutes - 1)
            if !taken[day].contains(where: { $0.overlaps(span) }) {
                taken[day].append(span)
                return start
            }
        }
        let start = own.first ?? 9 * 60
        taken[day].append(start...(start + template.minutes - 1))
        return start
    }

    /// Given names, dealt without repeats until the deck runs out.
    struct NameDeck {
        private var deck: [String]
        private var next = 0

        init(rng: inout SplitMix64) {
            deck = Week.names
            rng.shuffle(&deck)
        }

        func peek(_ offset: Int) -> String { deck[(next + offset) % deck.count] }

        mutating func advance(past used: Int, rng: inout SplitMix64) {
            next += used
            if next >= deck.count {
                next = 0
                rng.shuffle(&deck)
            }
        }
    }

    /// Fills one event's slots: every `{name}` in the event is the same person.
    struct Filler {
        let day: Int
        let start: Int
        let names: NameDeck
        private(set) var used = 0

        init(day: Int, start: Int, names: NameDeck) {
            self.day = day
            self.start = start
            self.names = names
        }

        mutating func fill(_ text: String) -> String {
            var out = text
            if out.contains("{name}") {
                used = 1
                out = out.replacingOccurrences(of: "{name}", with: names.peek(0))
            }
            out = out.replacingOccurrences(of: "{prevDay}", with: Week.fullDayNames[(day + 6) % 7])
            out = out.replacingOccurrences(of: "{slug}", with: Week.slug(day: day, start: start))
            while let open = out.range(of: "{t-"), let close = out[open.upperBound...].firstIndex(of: "}") {
                let n = Int(out[open.upperBound..<close]) ?? 0
                out.replaceSubrange(open.lowerBound...close, with: Week.clock(start - n))
            }
            return out
        }
    }

    /// A meeting-link path that differs by day and time: "wed-1400".
    static func slug(day: Int, start: Int) -> String {
        "\(dayNames[day].lowercased())-\(clock(start).replacingOccurrences(of: ":", with: ""))"
    }

    // MARK: - the templates

    /// Given names only, never with a surname.
    static let names = [
        "Ava", "Ben", "Chloe", "Dev", "Elena", "Felix", "Grace", "Hugo", "Iris", "Jonah", "Kira", "Leo",
        "Mina", "Noah", "Priya", "Quinn", "Rosa", "Sam", "Tara", "Victor", "Wren", "Yuki", "Marta", "Kai",
        "Nadia", "Ravi", "Ines", "Omar",
    ]

    static let templates: [Template] = [
        // nothing to prepare
        Template(.nothing, "Team standup", at: "Room 4B", notes: "weekly sync, no agenda",
                 times: ["09:00", "09:30"], minutes: 15, .weekday),
        Template(.nothing, "Lunch with {name}", times: ["12:00", "12:30", "13:00"], minutes: 60, .any),
        Template(.nothing, "Coffee with {name}", times: ["10:30", "15:00", "15:30"], minutes: 30, .any),
        Template(.nothing, "Focus time", notes: "no meetings", times: ["13:00", "14:00", "09:00"], minutes: 120, .weekday),
        Template(.nothing, "Morning run", at: "Riverside path", notes: "easy 5k", times: ["07:00", "07:30"],
                 minutes: 45, .any),
        Template(.nothing, "All-hands", at: "Main hall", notes: "quarterly update from the leads",
                 times: ["16:00", "11:00"], minutes: 60, .weekday),
        Template(.nothing, "Haircut", at: "the salon on Pine Street", times: ["11:00", "17:30"], minutes: 45, .any),
        Template(.nothing, "Brunch with {name}", at: "the corner café", times: ["10:30", "11:00"], minutes: 90, .weekend),

        // bring a document or an ID
        Template(.document, "Lease renewal signing", at: "the property office, 12 Weaver St",
                 notes: "bring two forms of ID and the signed lease copy", times: ["11:00", "16:00"], minutes: 45, .weekday),
        Template(.document, "Passport renewal appointment", at: "Civic Center, window 6",
                 notes: "bring old passport, birth certificate and two photos", times: ["09:30", "14:30"],
                 minutes: 30, .weekday),
        Template(.document, "Bank appointment", notes: "bring proof of address and last three payslips",
                 times: ["15:00", "10:00"], minutes: 45, .weekday),
        Template(.document, "Doctor's visit", at: "Clinic, 2nd floor",
                 notes: "bring your insurance card and the referral letter", times: ["08:30", "16:30"], minutes: 30, .weekday),
        Template(.document, "Car registration", at: "the licensing office",
                 notes: "bring the title, proof of insurance and your license", times: ["10:00", "13:30"],
                 minutes: 45, .weekday),
        Template(.document, "Tax prep with the accountant", notes: "bring last year's return and the receipts folder",
                 times: ["14:00", "10:00"], minutes: 60, .any),
        Template(.document, "Open house viewing", at: "44 Linden Avenue",
                 notes: "bring a photo ID and the mortgage pre-approval letter", times: ["11:00", "13:00"],
                 minutes: 45, .weekend),
        Template(.document, "Library card renewal", at: "Main library, front desk",
                 notes: "bring your old card and a utility bill", times: ["10:00", "15:00"], minutes: 20, .any),

        // prepare or send something first
        Template(.prepare, "Design review", at: "Room 2A", notes: "send the updated mockups to the group by noon",
                 times: ["16:30", "15:00"], minutes: 60, .weekday),
        Template(.prepare, "Budget meeting", notes: "prepare the Q4 spend summary and share it before the meeting",
                 times: ["13:00", "11:00"], minutes: 60, .weekday),
        Template(.prepare, "1:1 with {name}", notes: "write up the hiring plan draft first",
                 times: ["10:00", "14:30"], minutes: 30, .weekday),
        Template(.prepare, "Client pitch", at: "Conference room C",
                 notes: "finish the slides and send them to {name} the day before", times: ["14:00", "11:00"],
                 minutes: 60, .weekday),
        Template(.prepare, "Quarterly review", notes: "fill in the self-review form before the meeting",
                 times: ["11:00", "15:30"], minutes: 45, .weekday),
        Template(.prepare, "Interview panel", at: "Room 3C", notes: "read the resume and prepare three questions",
                 times: ["13:30", "10:30"], minutes: 45, .weekday),
        Template(.prepare, "Garage sale", at: "the driveway", notes: "price the boxes and put up the signs first",
                 times: ["09:00", "08:30"], minutes: 180, .weekend),
        Template(.prepare, "Neighborhood meeting", at: "Community room", notes: "print the agenda and send it round first",
                 times: ["18:00", "10:30"], minutes: 60, .any),

        // travel time: leave early
        Template(.travel, "Flight to the sales conference", at: "Terminal 2",
                 notes: "leave for the airport by {t-120}, gate closes {t-30}", times: ["07:15", "08:40", "06:50"],
                 minutes: 180, .any),
        Template(.travel, "Site visit at the north warehouse",
                 notes: "40 minute drive, leave by {t-45}, parking behind building C", times: ["12:00", "14:00"],
                 minutes: 90, .weekday),
        Template(.travel, "Train to Aunt {name}'s", at: "Central Station, platform 3",
                 notes: "the station is 30 minutes away, leave by {t-40}", times: ["08:00", "09:30"], minutes: 240, .weekend),
        Template(.travel, "Offsite at the lake lodge", notes: "90 minute drive, leave by {t-100}",
                 times: ["10:00", "09:30"], minutes: 240, .weekday),
        Template(.travel, "Airport pickup: {name}", at: "Arrivals, Terminal 1",
                 notes: "traffic is bad after 5, leave by {t-60}", times: ["17:30", "18:15"], minutes: 60, .any),
        Template(.travel, "Ferry to the island", at: "Pier 9",
                 notes: "boarding closes 15 minutes before, leave home by {t-50}", times: ["09:00", "10:15"],
                 minutes: 120, .weekend),
        Template(.travel, "Wedding rehearsal", at: "the chapel on Hill Road", notes: "an hour's drive, leave by {t-75}",
                 times: ["16:00", "15:00"], minutes: 120, .any),

        // join online: a link or dial-in
        Template(.online, "Vendor call", notes: "join via https://meet.example/{slug}, dial-in +1 555 0142",
                 times: ["14:00", "11:30"], minutes: 30, .weekday),
        Template(.online, "Book club", notes: "on video this week, link in the group chat",
                 times: ["17:00", "19:30"], minutes: 60, .any),
        Template(.online, "Yoga class", notes: "livestream, use the studio's link", times: ["10:00", "07:30"],
                 minutes: 60, .any),
        Template(.online, "Remote interview", notes: "video call, link in the invite: https://meet.example/{slug}",
                 times: ["15:30", "09:30"], minutes: 45, .weekday),
        Template(.online, "Family video call", notes: "{name} sends the link, or dial in at +1 555 0199",
                 times: ["18:00", "11:00"], minutes: 45, .weekend),
        Template(.online, "Webinar: planning for spring", notes: "join at https://events.example/{slug}",
                 times: ["12:00", "16:00"], minutes: 60, .weekday),
        Template(.online, "Team retro", notes: "remote this week: https://meet.example/{slug}",
                 times: ["16:00", "15:00"], minutes: 45, .weekday),

        // buy or bring something
        Template(.bring, "{name}'s birthday dinner", notes: "pick up the cake from the bakery and bring candles",
                 times: ["18:00", "19:00"], minutes: 120, .any),
        Template(.bring, "Potluck at {name}'s", notes: "you are down for the salad and plates",
                 times: ["19:00", "18:30"], minutes: 150, .any),
        Template(.bring, "Picnic in the park", at: "the east lawn", notes: "bring a blanket and the drinks",
                 times: ["12:00", "13:00"], minutes: 120, .weekend),
        Template(.bring, "Housewarming at {name}'s", notes: "buy a plant or a bottle on the way",
                 times: ["18:30", "19:30"], minutes: 120, .any),
        Template(.bring, "Team breakfast", at: "Kitchen, 3rd floor", notes: "your turn to bring the bagels",
                 times: ["08:30", "09:00"], minutes: 30, .weekday),
        Template(.bring, "Kids' soccer game", at: "Field 2", notes: "snack duty: orange slices and water for twelve",
                 times: ["10:00", "09:00"], minutes: 90, .weekend),
        Template(.bring, "Baby shower for {name}", notes: "get the gift from the registry", times: ["14:00", "15:00"],
                 minutes: 120, .weekend),

        // confirm or reply by a deadline
        Template(.confirm, "Dentist", notes: "reply to the clinic's text by {prevDay} to confirm the slot",
                 times: ["10:00", "16:00"], minutes: 45, .weekday),
        Template(.confirm, "Car service", at: "the garage on Mill Road",
                 notes: "call to confirm the loaner car is available", times: ["11:30", "08:30"], minutes: 60, .weekday),
        Template(.confirm, "Dinner reservation", at: "the bistro on Elm Street",
                 notes: "confirm the table for four by {prevDay} 5 pm or it is released", times: ["19:30", "20:00"],
                 minutes: 90, .any),
        Template(.confirm, "Plumber visit", notes: "text back by {prevDay} to confirm the 2-hour window",
                 times: ["09:00", "13:00"], minutes: 120, .any),
        Template(.confirm, "Vet appointment", notes: "reply YES to the reminder text by {prevDay}",
                 times: ["16:00", "10:30"], minutes: 30, .any),
        Template(.confirm, "Guest talk at the library", notes: "RSVP to the organizers by {prevDay}",
                 times: ["18:00", "17:00"], minutes: 60, .any),
        Template(.confirm, "Parent-teacher meeting", at: "Room 12",
                 notes: "confirm the time slot with the school office by {prevDay}", times: ["15:30", "17:00"],
                 minutes: 20, .weekday),
    ]
}

/// SplitMix64 (Steele, Lea, Flood 2014): a 64-bit state, one add and a mix per draw.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// 0..<n: the high word of next() × n (bias below n / 2^64).
    mutating func below(_ n: Int) -> Int {
        precondition(n > 0)
        return Int(next().multipliedFullWidth(by: UInt64(n)).high)
    }

    /// Fisher–Yates through `below(_:)`, so the order is the same on every platform.
    mutating func shuffle<T>(_ xs: inout [T]) {
        guard xs.count > 1 else { return }
        for i in stride(from: xs.count - 1, to: 0, by: -1) {
            xs.swapAt(i, below(i + 1))
        }
    }
}
