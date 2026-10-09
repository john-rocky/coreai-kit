// The paste contract: lines in the form the model reads come back as the same states, a line
// without a day and a time is skipped and counted, and an empty paste, a paste with nothing
// readable and a week over the limit are refused with the reason. No model is loaded here.

import Foundation
import Testing
import WeekCore

@Suite struct WeekInputTests {
    /// `week-cli dump`'s lines, as pasted.
    static func lines(_ count: Int, seed: UInt64 = 7) -> String {
        Week.generate(count: count, seed: seed).map(\.state).joined(separator: "\n")
    }

    @Test(arguments: [1, 20, 60, 200])
    func dumpedLinesReadBackAsTheSameStates(_ count: Int) {
        let reading = WeekInput.read(lines: Self.lines(count))
        #expect(reading.problem == nil)
        #expect(reading.skipped == 0)
        #expect(reading.note == nil)
        #expect(reading.events.map(\.state) == Week.generate(count: count, seed: 7).map(\.state))
        #expect(reading.events.map(\.id) == Array(0..<count))
    }

    @Test func aLineReadsLikeTheCalendar() throws {
        let text = """
            Monday 9:05 Dentist · Notes: none
            tue 14:30 Vendor call · Notes: join via https://meet.example/tue-1430 · dial-in +1 555 0142
            Wed 08:00 Site visit · the north warehouse · gate B
            """
        let reading = WeekInput.read(lines: text)
        #expect(reading.problem == nil && reading.skipped == 0)
        let e = reading.events
        #expect(e.count == 3)
        #expect(e[0].state == "Mon 09:05 Dentist · Notes: none")
        #expect(e[0].notes == nil && e[0].location == nil)
        #expect(e[1].notes == "join via https://meet.example/tue-1430 · dial-in +1 555 0142")
        #expect(e[1].location == nil)
        #expect(e[2].location == "the north warehouse · gate B")
        #expect(e[2].state == "Wed 08:00 Site visit · the north warehouse · gate B · Notes: none")
    }

    @Test func linesComeBackInTimeOrder() {
        let reading = WeekInput.read(lines: "Sun 10:00 Brunch\nMon 18:00 Book club\nMon 07:30 Morning run")
        #expect(reading.events.map(\.title) == ["Morning run", "Book club", "Brunch"])
        #expect(reading.events.map(\.id) == [0, 1, 2])
    }

    @Test func brokenLinesAreSkippedAndCounted() {
        var lines = Self.lines(10).components(separatedBy: "\n")
        lines[3] = "Lease renewal signing at eleven"          // no day, no time
        lines[7] = "Someday 11:00 Lease renewal signing"      // not a day
        let text = lines.joined(separator: "\n") + "\n\n   \n"  // blank lines are not counted
        let reading = WeekInput.read(lines: text)
        #expect(reading.problem == nil)
        #expect(reading.events.count == 8)
        #expect(reading.skipped == 2)
        #expect(reading.note == "2 lines skipped")
    }

    @Test(arguments: [
        "Mon 9:5 Late", "Mon 24:00 Late", "Mon 11:60 Late", "Mon 1100 Late", "Mon 11:00", "Mon 11:00   ",
        "Mon ９:００ Late", "Mon +9:00 Late", "11:00 Mon Late", "Mo 11:00 Late",
    ])
    func notALine(_ line: String) {
        #expect(WeekInput.event(line: line) == nil)
    }

    @Test func nothingReadableIsRefused() {
        let reading = WeekInput.read(lines: "groceries\ncall mom\nMon at 11")
        #expect(reading.events.isEmpty)
        #expect(reading.skipped == 3)
        #expect(reading.problem == .noEvents)
        #expect(reading.note == "No events found. One per line: Mon 11:00 Title · Place · Notes: …")
    }

    @Test(arguments: ["", "\n", "  \n\t\n"])
    func emptyIsNothingToPlan(_ text: String) {
        let reading = WeekInput.read(lines: text)
        #expect(reading.events.isEmpty && reading.skipped == 0)
        #expect(reading.problem == .empty)
        #expect(reading.note == "Nothing to plan")
    }

    @Test func overTheLimitIsRefusedWhole() {
        let over = WeekInput.read(lines: Self.lines(201))
        #expect(over.events.isEmpty)
        #expect(over.problem == .tooMany(201))
        #expect(over.note == "201 events: Week plans at most 200 at a time")
        #expect(WeekInput.read(lines: Self.lines(200)).events.count == 200)
    }

    @Test func longTextIsCleanedAndCut() throws {
        let notes = "Join the meeting:\n  https://meet.example/x\n" + String(repeating: "dial-in details ", count: 40)
        let e = try #require(WeekInput.event(
            day: 2, start: 600, minutes: 30, title: "  Weekly\nsync ", location: "", notes: notes))
        #expect(e.title == "Weekly sync")
        #expect(e.location == nil)
        #expect(e.notes?.hasPrefix("Join the meeting: https://meet.example/x dial-in") == true)
        #expect(e.notes?.count == WeekInput.notesLimit)
        #expect(e.notes?.hasSuffix("…") == true)
        #expect(!e.state.contains("\n"))
    }

    @Test func aDumpFileBecomesTheSameLines() throws {
        let week = Week.generate(count: 20, seed: 7)
        let data = try JSONEncoder().encode(week)
        let text = try WeekInput.lines(json: data)
        #expect(text == Self.lines(20))
        #expect(throws: WeekInput.InputError.self) { try WeekInput.lines(json: Data("{\"day\": 1}".utf8)) }
    }
}
