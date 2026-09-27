// The generator's contract: a seed is a week, the count is kept, and the week reads like a week.
// No model is loaded here.

import Foundation
import Testing
import WeekCore

@Suite struct WeekTests {
    @Test func sameSeedSameWeek() {
        #expect(Week.generate(count: 20, seed: 7) == Week.generate(count: 20, seed: 7))
        #expect(Week.generate(count: 20, seed: 7) != Week.generate(count: 20, seed: 8))
    }

    @Test(arguments: [0, 1, 6, 7, 20, 35, 60])
    func requestedCount(_ count: Int) {
        #expect(Week.generate(count: count, seed: 7).count == count)
    }

    @Test(arguments: [7, 20, 35, 60])
    func everyDayOfTheWeek(_ count: Int) {
        let days = Set(Week.generate(count: count, seed: 7).map(\.day))
        #expect(days == Set(0..<7))
    }

    @Test(arguments: [1, 20, 60])
    func noEmptyOrUnfilledText(_ count: Int) {
        for event in Week.generate(count: count, seed: 7) {
            #expect(!event.title.trimmingCharacters(in: .whitespaces).isEmpty)
            for text in [event.title, event.location ?? "", event.notes ?? ""] {
                #expect(!text.contains("{") && !text.contains("}"), "unfilled slot in \(text)")
            }
        }
    }

    @Test func timeOrderAndNoOverlaps() {
        let week = Week.generate(count: 20, seed: 7)
        #expect(week.map(\.id) == Array(0..<20))
        for (a, b) in zip(week, week.dropFirst()) {
            #expect((a.day, a.start) <= (b.day, b.start))
            if a.day == b.day { #expect(a.start + a.minutes <= b.start, "\(a.title) runs into \(b.title)") }
        }
    }

    @Test func twentyEventsInTheFixedMix() {
        let counts = Dictionary(grouping: Week.generate(count: 20, seed: 7), by: { $0.written! }).mapValues(\.count)
        #expect(counts == [.nothing: 4, .document: 3, .prepare: 3, .travel: 3, .online: 3, .bring: 2, .confirm: 2])
    }

    @Test func stateReadsLikeTheCalendar() {
        let event = WeekEvent(
            id: 0, day: 0, start: 11 * 60, minutes: 45, title: "Lease renewal signing",
            location: "the property office", notes: nil, written: .document)
        #expect(event.state == "Mon 11:00 Lease renewal signing · the property office · Notes: none")
    }

    @Test func fileFormRoundTrips() throws {
        let week = Week.generate(count: 20, seed: 7)
        let data = try JSONEncoder().encode(week)
        let back = try JSONDecoder().decode([WeekEvent].self, from: data).enumerated().map { $0.element.with(id: $0.offset) }
        #expect(back == week)
    }
}
