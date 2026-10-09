// The input contract: the sample inbox pasted one message a line reads back as the same messages,
// blocks separated by a blank line are one message each, a .csv file gives one message a row, a
// file that is not UTF-8 is refused with the reason, an empty paste and an inbox over the limit
// are refused, and the categories field reads 2 to 12 distinct names. No model is loaded here.

import Foundation
import InboxCore
import Testing

@Suite struct InboxInputTests {
    /// `textclassify-cli --inbox <count> --dump`'s lines, as pasted.
    static func lines(_ count: Int, seed: UInt64 = 7) -> String {
        Inbox.generate(count: count, seed: seed).map(\.text).joined(separator: "\n")
    }

    @Test(arguments: [1, 100, 2_000])
    func dumpedLinesReadBackAsTheSameMessages(_ count: Int) {
        let reading = InboxInput.read(text: Self.lines(count) + "\n")
        #expect(reading.problem == nil)
        #expect(reading.skipped == 0)
        #expect(reading.note == nil)
        #expect(reading.messages == Inbox.generate(count: count, seed: 7).map(\.text))
        let inbox = InboxInput.messages(reading.messages)
        #expect(inbox.map(\.id) == Array(0..<count))
        #expect(inbox.map(\.text) == reading.messages)
    }

    @Test func aBlankLineMakesBlocks() {
        let text = """
            Hi team,
            my order 41512 never arrived.
            Can you check?

            Love the new app.
               \t
            The refund of $20 is still missing.
            Order 10023.
            """
        let reading = InboxInput.read(text: text)
        #expect(reading.problem == nil)
        #expect(reading.messages == [
            "Hi team, my order 41512 never arrived. Can you check?",
            "Love the new app.",
            "The refund of $20 is still missing. Order 10023.",
        ])
    }

    @Test func blankLinesAtTheEndsKeepOneMessageALine() {
        let reading = InboxInput.read(text: "\n\n  first message  \r\nsecond\t\tmessage\n\n\n")
        #expect(reading.messages == ["first message", "second message"])
    }

    @Test func linesWithoutLettersOrDigitsAreSkippedAndCounted() {
        let reading = InboxInput.read(text: "-----\nRefund please, order 12.\n***\n=====\nThanks!")
        #expect(reading.messages == ["Refund please, order 12.", "Thanks!"])
        #expect(reading.skipped == 3)
        #expect(reading.note == "3 skipped (no letters or digits)")
        // Japanese is letters too.
        #expect(InboxInput.read(text: "注文がまだ届きません。").messages == ["注文がまだ届きません。"])
    }

    @Test func nothingToSortIsRefused() {
        for text in ["", "   ", "\n\n \t\n", "----\n...."] {
            let reading = InboxInput.read(text: text)
            #expect(reading.messages.isEmpty)
            #expect(reading.problem == .empty)
            #expect(reading.note == "Nothing to sort")
        }
    }

    @Test func overTheLimitIsRefusedWhole() {
        #expect(InboxInput.read(text: Self.lines(2_000)).messages.count == 2_000)
        let reading = InboxInput.read(text: Self.lines(2_001))
        #expect(reading.messages.isEmpty)
        #expect(reading.problem == .tooMany(2_001))
        #expect(reading.note == "2,001 messages: TextClassify sorts at most 2,000 at a time")
    }

    @Test func csvRowsLoseTheirQuotes() {
        let csv = """
            "Where is order 41512? It was due Monday, nothing yet."
            Ben,"Please refund the ""damaged"" kettle",order 10023

            "A message
            over two lines",
            ,,
            """
        let reading = InboxInput.read(csv: csv)
        #expect(reading.problem == nil)
        #expect(reading.messages == [
            "Where is order 41512? It was due Monday, nothing yet.",
            "Ben, Please refund the \"damaged\" kettle, order 10023",
            "A message over two lines",
        ])
    }

    @Test func filesAreReadByTheirKind() throws {
        let dir = FileManager.default.temporaryDirectory.appending(path: "InboxInputTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // UTF-8 with a byte-order mark, as some editors save it.
        let txt = dir.appending(path: "inbox.txt")
        try (Data([0xEF, 0xBB, 0xBF]) + Data("Refund please.\nWhere is my order?\n".utf8)).write(to: txt)
        #expect(InboxInput.read(contentsOf: txt).messages == ["Refund please.", "Where is my order?"])

        let md = dir.appending(path: "inbox.md")
        try Data("First line\nof one message.\n\nSecond message.".utf8).write(to: md)
        #expect(InboxInput.read(contentsOf: md).messages == ["First line of one message.", "Second message."])

        let csv = dir.appending(path: "inbox.CSV")
        try Data("\"one, with a comma\"\ntwo\n".utf8).write(to: csv)
        #expect(InboxInput.read(contentsOf: csv).messages == ["one, with a comma", "two"])

        // Shift_JIS is not UTF-8: refused with the reason, nothing guessed.
        let sjis = dir.appending(path: "inbox-sjis.txt")
        try #require("注文した商品がまだ届きません。\n返金をお願いします。\n".data(using: .shiftJIS)).write(to: sjis)
        let refused = InboxInput.read(contentsOf: sjis)
        #expect(refused.messages.isEmpty)
        #expect(refused.problem == .unreadable("not UTF-8 text"))
        #expect(refused.note == "Could not read the file: not UTF-8 text")

        let missing = InboxInput.read(contentsOf: dir.appending(path: "nowhere.txt"))
        #expect(missing.messages.isEmpty)
        if case .unreadable? = missing.problem {} else { Issue.record("a missing file reads as \(String(describing: missing.problem))") }
    }

    // MARK: - the categories field

    @Test func theDefaultCategoriesAreTheEightIntents() {
        let parsed = InboxCategories.parse(InboxCategories.defaultText)
        #expect(parsed.problem == nil)
        #expect(parsed.labels == InboxTasks.intent.labels)
        // The default categories ask exactly what the shipped inbox asks.
        let tasks = InboxCategories.tasks(parsed.labels)
        #expect(tasks.map(\.name) == InboxTasks.all.map(\.name))
        #expect(tasks.map(\.labels) == InboxTasks.all.map(\.labels))
        #expect(tasks.map(\.descriptions) == InboxTasks.all.map(\.descriptions))
        #expect(tasks.map(\.prompt) == InboxTasks.all.map(\.prompt))
    }

    @Test func yourOwnCategoriesReplaceTheIntentsOnly() {
        let parsed = InboxCategories.parse(" work,family , bills,\ttravel,  online   shopping, ")
        #expect(parsed.problem == nil)
        #expect(parsed.labels == ["work", "family", "bills", "travel", "online shopping"])
        let tasks = InboxCategories.tasks(parsed.labels)
        #expect(tasks[0].name == "intent" && tasks[0].labels == parsed.labels)
        #expect(tasks[1].labels == InboxTasks.urgency.labels && tasks[1].descriptions == InboxTasks.urgency.descriptions)
        #expect(tasks[2].labels == InboxTasks.sentiment.labels)
    }

    @Test func categoriesThatCannotBeSortedIntoSayWhy() {
        #expect(InboxCategories.parse("").problem == .tooFew(0))
        #expect(InboxCategories.parse(" , ,").problem == .tooFew(0))
        #expect(InboxCategories.parse("work").problem == .tooFew(1))
        #expect(InboxCategories.parse("work").problem?.message == "Name at least two categories, separated by commas")
        #expect(InboxCategories.parse("work, family, Work").problem == .duplicate("Work"))
        #expect(InboxCategories.parse("work, family, Work").problem?.message == "“Work” is named twice")
        let thirteen = (1...13).map { "c\($0)" }.joined(separator: ",")
        #expect(InboxCategories.parse(thirteen).problem == .tooMany(13))
        #expect(InboxCategories.parse((1...12).map { "c\($0)" }.joined(separator: ",")).problem == nil)
        let long = String(repeating: "x", count: 41)
        #expect(InboxCategories.parse("work, \(long)").problem == .tooLong(long))
        #expect(InboxCategories.parse("work, \(String(repeating: "x", count: 40))").problem == nil)
    }
}
