// InboxInput.swift — an inbox you hand over instead of the generator's, and the categories you sort
// it into. The app's Paste sheet, its Import file button and the autoplay harness (`-source paste`,
// `-source file`, `-categories`) all read through these functions, so every source reaches the
// model in one shape.
//
// Pasted text and a .txt or .md file: when a blank line separates two blocks of text, every block
// is one message (an email pasted with its line breaks); otherwise every line is one message. A
// .csv file: every row is one message, its fields joined with ", " and the quotes taken off. In
// every source, runs of white space become one space and the ends are trimmed; a message with no
// letter or digit (a rule like `-----`) is skipped and counted. A file must be UTF-8.
//
// The categories are the labels of the first question. The default is the support inbox's eight
// intents (`InboxTasks.intent`); any 2 to 12 names separated by commas replace them. Urgency and
// sentiment are asked as shipped, so the run's urgency order means the same whatever the
// categories are.

import CoreAIKitEmbeddings
import Foundation

public enum InboxInput {
    /// The most messages one sort may hold. A longer paste or file is refused whole, not cut.
    public static let limit = 2_000

    /// What reading a paste or a file gave.
    public struct Reading: Sendable, Equatable {
        /// The messages in order; empty when `problem` is set.
        public let messages: [String]
        /// Lines or blocks with no letter or digit, passed over.
        public let skipped: Int
        /// Why there is nothing to sort; nil when `messages` is not empty.
        public let problem: Problem?

        /// One line for the screen: the problem, else how many were skipped, else nil.
        public var note: String? {
            if let problem { return problem.message }
            return skipped == 0 ? nil : InboxInput.skippedNote(skipped)
        }
    }

    public enum Problem: Sendable, Equatable {
        /// Nothing but white space, or only lines with no letter or digit.
        case empty
        /// More messages than `limit`.
        case tooMany(Int)
        /// A file that could not be read as text.
        case unreadable(String)

        public var message: String {
            switch self {
            case .empty: return "Nothing to sort"
            case .tooMany(let n): return "\(grouped(n)) messages: TextClassify sorts at most \(grouped(InboxInput.limit)) at a time"
            case .unreadable(let why): return "Could not read the file: \(why)"
            }
        }
    }

    /// "1 skipped (no letters or digits)" / "3 skipped (no letters or digits)".
    public static func skippedNote(_ n: Int) -> String {
        "\(grouped(n)) skipped (no letters or digits)"
    }

    // MARK: - reading

    /// Pasted text (or a .txt / .md file): blocks when a blank line separates two of them, else lines.
    public static func read(text: String) -> Reading {
        var blocks: [[Substring]] = [[]]
        for line in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
            if line.allSatisfy(\.isWhitespace) {
                if !blocks[blocks.count - 1].isEmpty { blocks.append([]) }
            } else {
                blocks[blocks.count - 1].append(line)
            }
        }
        blocks.removeAll { $0.isEmpty }
        let pieces: [String] = blocks.count > 1
            ? blocks.map { $0.joined(separator: " ") }
            : (blocks.first ?? []).map(String.init)
        return checked(pieces)
    }

    /// A .csv file: one row a message, the quotes taken off, a row's fields joined with ", ".
    public static func read(csv: String) -> Reading {
        checked(rows(csv: csv).map { $0.map(clean).filter { !$0.isEmpty }.joined(separator: ", ") })
    }

    /// A file: `.csv` as CSV, anything else as pasted text. It must be UTF-8 (a byte-order mark is
    /// allowed).
    public static func read(contentsOf url: URL) -> Reading {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            return Reading(messages: [], skipped: 0, problem: .unreadable(error.localizedDescription))
        }
        guard let text = self.text(utf8: data) else {
            return Reading(messages: [], skipped: 0, problem: .unreadable("not UTF-8 text"))
        }
        return url.pathExtension.lowercased() == "csv" ? read(csv: text) : read(text: text)
    }

    /// The bytes as UTF-8 text, without a leading byte-order mark; nil when they are not UTF-8.
    public static func text(utf8 data: Data) -> String? {
        let body = data.starts(with: [0xEF, 0xBB, 0xBF]) ? data.dropFirst(3) : data[...]
        return String(data: body, encoding: .utf8)
    }

    /// Cleans the pieces, skips those with no letter or digit, applies the limit.
    static func checked(_ pieces: [String]) -> Reading {
        var messages: [String] = []
        var skipped = 0
        for piece in pieces {
            let message = clean(piece)
            if message.isEmpty { continue }
            if message.unicodeScalars.contains(where: { CharacterSet.alphanumerics.contains($0) }) {
                messages.append(message)
            } else {
                skipped += 1
            }
        }
        if messages.isEmpty { return Reading(messages: [], skipped: skipped, problem: .empty) }
        guard messages.count <= limit else { return Reading(messages: [], skipped: skipped, problem: .tooMany(messages.count)) }
        return Reading(messages: messages, skipped: skipped, problem: nil)
    }

    /// White space collapsed to single spaces, the ends trimmed.
    static func clean<S: StringProtocol>(_ text: S) -> String {
        text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    /// RFC 4180 rows: fields split at commas outside quotes; a quoted field may hold commas, line
    /// breaks and "" for one quote. An unclosed quote runs to the end of the text. Blank rows are
    /// dropped.
    static func rows(csv: String) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var quoted = false
        var chars = csv.makeIterator()
        var pending: Character? = nil
        func endRow() {
            row.append(field)
            if row.contains(where: { !$0.allSatisfy(\.isWhitespace) }) { rows.append(row) }
            row = []
            field = ""
        }
        while let c = pending ?? chars.next() {
            pending = nil
            if quoted {
                if c == "\"" {
                    if let next = chars.next() {
                        if next == "\"" { field.append("\"") } else { quoted = false; pending = next }
                    } else {
                        quoted = false
                    }
                } else {
                    field.append(c)
                }
            } else if c == "\"" {
                quoted = true
            } else if c == "," {
                row.append(field)
                field = ""
            } else if c.isNewline {
                endRow()
            } else {
                field.append(c)
            }
        }
        if !field.isEmpty || !row.isEmpty { endRow() }
        return rows
    }

    /// The texts as an inbox, ids from 0. Nothing was written for them, so the generator's labels
    /// are empty (`SortRun.agreement` is only read for the sample inbox).
    public static func messages(_ texts: [String]) -> [Message] {
        texts.enumerated().map {
            Message(id: $0.offset, text: $0.element, written: .init(intent: "", urgency: "", sentiment: ""))
        }
    }
}

/// The categories field: the names to sort into, separated by commas.
public enum InboxCategories {
    public static let range = 2...12
    /// Characters per name. Every name is part of every message's input, and a long one leaves less
    /// room for the message in the 256-token graph.
    public static let lengthLimit = 40
    /// The support inbox's eight intents.
    public static let defaults: [String] = InboxTasks.intent.labels
    public static let defaultText = defaults.joined(separator: ", ")

    public struct Parsed: Sendable, Equatable {
        /// The names in the order written, trimmed; empty when `problem` is set.
        public let labels: [String]
        public let problem: Problem?
    }

    public enum Problem: Sendable, Equatable {
        case tooFew(Int)
        case tooMany(Int)
        case duplicate(String)
        case tooLong(String)

        public var message: String {
            switch self {
            case .tooFew: return "Name at least two categories, separated by commas"
            case .tooMany(let n): return "\(n) categories: name at most \(InboxCategories.range.upperBound)"
            case .duplicate(let name): return "“\(name)” is named twice"
            case .tooLong(let name):
                return "“\(name.prefix(16))…” is too long: keep a category under \(InboxCategories.lengthLimit) characters"
            }
        }
    }

    /// The field's text as names: split at commas, white space collapsed, empty pieces dropped
    /// (a trailing comma is fine). Two names that differ only in case are the same name.
    public static func parse(_ text: String) -> Parsed {
        let names = text.split(separator: ",").map { InboxInput.clean($0) }.filter { !$0.isEmpty }
        var seen = Set<String>()
        for name in names {
            if name.count > lengthLimit { return Parsed(labels: [], problem: .tooLong(name)) }
            if !seen.insert(name.lowercased()).inserted { return Parsed(labels: [], problem: .duplicate(name)) }
        }
        if names.count < range.lowerBound { return Parsed(labels: [], problem: .tooFew(names.count)) }
        if names.count > range.upperBound { return Parsed(labels: [], problem: .tooMany(names.count)) }
        return Parsed(labels: names, problem: nil)
    }

    /// The three questions for these categories: the categories in place of the eight intents
    /// (asked as `intent`, the name the model reads), then urgency and sentiment as shipped. The
    /// default categories give exactly `InboxTasks.all`.
    public static func tasks(_ labels: [String]) -> [ClassificationTask] {
        [ClassificationTask(InboxTasks.intent.name, labels: labels), InboxTasks.urgency, InboxTasks.sentiment]
    }
}

/// 1,000 — the same grouping in every locale.
public func grouped(_ n: Int) -> String {
    n.formatted(.number.locale(Locale(identifier: "en_US")))
}
