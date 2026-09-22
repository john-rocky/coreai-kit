// ColumnsModel — screen: a table of texts, and typed columns the model fills. Open a CSV (or
// paste one text per line), write the columns as questions — a choice, a score, a yes/no —
// and every row is answered: its text prefilled once, one scored prompt per column. Sort by
// any column, save the CSV with the new columns in it. The bulk-classification shape of the
// System One posts (tickets, ads, emails, rows of a database) with nothing generated.
//
// Which questions read well on a short row is the whole craft. On these tickets MiniCPM5 2B
// names the topic 12/12 and what the customer wants 11/12 as choices, and places the mood on
// an upset–neutral–happy scale as expected; asked how *urgent* a row is, on any shape, it
// answers "today" for almost every row (2026-09-23) — so the sample has no urgency column.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class ColumnsModel {
    struct Column: Identifiable {
        let key: String
        let title: String
        let question: Decision.Question
        var id: String { key }
    }

    struct Row: Identifiable {
        let id: Int
        /// The fields as read from the file (header: value), or the one line.
        let fields: [(String, String)]
        /// What the model reads.
        let text: String
        var answers: [String: Decision.Answer] = [:]
        var milliseconds = 0.0
    }

    /// Characters read per row.
    static let rowLimit = 1200

    var rowsText = ColumnsModel.sampleRows
    var columnsText = ColumnsModel.sampleColumns
    var rows: [Row] = []
    var sortKey: String?
    var status = "Open a CSV, or use the sample, then Fill."
    var working = false
    var name = "sample tickets"
    var filled = 0

    var columns: [Column] { (try? Self.parseColumns(columnsText)) ?? [] }
    var totalMilliseconds: Double { rows.map(\.milliseconds).reduce(0, +) }
    var decisionsMade: Int { rows.map { $0.answers.count }.reduce(0, +) }
    /// One line of every row's cells, for the hands-off log.
    var detail: String { let columns = columns; return rows.map { row in "\(row.id):" + columns.map { label(row, $0) }.joined(separator: "/") }.joined(separator: " | ") }

    /// Rows in the sort column's order: choices grouped alphabetically, scores high first,
    /// yes before no; unsorted rows keep the file order.
    var sortedRows: [Row] {
        guard let sortKey, let column = columns.first(where: { $0.key == sortKey }) else { return rows }
        return rows.sorted { a, b in
            switch (a.answers[sortKey]?.value, b.answers[sortKey]?.value) {
            case (.some(.choice(let x)), .some(.choice(let y))):
                return x.id == y.id ? x.confidence > y.confidence : x.id < y.id
            case (.some(.score(let x)), .some(.score(let y))):
                return x.value > y.value
            case (.some(.noul(let x)), .some(.noul(let y))):
                return x > y
            case (.some, .none): return true
            default: return a.id < b.id
            }
            _ = column
        }
    }

    /// One column per line: `Title = choice: question | a | b`, `Title = score: question | low | high`,
    /// `Title = noul: question`. Without `Title =`, the kind is the title.
    static func parseColumns(_ text: String) throws -> [Column] {
        var out: [Column] = []
        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            var title: String?
            var spec = line
            if let eq = line.firstIndex(of: "="), !line[..<eq].contains(":") {
                title = line[..<eq].trimmingCharacters(in: .whitespaces)
                spec = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
            }
            guard let parsed = try ChecklistModel.parse(spec).first else { continue }
            let key = "c\(out.count + 1)"
            out.append(Column(key: key, title: title ?? spec.prefix { $0 != ":" }.capitalized, question: parsed.1))
        }
        return out
    }

    /// A row's cell for a column, in the column's own words.
    func label(_ row: Row, _ column: Column) -> String {
        guard let answer = row.answers[column.key] else { return "" }
        switch answer.value {
        case .choice(let c): return c.id
        case .noul(let p): return p >= 0.5 ? "yes" : "no"
        case .score(let s):
            if case .score(let levels) = column.question.kind, s.level < levels.count { return levels[s.level] }
            return "\(s.level)"
        }
    }

    // MARK: - Rows in

    func load(_ url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            status = "Could not read \(url.lastPathComponent) as text."
            return
        }
        rowsText = text
        name = url.lastPathComponent
        parseRows()
    }

    func loadSample() {
        rowsText = Self.sampleRows
        columnsText = Self.sampleColumns
        name = "sample tickets"
        parseRows()
    }

    /// A CSV with a header becomes `header: value` pairs per row; anything else is one row per line.
    func parseRows() {
        rows = []
        sortKey = nil
        filled = 0
        let records = Self.csvRecords(rowsText)
        if records.count >= 2, records[0].count >= 2, records.dropFirst().allSatisfy({ $0.count == records[0].count }) {
            let header = records[0]
            for (index, record) in records.dropFirst().enumerated() {
                let fields = Array(zip(header, record))
                let text = fields.map { "\($0.0): \($0.1)" }.joined(separator: "\n")
                rows.append(Row(id: index + 1, fields: fields, text: String(text.prefix(Self.rowLimit))))
            }
        } else {
            for (index, line) in rowsText.components(separatedBy: "\n").enumerated() {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { continue }
                rows.append(Row(id: index + 1, fields: [("text", trimmed)], text: String(trimmed.prefix(Self.rowLimit))))
            }
        }
        status = rows.isEmpty ? "No rows." : "\(rows.count) rows in \(name) — Fill."
    }

    /// Minimal CSV: commas, double quotes, doubled quotes inside a quoted field, CRLF or LF.
    nonisolated static func csvRecords(_ text: String) -> [[String]] {
        var records: [[String]] = []
        var record: [String] = []
        var field = ""
        var quoted = false
        var iterator = text.makeIterator()
        var pending: Character? = nil
        func next() -> Character? {
            if let p = pending { pending = nil; return p }
            return iterator.next()
        }
        while let ch = next() {
            if quoted {
                if ch == "\"" {
                    if let peek = iterator.next() {
                        if peek == "\"" { field.append("\"") } else { quoted = false; pending = peek }
                    } else { quoted = false }
                } else { field.append(ch) }
            } else {
                switch ch {
                case "\"": quoted = true
                case ",": record.append(field); field = ""
                case "\r": break
                case "\n":
                    record.append(field); field = ""
                    if record.contains(where: { !$0.isEmpty }) { records.append(record) }
                    record = []
                default: field.append(ch)
                }
            }
        }
        record.append(field)
        if record.contains(where: { !$0.isEmpty }) { records.append(record) }
        return records
    }

    // MARK: - The fill

    func run(_ runtime: DecideRuntime) {
        guard !working, !rows.isEmpty else { return }
        let columns: [Column]
        do {
            columns = try Self.parseColumns(columnsText)
        } catch {
            status = "Columns: \(error.localizedDescription)"
            return
        }
        guard !columns.isEmpty else {
            status = "Write at least one column, one per line."
            return
        }
        working = true
        filled = 0
        sortKey = nil
        for index in rows.indices {
            rows[index].answers = [:]
            rows[index].milliseconds = 0
        }
        Task {
            defer { working = false }
            do {
                let decider = try await runtime.ready()
                for index in rows.indices {
                    status = "Row \(index + 1) of \(rows.count)…"
                    let prefilled = try await decider.prefill(rows[index].text)
                    var total = prefilled.timing.milliseconds
                    for column in columns {
                        let answer = try await prefilled.decide(column.question)
                        rows[index].answers[column.key] = answer
                        total += answer.timing.milliseconds
                    }
                    rows[index].milliseconds = total
                    filled = index + 1
                }
                status = "\(rows.count) rows × \(columns.count) columns: \(decisionsMade) decisions in \(ms(totalMilliseconds))"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - CSV out

    /// The rows as read, plus one column per question, in the current sort order.
    func csv() -> String {
        let columns = columns
        guard let first = rows.first else { return "" }
        func cell(_ s: String) -> String {
            s.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" }) ? "\"\(s.replacingOccurrences(of: "\"", with: "\"\""))\"" : s
        }
        var lines = [(first.fields.map(\.0) + columns.map(\.title) + columns.map { "\($0.title) confidence" }).map(cell).joined(separator: ",")]
        for row in sortedRows {
            let values = row.fields.map(\.1) + columns.map { label(row, $0) }
                + columns.map { row.answers[$0.key].map { $0.confidence.formatted(.number.precision(.fractionLength(2))) } ?? "" }
            lines.append(values.map(cell).joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: - Sample

    static let sampleColumns = """
        # one column per line — Title = choice: q | a | b   /   Title = score: q | low | high   /   Title = noul: q
        Topic = choice: What is the message about? | billing or an invoice | delivery or shipping | a defective or wrong product | how to use the product | the account or the subscription | none of these
        Wants = choice: What does the customer want from us? | a refund | a replacement or an exchange | information or an answer | a correction to a document | an apology | nothing, it is just feedback
        Mood = score: How does the customer feel? | upset | neutral | happy
        """

    /// Twenty invented support tickets.
    static let sampleRows = """
        id,customer,message
        1041,Mika T.,"My package says delivered but nothing arrived. I need it for a wedding on Saturday."
        1042,Jonas B.,How do I change the language of the app?
        1043,Priya R.,"Charged twice for order 5521, please refund the duplicate."
        1044,Sam O.,The blender stopped working after two days. I want my money back.
        1045,Elena V.,"Great service, just wanted to say thanks!"
        1046,Tom H.,"I can't log in, the password reset email never arrives, and I have a presentation in an hour."
        1047,Aiko M.,Do you ship to Canada?
        1048,Lars K.,"The invoice has the wrong company name, accounting needs it corrected by end of month."
        1049,Nadia F.,"Received a blue one, ordered red. Can you swap it?"
        1050,Omar S.,Your delivery driver was rude and left the box in the rain.
        1051,Chen W.,Cancel my subscription immediately and confirm by email.
        1052,Julia P.,"The manual says 220V but the label says 110V, which is right?"
        1053,Ravi N.,Where can I download the invoice for last month?
        1054,Sofia L.,The kettle arrived cracked. Photos attached. Replacement please.
        1055,Ben A.,"Love the new update, the dark mode is perfect."
        1056,Hana K.,"My discount code didn't apply at checkout, I paid full price."
        1057,Diego M.,Tracking hasn't updated in 9 days. Is the parcel lost?
        1058,Yuki S.,Can I change the delivery address for order 5610 before it ships?
        1059,Grace O.,Two-factor codes never arrive by SMS since I changed my number.
        1060,Felix R.,The espresso machine leaks from the steam wand. Still under warranty.
        """
}
