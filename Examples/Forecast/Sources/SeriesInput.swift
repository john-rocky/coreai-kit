import Foundation

// SeriesInput.swift — your own series from pasted text or a CSV file. Shared by the app and the CLI.
//
// Accepted:
//   one number per line                    12.5\n13.1\n12.9
//   numbers on a line, any separator       12.5, 13.1, 12.9      (commas, semicolons, tabs or spaces)
//   a table with a header or a date column date,sales\n2026-01-01,12.5\n…   (the last column is the series)
// Missing values: an empty cell, NaN, NA, N/A, null, none, "-", "?" or inf — the models skip them.
// The decimal mark is "." (a "3,5" splits into two numbers on the comma).

/// A series read from text: the values in order, NaN where one is missing.
public struct SeriesInput: Sendable, Equatable {
    public var values: [Float]
    /// The column the values came from when the text is a table (`sales` for `date,sales`), else nil.
    public var column: String?

    public var missing: Int { values.reduce(0) { $0 + ($1.isNaN ? 1 : 0) } }
}

public enum SeriesInputError: Error, LocalizedError, Equatable {
    case empty
    case notANumber(line: Int, token: String)
    case outOfRange(line: Int, token: String)

    public var errorDescription: String? {
        switch self {
        case .empty:
            return "No values: paste numbers (one per line, or separated by commas) or choose a CSV file."
        case .notANumber(let line, let token):
            return "Line \(line): '\(token)' is not a number."
        case .outOfRange(let line, let token):
            return "Line \(line): '\(token)' is outside the float32 range."
        }
    }
}

public enum SeriesParser {
    static let missingMarkers: Set<String> = ["", "nan", "na", "n/a", "null", "none", "-", "?", "inf", "+inf", "-inf",
                                              "infinity", "+infinity", "-infinity"]

    /// One cell: a number, NaN for a missing marker, nil for anything else.
    static func cell(_ raw: Substring) -> Float?? {
        var t = raw.trimmingCharacters(in: .whitespaces)
        if t.count >= 2, t.first == "\"", t.last == "\"" { t = String(t.dropFirst().dropLast()) }
        if missingMarkers.contains(t.lowercased()) { return .some(.nan) }
        guard let v = Float(t) else { return .none }
        return .some(v)
    }

    public static func parse(_ text: String) throws -> SeriesInput {
        // (line number, fields), empty lines left out
        let separator: Character? = text.contains(",") ? "," : text.contains("\t") ? "\t" : text.contains(";") ? ";" : nil
        var rows: [(line: Int, fields: [Substring])] = []
        for (i, line) in text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).enumerated() {
            guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { continue }
            let fields = separator.map { s in line.split(separator: s, omittingEmptySubsequences: false) }
                ?? line.split(whereSeparator: { $0 == " " })
            rows.append((i + 1, fields))
        }
        guard !rows.isEmpty else { throw SeriesInputError.empty }

        // A table: two or more lines with the same number of fields, and one field a line, a header row, or a column
        // that is not numbers (dates). Its last column is the series. Anything else is a list read in order.
        let width = rows[0].fields.count
        let sameWidth = rows.count >= 2 && rows.allSatisfy { $0.fields.count == width }
        let header = sameWidth && rows[0].fields.contains { cell($0) == .none }
        let textColumn = sameWidth && width >= 2
            && rows.dropFirst().contains { $0.fields.dropLast().contains { cell($0) == .none } }
        if sameWidth && (width == 1 || header || textColumn) {
            var values: [Float] = []
            for row in rows.dropFirst(header ? 1 : 0) {
                values.append(try number(row.fields[width - 1], line: row.line))
            }
            guard !values.isEmpty else { throw SeriesInputError.empty }
            let name = header ? rows[0].fields[width - 1].trimmingCharacters(in: .whitespaces) : nil
            return SeriesInput(values: values, column: name)
        }
        var values: [Float] = []
        for row in rows {
            for f in row.fields { values.append(try number(f, line: row.line)) }
        }
        return SeriesInput(values: values, column: nil)
    }

    static func number(_ field: Substring, line: Int) throws -> Float {
        let token = field.trimmingCharacters(in: .whitespaces)
        guard let parsed = cell(field) else { throw SeriesInputError.notANumber(line: line, token: token) }
        let v = parsed ?? .nan
        if v.isInfinite && !missingMarkers.contains(token.lowercased()) {
            throw SeriesInputError.outOfRange(line: line, token: token)
        }
        return v.isInfinite ? .nan : v
    }
}
