// OrderedJSON.swift — a JSON value that keeps object key order, parsed from bytes and written
// back the way Python's `json.dumps(…, ensure_ascii=False)` writes it. A structured state in
// a `/v1/systemone` request is serialized with it, so the model reads the same bytes the
// publisher's Python client would have sent; Foundation's JSONSerialization reorders keys and
// escapes "/", which changes the tokens.

import Foundation

/// A JSON value with ordered object members. Numbers keep their source lexeme.
public indirect enum JSONValue: Equatable, Sendable {
    public struct Member: Equatable, Sendable {
        public let key: String
        public let value: JSONValue

        public init(_ key: String, _ value: JSONValue) {
            self.key = key
            self.value = value
        }
    }

    case null
    case bool(Bool)
    /// The number as written (`1`, `2.5`, `1e-05`); rewritten verbatim.
    case number(String)
    case string(String)
    case array([JSONValue])
    case object([Member])

    /// A double the way Python's `repr` and `json.dumps` write it (`0.5`, `1.0`, `1e-05`).
    public static func double(_ value: Double) -> JSONValue { .number(String(value)) }
    public static func int(_ value: Int) -> JSONValue { .number(String(value)) }

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var members: [Member]? {
        if case .object(let m) = self { return m }
        return nil
    }

    public var elements: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    public var doubleValue: Double? {
        if case .number(let n) = self { return Double(n) }
        return nil
    }

    /// The member `key` of an object, or nil.
    public subscript(key: String) -> JSONValue? {
        members?.first { $0.key == key }?.value
    }

    // MARK: - Writing

    /// The text `json.dumps(value, ensure_ascii=False)` produces: `", "` between items,
    /// `": "` after a key, non-ASCII verbatim, control characters escaped.
    public func dumps() -> String {
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .number(let n): return n
        case .string(let s): return DecisionPrompt.jsonString(s)
        case .array(let a): return "[" + a.map { $0.dumps() }.joined(separator: ", ") + "]"
        case .object(let m):
            return "{" + m.map { DecisionPrompt.jsonString($0.key) + ": " + $0.value.dumps() }.joined(separator: ", ") + "}"
        }
    }

    // MARK: - Parsing

    public struct ParseError: Error, LocalizedError, Equatable {
        public let message: String
        public let offset: Int
        public var errorDescription: String? { "\(message) at byte \(offset)" }
    }

    /// Parses one JSON document (UTF-8).
    public static func parse(_ data: Data) throws -> JSONValue {
        var parser = Parser(bytes: Array(data))
        parser.skipWhitespace()
        let value = try parser.value()
        parser.skipWhitespace()
        guard parser.atEnd else { throw parser.error("trailing characters") }
        return value
    }

    public static func parse(_ text: String) throws -> JSONValue {
        try parse(Data(text.utf8))
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        var atEnd: Bool { index >= bytes.count }

        func error(_ message: String) -> ParseError { ParseError(message: message, offset: index) }

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        }

        mutating func value() throws -> JSONValue {
            guard index < bytes.count else { throw error("unexpected end") }
            switch bytes[index] {
            case UInt8(ascii: "{"): return try object()
            case UInt8(ascii: "["): return try array()
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try literal("true"); return .bool(true)
            case UInt8(ascii: "f"): try literal("false"); return .bool(false)
            case UInt8(ascii: "n"): try literal("null"); return .null
            default: return try number()
            }
        }

        mutating func literal(_ word: String) throws {
            let expected = Array(word.utf8)
            guard index + expected.count <= bytes.count, Array(bytes[index..<index + expected.count]) == expected else {
                throw error("expected \(word)")
            }
            index += expected.count
        }

        mutating func number() throws -> JSONValue {
            let start = index
            while index < bytes.count {
                let b = bytes[index]
                if (b >= 0x30 && b <= 0x39) || [UInt8(ascii: "-"), UInt8(ascii: "+"), UInt8(ascii: "."), UInt8(ascii: "e"), UInt8(ascii: "E")].contains(b) {
                    index += 1
                } else { break }
            }
            guard index > start, let text = String(bytes: bytes[start..<index], encoding: .utf8), Double(text) != nil else {
                throw error("expected a value")
            }
            return .number(text)
        }

        mutating func object() throws -> JSONValue {
            index += 1
            var members: [Member] = []
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "}") { index += 1; return .object(members) }
            while true {
                skipWhitespace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else { throw error("expected a key") }
                let key = try string()
                skipWhitespace()
                guard index < bytes.count, bytes[index] == UInt8(ascii: ":") else { throw error("expected ':'") }
                index += 1
                skipWhitespace()
                members.append(Member(key, try value()))
                skipWhitespace()
                guard index < bytes.count else { throw error("unterminated object") }
                if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                if bytes[index] == UInt8(ascii: "}") { index += 1; return .object(members) }
                throw error("expected ',' or '}'")
            }
        }

        mutating func array() throws -> JSONValue {
            index += 1
            var elements: [JSONValue] = []
            skipWhitespace()
            if index < bytes.count, bytes[index] == UInt8(ascii: "]") { index += 1; return .array(elements) }
            while true {
                skipWhitespace()
                elements.append(try value())
                skipWhitespace()
                guard index < bytes.count else { throw error("unterminated array") }
                if bytes[index] == UInt8(ascii: ",") { index += 1; continue }
                if bytes[index] == UInt8(ascii: "]") { index += 1; return .array(elements) }
                throw error("expected ',' or ']'")
            }
        }

        mutating func string() throws -> String {
            index += 1  // opening quote
            var out: [UInt8] = []
            while index < bytes.count {
                let b = bytes[index]
                if b == UInt8(ascii: "\"") {
                    index += 1
                    guard let s = String(bytes: out, encoding: .utf8) else { throw error("invalid UTF-8 in string") }
                    return s
                }
                if b == UInt8(ascii: "\\") {
                    index += 1
                    guard index < bytes.count else { throw error("unterminated escape") }
                    let e = bytes[index]
                    index += 1
                    switch e {
                    case UInt8(ascii: "\""): out.append(0x22)
                    case UInt8(ascii: "\\"): out.append(0x5C)
                    case UInt8(ascii: "/"): out.append(0x2F)
                    case UInt8(ascii: "b"): out.append(0x08)
                    case UInt8(ascii: "f"): out.append(0x0C)
                    case UInt8(ascii: "n"): out.append(0x0A)
                    case UInt8(ascii: "r"): out.append(0x0D)
                    case UInt8(ascii: "t"): out.append(0x09)
                    case UInt8(ascii: "u"):
                        var scalar = try hex4()
                        if (0xD800...0xDBFF).contains(scalar) {
                            // a surrogate pair: the low half follows as another \uXXXX
                            guard index + 1 < bytes.count, bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u") else {
                                throw error("lone high surrogate")
                            }
                            index += 2
                            let low = try hex4()
                            guard (0xDC00...0xDFFF).contains(low) else { throw error("invalid low surrogate") }
                            scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                        }
                        guard let u = Unicode.Scalar(scalar) else { throw error("invalid code point") }
                        out.append(contentsOf: Array(String(Character(u)).utf8))
                    default: throw error("invalid escape")
                    }
                    continue
                }
                if b < 0x20 { throw error("control character in string") }
                out.append(b)
                index += 1
            }
            throw error("unterminated string")
        }

        mutating func hex4() throws -> UInt32 {
            guard index + 4 <= bytes.count, let text = String(bytes: bytes[index..<index + 4], encoding: .utf8),
                let v = UInt32(text, radix: 16) else { throw error("expected 4 hex digits") }
            index += 4
            return v
        }
    }
}
