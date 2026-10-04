// From the model zoo's apps/Kev/Sources/Kev/JSONValue.swift (9e06b5a, sha256 48b400ea0c71), identifiers prefixed Kev for the kit.
// Copied from apps/ClefFlash/Sources/ClefFlash/KevJSON.swift (zoo main 5ef2247); errors are KevError, the number
// texts come from KevPythonFormat.
//
// KevJSON — a request's JSON exactly as written: object members in their source order, numbers as their literal text.
//
// The author renders a value with Python's `str()` (`kev.api.render`), and what Python prints for a number depends on
// what `json.loads` made of its literal: `1250` is an int and prints `1250`, `1250.0` is a float and prints `1250.0`,
// `2 ** 70` keeps every digit — different token rows. `JSONSerialization` loses that distinction, so the request is
// parsed here: the literal is kept, and `KevPythonFormat` derives Python's text from it. The parser follows `json.loads`:
// strict strings (no raw control characters), `NaN` / `Infinity` / `-Infinity` accepted as floats, a duplicate key
// keeps its first position and its last value.

import Foundation

struct KevJSONMember: Sendable, Equatable {
    var key: String
    var value: KevJSON

    init(_ key: String, _ value: KevJSON) {
        self.key = key
        self.value = value
    }
}

indirect enum KevJSON: Sendable, Equatable {
    case null
    case bool(Bool)
    /// The number's literal as written in the source JSON (or Python's repr of a float computed here).
    case number(String)
    case string(String)
    case array([KevJSON])
    /// Members in source order.
    case object([KevJSONMember])

    static func int(_ v: Int) -> KevJSON { .number(String(v)) }
    /// A computed float, written the way Python's `json.dumps` writes it (`repr`).
    static func double(_ v: Double) -> KevJSON { .number(KevPythonFormat.floatRepr(v)) }

    subscript(key: String) -> KevJSON? {
        guard case .object(let m) = self else { return nil }
        return m.first(where: { $0.key == key })?.value
    }

    var string: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    var array: [KevJSON]? {
        if case .array(let a) = self { return a }
        return nil
    }

    var members: [KevJSONMember]? {
        if case .object(let m) = self { return m }
        return nil
    }

    var isNull: Bool { self == .null }

    /// The literal's value as a Double (an int literal converted, a float literal parsed).
    var double: Double? {
        guard case .number(let s) = self else { return nil }
        return KevPythonFormat.numberValue(s)
    }

    var intValue: Int? {
        guard case .number(let s) = self, !KevPythonFormat.isFloatLiteral(s) else { return nil }
        return Int(s)
    }

    var boolValue: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }
}

// MARK: - Parsing

enum KevJSONParser {
    static func parse(_ data: Data) throws -> KevJSON {
        var p = Parser(bytes: [UInt8](data))
        p.skipWhitespace()
        let v = try p.value(depth: 0)
        p.skipWhitespace()
        guard p.i == p.bytes.count else { throw p.error("extra data") }
        return v
    }

    static func parse(_ text: String) throws -> KevJSON { try parse(Data(text.utf8)) }

    struct Parser {
        let bytes: [UInt8]
        var i = 0

        init(bytes: [UInt8]) {
            // a UTF-8 byte order mark is not JSON; Python's json.loads(str) rejects it too
            self.bytes = bytes
        }

        func error(_ what: String) -> KevError { .json("\(what) at byte \(i)") }

        mutating func skipWhitespace() {
            while i < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[i]) { i += 1 }
        }

        mutating func value(depth: Int) throws -> KevJSON {
            guard depth < 512 else { throw error("nesting too deep") }
            guard i < bytes.count else { throw error("unexpected end") }
            switch bytes[i] {
            case UInt8(ascii: "{"): return try object(depth: depth)
            case UInt8(ascii: "["): return try array(depth: depth)
            case UInt8(ascii: "\""): return .string(try string())
            case UInt8(ascii: "t"): try literal("true"); return .bool(true)
            case UInt8(ascii: "f"): try literal("false"); return .bool(false)
            case UInt8(ascii: "n"): try literal("null"); return .null
            case UInt8(ascii: "N"): try literal("NaN"); return .number("NaN")
            case UInt8(ascii: "I"): try literal("Infinity"); return .number("Infinity")
            case UInt8(ascii: "-"):
                if i + 1 < bytes.count, bytes[i + 1] == UInt8(ascii: "I") {
                    i += 1
                    try literal("Infinity")
                    return .number("-Infinity")
                }
                return .number(try number())
            case UInt8(ascii: "0")...UInt8(ascii: "9"): return .number(try number())
            default: throw error("unexpected character")
            }
        }

        mutating func literal(_ word: String) throws {
            let w = Array(word.utf8)
            guard i + w.count <= bytes.count, Array(bytes[i..<(i + w.count)]) == w else { throw error("bad literal") }
            i += w.count
        }

        mutating func number() throws -> String {
            let start = i
            if bytes[i] == UInt8(ascii: "-") { i += 1 }
            func digits() -> Int {
                let s = i
                while i < bytes.count, (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(bytes[i]) { i += 1 }
                return i - s
            }
            guard i < bytes.count else { throw error("bad number") }
            if bytes[i] == UInt8(ascii: "0") {
                i += 1
            } else if digits() == 0 {
                throw error("bad number")
            }
            if i < bytes.count, bytes[i] == UInt8(ascii: ".") {
                let save = i
                i += 1
                if digits() == 0 { i = save }     // Python's scanner stops before a "." without digits
            }
            if i < bytes.count, bytes[i] == UInt8(ascii: "e") || bytes[i] == UInt8(ascii: "E") {
                let save = i
                i += 1
                if i < bytes.count, bytes[i] == UInt8(ascii: "+") || bytes[i] == UInt8(ascii: "-") { i += 1 }
                if digits() == 0 { i = save }
            }
            return String(decoding: bytes[start..<i], as: UTF8.self)
        }

        mutating func hex4() throws -> UInt32 {
            guard i + 4 <= bytes.count else { throw error("bad \\u escape") }
            var v: UInt32 = 0
            for _ in 0..<4 {
                let c = bytes[i]
                let d: UInt32
                switch c {
                case UInt8(ascii: "0")...UInt8(ascii: "9"): d = UInt32(c - UInt8(ascii: "0"))
                case UInt8(ascii: "a")...UInt8(ascii: "f"): d = UInt32(c - UInt8(ascii: "a") + 10)
                case UInt8(ascii: "A")...UInt8(ascii: "F"): d = UInt32(c - UInt8(ascii: "A") + 10)
                default: throw error("bad \\u escape")
                }
                v = v * 16 + d
                i += 1
            }
            return v
        }

        mutating func string() throws -> String {
            i += 1      // opening quote
            var scalars = String.UnicodeScalarView()
            var run = i
            func flush(_ end: Int) throws {
                if end > run {
                    guard let s = String(bytes: bytes[run..<end], encoding: .utf8) else { throw error("invalid UTF-8") }
                    scalars.append(contentsOf: s.unicodeScalars)
                }
            }
            while true {
                guard i < bytes.count else { throw error("unterminated string") }
                let c = bytes[i]
                if c == UInt8(ascii: "\"") {
                    try flush(i)
                    i += 1
                    return String(scalars)
                }
                if c < 0x20 { throw error("control character in string") }
                if c != UInt8(ascii: "\\") {
                    i += 1
                    continue
                }
                try flush(i)
                i += 1
                guard i < bytes.count else { throw error("unterminated escape") }
                let e = bytes[i]
                i += 1
                switch e {
                case UInt8(ascii: "\""): scalars.append("\"")
                case UInt8(ascii: "\\"): scalars.append("\\")
                case UInt8(ascii: "/"): scalars.append("/")
                case UInt8(ascii: "b"): scalars.append("\u{08}")
                case UInt8(ascii: "f"): scalars.append("\u{0C}")
                case UInt8(ascii: "n"): scalars.append("\n")
                case UInt8(ascii: "r"): scalars.append("\r")
                case UInt8(ascii: "t"): scalars.append("\t")
                case UInt8(ascii: "u"):
                    var u = try hex4()
                    if (0xD800..<0xDC00).contains(u), i + 6 <= bytes.count, bytes[i] == UInt8(ascii: "\\"),
                       bytes[i + 1] == UInt8(ascii: "u")
                    {
                        let save = i
                        i += 2
                        let lo = try hex4()
                        if (0xDC00..<0xE000).contains(lo) {
                            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00)
                        } else {
                            i = save
                        }
                    }
                    guard let s = Unicode.Scalar(u) else {
                        // Python keeps a lone surrogate in the str; a Swift String cannot hold one
                        throw error("lone surrogate \\u\(String(u, radix: 16)) is not supported")
                    }
                    scalars.append(s)
                default:
                    throw error("bad escape")
                }
                run = i
            }
        }

        mutating func array(depth: Int) throws -> KevJSON {
            i += 1
            var out: [KevJSON] = []
            skipWhitespace()
            if i < bytes.count, bytes[i] == UInt8(ascii: "]") {
                i += 1
                return .array(out)
            }
            while true {
                skipWhitespace()
                out.append(try value(depth: depth + 1))
                skipWhitespace()
                guard i < bytes.count else { throw error("unterminated array") }
                if bytes[i] == UInt8(ascii: ",") {
                    i += 1
                    continue
                }
                if bytes[i] == UInt8(ascii: "]") {
                    i += 1
                    return .array(out)
                }
                throw error("expected , or ]")
            }
        }

        mutating func object(depth: Int) throws -> KevJSON {
            i += 1
            var out: [KevJSONMember] = []
            var index: [String: Int] = [:]
            skipWhitespace()
            if i < bytes.count, bytes[i] == UInt8(ascii: "}") {
                i += 1
                return .object(out)
            }
            while true {
                skipWhitespace()
                guard i < bytes.count, bytes[i] == UInt8(ascii: "\"") else { throw error("expected a key") }
                let k = try string()
                skipWhitespace()
                guard i < bytes.count, bytes[i] == UInt8(ascii: ":") else { throw error("expected :") }
                i += 1
                skipWhitespace()
                let v = try value(depth: depth + 1)
                // Swift compares String keys by canonical equivalence; Python by code points
                let key = String(k.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: ","))
                if let at = index[key] {
                    out[at].value = v
                } else {
                    index[key] = out.count
                    out.append(KevJSONMember(k, v))
                }
                skipWhitespace()
                guard i < bytes.count else { throw error("unterminated object") }
                if bytes[i] == UInt8(ascii: ",") {
                    i += 1
                    continue
                }
                if bytes[i] == UInt8(ascii: "}") {
                    i += 1
                    return .object(out)
                }
                throw error("expected , or }")
            }
        }
    }
}

// MARK: - Writing (output files)

enum KevJSONWriter {
    /// Pretty JSON with one-space indent and members in their order (`json.dumps(indent=1, ensure_ascii=False)`'s
    /// shape); numbers as their literal, strings escaped as Python does.
    static func pretty(_ v: KevJSON) -> String {
        var out = ""
        write(v, indent: 0, into: &out)
        return out
    }

    private static func write(_ v: KevJSON, indent: Int, into out: inout String) {
        switch v {
        case .array(let a) where !a.isEmpty:
            out += "[\n"
            for (k, x) in a.enumerated() {
                out += String(repeating: " ", count: indent + 1)
                write(x, indent: indent + 1, into: &out)
                out += k + 1 < a.count ? ",\n" : "\n"
            }
            out += String(repeating: " ", count: indent) + "]"
        case .object(let m) where !m.isEmpty:
            out += "{\n"
            for (k, x) in m.enumerated() {
                out += String(repeating: " ", count: indent + 1) + KevPythonFormat.quote(x.key, asciiOnly: false) + ": "
                write(x.value, indent: indent + 1, into: &out)
                out += k + 1 < m.count ? ",\n" : "\n"
            }
            out += String(repeating: " ", count: indent) + "}"
        default:
            out += KevPythonFormat.dumps(v, asciiOnly: false)
        }
    }

    /// One line, no extra whitespace (for large arrays in output files).
    static func compact(_ v: KevJSON) -> String { KevPythonFormat.dumps(v, asciiOnly: false, separators: (",", ":")) }

    static func write(_ v: KevJSON, to url: URL, pretty: Bool = true) throws {
        let tmp = url.appendingPathExtension("tmp")
        try Data(((pretty ? Self.pretty(v) : compact(v)) + "\n").utf8).write(to: tmp)
        _ = try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: tmp, to: url)
    }
}

// MARK: - Building output values

extension KevJSON {
    static func doubles(_ xs: [Double]) -> KevJSON { .array(xs.map { .double($0) }) }
    static func floats(_ xs: [Float]) -> KevJSON { .array(xs.map { .double(Double($0)) }) }
    static func ints(_ xs: [Int]) -> KevJSON { .array(xs.map { .int($0) }) }
    static func strings(_ xs: [String]) -> KevJSON { .array(xs.map { .string($0) }) }
    static func obj(_ pairs: [(String, KevJSON)]) -> KevJSON { .object(pairs.map { KevJSONMember($0.0, $0.1) }) }
    static func optional(_ s: String?) -> KevJSON { s.map { .string($0) } ?? .null }
}
