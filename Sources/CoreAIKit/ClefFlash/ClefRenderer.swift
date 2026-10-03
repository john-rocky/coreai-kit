// From the model zoo's apps/ClefFlash/Sources/ClefFlash/Renderer.swift (5ef2247, sha256 963e0ff25b21, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// Renderer — Python's JSON text for a request's values, which is what the author's `render` puts in the prompt:
//
//   render(value) = value                                       a string, as is
//                 = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)   anything else
//
// Rules copied from CPython 3.11's json encoder (`c_make_encoder`, `encoder_encode_float`, `escape_unicode`) and
// float repr (`PyOS_double_to_string(x, 'r', 0, Py_DTSF_ADD_DOT_0)`):
//   * object keys sorted by code point (Python str order), at every level; no whitespace
//   * strings: `"` and `\` escaped, \b \f \n \r \t as such, other controls below U+0020 as \u00xx (lowercase hex),
//     everything else as is (`/` and non-ASCII unescaped)
//   * an int literal prints as Python's int (`-0` -> `0`); a float literal (one with `.`, `e` or `E`) prints as
//     repr(float(literal)): the shortest round-trip digits, positional when -4 < exponent <= 16 with `.0` added to
//     an integral value (`1250.0`), else `d.ddde±XX` with at least two exponent digits (`1e-05`, `1e+16`);
//     NaN / Infinity / -Infinity as those words
//   * true / false / null
// `pyRound` is Python's round(x, n) (correctly rounded decimal, ties to even, back to the nearest double).

import Foundation

enum ClefPythonJSON {
    /// `render` of the author's encode_record: a string as is, anything else canonical JSON.
    static func render(_ v: ClefJSON) -> String {
        if case .string(let s) = v { return s }
        return dumps(v, sortKeys: true)
    }

    /// `json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=sortKeys)`.
    static func dumps(_ v: ClefJSON, sortKeys: Bool = true) -> String {
        var out = ""
        write(v, sortKeys: sortKeys, into: &out)
        return out
    }

    private static func write(_ v: ClefJSON, sortKeys: Bool, into out: inout String) {
        switch v {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .number(let s): out += numberText(s)
        case .string(let s): out += quote(s)
        case .array(let a):
            out += "["
            for (k, x) in a.enumerated() {
                if k > 0 { out += "," }
                write(x, sortKeys: sortKeys, into: &out)
            }
            out += "]"
        case .object(let m):
            let members = sortKeys ? m.sorted { codePointLess($0.key, $1.key) } : m
            out += "{"
            for (k, x) in members.enumerated() {
                if k > 0 { out += "," }
                out += quote(x.key)
                out += ":"
                write(x.value, sortKeys: sortKeys, into: &out)
            }
            out += "}"
        }
    }

    /// Python's str comparison: lexicographic over code points (not Swift's canonical String order).
    static func codePointLess(_ a: String, _ b: String) -> Bool {
        a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars) { $0.value < $1.value }
    }

    private static let hex: [Character] = Array("0123456789abcdef")

    /// `escape_unicode` (ensure_ascii=False) with the surrounding quotes.
    static func quote(_ s: String) -> String {
        var out = String.UnicodeScalarView()
        out.append("\"")
        for c in s.unicodeScalars {
            switch c {
            case "\"": out.append(contentsOf: "\\\"".unicodeScalars)
            case "\\": out.append(contentsOf: "\\\\".unicodeScalars)
            case "\u{08}": out.append(contentsOf: "\\b".unicodeScalars)
            case "\u{0C}": out.append(contentsOf: "\\f".unicodeScalars)
            case "\n": out.append(contentsOf: "\\n".unicodeScalars)
            case "\r": out.append(contentsOf: "\\r".unicodeScalars)
            case "\t": out.append(contentsOf: "\\t".unicodeScalars)
            default:
                if c.value < 0x20 {
                    out.append(contentsOf: "\\u00".unicodeScalars)
                    out.append(contentsOf: String(hex[Int(c.value >> 4)]).unicodeScalars)
                    out.append(contentsOf: String(hex[Int(c.value & 0xF)]).unicodeScalars)
                } else {
                    out.append(c)
                }
            }
        }
        out.append("\"")
        return String(out)
    }

    /// A literal json.loads turns into a float (it has a fraction or an exponent, or is NaN / ±Infinity).
    static func isFloatLiteral(_ s: String) -> Bool {
        s.contains(where: { $0 == "." || $0 == "e" || $0 == "E" }) || s == "NaN" || s.hasSuffix("Infinity")
    }

    /// The value json.loads makes of a number literal, as a Double (ints converted).
    static func numberValue(_ s: String) -> Double {
        switch s {
        case "NaN": return .nan
        case "Infinity": return .infinity
        case "-Infinity": return -.infinity
        default: return Double(s) ?? .nan
        }
    }

    /// What `json.dumps` prints for the value `json.loads` made of the literal `s`.
    static func numberText(_ s: String) -> String {
        if isFloatLiteral(s) { return floatRepr(numberValue(s)) }
        // an int: Python prints its decimal digits; "-0" is 0
        let digits = s.hasPrefix("-") ? String(s.dropFirst()) : s
        if digits.allSatisfy({ $0 == "0" }) { return "0" }
        return s
    }

    /// Python's `repr(float)` (and json.dumps's text for a float; non-finite values as json.dumps writes them).
    static func floatRepr(_ x: Double) -> String {
        if x.isNaN { return "NaN" }
        if x.isInfinite { return x < 0 ? "-Infinity" : "Infinity" }
        if x == 0 { return x.sign == .minus ? "-0.0" : "0.0" }
        // Swift's description is the shortest digit string that round-trips (the closest one when several do), the
        // same digits as Python's dtoa mode 0; only the layout differs, so take the digits and the exponent.
        var s = x.magnitude.description
        var exp10 = 0
        if let e = s.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            exp10 = Int(s[s.index(after: e)...].replacingOccurrences(of: "+", with: ""))!
            s = String(s[..<e])
        }
        let parts = s.split(separator: ".", omittingEmptySubsequences: false)
        let intPart = String(parts[0])
        let frac = parts.count > 1 ? String(parts[1]) : ""
        var digits = Array(intPart + frac)
        var decpt = intPart.count + exp10            // value = 0.d1d2... x 10^decpt
        while let f = digits.first, f == "0" {
            digits.removeFirst()
            decpt -= 1
        }
        while let l = digits.last, l == "0" { digits.removeLast() }
        let sign = x < 0 ? "-" : ""
        if decpt <= -4 || decpt > 16 {
            let exp = decpt - 1
            var m = String(digits[0])
            if digits.count > 1 { m += "." + String(digits[1...]) }
            let e = abs(exp) < 10 ? "0\(abs(exp))" : "\(abs(exp))"
            return sign + m + "e" + (exp < 0 ? "-" : "+") + e
        }
        if decpt <= 0 {
            return sign + "0." + String(repeating: "0", count: -decpt) + String(digits)
        }
        if decpt < digits.count {
            return sign + String(digits[..<decpt]) + "." + String(digits[decpt...])
        }
        return sign + String(digits) + String(repeating: "0", count: decpt - digits.count) + ".0"
    }

    /// Python's `round(x, ndigits)` for a float: the correctly rounded decimal with `ndigits` places (ties to even on
    /// the exact binary value, as dtoa mode 3 and the C library's `%.*f` both do), read back as the nearest double.
    static func pyRound(_ x: Double, _ ndigits: Int = 4) -> Double {
        guard x.isFinite else { return x }
        return Double(String(format: "%.\(ndigits)f", x))!
    }
}
