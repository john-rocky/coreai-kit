// KokoroG2P.swift — dictionary-first English G2P for Kokoro-82M.
//
// Kokoro consumes misaki phonemes (hexgrad/misaki, Apache-2.0). The full misaki stack is a
// Python pipeline (spaCy POS pass + a seq2seq fallback for out-of-dictionary words); the
// established Swift port (MisakiSwift) carries an MLX dependency for that fallback — too heavy
// a transitive dependency for the kit. This is the dictionary core alone: the misaki US
// gold/silver lexicons (~180k entries, downloaded with the model's host glue), integer→words
// expansion, and letter-name spelling for out-of-dictionary words. Homographs take the
// lexicon's DEFAULT reading (no POS pass). Punctuation rides through — the Kokoro vocab
// maps it and the model uses it for prosody.

import Foundation

struct KokoroG2P: Sendable {
    private let lexicon: [String: String]

    /// Loads the misaki lexicons. Silver loads first so gold wins on shared keys.
    init(goldURL: URL, silverURL: URL) throws {
        var lexicon: [String: String] = [:]
        for url in [silverURL, goldURL] {
            let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            guard let entries = raw as? [String: Any] else { continue }
            for (word, value) in entries {
                if let phonemes = value as? String {
                    lexicon[word] = phonemes
                } else if let variants = value as? [String: Any] {
                    // Homograph entry: {"DEFAULT": "...", "NOUN": ...}. Take DEFAULT, else the
                    // first non-null reading.
                    if let phonemes = variants["DEFAULT"] as? String {
                        lexicon[word] = phonemes
                    } else if let phonemes = variants.values.compactMap({ $0 as? String }).first {
                        lexicon[word] = phonemes
                    }
                }
            }
        }
        self.lexicon = lexicon
    }

    /// Text → misaki phoneme string. Words become phoneme runs separated by spaces;
    /// punctuation stays in place (unknown characters are dropped later at vocab mapping).
    func phonemize(_ text: String) -> String {
        var out = ""
        var word = ""

        func flushWord() {
            guard !word.isEmpty else { return }
            if !out.isEmpty, out.last != " " { out += " " }
            out += pronounce(word)
            word = ""
        }

        for ch in text {
            if ch.isLetter || ch.isNumber || ch == "'" || ch == "’" {
                word.append(ch == "’" ? "'" : ch)
            } else if ch.isWhitespace {
                flushWord()
                if !out.isEmpty, out.last != " " { out += " " }
            } else {
                // Word-internal hyphen splits the word; other punctuation attaches to the
                // phoneme stream directly (".", ",", "!", "?", ";", ":" are vocab tokens).
                flushWord()
                if ch != "-" { out.append(ch) }
                else if !out.isEmpty, out.last != " " { out += " " }
            }
        }
        flushWord()
        return out.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Word resolution

    private func pronounce(_ word: String) -> String {
        if let hit = lookup(word) { return hit }

        // Mixed alphanumerics ("mp3", "iPhone17"): split into letter/digit runs.
        if word.contains(where: \.isNumber), word.contains(where: \.isLetter) {
            var runs: [String] = []
            var current = ""
            for ch in word {
                if let last = current.last, last.isNumber != ch.isNumber {
                    runs.append(current)
                    current = ""
                }
                current.append(ch)
            }
            if !current.isEmpty { runs.append(current) }
            return runs.map(pronounce).joined(separator: " ")
        }

        // Numbers: integers → words ("42" → "forty two"), decimals digit-by-digit after
        // "point", anything longer than 12 digits digit-by-digit.
        if word.allSatisfy(\.isNumber) {
            return pronounceNumber(word)
        }

        // Out-of-dictionary: spell it (letter names are lexicon entries).
        return word.compactMap { lookup(String($0).uppercased()) }.joined(separator: " ")
    }

    private func lookup(_ word: String) -> String? {
        lexicon[word] ?? lexicon[word.lowercased()]
    }

    private func pronounceNumber(_ digits: String) -> String {
        if digits.count <= 12, let value = Int(digits) {
            return spellOut(value).compactMap { lookup($0) }.joined(separator: " ")
        }
        return digits.compactMap { lookup(digitName($0)) }.joined(separator: " ")
    }

    private func digitName(_ ch: Character) -> String {
        ["0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
         "6": "six", "7": "seven", "8": "eight", "9": "nine"][String(ch), default: ""]
    }

    /// 1234 → ["one", "thousand", "two", "hundred", "thirty", "four"].
    private func spellOut(_ value: Int) -> [String] {
        let ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                    "sixteen", "seventeen", "eighteen", "nineteen"]
        let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                    "eighty", "ninety"]
        if value < 20 { return [ones[value]] }
        if value < 100 {
            let rest = value % 10
            return [tens[value / 10]] + (rest > 0 ? [ones[rest]] : [])
        }
        if value < 1_000 {
            let rest = value % 100
            return [ones[value / 100], "hundred"] + (rest > 0 ? spellOut(rest) : [])
        }
        for (limit, name) in [(1_000_000_000_000, "billion"), (1_000_000_000, "million"),
                              (1_000_000, "thousand")] where value >= limit / 1_000 {
            let scale = limit / 1_000
            let rest = value % scale
            return spellOut(value / scale) + [name] + (rest > 0 ? spellOut(rest) : [])
        }
        return [ones.indices.contains(value) ? ones[value] : ""]
    }
}
