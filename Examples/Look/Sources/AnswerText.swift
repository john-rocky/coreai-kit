// AnswerText — the answer as the screen sets it. The model writes Markdown; the screen reads it
// inline: `**bold**` in bold, a `#` heading as a bold line, a `- ` item as `• `, a line break inside
// a paragraph (two trailing spaces, or one newline) as a space, a blank line as the gap between
// two paragraphs. A `>` quote mark and a `---` rule are dropped. The record keeps the text exactly
// as the model wrote it; only the screen reads it this way.
//
// Why the breaks become spaces: asked what a sign says, this model writes the sign's words one per
// line in bold (`**Please␣␣⏎KEEP␣␣⏎OFF␣␣⏎THE GRASS**`, the same in Hugging Face fp32). Read as
// spaces they stay one bold phrase, and the whole answer fits the box from its first line.

import Foundation

enum AnswerText {
    /// One line of a paragraph as the screen sets it; a heading line is set in bold.
    private struct Line {
        var text: String
        var heading: Bool
    }

    /// The answer's paragraphs as the screen sets them: each is one block of text, its lines
    /// separated by a newline (a heading, a list item); the screen puts a gap between blocks.
    static func paragraphs(_ raw: String) -> [AttributedString] {
        blocks(raw).map { lines in
            var out = AttributedString()
            for (index, line) in lines.enumerated() {
                if index > 0 { out += AttributedString("\n") }
                var piece = inline(line.text)
                if line.heading {
                    for (range, intent) in piece.runs.map({ ($0.range, $0.inlinePresentationIntent) }) {
                        piece[range].inlinePresentationIntent = (intent ?? []).union(.stronglyEmphasized)
                    }
                }
                out += piece
            }
            return out
        }
    }

    /// The same as plain text, a blank line between paragraphs: the record's `answer_shown`.
    static func plain(_ raw: String) -> String {
        paragraphs(raw).map { String($0.characters) }.joined(separator: "\n\n")
    }

    /// Lines grouped into paragraphs; a line that only continues the one before is joined to it.
    private static func blocks(_ raw: String) -> [[Line]] {
        var blocks: [[Line]] = [[]]
        for rawLine in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            var line = rawLine.drop { $0 == " " || $0 == "\t" }
            while line.hasPrefix(">") { line = line.dropFirst().drop { $0 == " " } }  // a quote mark
            var text = line.trimmingCharacters(in: .whitespaces)
            if text.isEmpty || isRule(text) {  // a blank line or a rule: the paragraph ends
                if !blocks[blocks.count - 1].isEmpty { blocks.append([]) }
                continue
            }
            if text.hasSuffix("\\") && !text.hasSuffix("\\\\") { text.removeLast() }  // a backslash break
            let last = blocks.count - 1
            if text.hasPrefix("#") {
                let body = text.drop { $0 == "#" }.drop { $0 == " " }
                blocks[last].append(Line(text: String(body), heading: true))
            } else if text.hasPrefix("- ") || text.hasPrefix("* ") || text.hasPrefix("+ ") {
                blocks[last].append(Line(text: "• " + String(text.dropFirst(2)), heading: false))
            } else if isNumberedItem(text) {
                blocks[last].append(Line(text: text, heading: false))
            } else if let previous = blocks[last].last, !previous.heading {
                blocks[last][blocks[last].count - 1].text = previous.text + " " + text  // a break: a space
            } else {
                blocks[last].append(Line(text: text, heading: false))
            }
        }
        return blocks.filter { !$0.isEmpty }
    }

    /// The inline Markdown of one line. A `**` still open while the answer streams is closed,
    /// so the marks never flash on the screen.
    private static func inline(_ text: String) -> AttributedString {
        var text = text
        if text.components(separatedBy: "**").count % 2 == 0 { text += "**" }
        let options = AttributedString.MarkdownParsingOptions(
            interpretedSyntax: .inlineOnlyPreservingWhitespace, failurePolicy: .returnPartiallyParsedIfPossible)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }

    /// `---`, `***`, `___` (three or more, spaces allowed): a thematic break.
    private static func isRule(_ text: String) -> Bool {
        let marks = text.filter { $0 != " " }
        guard marks.count >= 3, let first = marks.first, "-*_".contains(first) else { return false }
        return marks.allSatisfy { $0 == first }
    }

    /// `1. ` or `1) `: a numbered item keeps its own line and its number.
    private static func isNumberedItem(_ text: String) -> Bool {
        let digits = text.prefix { $0.isASCII && $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3 else { return false }
        let rest = text.dropFirst(digits.count)
        return rest.hasPrefix(". ") || rest.hasPrefix(") ")
    }
}
