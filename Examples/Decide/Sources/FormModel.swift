// FormModel — screen: copy an email (or any message) and every field of the checkout form
// fills at once. The copied text is prefilled once; then one typed question per field picks
// the line that holds it — "which line has the sender's full name?", "…the street of the
// address the order should be shipped to?" — as a choice among the text's lines with "none of
// these" at the end. The chosen line is trimmed to the part the field takes (the address inside
// "<…>", the number after "Phone:"); nothing is generated. A copy of a single piece (a name, a
// phone number) goes through the nine-kind question instead and lands in its one field.
//
// Watch polls the pasteboard, so a ⌘C in Mail is enough; Paste reads it once (the iPhone way).
// The sample email carries an old address as a decoy: the model has to pick the new one, and
// on MiniCPM5 2B it does (2026-09-22, both address lines at 0.44–0.64 against the decoy).

import CoreAIOps
import Foundation
import Observation

#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif

@MainActor
@Observable
final class FormModel {
    struct Field: Identifiable {
        let id: String        // the HTML element id
        let label: String
        let kind: String      // the single-piece kind that routes here
        /// The question that picks this field's line out of a copied text.
        let lineQuestion: String
    }

    /// One field's fill: where it came from and how sure the model was.
    struct Fill: Identifiable {
        let id: String        // field id
        let label: String
        let value: String
        let sourceLine: String
        let confidence: Double
    }

    nonisolated static let fields: [Field] = [
        Field(id: "name", label: "Full name", kind: "a person's name",
              lineQuestion: "Which line has the sender's full name?"),
        Field(id: "address", label: "Shipping address", kind: "a postal address",
              lineQuestion: "Which line has the street of the address the order should be shipped to?"),
        Field(id: "phone", label: "Phone", kind: "a phone number",
              lineQuestion: "Which line has the phone number?"),
        Field(id: "email", label: "Email", kind: "an email address",
              lineQuestion: "Which line has the email address?"),
        Field(id: "order", label: "Order reference", kind: "an order number",
              lineQuestion: "Which line has the order number?"),
        Field(id: "note", label: "Delivery note", kind: "a note to the courier",
              lineQuestion: "Which line is an instruction for the courier?"),
    ]
    /// The address may span lines: asked over the lines right after its first one.
    nonisolated static let addressEndQuestion =
        "Which of these lines is the last line of that shipping address (city, postal code, country)?"
    nonisolated static let secretKind = "a secret key, token or password"
    nonisolated static let kinds: [String] = fields.map(\.kind) + ["a date or a time", secretKind, "something else"]
    nonisolated static let kindQuestion = Decision.Question.choice("Which one is this text?", FormModel.kinds)
    /// Below this the chosen line is not trusted and the field stays empty.
    nonisolated static let minimumConfidence = 0.3
    /// A text with more lines than this is asked by paragraph instead of by line.
    nonisolated static let maxLines = 15

    nonisolated static let sampleEmail = """
        From: Mika Tanaka <mika.tanaka@example.com>
        Subject: Order HL-88213 — please ship to my new address

        Hi, I moved last week. Please send the order to my new place, not to the old one.

        Old address: 5-1 Aoyama, Minato-ku, Tokyo 107-0062

        New address:
        Mika Tanaka
        2-4-8 Sakuragaoka, Shibuya-ku
        Tokyo 150-0031, Japan
        Phone: +81 90 1234 5678

        Please leave the parcel with the concierge if I am out.

        Thanks,
        Mika
        """

    var values: [String: String] = [:]
    /// Bumped once per fill so the web view applies all fields together.
    var fillVersion = 0
    var fills: [Fill] = []
    var lastText = ""
    var watching = false
    var working = false
    var status = "Copy an email or a message — every field fills at once. Or press Paste."
    var decisions = 0
    var milliseconds = 0.0
    var prefillTokens = 0
    private var lastChangeCount = -1
    private var lastSeen = ""
    private var watchTask: Task<Void, Never>?

    var filledCount: Int { values.values.filter { !$0.isEmpty }.count }

    private var changeCount: Int {
        #if canImport(UIKit)
        UIPasteboard.general.changeCount
        #else
        NSPasteboard.general.changeCount
        #endif
    }

    private var pasteboardText: String? {
        #if canImport(UIKit)
        UIPasteboard.general.string
        #else
        NSPasteboard.general.string(forType: .string)
        #endif
    }

    func clear() {
        values = [:]
        fills = []
        lastText = ""
        fillVersion += 1
        decisions = 0
        milliseconds = 0
        status = watching ? "Watching — copy an email or a message." : "Copy an email or a message — every field fills at once. Or press Paste."
    }

    /// Fills from whatever is on the pasteboard now.
    func paste(_ runtime: DecideRuntime) {
        guard let text = pasteboardText?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            status = "The clipboard holds no text."
            return
        }
        Task { await fill(from: text, runtime: runtime) }
    }

    /// Puts the sample email on the pasteboard — what a ⌘C in Mail does — so Watch fills the
    /// form from it; without Watch, fills from it directly.
    func useSample(_ runtime: DecideRuntime) {
        if watching {
            #if canImport(UIKit)
            UIPasteboard.general.string = Self.sampleEmail
            #else
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(Self.sampleEmail, forType: .string)
            #endif
        } else {
            Task { await fill(from: Self.sampleEmail, runtime: runtime) }
        }
    }

    func setWatching(_ on: Bool, runtime: DecideRuntime) {
        watching = on
        watchTask?.cancel()
        watchTask = nil
        guard on else {
            status = "Stopped watching."
            return
        }
        lastChangeCount = changeCount
        lastSeen = ""
        status = "Watching — copy an email or a message."
        watchTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(300))
                guard let self, self.watching else { return }
                let count = self.changeCount
                guard count != self.lastChangeCount, !self.working else { continue }
                self.lastChangeCount = count
                guard let raw = self.pasteboardText else { continue }
                let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty, text != self.lastSeen else { continue }
                self.lastSeen = text
                await self.fill(from: text, runtime: runtime)
            }
        }
    }

    // MARK: - The fill

    func fill(from text: String, runtime: DecideRuntime) async {
        guard !working else { return }
        working = true
        defer { working = false }
        lastText = text
        status = "Reading…"
        let start = Date()
        do {
            let decider = try await runtime.ready()
            let lines = text.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            var newValues: [String: String] = [:]
            var newFills: [Fill] = []
            var count = 0
            if lines.count <= 2 {
                // A single piece: which kind is it, and so which field.
                let answer = try await decider.decide(text, Self.kindQuestion)
                count = 1
                if let choice = answer.choice, let field = Self.fields.first(where: { $0.kind == choice }) {
                    let value = Self.trim(text, for: field.id)
                    newValues[field.id] = value
                    newFills.append(Fill(id: field.id, label: field.label, value: value, sourceLine: text, confidence: answer.confidence))
                    status = "\(field.label) ← \(choice)"
                } else if answer.choice == Self.secretKind {
                    status = "Not pasted: that looks like a secret key or a password."
                } else {
                    status = "No field for \(answer.choice ?? "that")."
                }
            } else {
                // The whole text: prefill once, then one question per field over its lines.
                let candidates = lines.count <= Self.maxLines ? lines : Self.paragraphs(of: text)
                let options = candidates.enumerated().map { Decision.Option(id: "L\($0.offset)", description: $0.element) }
                    + [Decision.Option(id: "none", description: "none of these")]
                let prefilled = try await decider.prefill(text)
                prefillTokens = prefilled.tokens
                func pick(_ question: String) async throws -> (index: Int, confidence: Double)? {
                    let answer = try await prefilled.decide(.choice(question, options: options))
                    count += 1
                    guard let id = answer.choice, id != "none", answer.confidence >= Self.minimumConfidence,
                        let index = Int(id.dropFirst()) else { return nil }
                    return (index, answer.confidence)
                }
                for field in Self.fields {
                    guard let first = try await pick(field.lineQuestion) else { continue }
                    var value = candidates[first.index]
                    var source = value
                    if field.id == "address", lines.count <= Self.maxLines, first.index + 1 < candidates.count {
                        // The address may run on: ask which of the next lines ends it, and only those
                        // lines are on offer, so a decoy address elsewhere cannot be picked.
                        let tail = Array(candidates[(first.index + 1)...min(first.index + 3, candidates.count - 1)])
                        let tailOptions = tail.enumerated().map { Decision.Option(id: "T\($0.offset)", description: $0.element) }
                            + [Decision.Option(id: "none", description: "none of these: the address is the one line")]
                        let answer = try await prefilled.decide(.choice(Self.addressEndQuestion, options: tailOptions))
                        count += 1
                        if let id = answer.choice, id != "none", answer.confidence >= Self.minimumConfidence, let end = Int(id.dropFirst()) {
                            let span = [candidates[first.index]] + tail[0...end]
                            value = span.joined(separator: "\n")
                            source = span.joined(separator: " / ")
                        }
                    }
                    value = Self.trim(value, for: field.id)
                    guard !value.isEmpty else { continue }
                    newValues[field.id] = value
                    newFills.append(Fill(id: field.id, label: field.label, value: value, sourceLine: source, confidence: first.confidence))
                }
                status = "\(newFills.count) of \(Self.fields.count) fields filled from \(lines.count) lines"
            }
            values = newValues
            fills = newFills
            fillVersion += 1
            decisions = count
            milliseconds = Date().timeIntervalSince(start) * 1000
            if !newFills.isEmpty {
                status += " · \(count) decisions in \(ms(milliseconds))"
            }
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    /// Paragraphs (blank-line separated), for a long text.
    nonisolated static func paragraphs(of text: String) -> [String] {
        text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    /// The part of a chosen line the field takes: the address inside `<…>`, the digits after
    /// `Phone:`, the number in the subject line; labels like `From:` / `New address:` dropped.
    nonisolated static func trim(_ line: String, for field: String) -> String {
        func first(_ pattern: String) -> String? {
            guard let regex = try? Regex(pattern), let match = line.firstMatch(of: regex) else { return nil }
            return String(line[match.range])
        }
        func dropLabel(_ text: String) -> String {
            let labels = ["from", "to", "name", "phone", "tel", "mobile", "email", "e-mail", "order", "order number", "ref",
                          "address", "new address", "shipping address", "ship to", "note", "delivery note", "subject"]
            var out = text
            for label in labels {
                if let regex = try? Regex("(?i)^\(label)\\s*[:：]\\s*"), let m = out.firstMatch(of: regex) {
                    out.removeSubrange(m.range)
                }
            }
            return out.trimmingCharacters(in: .whitespaces)
        }
        switch field {
        case "email":
            return first(#"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#) ?? dropLabel(line)
        case "phone":
            return first(#"\+?[0-9][0-9 ()\-]{6,}[0-9]"#) ?? dropLabel(line)
        case "order":
            return first(#"[A-Z]{1,4}-?[0-9]{4,}"#) ?? first(#"[0-9]{5,}"#) ?? dropLabel(line)
        case "name":
            var out = dropLabel(line)
            if let regex = try? Regex(#"\s*<[^>]*>"#) { out = out.replacing(regex, with: "") }
            return out.trimmingCharacters(in: .whitespaces)
        case "address":
            return line.components(separatedBy: "\n").map { dropLabel($0) }.filter { !$0.isEmpty }.joined(separator: "\n")
        default:
            return dropLabel(line)
        }
    }
}
