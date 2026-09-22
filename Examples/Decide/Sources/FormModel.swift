// FormModel — screen (f): copy anywhere, and the right field of a form fills itself. Watch
// polls the pasteboard; every new copy is one decision — which of nine kinds of thing is it —
// and the kinds that map to a field of the checkout form land there, on the spot. A secret
// (a key, a password) is refused and said so; a kind the form has no field for is left where
// it is. One scored prompt per copy, nothing generated, the milliseconds next to each.
//
// The source pane on the left is a sample email to copy from; the copies in a hands-off run
// come from `pbcopy`, and the pane highlights the piece that was copied so the eye can follow
// it into the form. The form itself is HTML in a web view (Form.html), filled through one
// JavaScript call per field — the same page fills on a Mac and on an iPhone.

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
        let id: String       // the HTML element id
        let label: String
        let kind: String     // the option that routes here
    }

    struct Event: Identifiable {
        let id = UUID()
        let text: String
        let kind: String
        let confidence: Double
        let field: Field?
        let refused: Bool
        let milliseconds: Double

        var line: String {
            if refused { return "\(kind) — not pasted" }
            if let field { return "\(kind) → \(field.label)" }
            return "\(kind) — no field for it"
        }
    }

    nonisolated static let fields: [Field] = [
        Field(id: "name", label: "Full name", kind: "a person's name"),
        Field(id: "address", label: "Shipping address", kind: "a postal address"),
        Field(id: "phone", label: "Phone", kind: "a phone number"),
        Field(id: "email", label: "Email", kind: "an email address"),
        Field(id: "order", label: "Order reference", kind: "an order number"),
        Field(id: "note", label: "Delivery note", kind: "a note to the courier"),
    ]
    nonisolated static let secretKind = "a secret key, token or password"
    /// Everything the model can answer: the six fields' kinds, then the kinds that stay off the form.
    nonisolated static let kinds: [String] = fields.map(\.kind) + ["a date or a time", secretKind, "something else"]
    nonisolated static let question = Decision.Question.choice("Which one is this text?", FormModel.kinds)

    nonisolated static let sampleSource = """
        From: Mika Tanaka <mika.tanaka@example.com>
        Subject: Order HL-88213 — new delivery address

        Hi, could you send my order to the new place instead?

        Mika Tanaka
        2-4-8 Sakuragaoka, Shibuya-ku
        Tokyo 150-0031, Japan
        Phone: +81 90 1234 5678

        Please leave the parcel with the concierge if I am out.

        Thanks,
        Mika
        """

    var source = FormModel.sampleSource
    var values: [String: String] = [:]
    /// Bumped on every fill so the web view applies it once.
    var fillVersion = 0
    var events: [Event] = []
    var highlight: Range<String.Index>?
    var watching = false
    var working = false
    var status = "Turn on Watch, then copy something — the matching field fills itself."
    private var lastChangeCount = -1
    private var lastText = ""
    private var watchTask: Task<Void, Never>?

    var filledCount: Int { values.values.filter { !$0.isEmpty }.count }
    var totalMilliseconds: Double { events.map(\.milliseconds).reduce(0, +) }

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

    func reset() {
        values = [:]
        events = []
        highlight = nil
        fillVersion += 1
        status = watching ? "Watching — copy something." : "Turn on Watch, then copy something."
    }

    /// Starts or stops deciding every new copy. Whatever is on the pasteboard when Watch
    /// turns on is skipped; only copies made afterwards are judged.
    func setWatching(_ on: Bool, runtime: DecideRuntime) {
        watching = on
        watchTask?.cancel()
        watchTask = nil
        guard on else {
            status = "Stopped watching."
            return
        }
        lastChangeCount = changeCount
        lastText = ""
        status = "Watching — copy something."
        watchTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(300))
                guard let self, self.watching else { return }
                await self.poll(runtime)
            }
        }
    }

    private func poll(_ runtime: DecideRuntime) async {
        let count = changeCount
        guard count != lastChangeCount, !working else { return }
        lastChangeCount = count
        guard let raw = pasteboardText else { return }
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, text != lastText else { return }
        lastText = text
        highlight = Self.find(text, in: source)
        working = true
        defer { working = false }
        do {
            let decider = try await runtime.ready()
            let answer = try await decider.decide(text, Self.question)
            guard case .choice(let choice) = answer.value else { return }
            let field = Self.fields.first { $0.kind == choice.id }
            let refused = choice.id == Self.secretKind
            if let field, !refused {
                values[field.id] = text
                fillVersion += 1
            }
            let event = Event(
                text: text, kind: choice.id, confidence: choice.confidence, field: field,
                refused: refused, milliseconds: answer.timing.milliseconds)
            events.insert(event, at: 0)
            status = "\(event.line) · \(ms(event.milliseconds)) · \(filledCount) of \(Self.fields.count) fields filled"
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    /// The copied text's place in the source, tolerant of the line breaks a copy may lose.
    static func find(_ text: String, in source: String) -> Range<String.Index>? {
        if let range = source.range(of: text) { return range }
        let flat = text.replacingOccurrences(of: "\n", with: " ")
        let lines = flat.components(separatedBy: ", ")
        guard let first = lines.first, let start = source.range(of: first) else { return nil }
        var end = start.upperBound
        for piece in lines.dropFirst() {
            guard let next = source.range(of: piece, range: end..<source.endIndex) else { return nil }
            end = next.upperBound
        }
        return start.lowerBound..<end
    }
}
