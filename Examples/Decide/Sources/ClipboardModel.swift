// ClipboardModel — screen (b): what is on the clipboard, and is it what the user needs right
// now? Two typed questions on the pasteboard text: which kind of thing it is (choice, eight
// kinds) and whether it is the thing the user said they need (score: no / partly / yes,
// completely — "Is this text a shipping address?"). The same shapes are exposed to Shortcuts
// as actions (Intents.swift), so a shortcut can branch on one without opening the app.
//
// Why these two and not a yes/no on the purpose: asked "is this what the purpose needs?",
// MiniCPM5 2B says yes to everything on the clipboard (P(yes) ≥ 0.7 for a phone number, a URL
// and an API key against "fill in the shipping address"), and asked "how safe is it to paste
// this for the purpose?" it splits 50/50 on the one address. Asked whether the text *is* the
// named thing, on a no / partly / completely scale, the same model puts the address at
// "completely" and the rest at "no" (measured 2026-09-22 on the eight sample copies, and the
// same with "a phone number to call back" and "an email address" as the need).
//
// Watch mode polls the pasteboard's change count and decides every new copy as it lands —
// the "at the event" shape: on the Mac the verdict also sits in the menu bar. (On iPhone the
// poll runs while the app is in front; the Shortcuts actions are the background path.)

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
final class ClipboardModel {
    /// The kinds, as the answer reports them and as the model reads them. The wording the
    /// model reads matters: with bare nouns MiniCPM5 2B filed an API key, a URL and a plain
    /// sentence all under "order or tracking number"; with "which one is this text?" and an
    /// article per option it named all eight sample copies (2026-09-22).
    static let kinds: [Decision.Option] = [
        .init(id: "address", description: "an address"),
        .init(id: "phone number", description: "a phone number"),
        .init(id: "email address", description: "an email address"),
        .init(id: "link", description: "a link"),
        .init(id: "date or time", description: "a date or a time"),
        .init(id: "order or tracking number", description: "an order or tracking number"),
        .init(id: "secret key or password", description: "a secret key or password"),
        .init(id: "ordinary prose", description: "ordinary prose"),
    ]

    /// One watched copy and its verdict.
    struct Verdict: Identifiable {
        let id = UUID()
        let text: String
        let kind: String
        let kindConfidence: Double
        let fitLevel: Int
        let fitConfidence: Double
        let milliseconds: Double
        let time: Date

        var isSecret: Bool { kind == ClipboardModel.secretKind }
        var line: String { "\(isSecret ? "⚠︎ " : "")\(kind) · \(ClipboardModel.fitLabels[fitLevel])" }
    }

    /// The fit scale as the verdict reports it.
    nonisolated static let fitLabels = ["not what you need", "part of what you need", "paste as-is"]
    nonisolated static let secretKind = "secret key or password"

    /// What the user is about to paste — the thing the fit question names — optionally with
    /// what "partly" and "completely" mean for it: `need | partly | completely`. The level
    /// wording matters to a small model: with "only a name or a postal code" as the middle
    /// level, a URL and a phone number stay at "no"; with a generic middle level they drift up.
    var need = "a shipping address | only a name or a postal code | a complete address"

    /// The need and the three level descriptions the model reads.
    var needSpec: (need: String, levels: [String]) {
        let parts = need.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        let name = parts.first ?? ""
        let partly = parts.count > 1 && !parts[1].isEmpty ? "partly: \(parts[1])" : "partly"
        let fully = parts.count > 2 && !parts[2].isEmpty ? "yes, \(parts[2])" : "yes, completely"
        return (name, ["no", partly, fully])
    }
    var clipboard = ""
    var status = "Read the clipboard, or paste text above, then Decide."
    var working = false
    var answers: [String: Decision.Answer] = [:]
    var watching = false
    var history: [Verdict] = []
    private var lastChangeCount = -1
    private var lastWatchedText = ""
    private var watchTask: Task<Void, Never>?

    /// What the menu bar shows: the last copy's kind and fit verdict.
    var menuBarLabel: String {
        guard watching else { return "Decide" }
        return history.first?.line ?? "watching…"
    }

    var totalMilliseconds: Double { answers.values.map(\.timing.milliseconds).reduce(0, +) }

    static let sample =
        "Taro Yamada\n1-2-3 Sakura-cho, Chuo-ku\nTokyo 100-0001\nJapan\n+81 3 0000 0000"

    func readClipboard() {
        #if canImport(UIKit)
        clipboard = UIPasteboard.general.string ?? ""
        #else
        clipboard = NSPasteboard.general.string(forType: .string) ?? ""
        #endif
        status = clipboard.isEmpty ? "The clipboard holds no text." : "Clipboard read — tap Decide."
    }

    func loadSample() {
        clipboard = Self.sample
        status = "Sample loaded — tap Decide."
    }

    private var changeCount: Int {
        #if canImport(UIKit)
        UIPasteboard.general.changeCount
        #else
        NSPasteboard.general.changeCount
        #endif
    }

    /// Starts or stops deciding every new copy. The first tick skips whatever is already
    /// on the pasteboard, so only copies made while watching are judged.
    func setWatching(_ on: Bool, runtime: DecideRuntime) {
        watching = on
        watchTask?.cancel()
        watchTask = nil
        guard on else {
            status = "Stopped watching."
            return
        }
        lastChangeCount = changeCount
        lastWatchedText = ""
        status = "Watching the clipboard — copy something."
        watchTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(400))
                guard let self, self.watching else { return }
                await self.poll(runtime)
            }
        }
    }

    private func poll(_ runtime: DecideRuntime) async {
        let count = changeCount
        guard count != lastChangeCount, !working else { return }
        lastChangeCount = count
        readClipboard()
        let text = clipboard.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, text != lastWatchedText else { return }
        lastWatchedText = text
        working = true
        defer { working = false }
        do {
            let decider = try await runtime.ready()
            let start = Date()
            let a = try await decider.decide(text, questions(need: need))
            answers = a
            guard case .choice(let kind)? = a["kind"]?.value, case .score(let fit)? = a["fit"]?.value else { return }
            let verdict = Verdict(
                text: text, kind: kind.id, kindConfidence: kind.confidence,
                fitLevel: fit.level, fitConfidence: fit.confidence,
                milliseconds: a.values.map(\.timing.milliseconds).reduce(0, +), time: start)
            history.insert(verdict, at: 0)
            if history.count > 20 { history.removeLast() }
            status = "\(verdict.line) · 2 decisions in \(ms(verdict.milliseconds))"
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    /// The questions an app asks before it pastes for the user. The need rides inside the
    /// question, the clipboard text is the state — so the questions share the prefilled
    /// clipboard and each costs only its own tail.
    func questions(need: String) -> [String: Decision.Question] {
        let spec = needSpec
        return [
            "kind": .choice("Which one is this text?", options: Self.kinds),
            "fit": .score("Is this text \(spec.need)?", levels: spec.levels),
        ]
    }

    func decide(_ runtime: DecideRuntime) {
        guard !working, !clipboard.isEmpty else { return }
        working = true
        answers = [:]
        status = "Deciding…"
        Task {
            defer { working = false }
            do {
                let decider = try await runtime.ready()
                answers = try await decider.decide(clipboard, questions(need: need))
                status = "2 decisions in \(ms(totalMilliseconds))"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }
}
