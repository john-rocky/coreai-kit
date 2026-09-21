// ClipboardModel — screen (b): what is on the clipboard, and does it fit what the user is
// about to do? Three typed questions on the pasteboard text: what kind of information it is
// (choice), whether it is what the stated purpose needs (noul), and how safe it is to paste
// as-is (score). The same decisions are exposed to Shortcuts as actions (Intents.swift), so
// a shortcut can branch on one without opening the app.

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
    static let kinds = [
        "postal address", "phone number", "email address", "web link", "date or time",
        "order or tracking number", "password or key", "plain text",
    ]

    var purpose = "Fill in the shipping address on a checkout form"
    var clipboard = ""
    var status = "Read the clipboard, or paste text above, then Decide."
    var working = false
    var answers: [String: Decision.Answer] = [:]

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

    /// The questions an app asks before it pastes for the user. The purpose rides inside the
    /// question, the clipboard text is the state — so the three questions share the prefilled
    /// clipboard and each costs only its own tail.
    func questions(purpose: String) -> [String: Decision.Question] {
        [
            "kind": .choice("What kind of information is this?", Self.kinds),
            "fits": .noul("Is this the information needed for the purpose: \(purpose)?"),
            "safe": .score(
                "How safe is it to paste this text as-is for the purpose: \(purpose)?",
                levels: ["do not paste", "paste part of it", "paste as-is"]),
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
                answers = try await decider.decide(clipboard, questions(purpose: purpose))
                status = "3 decisions in \(ms(totalMilliseconds))"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }
        }
    }
}
