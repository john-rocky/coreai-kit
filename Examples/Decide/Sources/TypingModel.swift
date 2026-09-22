// TypingModel — screen: while you type, three decisions run on the text so far — its tone
// (a score), what you are doing (a choice) and the emoji that fits (a choice) — and the chips
// under the field update at each pause. The as-you-type shape of the System One posts (a
// typewriter that reads sentiment and intent, emoji suggestions, a launcher that reads a
// half-typed intent), with the model on the device and nothing generated.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class TypingModel {
    nonisolated static let toneQuestion = Decision.Question.score(
        "What is the tone of what the user is typing?", levels: ["negative", "neutral", "positive"])
    nonisolated static let intentQuestion = Decision.Question.choice(
        "What is the user doing in this message?",
        ["asking a question", "asking for something", "sharing news", "complaining", "greeting", "thanking", "suggesting or inviting"])
    nonisolated static let emoji: [(symbol: String, name: String)] = [
        ("😀", "grinning face"), ("👍", "thumbs up"), ("❤️", "red heart"), ("😂", "face with tears of joy"),
        ("😢", "crying face"), ("😡", "angry face"), ("🙏", "folded hands (please or thanks)"),
        ("🎉", "party popper"), ("🤔", "thinking face"), ("🔥", "fire"), ("😴", "sleeping face"), ("🍕", "pizza"),
    ]
    nonisolated static let emojiQuestion = Decision.Question.choice(
        "Which emoji fits this message best?", options: emoji.map { .init(id: $0.symbol, description: $0.name) })
    /// Seconds of no typing before the text is read.
    nonisolated static let pause = 0.35

    var text = ""
    var tone: Decision.Answer?
    var intent: Decision.Answer?
    var emoji: Decision.Answer?
    var milliseconds = 0.0
    var decisions = 0
    var status = "Type — the chips read the text at every pause."
    private var pending: Task<Void, Never>?
    private var working = false
    private var evaluated = ""

    var toneLevel: String? { tone.map { Self.toneQuestion.levelName($0) } ?? nil }
    /// The chips in one line, for the hands-off log.
    var detail: String { "tone=\(toneLevel ?? "-") intent=\(intent?.choice ?? "-") emoji=\(emoji?.choice ?? "-") text=\(evaluated)" }

    /// Called on every change of `text`: waits for the pause, then reads the text once.
    func changed(_ runtime: DecideRuntime) {
        pending?.cancel()
        let snapshot = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !snapshot.isEmpty else {
            tone = nil; intent = nil; emoji = nil
            return
        }
        pending = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.pause))
            guard !Task.isCancelled, let self else { return }
            // The read runs in its own task: a keystroke cancels a *wait*, never a decision in flight.
            Task { await self.evaluate(snapshot, runtime: runtime) }
        }
    }

    func evaluate(_ snapshot: String, runtime: DecideRuntime) async {
        guard !working, snapshot != evaluated else { return }
        working = true
        defer { working = false }
        do {
            let decider = try await runtime.ready()
            let prefilled = try await decider.prefill(snapshot)
            let t = try await prefilled.decide(Self.toneQuestion)
            let i = try await prefilled.decide(Self.intentQuestion)
            let e = try await prefilled.decide(Self.emojiQuestion)
            tone = t; intent = i; emoji = e
            evaluated = snapshot
            decisions += 3
            milliseconds = prefilled.timing.milliseconds + t.timing.milliseconds + i.timing.milliseconds + e.timing.milliseconds
            status = "3 decisions in \(ms(milliseconds)) · \(snapshot.count) characters"
        } catch is CancellationError {
            // a stale read; the newest text is read below
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
        // The text moved on while we read: read the newest.
        let latest = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if latest != evaluated, !latest.isEmpty { changed(runtime) }
    }

    /// Appends the suggested emoji to the text.
    func acceptEmoji() {
        guard let symbol = emoji?.choice else { return }
        text += (text.hasSuffix(" ") || text.isEmpty ? "" : " ") + symbol
    }

    /// Types `sample` in, a character at a time with a longer pause at each word, so the chips
    /// update as the sentence forms (the hands-off run).
    func type(_ sample: String, runtime: DecideRuntime) async {
        text = ""
        for character in sample {
            text.append(character)
            changed(runtime)
            try? await Task.sleep(for: .milliseconds(character == " " ? 520 : 55))
        }
        try? await Task.sleep(for: .seconds(1.2))
    }

    nonisolated static let sample = "just got the job offer!! dinner tonight to celebrate?"
}

extension Decision.Question {
    /// The level's own words for a score answer to this question.
    func levelName(_ answer: Decision.Answer) -> String? {
        guard case .score(let levels) = kind, case .score(let s) = answer.value, s.level < levels.count else { return nil }
        return levels[s.level]
    }
}
