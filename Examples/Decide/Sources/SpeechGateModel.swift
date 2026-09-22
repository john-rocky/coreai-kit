// SpeechGateModel — screen (a): a speech gate. Record, split the recording into utterances,
// transcribe each on device, and ask one question per utterance: is this for the assistant?
// Only the utterances that pass would go to a language model; everything else is dropped at
// the cost of one scored prompt. The whole loop is kit API: `MicRecorder`,
// `VoiceActivityDetector.segments(in:)`, Apple's `SystemTranscriber`, `TypedDecisions`.
//
// The gate is a two-way choice, not a yes/no: asked "is the speaker asking the assistant to
// do something?", MiniCPM5 2B says yes to three of the four remarks in the sample (P(yes)
// 0.58–0.84); asked to file the utterance as "a request or a question for the assistant" or
// "small talk, a remark, or talking to someone else", it files seven of eight as intended
// (2026-09-22). The first option is what passes.

import CoreAIOps
import Foundation
import Observation

@MainActor
@Observable
final class SpeechGateModel {
    struct Utterance: Identifiable {
        let id = UUID()
        let text: String
        let seconds: Double
        let answer: Decision.Answer
        /// The option that passes, and its probability.
        let passOption: String
        var passes: Bool { answer.choice == passOption }
        var passProbability: Double {
            if case .choice(let c) = answer.value { return c.probabilities[passOption] ?? 0 }
            return answer.noul ?? 0
        }
    }

    /// What the gate asks about every utterance: `question | option that passes | other option…`.
    /// Editable: the same loop gates on anything a choice can express.
    var question = "What is this utterance? | a request or a question for the assistant | small talk, a remark, or talking to someone else"

    /// The gate question and the option that passes; nil unless the line has two options.
    var gateQuestion: (question: Decision.Question, passOption: String)? {
        let parts = question.split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count >= 3, !parts[0].isEmpty else { return nil }
        let options = Array(parts.dropFirst()).filter { !$0.isEmpty }
        guard options.count >= 2 else { return nil }
        return (.choice(parts[0], options), options[0])
    }
    var utterances: [Utterance] = []
    var status = "Record a few sentences, or run the sample."
    var recording = false
    var working = false

    private let recorder = MicRecorder()

    /// Eight sentences a phone hears in a day: four are requests, four are not. Runs the gate
    /// without a microphone, which is also how the numbers in the README were taken.
    static let sample: [String] = [
        "Remind me to call the dentist tomorrow at nine.",
        "Yeah I think the meeting went okay, honestly.",
        "What's the weather going to be like this weekend?",
        "Hold on, let me find my keys.",
        "Add oat milk and coffee to the shopping list.",
        "That movie last night was way too long.",
        "How many grams are in an ounce?",
        "Okay, see you later, drive safe.",
    ]

    var passed: Int { utterances.filter(\.passes).count }
    var medianMilliseconds: Double { median(utterances.map(\.answer.timing.milliseconds)) }

    func runSample(_ runtime: DecideRuntime) {
        guard !working else { return }
        Task { await gate(Self.sample.map { ($0, 0) }, runtime: runtime) }
    }

    func toggleRecord(_ runtime: DecideRuntime) {
        if recording {
            recording = false
            let samples = recorder.stop()
            guard !samples.isEmpty else {
                status = "No audio captured."
                return
            }
            Task { await transcribeAndGate(samples, runtime: runtime) }
        } else {
            Task {
                do {
                    try await recorder.start()
                    recording = true
                    status = "Recording… tap Stop when done."
                } catch {
                    status = error.localizedDescription
                }
            }
        }
    }

    /// Split the clip into utterances and transcribe each with Apple's on-device recognizer
    /// (no download). Swap in `KitTranscriber(catalog:)` for a catalog ASR model.
    private func transcribeAndGate(_ samples: [Float], runtime: DecideRuntime) async {
        working = true
        defer { working = false }
        status = "Transcribing…"
        do {
            let transcriber = try await SystemTranscriber(locale: .current)
            var spoken: [(String, Double)] = []
            for segment in VoiceActivityDetector.segments(in: samples, options: .patient) {
                let text = try await transcriber.transcribe(samples: segment.samples(from: samples)).text
                if !text.isEmpty { spoken.append((text, segment.duration)) }
            }
            guard !spoken.isEmpty else {
                status = "No speech found in the recording."
                return
            }
            await gate(spoken, runtime: runtime)
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }

    private func gate(_ spoken: [(String, Double)], runtime: DecideRuntime) async {
        working = true
        defer { working = false }
        utterances = []
        guard let gate = gateQuestion else {
            status = "The gate line wants a question and at least two options, separated by |."
            return
        }
        do {
            let decider = try await runtime.ready()
            for (text, seconds) in spoken {
                status = "Deciding: \(text.prefix(40))…"
                let answer = try await decider.decide(text, gate.question)
                utterances.append(Utterance(text: text, seconds: seconds, answer: answer, passOption: gate.passOption))
            }
            status = "\(passed) of \(utterances.count) utterances need the assistant · median \(ms(medianMilliseconds)) per decision"
        } catch {
            status = "Error: \(error.localizedDescription)"
        }
    }
}
