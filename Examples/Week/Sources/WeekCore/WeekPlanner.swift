// WeekPlanner.swift — the demo's one ML surface: every event of the week gets one typed decision
// from decider-0.8b, "what does this event need you to do before it?", in time order, on this
// actor and never on the main one. The app and `week-cli run` both run it, so the screen and the
// CLI's JSON report the same measurements.
//
// Each event is its own state (`WeekEvent.state`, one line), not one item of a list: asked about
// "event k" of a whole week in one state, the 0.8B model loses track of which event is k. The
// seven options are short on purpose: every option is part of every prompt, and on this S = 1
// graph a prompt costs its token count in steps.
//
// An event's time is its whole `decide` call, measured with ContinuousClock: render, the engine
// rewinding to what it shares with the previous prompt, the steps, the readout.

import CoreAIKit
import Foundation

/// What an event needs before it: the seven answers, `nothing` at the top.
public enum WeekBin: String, CaseIterable, Sendable, Codable {
    case nothing, document, prepare, travel, online, bring, confirm

    /// The option as the model reads it.
    public var option: String {
        switch self {
        case .nothing: return "nothing to prepare"
        case .document: return "bring a document or an ID"
        case .prepare: return "prepare or send something first"
        case .travel: return "travel time: leave early"
        case .online: return "join online: a link or dial-in"
        case .bring: return "buy or bring something"
        case .confirm: return "confirm or reply by a deadline"
        }
    }

    /// What the chip, the bars and a reminder's title call it.
    public var label: String {
        switch self {
        case .nothing: return "nothing"
        case .document: return "documents"
        case .prepare: return "prepare"
        case .travel: return "leave early"
        case .online: return "join online"
        case .bring: return "buy or bring"
        case .confirm: return "confirm"
        }
    }

    public init?(option: String) {
        guard let bin = WeekBin.allCases.first(where: { $0.option == option }) else { return nil }
        self = bin
    }
}

/// The one question every event is asked.
public enum WeekQuestion {
    public static let instructions = "What does this calendar event need you to do before it?"
    public static let question = Decision.Question.choice(instructions, WeekBin.allCases.map(\.option))
    /// The catalog model the app downloads when no bundle is given.
    public static let catalogID = "decider-0.8b"
}

/// One event's answer and what it cost.
public struct PlannedEvent: Sendable, Equatable {
    /// The event's position in the week (`WeekEvent.id`).
    public let index: Int
    public let bin: WeekBin
    /// Probability of `bin`.
    public let confidence: Double
    /// Every option's probability, in `WeekBin.allCases` order.
    public let probabilities: [Double]
    /// Tokens in the rendered prompt, and how many of them the engine already held.
    public let promptTokens: Int
    public let reusedTokens: Int
    /// The `decide` call, in seconds.
    public let seconds: Double
    /// When the answer was ready, in seconds after the run's start.
    public let at: Double
}

public actor WeekPlanner {
    public nonisolated let decider: TypedDecisions

    public init(decider: TypedDecisions) {
        self.decider = decider
    }

    /// Loads a local bundle, or downloads the catalog's decider-0.8b once and caches it. A local
    /// bundle is read in the decider form, one token per prefill step when its graph is a
    /// decode-only (S = 1) export — the catalog's is; a directory renamed on a phone
    /// (Documents/decider-0.8b) no longer says so in its name, so the graph file is read instead.
    public static func load(
        bundle: URL?, progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> TypedDecisions {
        guard let bundle else {
            return try await TypedDecisions(catalog: WeekQuestion.catalogID, downloadProgress: progress)
        }
        var configuration = TypedDecisions.Configuration()
        configuration.format = .decider
        configuration.singleTokenPrefill = isDecodeOnly(bundle)
        return try await TypedDecisions(bundleAt: bundle, configuration: configuration)
    }

    /// True when the bundle's graph (its metadata name or an .aimodel inside) is a `_decode_`
    /// export; nil (the kit's own guess from the directory name) when it cannot tell.
    static func isDecodeOnly(_ bundle: URL) -> Bool? {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: bundle.path)) ?? [])
            .filter { $0.hasSuffix(".aimodel") }
        if names.contains(where: { $0.contains("_decode_") }) { return true }
        return names.isEmpty ? nil : false
    }

    /// One throwaway decision, so that warming the engine (its pipelines, its caches) is part of
    /// getting ready and not of Monday's earliest event. Returns its seconds.
    public func warm() async throws -> Double {
        let t0 = ContinuousClock.now
        _ = try await decider.decide("Mon 08:00 Warm-up · Notes: none", WeekQuestion.question)
        return t0.duration(to: .now).inSeconds
    }

    /// One event, one decision.
    public func plan(_ event: WeekEvent, start: ContinuousClock.Instant) async throws -> PlannedEvent {
        let t0 = ContinuousClock.now
        let answer = try await decider.decide(event.state, WeekQuestion.question)
        let t1 = ContinuousClock.now
        guard case .choice(let choice) = answer.value, let bin = WeekBin(option: choice.id) else {
            throw WeekError.unexpectedAnswer(String(describing: answer.value))
        }
        return PlannedEvent(
            index: event.id, bin: bin, confidence: choice.confidence,
            probabilities: WeekBin.allCases.map { choice.probabilities[$0.option] ?? 0 },
            promptTokens: answer.timing.promptTokens, reusedTokens: answer.timing.reusedTokens,
            seconds: t0.duration(to: t1).inSeconds, at: start.duration(to: t1).inSeconds)
    }

    /// Plans `events` in order, one at a time. Each answer goes to `deliver` as it arrives (one
    /// decision takes most of a second, so there is nothing to batch); `deliver` must return at
    /// once. `start` is when the run began for its caller (a button press); `PlanRun.totalSeconds`
    /// runs from it to the last answer.
    public func plan(
        _ events: [WeekEvent], start: ContinuousClock.Instant = .now,
        deliver: @Sendable (PlannedEvent) -> Void = { _ in }
    ) async throws -> PlanRun {
        var all: [PlannedEvent] = []
        all.reserveCapacity(events.count)
        for event in events {
            try Task.checkCancellation()
            let r = try await plan(event, start: start)
            all.append(r)
            deliver(r)
        }
        return PlanRun(results: all, totalSeconds: all.last?.at ?? 0)
    }
}

public enum WeekError: Error, LocalizedError {
    case unexpectedAnswer(String)

    public var errorDescription: String? {
        switch self {
        case .unexpectedAnswer(let what): return "The model answered outside the seven options: \(what)"
        }
    }
}

/// A finished run and the numbers the screen and the CLI report from it.
public struct PlanRun: Sendable {
    public let results: [PlannedEvent]
    /// From the run's start to the last answer.
    public let totalSeconds: Double

    public init(results: [PlannedEvent], totalSeconds: Double) {
        self.results = results
        self.totalSeconds = totalSeconds
    }

    public var count: Int { results.count }
    /// Per-event times in milliseconds, ascending.
    public var milliseconds: [Double] { results.map { $0.seconds * 1000 }.sorted() }
    public var eventsPerSecond: Double { totalSeconds > 0 ? Double(count) / totalSeconds : 0 }

    /// Every bin with the events given it, zero or not.
    public var binCounts: [WeekBin: Int] {
        var out = Dictionary(uniqueKeysWithValues: WeekBin.allCases.map { ($0, 0) })
        for r in results { out[r.bin, default: 0] += 1 }
        return out
    }

    /// The share of events the generator wrote whose answer is the bin it wrote them as; nil when
    /// none of them was written by the generator.
    public func agreement(with events: [WeekEvent]) -> Double? {
        let written = Dictionary(uniqueKeysWithValues: events.compactMap { e in e.written.map { (e.id, $0) } })
        let judged = results.filter { written[$0.index] != nil }
        guard !judged.isEmpty else { return nil }
        return Double(judged.filter { written[$0.index] == $0.bin }.count) / Double(judged.count)
    }

    /// The numbers both the CLI and the app write: times, rates, tokens and the bin counts, and
    /// every event's answer and series row ([seconds after the start, milliseconds, prompt tokens]).
    public func summary(events: [WeekEvent]) -> [String: JSONValue] {
        let ms = milliseconds
        let tokens = results.map { Double($0.promptTokens) }.sorted()
        let byIndex = Dictionary(uniqueKeysWithValues: events.map { ($0.id, $0) })
        var out: [String: JSONValue] = [
            "count": .int(count),
            "total_s": .rounded(totalSeconds, 3),
            "median_ms": .rounded(Stats.percentile(ms, 0.5), 1),
            "p90_ms": .rounded(Stats.percentile(ms, 0.9), 1),
            "max_ms": .rounded(ms.last ?? 0, 1),
            "events_per_s": .rounded(eventsPerSecond, 3),
            "bins": .object(Dictionary(uniqueKeysWithValues: binCounts.map { ($0.key.rawValue, .int($0.value)) })),
            "prompt_tokens": .object([
                "min": .int(Int(tokens.first ?? 0)),
                "median": .rounded(Stats.percentile(tokens, 0.5), 1),
                "max": .int(Int(tokens.last ?? 0)),
                "reused_max": .int(results.map(\.reusedTokens).max() ?? 0),
            ]),
            "question": .object([
                "instructions": .string(WeekQuestion.instructions),
                "options": .array(WeekBin.allCases.map { .string($0.option) }),
            ]),
            "series": .array(results.map {
                .array([.rounded($0.at, 3), .rounded($0.seconds * 1000, 1), .int($0.promptTokens)])
            }),
            "answers": .array(results.map { r in
                let event = byIndex[r.index]
                var row: [String: JSONValue] = [
                    "event": .string(event?.state ?? "#\(r.index)"),
                    "answer": .string(r.bin.rawValue),
                    "confidence": .rounded(r.confidence, 3),
                    "probabilities": .object(Dictionary(uniqueKeysWithValues: zip(WeekBin.allCases, r.probabilities)
                        .map { ($0.0.rawValue, JSONValue.rounded($0.1, 3)) })),
                ]
                if let written = event?.written { row["written"] = .string(written.rawValue) }
                return .object(row)
            }),
        ]
        if let agree = agreement(with: events) { out["agree_with_written"] = .rounded(agree, 3) }
        return out
    }
}

public enum Stats {
    /// NumPy's default (linear) percentile of an ascending array; the median of an even count is
    /// the mean of the middle two. 0 for an empty array.
    public static func percentile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let x = q * Double(sorted.count - 1)
        let lo = Int(x.rounded(.down)), hi = min(lo + 1, sorted.count - 1)
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (x - Double(lo))
    }
}

/// A JSON value with keys written in sorted order and numbers in their shortest form.
public enum JSONValue: Encodable, Sendable {
    case int(Int)
    case double(Double)
    case string(String)
    case bool(Bool)
    case null
    case array([JSONValue])
    case object([String: JSONValue])

    /// `x` rounded to `places` decimals.
    public static func rounded(_ x: Double, _ places: Int) -> JSONValue {
        let p = pow(10, Double(places))
        return .double((x * p).rounded() / p)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }

    /// One line (or indented with `pretty`).
    public func json(pretty: Bool = false) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = pretty ? [.sortedKeys, .withoutEscapingSlashes, .prettyPrinted] : [.sortedKeys, .withoutEscapingSlashes]
        return (try? enc.encode(self)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }
}

extension Duration {
    public var inSeconds: Double {
        let c = components
        return Double(c.seconds) + Double(c.attoseconds) * 1e-18
    }
}
