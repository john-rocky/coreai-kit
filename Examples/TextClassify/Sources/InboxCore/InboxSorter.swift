// InboxSorter.swift — the demo's one ML surface: every message of the inbox gets three answers
// (intent, urgency, sentiment) from one forward of GLiNER2.5-Decide, in inbox order, on this actor
// and never on the main one. The app and `textclassify-cli --inbox` both run it, so the screen and
// the CLI's JSON report the same measurements.
//
// A message's time is its whole trip through the classifier: collate (tokenize + schema) → graph →
// decide, measured with ContinuousClock. That is `TextClassifier.classify` split into its three
// steps, so the text is tokenized once and the sequence length the graph ran at is recorded.

import CoreAIKitEmbeddings
import Foundation

/// The three questions every message is asked: 15 labels of the bundle's 32, one answer each.
public enum InboxTasks {
    public static let intent = ClassificationTask("intent", labels: [
        "order_status", "refund_request", "cancel_subscription", "technical_issue",
        "billing_question", "account_access", "feature_request", "general_feedback",
    ])
    /// With the bare labels half the inbox came back "low" and an angry but routine message read
    /// as "critical"; saying what each level means moves both (Mac, 1,000 messages, seed 7).
    public static let urgency = ClassificationTask(
        "urgency", labels: ["low", "normal", "high", "critical"],
        descriptions: [
            "low": "no action needed, such as a compliment or an idea for later",
            "normal": "a routine request or question that needs a reply, however the customer feels",
            "high": "the customer is waiting on money or a delivery, or asks for a reply today",
            "critical": "the customer cannot use the app or their account at all, or needs it fixed right now",
        ])
    public static let sentiment = ClassificationTask("sentiment", labels: ["positive", "neutral", "negative"])
    public static let all: [ClassificationTask] = [intent, urgency, sentiment]
}

/// One message's answers and what they cost.
public struct SortedMessage: Sendable, Equatable {
    /// The message's position in the inbox (`Message.id`).
    public let index: Int
    /// Task name → the chosen label.
    public let labels: [String: String]
    /// Task name → the chosen label's probability.
    public let confidence: [String: Float]
    /// The sequence length of the graph the message ran on (256 or 512).
    public let sequenceLength: Int
    /// Tokens the tasks and the text took; the graph pads them to `sequenceLength`.
    public let tokens: Int
    /// The text was cut to fit the largest graph.
    public let truncated: Bool
    /// Collate + graph + decide, in seconds.
    public let seconds: Double
    /// When the answers were ready, in seconds after the run's start.
    public let at: Double
}

public actor InboxSorter {
    public nonisolated let classifier: TextClassifier
    public nonisolated let tasks: [ClassificationTask]

    public init(classifier: TextClassifier, tasks: [ClassificationTask] = InboxTasks.all) {
        self.classifier = classifier
        self.tasks = tasks
    }

    /// Runs one throwaway input through the graph of every sequence length, so that loading the
    /// graphs (they load on first use) is part of getting ready, not of the first messages' times.
    /// Returns the seconds each took, by sequence length.
    public func warm() async throws -> [Int: Double] {
        var out: [Int: Double] = [:]
        var words = 1
        for S in classifier.sequenceLengths {
            var c = try classifier.collate(Self.filler(words), tasks: tasks)
            while c.sequenceLength < S, !c.truncated {
                words *= 2
                c = try classifier.collate(Self.filler(words), tasks: tasks)
            }
            guard c.sequenceLength == S else { continue }
            let t0 = ContinuousClock.now
            _ = try await classifier.logits(for: c)
            out[S] = t0.duration(to: .now).inSeconds
        }
        return out
    }

    static func filler(_ words: Int) -> String { Array(repeating: "ok", count: words).joined(separator: " ") }

    /// One message: all tasks in one forward.
    public func sort(_ text: String, index: Int, start: ContinuousClock.Instant) async throws -> SortedMessage {
        let t0 = ContinuousClock.now
        let c = try classifier.collate(text, tasks: tasks)
        let rows = try await classifier.logits(for: c)
        var labels: [String: String] = [:]
        var confidence: [String: Float] = [:]
        for (task, row) in zip(tasks, rows) {
            let r = TextClassifier.decide(row, task: task, truncated: c.truncated)
            labels[task.name] = r.labels.joined(separator: ",")
            confidence[task.name] = r.probabilities.first { $0.label == r.labels[0] }?.probability
        }
        let t1 = ContinuousClock.now
        return SortedMessage(
            index: index, labels: labels, confidence: confidence, sequenceLength: c.sequenceLength,
            tokens: c.inputIds.count, truncated: c.truncated, seconds: t0.duration(to: t1).inSeconds,
            at: start.duration(to: t1).inSeconds)
    }

    /// Sorts `messages` in order, one at a time. Results go to `deliver` in batches, at most one
    /// batch per `interval` (and the rest at the end), so a screen redraws at a bounded rate
    /// whatever the per-message time; `deliver` must return at once — the loop does not wait on it.
    /// `start` is when the run began for its caller (a button press); `SortRun.totalSeconds` runs
    /// from it to the last answer.
    public func sort(
        _ messages: [Message], start: ContinuousClock.Instant = .now, interval: Duration = .milliseconds(50),
        deliver: @Sendable ([SortedMessage]) -> Void = { _ in }
    ) async throws -> SortRun {
        var all: [SortedMessage] = []
        all.reserveCapacity(messages.count)
        var pending: [SortedMessage] = []
        var lastFlush = start
        for m in messages {
            try Task.checkCancellation()
            let r = try await sort(m.text, index: m.id, start: start)
            all.append(r)
            pending.append(r)
            let now = ContinuousClock.now
            if lastFlush.duration(to: now) >= interval {
                deliver(pending)
                pending.removeAll(keepingCapacity: true)
                lastFlush = now
            }
        }
        if !pending.isEmpty { deliver(pending) }
        return SortRun(results: all, totalSeconds: all.last?.at ?? 0, tasks: tasks)
    }
}

/// A finished run and the numbers the screen and the CLI report from it.
public struct SortRun: Sendable {
    public let results: [SortedMessage]
    /// From the run's start to the last answer.
    public let totalSeconds: Double
    public let tasks: [ClassificationTask]

    public init(results: [SortedMessage], totalSeconds: Double, tasks: [ClassificationTask]) {
        self.results = results
        self.totalSeconds = totalSeconds
        self.tasks = tasks
    }

    public var count: Int { results.count }
    /// Per-message times in milliseconds, ascending.
    public var milliseconds: [Double] { results.map { $0.seconds * 1000 }.sorted() }
    public var messagesPerSecond: Double { totalSeconds > 0 ? Double(count) / totalSeconds : 0 }
    public var truncatedCount: Int { results.filter(\.truncated).count }

    /// Sequence length → messages that ran on it.
    public var sequenceLengthCounts: [Int: Int] {
        results.reduce(into: [:]) { $0[$1.sequenceLength, default: 0] += 1 }
    }

    /// Task → label → messages given it; every label of every task is present, zero or not.
    public var labelCounts: [String: [String: Int]] {
        var out: [String: [String: Int]] = [:]
        for t in tasks { out[t.name] = Dictionary(uniqueKeysWithValues: t.labels.map { ($0, 0) }) }
        for r in results {
            for (task, label) in r.labels { out[task, default: [:]][label, default: 0] += 1 }
        }
        return out
    }

    /// Task → the share of messages whose answer is the label the generator wrote them as.
    public func agreement(with messages: [Message]) -> [String: Double] {
        let written = Dictionary(uniqueKeysWithValues: messages.map { ($0.id, $0.written) })
        var out: [String: Double] = [:]
        for t in tasks {
            let hits = results.filter { r in
                guard let w = written[r.index] else { return false }
                let want = t.name == "intent" ? w.intent : t.name == "urgency" ? w.urgency : w.sentiment
                return r.labels[t.name] == want
            }.count
            out[t.name] = results.isEmpty ? 0 : Double(hits) / Double(results.count)
        }
        return out
    }

    /// Task → written label → the labels the model gave those messages.
    public func confusion(_ task: String, with messages: [Message]) -> [String: [String: Int]] {
        let written = Dictionary(uniqueKeysWithValues: messages.map { ($0.id, $0.written) })
        var out: [String: [String: Int]] = [:]
        for r in results {
            guard let w = written[r.index], let got = r.labels[task] else { continue }
            let want = task == "intent" ? w.intent : task == "urgency" ? w.urgency : w.sentiment
            out[want, default: [:]][got, default: 0] += 1
        }
        return out
    }

    /// The numbers both the CLI and the app write: times, rates, the sequence-length split and
    /// the label counts.
    public func summary() -> [String: JSONValue] {
        let ms = milliseconds
        return [
            "count": .int(count),
            "total_s": .rounded(totalSeconds, 3),
            "p50_ms": .rounded(Stats.percentile(ms, 0.5), 2),
            "p90_ms": .rounded(Stats.percentile(ms, 0.9), 2),
            "max_ms": .rounded(ms.last ?? 0, 2),
            "msg_per_s": .rounded(messagesPerSecond, 2),
            "s_counts": .object(Dictionary(uniqueKeysWithValues: sequenceLengthCounts.map { ("\($0.key)", .int($0.value)) })),
            "labels": .object(labelCounts.mapValues { .object($0.mapValues { .int($0) }) }),
            "truncated": .int(truncatedCount),
            "tokens_max": .int(results.map(\.tokens).max() ?? 0),
        ]
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
