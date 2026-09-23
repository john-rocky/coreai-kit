// DecisionCalibration.swift — the temperature a decision's answer-slot logits are read at, and
// the arithmetic that fits one: a distribution re-read at another temperature, the temperature
// that minimises NLL on labelled rows, and the numbers that say whether the probabilities mean
// what they say (accuracy, NLL, Brier, top-label ECE).
//
// Which temperature, for each question type:
//
//   1. `TypedDecisions.Configuration.temperature` — the app's own, for every type;
//   2. the catalog entry's `calibration` — fitted by the maintainer (`decide-cli calibrate`);
//      its type's value when the types were fitted apart, else its one value;
//   3. what the bundle declares — a slot head one per type, a scalar head or a letter list one;
//   4. the prompt form's default — 1.03 for `.decider` (its card's), else 1 (the raw distribution).
//
// A temperature divides every option's logit by the same positive number, so it never changes
// which option wins; it changes how sure the answer says it is.
//
// The fitting arithmetic is `Examples/Decide/conformance/calibration.py`'s, definition for
// definition, so the two can check each other: the same temperature grid, the same ECE bins,
// SemIf's multi-class Brier, balanced accuracy over the right options' ids.

import Foundation

/// The temperature a loaded model reads each question type at.
struct DecisionTemperatures: Sendable, Equatable {
    let choice: Double
    let score: Double
    let noul: Double

    func temperature(for kind: Decision.Question.Kind) -> Double {
        switch kind {
        case .choice: return choice
        case .score: return score
        case .noul: return noul
        }
    }
}

extension TypedDecisions {
    /// The temperature per question type, in the order above. A catalog temperature that is not
    /// a positive number fails the load, as a bundle's own declaration does.
    static func resolveTemperatures(
        id: String, configured: Double?, catalog: CatalogEntry.Calibration?, format: Decision.Format,
        slot: SlotPrompt.Layout?, scalar: ScalarPrompt.Layout?, letters: LetterListPrompt.Layout?
    ) throws -> DecisionTemperatures {
        if let configured {
            return DecisionTemperatures(choice: configured, score: configured, noul: configured)
        }
        if let catalog {
            let resolved = DecisionTemperatures(
                choice: catalog.temperature(forType: "choice"), score: catalog.temperature(forType: "score"),
                noul: catalog.temperature(forType: "noul"))
            for value in [resolved.choice, resolved.score, resolved.noul] where !(value.isFinite && value > 0) {
                throw DecisionError.unsupportedModel(
                    id: id, reason: "its catalog entry declares a calibration temperature that is not a positive number")
            }
            return resolved
        }
        switch format {
        case .slot:
            return DecisionTemperatures(
                choice: slot?.choiceTemperature ?? 1, score: slot?.scoreTemperature ?? 1,
                noul: slot?.noulTemperature ?? 1)
        case .scalar:
            let t = scalar?.temperature ?? 1
            return DecisionTemperatures(choice: t, score: t, noul: t)
        case .letterList:
            let t = letters?.temperature ?? 1
            return DecisionTemperatures(choice: t, score: t, noul: t)
        case .decider:
            let t = DeciderPrompt.defaultTemperature
            return DecisionTemperatures(choice: t, score: t, noul: t)
        case .chat, .sharedState, .decisionFunction:
            return DecisionTemperatures(choice: 1, score: 1, noul: 1)
        }
    }
}

/// Temperature calibration of decision probabilities: re-read a distribution at another
/// temperature, fit one on labelled rows, and measure what the probabilities are worth.
public enum DecisionCalibration {
    /// `probabilities` read at temperature `from`, re-read at temperature `to`: the softmax of
    /// log p · from / to. When the distribution is a softmax of logits at `from` — every answer
    /// of `TypedDecisions` except a `.decider` score and a `.letterList` yes/no, which combine
    /// several — this is the softmax of the same logits at `to`, exactly; the logits are not
    /// needed. A probability of 0 is floored at 1e-12. Both temperatures are positive.
    public static func rescale(_ probabilities: [Double], from: Double = 1, to: Double) -> [Double] {
        DecisionPrompt.probabilities(
            logits: probabilities.map { log($0 > 0 ? $0 : 1e-12) * from }, temperature: to)
    }

    /// One labelled question: its probabilities in option order and the index of the right option.
    public struct Row: Sendable, Equatable {
        public var probabilities: [Double]
        /// Index of the right option in `probabilities`.
        public var label: Int
        /// The option ids in the same order. The right option's id is the class balanced accuracy
        /// averages over, so a fixture that shuffles its options per row keeps its classes; nil
        /// counts the position instead.
        public var optionIDs: [String]?
        /// The group a table reports the row under (SemIf's task family).
        public var family: String?

        public init(probabilities: [Double], label: Int, optionIDs: [String]? = nil, family: String? = nil) {
            self.probabilities = probabilities
            self.label = label
            self.optionIDs = optionIDs
            self.family = family
        }

        /// The class of the right option: its id, or its position when there are no ids.
        var gold: String {
            if let optionIDs, optionIDs.indices.contains(label) { return optionIDs[label] }
            return String(label)
        }
    }

    /// The temperatures `fitTemperature` tries: exp(x / 40) for x in -60..<100, 0.22 to 11.9.
    public static let temperatureGrid: [Double] = (-60..<100).map { exp(Double($0) / 40) }

    /// The temperature in `temperatureGrid` that minimises the negative log-likelihood of the
    /// right options, a tie going to the lower one. `rows` are read at temperature 1 —
    /// `rescale(p, from: t, to: 1)` recovers them from answers read at t — so the result is the
    /// temperature to read the logits at (`TypedDecisions.Configuration.temperature`, or a
    /// catalog record). 1 when there are no rows.
    public static func fitTemperature(_ rows: [Row]) -> Double {
        guard !rows.isEmpty else { return 1 }
        var best = (temperature: 1.0, nll: Double.infinity)
        for t in temperatureGrid {
            let nll = rows.reduce(0.0) { $0 - log(max(rescale($1.probabilities, to: t)[$1.label], 1e-12)) }
            if nll < best.nll { best = (t, nll) }
        }
        return best.temperature
    }

    /// What a set of answers is worth against their labels.
    public struct Metrics: Sendable, Equatable {
        public let n: Int
        /// Rows whose most probable option (the first, on a tie) is the right one.
        public let accuracy: Double
        /// Mean over the right options' classes of the share of each answered correctly.
        public let balancedAccuracy: Double
        /// Mean −log P(right option), each probability floored at 1e-12.
        public let nll: Double
        /// Multi-class Brier score: Σ over options of (p − 1[right])², averaged over rows.
        public let brier: Double
        /// Top-label expected calibration error: the rows binned by their answer's probability
        /// into equal-width bins, |accuracy − mean probability| per bin, weighted by its rows.
        public let ece: Double
        /// Mean probability of the answer each row gives.
        public let meanConfidence: Double

        init(n: Int, accuracy: Double, balancedAccuracy: Double, nll: Double, brier: Double, ece: Double, meanConfidence: Double) {
            self.n = n
            self.accuracy = accuracy
            self.balancedAccuracy = balancedAccuracy
            self.nll = nll
            self.brier = brier
            self.ece = ece
            self.meanConfidence = meanConfidence
        }
    }

    /// The metrics of `rows` as they are (rescale them first to measure another temperature).
    /// NaN when there are no rows.
    public static func metrics(_ rows: [Row], bins: Int = 10) -> Metrics {
        guard !rows.isEmpty else {
            return Metrics(n: 0, accuracy: .nan, balancedAccuracy: .nan, nll: .nan, brier: .nan, ece: .nan, meanConfidence: .nan)
        }
        let bins = max(1, bins)
        let n = Double(rows.count)
        var correct = 0.0, nll = 0.0, brier = 0.0, confidence = 0.0
        var binRows = [Double](repeating: 0, count: bins)
        var binCorrect = [Double](repeating: 0, count: bins)
        var binConfidence = [Double](repeating: 0, count: bins)
        var byClass: [String: (rows: Double, correct: Double)] = [:]
        for row in rows {
            let p = row.probabilities
            let best = argmax(p)
            let right: Double = best == row.label ? 1 : 0
            let conf = p[best]
            correct += right
            nll -= log(max(p[row.label], 1e-12))
            brier += p.indices.reduce(0.0) { sum, k in
                let d = p[k] - (k == row.label ? 1 : 0)
                return sum + d * d
            }
            confidence += conf
            let scaled = conf * Double(bins)
            let bin = scaled.isFinite ? min(bins - 1, max(0, Int(scaled))) : bins - 1
            binRows[bin] += 1
            binCorrect[bin] += right
            binConfidence[bin] += conf
            byClass[row.gold, default: (0, 0)].rows += 1
            byClass[row.gold, default: (0, 0)].correct += right
        }
        var ece = 0.0
        for b in 0..<bins where binRows[b] > 0 {
            ece += binRows[b] / n * abs(binCorrect[b] / binRows[b] - binConfidence[b] / binRows[b])
        }
        let recalls = byClass.keys.sorted().map { byClass[$0]!.correct / byClass[$0]!.rows }
        return Metrics(
            n: rows.count, accuracy: correct / n, balancedAccuracy: recalls.reduce(0, +) / Double(recalls.count),
            nll: nll / n, brier: brier / n, ece: ece, meanConfidence: confidence / n)
    }

    /// The table calibration.py prints: one row per family, sorted by name, then `all` — every
    /// row pooled, except that its balanced accuracy is the families' mean (SemIf's summary
    /// figure). A row without a family counts in `all` only.
    public static func familyMetrics(_ rows: [Row], bins: Int = 10) -> [(family: String, metrics: Metrics)] {
        var families: [String: [Row]] = [:]
        for row in rows {
            if let family = row.family { families[family, default: []].append(row) }
        }
        var table = families.keys.sorted().map { (family: $0, metrics: metrics(families[$0]!, bins: bins)) }
        let pooled = metrics(rows, bins: bins)
        let balanced = table.isEmpty
            ? pooled.balancedAccuracy
            : table.map(\.metrics.balancedAccuracy).reduce(0, +) / Double(table.count)
        table.append(
            (family: "all",
             metrics: Metrics(
                n: pooled.n, accuracy: pooled.accuracy, balancedAccuracy: balanced, nll: pooled.nll,
                brier: pooled.brier, ece: pooled.ece, meanConfidence: pooled.meanConfidence)))
        return table
    }

    /// Index of the largest probability, the first on a tie (Python's `max` over the indices).
    static func argmax(_ p: [Double]) -> Int {
        var best = 0
        for i in p.indices where p[i] > p[best] { best = i }
        return best
    }
}
