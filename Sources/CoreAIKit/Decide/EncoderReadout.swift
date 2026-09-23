// EncoderReadout.swift — what an encoder-type decision model's outputs mean
// (`Decision.Format.encoder`, laya): the publisher's host decoder (laya 0.3.4 `Agent.predict`,
// and the NumPy `host_decode.py` its LiteRT port was gated against), step for step.
//
//   1. The option logits are the graph's token logits at the marker positions, in option
//      order. No temperature is applied inside the graph.
//   2. The act features come from the RAW softmax of those logits — the largest probability,
//      its lead over the second, the entropy over ln(max(K, 2)), and max(K, 2) / 255 — never
//      from calibrated probabilities. They go, with the pooled [CLS] state, into the `act`
//      function.
//   3. The act probability is softmax(act logits)[0]: class 0 is the model's "answer directly"
//      action. No temperature touches it, and nothing here turns it into a policy.
//   4. The option probabilities are softmax(z / max(1e-3, T)), T looked up by bucket first —
//      `choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, the same for score and noul —
//      and by question type when the bucket has none.
//
// A choice's or a score's confidence (the publisher's 1 − normalised entropy,
// `Decision.Choice.certainty`), a score's expected level and a noul's P(true) are what
// `DecisionPrompt.answer` makes of any format's probabilities, the way the publisher computes
// them. The arithmetic is Double where the reference's is float32; on the publisher's worked
// rows the two agree to about 1e-7.

import Foundation

/// Low level: the stable API is `TypedDecisions`, which reads its answers with this.
public enum EncoderReadout {
    /// The question types in the model's order: the index of its type embedding (the graph's
    /// `qtype_onehot`) and of its per-type temperatures.
    public static let questionTypes = ["choice", "score", "noul"]

    /// The index of a question's type in `questionTypes`.
    public static func qtype(of kind: Decision.Question.Kind) -> Int {
        switch kind {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }

    /// The softmax temperature of a question, by its type and its option count.
    public struct Temperatures: Sendable, Equatable {
        /// One per question type, in `questionTypes` order.
        public let byType: [Double]
        /// By bucket (`choice:3-5`, `score:3-5`, `noul:2` …); a bucket listed here wins over
        /// its type's value.
        public let byOptions: [String: Double]

        public init(byType: [Double], byOptions: [String: Double] = [:]) {
            precondition(byType.count == EncoderReadout.questionTypes.count, "one temperature per question type")
            self.byType = byType
            self.byOptions = byOptions
        }

        /// Temperature 1 everywhere: the raw distribution.
        public static let one = Temperatures(byType: [1, 1, 1])

        /// The bucket a question falls in: its type, then `2`, `3-5`, `6-10` or `11+` options.
        public static func bucket(qtype: Int, options count: Int) -> String {
            let size = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
            return EncoderReadout.questionTypes[qtype] + ":" + size
        }

        public func temperature(qtype: Int, options count: Int) -> Double {
            byOptions[Self.bucket(qtype: qtype, options: count)] ?? byType[qtype]
        }

        public func temperature(for question: Decision.Question) -> Double {
            temperature(qtype: EncoderReadout.qtype(of: question.kind), options: question.optionIDs.count)
        }
    }

    /// The option logits: the token logits at the marker positions, in option order.
    public static func optionLogits(tokenLogits: [Float], markers: [Int32]) -> [Float] {
        markers.map { tokenLogits[Int($0)] }
    }

    /// The option probabilities at `temperature`: softmax(z / max(1e-3, T)).
    public static func probabilities(logits: [Float], temperature: Double) -> [Double] {
        DecisionPrompt.probabilities(logits: logits.map(Double.init), temperature: max(1e-3, temperature))
    }

    /// The four act features, from the raw (untempered) softmax of the option logits.
    public static func actFeatures(logits: [Float]) -> [Float] {
        let p = DecisionPrompt.probabilities(logits: logits.map(Double.init), temperature: 1)
        let k = Double(max(p.count, 2))
        let sorted = p.sorted(by: >)
        let top1 = sorted.first ?? 0
        let top2 = sorted.count > 1 ? sorted[1] : 0
        let entropy = -p.reduce(0.0) { $0 + $1 * log(max($1, 1e-9)) } / log(k)
        return [Float(top1), Float(top1 - top2), Float(entropy), Float(k / 255)]
    }

    /// The act probability: softmax(act logits)[0], class 0 being the direct-answer action.
    public static func actProbability(actLogits: [Float]) -> Double {
        DecisionPrompt.probabilities(logits: actLogits.map(Double.init), temperature: 1).first ?? 0
    }
}
