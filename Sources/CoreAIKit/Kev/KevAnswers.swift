// From the model zoo's apps/Kev/Sources/Kev/SystemOne.swift (9e06b5a, sha256 930a39a91163), identifiers prefixed Kev for the kit.
// SystemOne — the answers and the response body (conversion/kev/host.py §6–7; kev/api.py `to_answers`,
// `choice_confidence`, `score_confidence`, `round_prob`, kev/serve.py `Server._body` at tag kev-1.0):
//
//   noul   {"type": "noul", "noul": round4(p[1])}                                       (options: no, yes)
//   choice {"type": "choice", "choice": keys[first argmax p], "confidence": round4((max(n) - 1/K) / (1 - 1/K)),
//           "probabilities": {key: round4(p)}}                                          (confidence 1.0 when K = 1)
//   score  {"type": "score", "score": round4(sum_i i * p_i), "legend": legend, "probabilities": {"i": round4(p_i)},
//           "confidence": round4(max(0, 1 - sum_i n_i |i - mode| / D))}                 (confidence 1.0 when L = 1)
//     n = p / sum(p) (all zeros -> uniform), mode = the first argmax of n, D = sum_i |i - (L - 1) / 2| / L,
//     p = the Float probabilities as Doubles, round4 = Python's round(x, 4), every sum Python 3.12's sum()
//   body   {"model", "answers", "usage": {"input_tokens", "output_tokens": tokens of json.dumps(answers)},
//           "latency_ms": round(ms, 1)}

import Foundation

enum KevAnswers {
    /// Python 3.12's `sum()` of floats (bltinmodule.c): the first value added to int 0, then left to right with
    /// Neumaier's compensation, the compensation added at the end when it is nonzero and finite.
    static func pySum(_ values: [Double]) -> Double {
        guard let first = values.first else { return 0 }
        var f = 0.0 + first
        var c = 0.0
        for x in values.dropFirst() {
            let t = f + x
            if abs(f) >= abs(x) {
                c += (f - t) + x
            } else {
                c += (x - t) + f
            }
            f = t
        }
        if c != 0 && c.isFinite { f += c }
        return f
    }

    static func normalize(_ p: [Double]) -> [Double] {
        let t = pySum(p)
        return t == 0 ? [Double](repeating: 1.0 / Double(p.count), count: p.count) : p.map { $0 / t }
    }

    /// The first index of the largest value (Python's max over range(n) with a key: a later equal value never wins).
    static func firstArgmax(_ p: [Double]) -> Int {
        var best = 0
        for i in 1..<p.count where p[i] > p[best] { best = i }
        return best
    }

    /// `kev.api.choice_confidence`: 0 at uniform, 1 at certainty; 1 when K = 1.
    static func choiceConfidence(_ p: [Double]) -> Double {
        let K = p.count
        if K == 1 { return 1.0 }
        let n = normalize(p)
        var m = n[0]
        for v in n.dropFirst() where v > m { m = v }
        return (m - 1.0 / Double(K)) / (1.0 - 1.0 / Double(K))
    }

    /// `kev.api.score_confidence`: max(0, 1 - E|level - mode| / D); 1 when L = 1.
    static func scoreConfidence(_ p: [Double]) -> Double {
        let L = p.count
        if L == 1 { return 1.0 }
        let n = normalize(p)
        let mode = firstArgmax(n)
        let half = Double(L - 1) / 2
        let D = pySum((0..<L).map { abs(Double($0) - half) }) / Double(L)
        let x = 1.0 - pySum(n.enumerated().map { $0.element * Double(abs($0.offset - mode)) }) / D
        return x > 0.0 ? x : 0.0
    }

    static func r4(_ x: Double) -> KevJSON { .double(KevPythonFormat.pyRound(x, 4)) }

    /// `kev.api.to_answers`: per question its SystemOne answer, in request order (`probs` in option order).
    static func answers(probs: [[Float]], meta: [KevQuestionMeta]) -> KevJSON {
        var out: [(String, KevJSON)] = []
        for (pf, m) in zip(probs, meta) {
            let p = pf.map(Double.init)
            switch m.type {
            case "noul":
                out.append((m.id, .obj([("type", .string("noul")), ("noul", r4(p[1]))])))
            case "choice":
                out.append((m.id, .obj([("type", .string("choice")), ("choice", .string(m.keys[firstArgmax(p)])),
                                        ("confidence", r4(choiceConfidence(p))),
                                        ("probabilities", .obj(zip(m.keys, p).map { ($0, r4($1)) }))])))
            default:
                let score = pySum(p.enumerated().map { Double($0.offset) * $0.element })
                out.append((m.id, .obj([("type", .string("score")), ("score", r4(score)),
                                        ("legend", .obj((m.legend ?? []).map { ($0.0, .string($0.1)) })),
                                        ("probabilities", .obj(p.enumerated().map { (String($0.offset), r4($0.element)) })),
                                        ("confidence", r4(scoreConfidence(p)))])))
            }
        }
        return .obj(out)
    }

    /// The /v1/systemone body (kev.serve.Server._body without the truncation fields a default server never sends).
    static func response(model: String, answers: KevJSON, inputTokens: Int, outputTokens: Int,
                                latencyMs: Double) -> KevJSON {
        .obj([("model", .string(model)), ("answers", answers),
              ("usage", .obj([("input_tokens", .int(inputTokens)), ("output_tokens", .int(outputTokens))])),
              ("latency_ms", .double(KevPythonFormat.pyRound(latencyMs, 1)))])
    }
}
