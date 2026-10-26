// TailReport — the summary of `WeVisOneDecoder.tailBench`: per tail length r, each form's median, min and max seconds,
// the ratio, and whether the two forms agree (the next token's logits: cosine, max |Δ|, argmax; the 16 greedy ids after).

import Foundation

@available(macOS 27, iOS 27, *)
public enum TailReport {
    static func median(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        guard !s.isEmpty else { return .nan }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    public static func make(_ runs: [WeVisOneDecoder.TailRun], promptLength: Int) -> [String: Any] {
        var summary: [[String: Any]] = []
        for r in Array(Set(runs.map(\.r))).sorted() {
            let mine = runs.filter { $0.r == r }
            let s1 = mine.filter { $0.form == "s1" }, pad = mine.filter { $0.form == "pad" }
            let ms1 = s1.map { $0.seconds * 1e3 }, mpad = pad.map { $0.seconds * 1e3 }
            var agree: [String: Any] = [:]
            if let a = s1.first(where: { !$0.logits.isEmpty }), let b = pad.first(where: { !$0.logits.isEmpty }) {
                var dot = 0.0, na = 0.0, nb = 0.0, maxd = 0.0
                for i in 0..<a.logits.count {
                    let x = Double(a.logits[i]), y = Double(b.logits[i])
                    dot += x * y; na += x * x; nb += y * y
                    maxd = max(maxd, abs(x - y))
                }
                let cos: Double = dot / (na.squareRoot() * nb.squareRoot())
                agree["cos"] = cos
                agree["max_abs_d"] = maxd
                agree["argmax_equal"] = a.argmax == b.argmax
                agree["follow_equal"] = a.follow == b.follow
                agree["follow_s1"] = a.follow
                agree["follow_pad"] = b.follow
            }
            let s1Stats: [String: Double] = ["median": median(ms1), "min": ms1.min() ?? .nan, "max": ms1.max() ?? .nan]
            let padStats: [String: Double] = ["median": median(mpad), "min": mpad.min() ?? .nan, "max": mpad.max() ?? .nan]
            var entry: [String: Any] = [:]
            entry["r"] = r
            entry["prefix"] = mine.first?.prefix ?? 0
            entry["n_s1"] = ms1.count
            entry["n_pad"] = mpad.count
            entry["s1_ms"] = s1Stats
            entry["pad_ms"] = padStats
            entry["s1_over_pad"] = median(ms1) / median(mpad)
            entry["agree"] = agree
            summary.append(entry)
        }
        return ["prompt_length": promptLength, "summary": summary,
                "runs": runs.map { ["r": $0.r, "form": $0.form, "ms": $0.seconds * 1e3, "argmax": $0.argmax] }]
    }
}
