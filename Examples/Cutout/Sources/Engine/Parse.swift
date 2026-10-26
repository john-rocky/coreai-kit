// Parse — the boxes in an answer, by the package's rules (engine.py, Tencent/WeVisOne-VL 40f735b8):
//
//   parse_seg_boxes   the text up to <|im_end|>; each <box>…</box> (or <box>… to the end); in it, each stretch before
//                     an <OBJ>; its coordinates <n> (1–4 digits); the last four, x1 y1 x2 y2 in 0 … 1000 with x2 > x1
//                     and y2 > y1, are one box (/ 1000) with a matte — the cut-out answer
//   detection boxes   the same box tags without <OBJ>: the last four coordinates of each <box>, for the boxes answer
//
// The model answers "None" when nothing in the photo matches the words: no box, nothing to cut out.

import Foundation

public enum AnswerParse {
    static let boxRE = try! NSRegularExpression(pattern: "<box>(.*?)(?:</box>|$)", options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let coordRE = try! NSRegularExpression(pattern: "<(\\d{1,4})>")
    static let segRE = try! NSRegularExpression(pattern: "<OBJ>", options: [.caseInsensitive])

    static func matches(_ re: NSRegularExpression, _ s: String) -> [NSTextCheckingResult] {
        re.matches(in: s, range: NSRange(s.startIndex..., in: s))
    }

    static func group(_ m: NSTextCheckingResult, _ i: Int, _ s: String) -> String {
        guard let r = Range(m.range(at: i), in: s) else { return "" }
        return String(s[r])
    }

    static func bodies(_ text: String) -> [String] {
        let t = text.components(separatedBy: "<|im_end|>")[0]
        return matches(boxRE, t).map { group($0, 1, t) }
    }

    static func lastFour(_ s: String) -> [Int]? {
        let numbers = matches(coordRE, s).compactMap { Int(group($0, 1, s)) }
        guard numbers.count >= 4 else { return nil }
        let b = Array(numbers.suffix(4))
        guard b.allSatisfy({ (0...1000).contains($0) }), b[2] > b[0], b[3] > b[1] else { return nil }
        return b
    }

    /// `parse_seg_boxes`: the boxes followed by <OBJ>, as integers 0 … 1000 (x1, y1, x2, y2).
    public static func segBoxes(_ text: String) -> [[Int]] {
        var out: [[Int]] = []
        for body in bodies(text) {
            // re.split(<OBJ>)[:-1]: every stretch that an <OBJ> ends
            var pieces: [String] = []
            var last = body.startIndex
            for m in matches(segRE, body) {
                guard let r = Range(m.range, in: body) else { continue }
                pieces.append(String(body[last..<r.lowerBound]))
                last = r.upperBound
            }
            for p in pieces { if let b = lastFour(p) { out.append(b) } }
        }
        return out
    }

    /// The boxes of a detection answer: the last four coordinates of each box tag.
    public static func detectionBoxes(_ text: String) -> [[Int]] {
        bodies(text).compactMap { body in
            // a tag can hold several boxes separated by commas (the visual-prompt form); read each
            let parts = body.components(separatedBy: ",")
            let boxes = parts.compactMap(lastFour)
            return boxes.isEmpty ? nil : boxes
        }.flatMap { $0 }
    }

    /// Every box in the answer with the label the model wrote before it (`<ref>label</ref><box>…</box>`) and whether an
    /// <OBJ> follows it, in order. The cut-out boxes are the `obj` ones, in the order `segBoxes` reads them.
    public static func labeled(_ text: String) -> [(label: String, box: [Int], obj: Bool)] {
        let t = text.components(separatedBy: "<|im_end|>")[0]
        let re = try! NSRegularExpression(pattern: "(?:<ref>(.*?)</ref>)?\\s*<box>(.*?)(?:</box>|$)",
                                          options: [.caseInsensitive, .dotMatchesLineSeparators])
        var out: [(String, [Int], Bool)] = []
        var label = ""
        for m in matches(re, t) {
            let l = group(m, 1, t)
            if !l.isEmpty { label = l }
            for part in group(m, 2, t).components(separatedBy: ",") {
                let hasObj = matches(segRE, part).count > 0
                let before = hasObj ? String(part[..<(Range(matches(segRE, part)[0].range, in: part)!.lowerBound)]) : part
                if let b = lastFour(before) { out.append((label, b, hasObj)) }
            }
        }
        return out
    }

    /// fp32 of x / 1000 (the package's float division, then the tensor's float32).
    public static func unit(_ box: [Int]) -> [Float] { box.map { Float(Double($0) / 1000.0) } }
}
