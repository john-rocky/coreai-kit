// TextClassifier.swift — on-device zero-shot text classification (GLiNER2.5-Decide export): any
// tasks and labels at call time, several decisions from one forward. One static graph per sequence
// length S runs DeBERTa-v3-large, gathers the [L] marker row of every label and scores it with the
// 1024-2048-1 label head:
//
//     input_ids[1,S] + attention_mask[1,S] + label_idx[1,MMAX] -> logits[1,MMAX]
//
// The labels are not baked in: this file (the host) linearizes the tasks into gliner2's schema
// layout ahead of the text, so one bundle answers any label set up to MMAX labels per call. The
// collator is gliner2 2.0.0's (`_collate_batch` -> `_transform_record` -> `_format_input_with_mapping`),
// input_ids bit-identical on the zoo's oracle fixtures; the decision is `_extract_classification_result`
// at temperature 1 (single-label: softmax, argmax; multi-label: sigmoid >= threshold, none -> argmax).

@_exported import CoreAIKitCore
@_exported import CoreAIKitVision

import Foundation
import Tokenizers

/// One question to ask of a text: its name, the labels to choose from, and how to choose.
@available(macOS 27, iOS 27, *)
public struct ClassificationTask: Sendable {
    public var name: String
    public var labels: [String]
    /// Every label whose probability (sigmoid) reaches `threshold`, instead of the one best label.
    public var multiLabel: Bool
    /// Multi-label only; when no label reaches it, the best one is returned alone.
    public var threshold: Float
    /// Extra instruction appended to the task name (`name: prompt`), e.g. the question to answer.
    public var prompt: String?
    /// label -> what it means. Written into the schema in `labels` order.
    public var descriptions: [String: String]

    public init(
        _ name: String, labels: [String], multiLabel: Bool = false, threshold: Float = 0.5,
        prompt: String? = nil, descriptions: [String: String] = [:]
    ) {
        self.name = name
        self.labels = labels
        self.multiLabel = multiLabel
        self.threshold = threshold
        self.prompt = prompt
        self.descriptions = descriptions
    }
}

/// The answer to one task.
@available(macOS 27, iOS 27, *)
public struct ClassificationResult: Sendable {
    public let task: String
    /// The chosen labels: one for a single-label task, one or more for a multi-label task.
    public let labels: [String]
    /// Every label with its probability, in the task's label order (softmax over the labels for a
    /// single-label task, an independent sigmoid per label for a multi-label task).
    public let probabilities: [(label: String, probability: Float)]
    /// The text did not fit the largest bundle: words were dropped from its end.
    public let truncated: Bool

    let chosen: [Int]
}

/// Zero-shot classification over a GLiNER2.5-Decide bundle. Await each call before starting the
/// next on the same instance: the calls share the loaded graphs, and overlapping runs on one shared
/// graph are what corrupted InformationExtractor's output.
@available(macOS 27, iOS 27, *)
public final class TextClassifier: @unchecked Sendable {
    public enum ClassificationError: Error, LocalizedError, Sendable {
        case noTasks
        case noLabels(task: String)
        case duplicateTask(String)
        case tooManyLabels(count: Int, max: Int)
        case schemaTooLong(tokens: Int, max: Int)

        public var errorDescription: String? {
            switch self {
            case .noTasks: return "at least one task is needed"
            case .noLabels(let task): return "task '\(task)' has no labels"
            case .duplicateTask(let task): return "task '\(task)' appears twice"
            case .tooManyLabels(let count, let max):
                return "\(count) labels over all tasks; the bundle takes at most \(max) per call"
            case .schemaTooLong(let tokens, let max):
                return "the tasks alone take \(tokens) tokens; the largest bundle holds \(max)"
            }
        }
    }

    /// What `classify` feeds the graph: gliner2's collated input, unpadded, with the marker
    /// positions and the shape it runs on. `collate` returns it without running the graph.
    public struct Collated: Sendable {
        /// Sub-word ids, schema then text (gliner2's `input_ids`; no CLS/SEP in this layout).
        public let inputIds: [Int32]
        /// The piece behind each id (gliner2's `subword_list`). A run of characters the vocab lacks is
        /// one [UNK]: its piece is the whole run in gliner2, the run's first character here.
        public let pieces: [String]
        /// Per task, the linearized schema (gliner2's `schema_tokens_list`).
        public let schemaTokens: [[String]]
        /// The text words that made it in: '.'-terminated, lowercased (gliner2's `text_tokens`).
        public let words: [String]
        /// Per task, the position of its [P] marker.
        public let promptPositions: [Int]
        /// Per task, the position of each label's [L] marker, in label order.
        public let labelPositions: [[Int]]
        /// The S of the bundle this runs on: the smallest one the ids fit.
        public let sequenceLength: Int
        /// Words were dropped from the end of the text to fit the largest bundle.
        public let truncated: Bool
    }

    private struct Config: Decodable {
        struct Shape: Decodable { let S: Int; let bundle: String }
        struct Pad: Decodable { let input_ids: Int }
        let MMAX: Int
        let shapes: [Shape]
        let marker_ids: [String: Int]
        let pad: Pad
    }

    /// The sequence lengths of the bundle's graphs, ascending.
    public let sequenceLengths: [Int]
    /// Labels per call, summed over the tasks.
    public let maxLabels: Int

    private let bundleURL: URL
    private let bundles: [Int: String]
    private let computeUnits: GraphModel.ComputeUnits
    private let tokenizer: any Tokenizer
    // GLiNER markers ([P], [L], [SEP_TEXT], ...) and [MASK] are added tokens outside the Unigram
    // vocab, which maps them to [UNK]; their ids come from classifier.json instead.
    private let markerIds: [String: Int]
    private let padId: Int32
    private let graphs = GraphCache()

    /// Loads a bundle directory holding `classifier.json`, a `tokenizer/` folder and the `.aimodel`
    /// each entry of `classifier.json`'s `shapes` names. A graph loads on its first use.
    ///
    /// The tokenizer folder's `tokenizer_config.json` must name `XLMRobertaTokenizer`:
    /// swift-transformers has no DeBERTa-v2 class, and DeBERTa-v3's SentencePiece Unigram model is
    /// the one it runs under that name.
    public init(bundleAt url: URL, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        let cfg = try JSONDecoder().decode(
            Config.self, from: Data(contentsOf: url.appendingPathComponent("classifier.json")))
        guard !cfg.shapes.isEmpty, cfg.MMAX > 0 else {
            throw VisionError.bundleLayout("classifier.json lists no shapes")
        }
        for m in ["[P]", "[L]", "[SEP_STRUCT]", "[SEP_TEXT]"] where cfg.marker_ids[m] == nil {
            throw VisionError.bundleLayout("classifier.json has no id for \(m)")
        }
        self.bundleURL = url
        self.bundles = Dictionary(cfg.shapes.map { ($0.S, $0.bundle) }, uniquingKeysWith: { a, _ in a })
        self.sequenceLengths = bundles.keys.sorted()
        self.maxLabels = cfg.MMAX
        self.markerIds = cfg.marker_ids
        self.padId = Int32(cfg.pad.input_ids)
        self.computeUnits = computeUnits
        self.tokenizer = try await AutoTokenizer.from(
            modelFolder: url.appendingPathComponent("tokenizer"))
    }

    // MARK: - Classification

    /// Answers every task about `text` in one forward. Returns task name -> result. The labels of
    /// all tasks together may number up to `maxLabels`.
    public func classify(
        _ text: String, tasks: [ClassificationTask]
    ) async throws -> [String: ClassificationResult] {
        let c = try collate(text, tasks: tasks)
        let rows = try await logits(for: c)
        var out: [String: ClassificationResult] = [:]
        for (task, row) in zip(tasks, rows) {
            out[task.name] = Self.decide(row, task: task, truncated: c.truncated)
        }
        return out
    }

    /// One single-label question: the best label and its (softmax) probability.
    public func classify(
        _ text: String, labels: [String], task: String = "label"
    ) async throws -> (label: String, probability: Float) {
        let t = ClassificationTask(task, labels: labels)
        let c = try collate(text, tasks: [t])
        let r = Self.decide(try await logits(for: c)[0], task: t, truncated: c.truncated)
        return r.probabilities[r.chosen[0]]
    }

    /// The raw label scores of a collated input, per task in label order (gliner2's logits before
    /// softmax / sigmoid).
    public func logits(for c: Collated) async throws -> [[Float]] {
        let S = c.sequenceLength
        guard let name = bundles[S] else {
            throw VisionError.bundleLayout("no bundle for S=\(S) (have \(sequenceLengths))")
        }
        let graph = try await graphs.graph(S) { [bundleURL, computeUnits, maxLabels] in
            let g = try await GraphModel(
                contentsOf: bundleURL.appendingPathComponent(name), computeUnits: computeUnits)
            guard g.shape(ofInput: "input_ids") == [1, S], g.shape(ofInput: "attention_mask") == [1, S],
                g.shape(ofInput: "label_idx") == [1, maxLabels], g.outputNames.contains("logits")
            else {
                throw VisionError.bundleLayout(
                    "\(name): unexpected graph contract, inputs \(g.inputNames) outputs \(g.outputNames)")
            }
            return g
        }

        let n = c.inputIds.count
        let ids = c.inputIds + [Int32](repeating: padId, count: S - n)
        let mask = [Int32](repeating: 1, count: n) + [Int32](repeating: 0, count: S - n)
        // label_idx: every task's [L] positions in task order; unused slots repeat the first one
        // (a safe gather nobody reads).
        let flat = c.labelPositions.flatMap { $0 }
        var labelIdx = [Int32](repeating: Int32(flat[0]), count: maxLabels)
        for (k, p) in flat.enumerated() { labelIdx[k] = Int32(p) }

        let outputs = try await graph.run([
            "input_ids": .int32(ids, shape: [1, S]),
            "attention_mask": .int32(mask, shape: [1, S]),
            "label_idx": .int32(labelIdx, shape: [1, maxLabels]),
        ])
        guard let row = outputs["logits"]?.floats() else { throw VisionError.missingOutput("logits") }
        var rows: [[Float]] = []
        var start = 0
        for positions in c.labelPositions {
            rows.append(Array(row[start..<(start + positions.count)]))
            start += positions.count
        }
        return rows
    }

    /// gliner2's decision on one task's logits. Probabilities are computed in Float64 in NumPy's
    /// summation order, so they match the zoo's Python host bit for bit.
    public static func decide(
        _ logits: [Float], task: ClassificationTask, truncated: Bool = false
    ) -> ClassificationResult {
        let x = logits.map(Double.init)
        let probs: [Double]
        if task.multiLabel {
            probs = x.map { 1.0 / (1.0 + exp(-$0)) }
        } else {
            let m = nanMax(x)
            let e = x.map { exp($0 - m) }
            let s = pairwiseSum(e[...])
            probs = e.map { $0 / s }
        }
        var chosen: [Int]
        if task.multiLabel {
            // The threshold as written (0.4, not Float(0.4) = 0.40000000596): gliner2 compares with
            // a Python float.
            let thr = Double("\(task.threshold)") ?? Double(task.threshold)
            chosen = probs.indices.filter { probs[$0] >= thr }
            if chosen.isEmpty { chosen = [argmax(probs)] }
        } else {
            chosen = [argmax(probs)]
        }
        return ClassificationResult(
            task: task.name, labels: chosen.map { task.labels[$0] },
            probabilities: zip(task.labels, probs).map { (label: $0, probability: Float($1)) },
            truncated: truncated, chosen: chosen)
    }

    // MARK: - Collation (gliner2 2.0.0, classification path)

    /// The graph input for `text` and `tasks`, without running the graph.
    public func collate(_ text: String, tasks: [ClassificationTask]) throws -> Collated {
        guard !tasks.isEmpty else { throw ClassificationError.noTasks }
        var names = Set<String>()
        for t in tasks {
            guard !t.labels.isEmpty else { throw ClassificationError.noLabels(task: t.name) }
            guard names.insert(t.name).inserted else { throw ClassificationError.duplicateTask(t.name) }
        }
        let labelCount = tasks.reduce(0) { $0 + $1.labels.count }
        guard labelCount <= maxLabels else {
            throw ClassificationError.tooManyLabels(count: labelCount, max: maxLabels)
        }

        var ids: [Int32] = []
        var pieces: [String] = []
        func append(_ ps: [String]) {
            for p in ps {
                pieces.append(p)
                ids.append(pieceId(p))
            }
        }

        // ( [P] prompt ( [L] l1 [L] l2 ... ) ) [SEP_STRUCT] ( ... ) [SEP_TEXT] words; each piece
        // tokenized on its own. The routed markers are [P] at 1 and [L] at 4, 6, ..., count - 3.
        let schemaTokens = tasks.map(Self.schemaTokens(for:))
        var promptPositions: [Int] = []
        var labelPositions: [[Int]] = []
        for (j, tokens) in schemaTokens.enumerated() {
            if j > 0 { append(tokenize("[SEP_STRUCT]")) }
            var positions: [Int] = []
            for (k, token) in tokens.enumerated() {
                let pos = ids.count
                append(tokenize(token))
                if k == 1 {
                    promptPositions.append(pos)
                } else if k >= 4, k < tokens.count - 2, k % 2 == 0 {
                    positions.append(pos)
                }
            }
            labelPositions.append(positions)
        }
        append(tokenize("[SEP_TEXT]"))

        let largest = sequenceLengths.last!
        guard ids.count <= largest else {
            throw ClassificationError.schemaTooLong(tokens: ids.count, max: largest)
        }
        // Word-level truncation, like gliner2's max_len: keep the longest prefix of words that fits.
        var words: [String] = []
        var truncated = false
        for w in Self.words(text) {
            let ps = tokenize(w)
            if ids.count + ps.count > largest {
                truncated = true
                break
            }
            append(ps)
            words.append(w)
        }
        let S = sequenceLengths.first { ids.count <= $0 }!
        return Collated(
            inputIds: ids, pieces: pieces, schemaTokens: schemaTokens, words: words,
            promptPositions: promptPositions, labelPositions: labelPositions, sequenceLength: S,
            truncated: truncated)
    }

    /// One task as gliner2's `_transform_schema` writes it at inference: the prompt string gets
    /// `: prompt` when there is one, then ` [DESCRIPTION] label: description` for each described
    /// label (once per label, in label order); every label follows an [L].
    static func schemaTokens(for task: ClassificationTask) -> [String] {
        var prompt = task.name
        if let p = task.prompt, !p.isEmpty { prompt += ": \(p)" }
        var described = Set<String>()
        for label in task.labels where described.insert(label).inserted {
            if let d = task.descriptions[label] { prompt += " [DESCRIPTION] \(label): \(d)" }
        }
        var tokens = ["(", "[P]", prompt, "("]
        for label in task.labels { tokens += ["[L]", label] }
        return tokens + [")", ")"]
    }

    /// The text words: '.' appended unless the text ends in . ! ? (empty -> "."), split with
    /// gliner2's WhitespaceTokenSplitter, each word lowercased as Python's `str.lower` does.
    static func words(_ text: String) -> [String] {
        var t = text
        if let last = t.unicodeScalars.last {
            if !(last == "." || last == "!" || last == "?") { t += "." }
        } else {
            t = "."
        }
        let ns = t as NSString
        var out: [String] = []
        wordRegex.enumerateMatches(in: t, range: NSRange(location: 0, length: ns.length)) { m, _, _ in
            guard let r = m?.range else { return }
            out.append(pythonLowercased(ns.substring(with: r)))
        }
        return out
    }

    // gliner2's WhitespaceTokenSplitter._PATTERN: URLs / emails / @handles (case-insensitive) |
    // words joined by - or _ | any other single non-space. \w and \s are spelled out as Python's:
    // ICU's \w also takes marks and joiners but not ² or ½, and its \s misses \v, \x1c-\x1f and
    // \x85. [\p{L}\p{N}_] equals Python's \w on every code point (Unicode 15, gliner2's Python 3.12).
    private static let wordRegex: NSRegularExpression = {
        let space = "\\t\\n\\x{0B}\\f\\r\\x{1C}-\\x{1F} \\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}"
            + "\\x{2028}\\x{2029}\\x{202F}\\x{205F}\\x{3000}"
        let word = "[\\p{L}\\p{N}_]"
        let pattern =
            "(?i:https?://[^\(space)]+|www\\.[^\(space)]+)"
            + "|(?i:[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,})"
            + "|(?i:@[a-z0-9_]+)"
            + "|\(word)+(?:[-_]\(word)+)*"
            + "|[^\(space)]"
        return try! NSRegularExpression(pattern: pattern)
    }()

    /// Python's `str.lower`: each scalar's full lowercase mapping, and capital sigma as final sigma
    /// where Unicode's Final_Sigma context holds (Swift's `lowercased()` maps every Σ to σ).
    static func pythonLowercased(_ s: String) -> String {
        let scalars = Array(s.unicodeScalars)
        var out = String.UnicodeScalarView()
        for (i, c) in scalars.enumerated() {
            guard c == "\u{03A3}" else {
                out.append(contentsOf: c.properties.lowercaseMapping.unicodeScalars)
                continue
            }
            var j = i - 1
            while j >= 0, scalars[j].properties.isCaseIgnorable { j -= 1 }
            var final = j >= 0 && scalars[j].properties.isCased
            if final {
                var k = i + 1
                while k < scalars.count, scalars[k].properties.isCaseIgnorable { k += 1 }
                final = k == scalars.count || !scalars[k].properties.isCased
            }
            out.append(final ? "\u{03C2}" : "\u{03C3}")
        }
        return String(out)
    }

    /// `tokenizer.tokenize(piece)` as the HF fast tokenizer gives it. A piece its normalizer strips
    /// to nothing (empty or all whitespace) has no tokens there; swift-transformers' Metaspace would
    /// still emit a lone "▁" for it.
    private func tokenize(_ piece: String) -> [String] {
        if piece.unicodeScalars.allSatisfy(\.properties.isWhitespace) { return [] }
        return tokenizer.tokenize(text: piece)
    }

    private func pieceId(_ piece: String) -> Int32 {
        if let id = markerIds[piece] { return Int32(id) }
        return Int32(tokenizer.convertTokenToId(piece) ?? tokenizer.unknownTokenId ?? 3)
    }

    // MARK: - numerics (NumPy's, so the decision matches the Python host)

    /// numpy.add.reduce over a contiguous float64 vector: pairwise_sum (below 8 elements a plain
    /// loop from -0.0, up to 128 eight accumulators, above that halves).
    static func pairwiseSum(_ a: ArraySlice<Double>) -> Double {
        let n = a.count, b = a.startIndex
        if n < 8 {
            var res = -0.0
            for x in a { res += x }
            return res
        }
        if n <= 128 {
            var r = Array(a[b..<(b + 8)])
            var i = 8
            while i < n - (n % 8) {
                for j in 0..<8 { r[j] += a[b + i + j] }
                i += 8
            }
            var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while i < n {
                res += a[b + i]
                i += 1
            }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return pairwiseSum(a[b..<(b + n2)]) + pairwiseSum(a[(b + n2)...])
    }

    /// numpy.max: NaN if any element is NaN.
    static func nanMax(_ xs: [Double]) -> Double {
        var m = -Double.infinity
        for x in xs {
            if x.isNaN { return .nan }
            m = max(m, x)
        }
        return m
    }

    /// numpy.argmax: the first maximum, or the first NaN.
    static func argmax(_ xs: [Double]) -> Int {
        guard !xs.isEmpty else { return 0 }
        if let k = xs.firstIndex(where: { $0.isNaN }) { return k }
        var best = 0
        for k in 1..<xs.count where xs[k] > xs[best] { best = k }
        return best
    }
}

/// One graph per S, loaded on first use; concurrent first calls share one load.
@available(macOS 27, iOS 27, *)
private actor GraphCache {
    private var loads: [Int: Task<GraphModel, Error>] = [:]

    func graph(_ s: Int, load: @escaping @Sendable () async throws -> GraphModel) async throws -> GraphModel {
        if let t = loads[s] { return try await t.value }
        let t = Task { try await load() }
        loads[s] = t
        do {
            return try await t.value
        } catch {
            loads[s] = nil
            throw error
        }
    }
}
