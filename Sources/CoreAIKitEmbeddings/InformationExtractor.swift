// InformationExtractor.swift — on-device schema-driven information extraction (GLiNER2 export):
// zero-shot named-entity / PII extraction with an arbitrary label set supplied at call time. The
// flagship use is PII redaction. One static fused graph runs mDeBERTa-v3 -> "first" word pooling
// -> SpanMarker -> CountLSTM -> einsum -> sigmoid, emitting span_scores[1, MMAX, T, K]; the host
// collator (this file) tokenizes the text, linearizes the label schema into the graph's input_ids,
// supplies the gather indices, and decodes spans (per-field threshold + confidence-desc greedy NMS).
//
// The graph is schema-agnostic: the label set is NOT baked in — pass any labels per call (up to
// MMAX). Competitor GLiNER2Swift is macOS-CPU/MLX only; this runs the GPU (and, once AOT-compiled,
// the ANE) on iPhone too. The Swift collator is byte-identical to GLiNER2's Python
// `collate_fn_inference`, and decode is byte-identical to `_format_spans`, so `extract` reproduces
// `ext.extract` exactly (verified on the demo PII suite).

@_exported import CoreAIKitCore
@_exported import CoreAIKitVision

import Foundation
import Tokenizers

public final class InformationExtractor: @unchecked Sendable {
    private let graph: GraphModel
    private let tokenizer: any Tokenizer

    private let idsInput: String
    private let maskInput: String
    private let wordInput: String
    private let schemaInput: String
    private let outputName: String

    private let seqLen: Int          // S — padded input_ids length (e.g. 256)
    private let maxWords: Int        // T — padded word-gather length (e.g. 96)
    private let maxLabels: Int       // MMAX — max schema fields (e.g. 16)
    private let spanWidth: Int       // K — max span width (e.g. 8)
    private let defaultThreshold: Float

    // GLiNER special markers live in added_tokens ABOVE the Unigram vocab, so the Unigram model
    // maps them to [UNK]; the collator emits their ids straight from this map instead.
    private let markerIds: [String: Int]      // "[P]","[E]","[SEP_TEXT]" -> id
    private let poolIds: Set<Int>             // {[P],[C],[E],[R],[L]} -> triggers schema pooling

    // WhitespaceTokenSplitter._PATTERN (GLiNER2), applied to lowercased text. URLs / emails /
    // @handles / hyphen-or-underscore-joined words / any single non-space char.
    private static let wordRegex = try! NSRegularExpression(
        pattern:
            "(?:https?://[^\\s]+|www\\.[^\\s]+)" +
            "|[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}" +
            "|@[a-z0-9_]+" +
            "|\\w+(?:[-_]\\w+)*" +
            "|\\S",
        options: [.caseInsensitive])

    private struct ExtractorConfig: Decodable {
        let S: Int; let T: Int; let MMAX: Int; let K: Int; let threshold: Float
        let marker_ids: [String: Int]; let pool_ids: [Int]
    }

    /// Loads a bundle directory holding one `*.aimodel`, a `tokenizer/` folder, and `extractor.json`.
    public init(bundleAt url: URL, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        guard let modelURL = try FileManager.default.contentsOfDirectory(
            at: url, includingPropertiesForKeys: nil
        ).first(where: { $0.pathExtension == "aimodel" }) else {
            throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
        }
        let cfg = try JSONDecoder().decode(
            ExtractorConfig.self,
            from: Data(contentsOf: url.appendingPathComponent("extractor.json")))
        self.graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.tokenizer = try await AutoTokenizer.from(
            modelFolder: url.appendingPathComponent("tokenizer"))

        guard
            let ids = graph.inputNames.first(where: { $0.contains("input_ids") }),
            let mask = graph.inputNames.first(where: { $0.contains("attention_mask") }),
            let word = graph.inputNames.first(where: { $0.contains("text_word_idx") }),
            let schema = graph.inputNames.first(where: { $0.contains("schema_idx") }),
            let out = graph.outputNames.first
        else {
            throw VisionError.bundleLayout(
                "unexpected extractor graph contract: in \(graph.inputNames) out \(graph.outputNames)")
        }
        self.idsInput = ids; self.maskInput = mask; self.wordInput = word
        self.schemaInput = schema; self.outputName = out
        self.seqLen = cfg.S; self.maxWords = cfg.T; self.maxLabels = cfg.MMAX; self.spanWidth = cfg.K
        self.defaultThreshold = cfg.threshold
        self.markerIds = cfg.marker_ids
        self.poolIds = Set(cfg.pool_ids)
    }

    /// Downloads the bundle from the Hub if needed (or uses a sideloaded copy already at the store
    /// path), then loads it. Defaults to the GLiNER2-PII fp16 bundle.
    public convenience init(
        model: ModelID = .gliner2PII,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits)
    }

    // MARK: - Extraction

    /// One detected entity: its label, the matched substring, its character range in the original
    /// `text` (for highlighting / in-place redaction), and the model's confidence.
    public struct EntitySpan: Sendable {
        public let label: String
        public let text: String
        public let range: Range<String.Index>
        public let confidence: Float
    }

    /// Extracts entities for the given labels from `text`. Returns label -> matched spans (verbatim
    /// substrings of `text`), in confidence-descending, non-overlapping order. Pass any labels you
    /// like (zero-shot), up to the bundle's MMAX. `threshold` overrides the bundle default.
    public func extract(
        from text: String, entities labels: [String], threshold: Float? = nil
    ) async throws -> [String: [String]] {
        var out: [String: [String]] = [:]
        for g in try await extractGrouped(from: text, entities: labels, threshold: threshold)
        where !g.spans.isEmpty {
            out[g.label] = g.spans.map { $0.text }
        }
        return out
    }

    /// Like `extract`, but returns every matched span with its character range in `text` and its
    /// confidence — what a highlighting / redaction UI needs. A token may match more than one label
    /// (per-label scoring), so ranges can overlap across labels; use `redact` for a single
    /// non-overlapping rewrite.
    public func extractSpans(
        from text: String, entities labels: [String], threshold: Float? = nil
    ) async throws -> [EntitySpan] {
        try await extractGrouped(from: text, entities: labels, threshold: threshold)
            .flatMap { $0.spans }
    }

    /// Rewrites `text` with every detected entity replaced. Cross-label overlaps are resolved by
    /// confidence (highest wins). `replacement` maps a span to its replacement (default `[LABEL]`);
    /// pass e.g. `{ _ in "██████" }` for block redaction.
    public func redact(
        _ text: String, entities labels: [String], threshold: Float? = nil,
        replacement: (EntitySpan) -> String = { "[\($0.label.uppercased())]" }
    ) async throws -> String {
        let spans = try await extractSpans(from: text, entities: labels, threshold: threshold)
        var kept: [EntitySpan] = []
        for s in spans.sorted(by: { $0.confidence > $1.confidence })
        where !kept.contains(where: { $0.range.overlaps(s.range) }) {
            kept.append(s)
        }
        var result = text
        for s in kept.sorted(by: { $0.range.lowerBound > $1.range.lowerBound }) {
            result.replaceSubrange(s.range, with: replacement(s))
        }
        return result
    }

    private func extractGrouped(
        from text: String, entities labels: [String], threshold: Float?
    ) async throws -> [(label: String, spans: [EntitySpan])] {
        precondition(!labels.isEmpty && labels.count <= maxLabels,
                     "labels must be 1...\(maxLabels)")
        let thr = threshold ?? defaultThreshold
        let c = collate(text: text, labels: labels)

        let outputs = try await graph.run([
            idsInput: .int32(c.inputIds, shape: [1, seqLen]),
            maskInput: .int32(c.mask, shape: [1, seqLen]),
            wordInput: .int32(c.wordIdx, shape: [1, maxWords]),
            schemaInput: .int32(c.schemaIdx, shape: [1, maxLabels + 1]),
        ])
        guard let scores = outputs[outputName]?.floats() else {
            throw VisionError.missingOutput(outputName)
        }
        return decodeGrouped(scores: scores, labels: labels, text: text,
                             startMap: c.startMap, endMap: c.endMap, threshold: thr)
    }

    // MARK: - Collation (byte-identical to GLiNER2 SchemaTransformer, entities-only inference path)

    private struct Collated {
        let inputIds: [Int32]; let mask: [Int32]; let wordIdx: [Int32]; let schemaIdx: [Int32]
        let startMap: [Int]; let endMap: [Int]
    }

    private func wordSplit(_ text: String) -> [(String, Int, Int)] {
        let lower = text.lowercased()
        let ns = lower as NSString
        var out: [(String, Int, Int)] = []
        Self.wordRegex.enumerateMatches(in: lower, range: NSRange(location: 0, length: ns.length)) {
            m, _, _ in
            guard let r = m?.range else { return }
            out.append((ns.substring(with: r), r.location, r.location + r.length))
        }
        return out
    }

    private func pieceToId(_ piece: String) -> Int32 {
        if let sid = markerIds[piece] { return Int32(sid) }        // GLiNER markers (beyond vocab)
        return Int32(tokenizer.convertTokensToIds([piece]).first.flatMap { $0 } ?? 3)  // 3 = [UNK]
    }

    private func collate(text: String, labels: [String]) -> Collated {
        // schema linearization (entities-only, no descriptions): ( [P] entities ( [E] l0 [E] l1 ... ) )
        var schemaTokens: [String] = ["(", "[P]", "entities", "("]
        for l in labels { schemaTokens.append("[E]"); schemaTokens.append(l) }
        schemaTokens.append(")"); schemaTokens.append(")")

        var words = wordSplit(text)
        if words.count > maxWords { words = Array(words.prefix(maxWords)) }  // word-level truncation

        // combined token stream, tagged: 0=schema, 1=sep, 2=text
        var combined: [(String, Int)] = schemaTokens.map { ($0, 0) }
        combined.append(("[SEP_TEXT]", 1))
        for w in words { combined.append((w.0, 2)) }

        var subwords: [Int32] = []
        var textWordFirst: [Int] = []
        var schemaSpecial: [Int] = []
        for (token, seg) in combined {
            let pieces = tokenizer.tokenize(text: token)
            let pos = subwords.count
            if seg == 2 {  // drop trailing text words that would overflow the S budget
                let ids = pieces.map { pieceToId($0) }
                if subwords.count + ids.count > seqLen { break }
                subwords.append(contentsOf: ids)
                if !pieces.isEmpty { textWordFirst.append(pos) }
            } else {
                for p in pieces { subwords.append(pieceToId(p)) }
                if seg == 0, let first = pieces.first, poolIds.contains(Int(pieceToId(first))) {
                    schemaSpecial.append(pos)
                }
            }
        }

        let startMap = Array(words.prefix(textWordFirst.count)).map { $0.1 }
        let endMap = Array(words.prefix(textWordFirst.count)).map { $0.2 }

        let inputIds = pad(subwords, to: seqLen, with: 0)
        let mask = pad([Int32](repeating: 1, count: subwords.count), to: seqLen, with: 0)
        let wordIdx = pad(textWordFirst.map(Int32.init), to: maxWords, with: 0)
        // schema_idx = [P_pos] + field positions, padded with the [P] position (a safe gather).
        let schemaIdx = pad(schemaSpecial.map(Int32.init), to: maxLabels + 1,
                            with: Int32(schemaSpecial.first ?? 0))
        return Collated(inputIds: inputIds, mask: mask, wordIdx: wordIdx, schemaIdx: schemaIdx,
                        startMap: startMap, endMap: endMap)
    }

    private func pad(_ v: [Int32], to n: Int, with fill: Int32) -> [Int32] {
        v.count >= n ? Array(v.prefix(n)) : v + [Int32](repeating: fill, count: n - v.count)
    }

    // MARK: - Decode (byte-identical to GLiNER2 _find_spans + _format_spans)

    private func decodeGrouped(
        scores: [Float], labels: [String], text: String,
        startMap: [Int], endMap: [Int], threshold: Float
    ) -> [(label: String, spans: [EntitySpan])] {
        let ns = text as NSString
        let tl = startMap.count                       // real words (== textWordFirst.count)
        let strideM = maxWords * spanWidth            // graph output is [MMAX, T, K], row-major
        var out: [(label: String, spans: [EntitySpan])] = []
        for (idx, name) in labels.enumerated() {
            var cand: [(text: String, conf: Float, cs: Int, ce: Int, ord: Int)] = []
            var ord = 0
            for start in 0..<tl {
                for width in 0..<spanWidth {
                    let end = start + width + 1
                    if end > tl { ord += 1; continue }
                    let conf = scores[idx * strideM + start * spanWidth + width]
                    if conf >= threshold {
                        let cs = startMap[start], ce = endMap[end - 1]
                        let raw = ns.substring(with: NSRange(location: cs, length: ce - cs))
                        let span = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !span.isEmpty { cand.append((span, conf, cs, ce, ord)) }
                    }
                    ord += 1
                }
            }
            // confidence desc, stable (ascending ord breaks ties == Python's stable sort)
            cand.sort { $0.conf != $1.conf ? $0.conf > $1.conf : $0.ord < $1.ord }
            var sel: [(text: String, conf: Float, cs: Int, ce: Int)] = []
            for x in cand where !sel.contains(where: { !(x.ce <= $0.cs || x.cs >= $0.ce) }) {
                sel.append((x.text, x.conf, x.cs, x.ce))
            }
            let spans = sel.map { s in
                EntitySpan(label: name, text: s.text,
                           range: Range(NSRange(location: s.cs, length: s.ce - s.cs), in: text)!,
                           confidence: s.conf)
            }
            out.append((name, spans))
        }
        return out
    }
}
