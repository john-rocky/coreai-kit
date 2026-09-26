// TextClassifierTests.swift — TextClassifier's host without a tokenizer or weights: the schema it
// linearizes and the words it splits the text into, against gliner2 2.0.0 on the 21 classify_text
// examples of the GLiNER2.5-Decide card (Fixtures/gliner25_decide_readme21_schema.json, from the zoo's
// oracle fixture: schema_tokens_list as gliner2 collated it, words = gliner2's WhitespaceTokenSplitter
// on the '.'-terminated text), plus the decision rule.
//
// The tokenized input (input_ids and [L] positions on all 454 oracle cases) and the graph's decisions
// need the bundle; Examples/TextClassify's `textclassify-cli --gate` checks those.

import Foundation
import Testing

@testable import CoreAIKitEmbeddings

struct TextClassifierTests {
    /// One task of a case: gliner2's classify_text input, `[labels]` or
    /// `{"labels": [..] | {label: description}, "multi_label", "cls_threshold", "prompt"}`.
    struct Spec: Decodable {
        var labels: [String]?
        var descriptions: [String: String] = [:]
        var multiLabel = false
        var threshold = 0.5
        var prompt: String?

        enum K: String, CodingKey { case labels, multi_label, cls_threshold, prompt }

        init(from decoder: Decoder) throws {
            if let list = try? decoder.singleValueContainer().decode([String].self) {
                labels = list
                return
            }
            let c = try decoder.container(keyedBy: K.self)
            if let list = try? c.decode([String].self, forKey: .labels) {
                labels = list
            } else {
                descriptions = try c.decode([String: String].self, forKey: .labels)
            }
            multiLabel = try c.decodeIfPresent(Bool.self, forKey: .multi_label) ?? false
            threshold = try c.decodeIfPresent(Double.self, forKey: .cls_threshold) ?? 0.5
            prompt = try c.decodeIfPresent(String.self, forKey: .prompt)
        }
    }

    struct Case: Decodable {
        let id: String
        let text_raw: String
        let tasks: [String: Spec]
        // Foundation drops JSON key order; the oracle's task_results order is gliner2's schema order.
        let task_order: [String]
        let label_order: [[String]]
        let schema_tokens_list: [[String]]
        let words: [String]
    }

    struct Fixture: Decodable { let cases: [Case] }

    static func readme21() throws -> [Case] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("Fixtures/gliner25_decide_readme21_schema.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url)).cases
    }

    /// Scalar for scalar, as Python compares (Swift's == is canonical equivalence).
    static func scalars(_ xs: [String]) -> [[Unicode.Scalar]] { xs.map { Array($0.unicodeScalars) } }

    @available(macOS 27, iOS 27, *)
    @Test func theSchemaIsGliner2sOnTheCardExamples() throws {
        let cases = try Self.readme21()
        #expect(cases.count == 21)
        for c in cases {
            #expect(Set(c.tasks.keys) == Set(c.task_order) && c.task_order.count == c.schema_tokens_list.count, "\(c.id)")
            for (j, name) in c.task_order.enumerated() {
                let spec = try #require(c.tasks[name])
                let labels = c.label_order[j]
                #expect(spec.labels.map { $0 == labels } ?? (Set(spec.descriptions.keys) == Set(labels)), "\(c.id)")
                let task = ClassificationTask(
                    name, labels: labels, multiLabel: spec.multiLabel, threshold: Float(spec.threshold),
                    prompt: spec.prompt, descriptions: spec.descriptions)
                #expect(Self.scalars(TextClassifier.schemaTokens(for: task)) == Self.scalars(c.schema_tokens_list[j]),
                        "\(c.id) \(name)")
            }
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func theWordsAreGliner2sOnTheCardExamples() throws {
        for c in try Self.readme21() {
            #expect(Self.scalars(TextClassifier.words(c.text_raw)) == Self.scalars(c.words), "\(c.id)")
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func wordsSplitAndLowercaseAsPythonDoes() {
        // '.' unless the text already ends in . ! ?; an empty text is "."
        #expect(TextClassifier.words("") == ["."])
        #expect(TextClassifier.words("Is it?") == ["is", "it", "?"])
        #expect(TextClassifier.words("Mail A@B.io or see https://X.io/a now") == ["mail", "a@b.io", "or", "see", "https://x.io/a", "now", "."])
        // Python's \w takes ² (category No): "mm²" is one word. ICU's \w would split off the ².
        #expect(TextClassifier.words("0.21 mm² per s") == ["0", ".", "21", "mm²", "per", "s", "."])
        // Python's \w does not take combining marks: a decomposed é is two words.
        #expect(Self.scalars(TextClassifier.words("e\u{301}")) == Self.scalars(["e", "\u{301}", "."]))
        // str.lower keeps a final sigma final; Swift's lowercased() would not.
        #expect(TextClassifier.pythonLowercased("ΟΔΟΣ ΣΑΣ") == "οδος σας")
        #expect(TextClassifier.pythonLowercased("İ") == "i\u{307}")
    }

    @available(macOS 27, iOS 27, *)
    @Test func theDecisionRuleIsGliner2s() {
        let single = ClassificationTask("t", labels: ["a", "b", "c"])
        let s = TextClassifier.decide([0.1, 2.0, -1.0], task: single)
        #expect(s.labels == ["b"] && s.probabilities.map(\.label) == ["a", "b", "c"])
        #expect(abs(s.probabilities.map(\.probability).reduce(0, +) - 1) < 1e-6)

        var multi = ClassificationTask("m", labels: ["a", "b", "c"], multiLabel: true, threshold: 0.4)
        // sigmoid(Float(-0.4054651)) = 0.4000000028: over 0.4 as written, under Float(0.4) = 0.4000000060
        #expect(TextClassifier.decide([2.0, -0.4054651, -3.0], task: multi).labels == ["a", "b"])
        multi.threshold = 0.99
        #expect(TextClassifier.decide([0.5, 1.5, -3.0], task: multi).labels == ["b"])   // none reach it -> argmax
    }
}
