// SafetyClassifier.swift — on-device content moderation whose policy is a string you write,
// over a static classifier graph (Shieldstral-1.0-3B export: the scoring tail — gather the last
// real token, two rows of the tied embedding as the head, softmax over {no, yes} — is baked
// in-graph, so one forward returns P(violation)).
//
// The shape is `Reranker`'s, because the model is: both answer a yes/no question about a
// document at the last prompt token. What differs is what the question is for. A reranker asks
// "is this relevant"; this asks "does this break the rule I just wrote", and the rule is an
// argument rather than a retrained model.

@_exported import CoreAIKitCore
@_exported import CoreAIKitVision

import Foundation
import Tokenizers

public final class SafetyClassifier: @unchecked Sendable {
    /// What to flag, and the question asked of the content. Both are plain language: this is the
    /// whole point of the model, not a convenience wrapper over a fixed taxonomy.
    ///
    /// The presets are the four policies the port's gate suite scored against fp32, so they are
    /// the ones with measured behaviour — including on their near-misses (a weapons *safety*
    /// question, a refusal to dox, and help-seeking all score below 0.001).
    public struct Policy: Sendable {
        public var instruction: String
        public var query: String

        public init(instruction: String, query: String) {
            self.instruction = instruction
            self.query = query
        }

        public static let violence = Policy(
            instruction: "You are a strict safety moderator reviewing potentially harmful "
                + "content. Apply a low tolerance threshold.",
            query: "Does this content promote physical violence?")

        public static let weapons = Policy(
            instruction: "You are a content moderator. Flag content that provides instructions "
                + "for creating weapons or explosives.",
            query: "Does this content contain weapon-making instructions?")

        public static let privacy = Policy(
            instruction: "Flag any content that solicits or discloses another person's private "
                + "information without consent.",
            query: "Does this content violate someone's privacy?")

        public static let selfHarm = Policy(
            instruction: "You are a strict safety moderator. Flag self-harm promotion; do not "
                + "flag help-seeking or support resources.",
            query: "Does this content promote self-harm?")
    }

    /// A probability and the call made from it. `probability` is the number to keep: a verdict
    /// that holds at 0.51 and one that holds at 0.99 are not the same evidence, and only this
    /// says which. Compression moves probabilities well before it moves verdicts.
    public struct Verdict: Sendable {
        public let violates: Bool
        public let probability: Float
    }

    /// The system prompt is fixed by training and travels with the bundle's `reference.json`;
    /// only the policy is the caller's.
    public struct Template: Sendable {
        public var system: String
        public var openSystem: String
        public var closeSystem: String
        public var openInstruction: String
        public var closeInstruction: String

        public init(
            system: String, openSystem: String, closeSystem: String,
            openInstruction: String, closeInstruction: String
        ) {
            self.system = system
            self.openSystem = openSystem
            self.closeSystem = closeSystem
            self.openInstruction = openInstruction
            self.closeInstruction = closeInstruction
        }

        /// Shieldstral's chat template, flattened. The device has no Jinja, and the flattening
        /// is gated in the zoo against the tokenizer's own rendering (bit-identical ids on all
        /// nine suite cases, Japanese included).
        public static let shieldstral = Template(
            system: "Judge whether the Document meets the requirements based on the Query and "
                + "the Instruction provided. Note that the answer can only be \"yes\" or \"no\".",
            openSystem: "<s>[SYSTEM_PROMPT]",
            closeSystem: "[/SYSTEM_PROMPT]",
            openInstruction: "[INST]",
            closeInstruction: "[/INST]")

        var prefix: String { openSystem + system + closeSystem + openInstruction }
        var suffix: String { closeInstruction }

        func body(policy: Policy, document: String) -> String {
            "<Instruct>: \(policy.instruction)\n\n<Query>: \(policy.query)\n\n"
                + "<Document>: \(document)"
        }
    }

    private let graph: GraphModel
    private let tokenizer: any Tokenizer
    private let template: Template
    private let idsInput: String
    private let maskInput: String
    private let outputName: String
    private let padToken: Int32

    /// Probability at or above which `Verdict.violates` is true. 0.5 is where the model was
    /// trained to sit; raise it for a quieter filter, but tune it against **this bundle** — an
    /// int4 graph and fp32 agree on every verdict in the gate suite while differing by up to
    /// 0.03 in probability, all of it on the case fp32 was least sure about.
    public var threshold: Float

    /// The graph's fixed grid. A verdict costs this many tokens of compute whatever the document
    /// length, so it is the number to pick a bundle by.
    public let sequenceLength: Int

    /// Loads a bundle directory holding one `*.aimodel` plus a `tokenizer/` folder.
    public init(
        bundleAt url: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        template: Template = .shieldstral,
        threshold: Float = 0.5,
        padToken: Int32 = 11
    ) async throws {
        guard
            let modelURL = try FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil
            ).first(where: { $0.pathExtension == "aimodel" })
        else {
            throw VisionError.bundleLayout("no .aimodel found under \(url.path)")
        }
        let graph = try await GraphModel(contentsOf: modelURL, computeUnits: computeUnits)
        self.graph = graph
        self.template = template
        self.threshold = threshold
        self.padToken = padToken
        self.tokenizer = try await AutoTokenizer.from(
            modelFolder: url.appendingPathComponent("tokenizer"))

        guard
            let idsInput = graph.inputNames.first(where: { $0.contains("input_ids") }),
            let maskInput = graph.inputNames.first(where: { $0.contains("mask") }),
            let idsShape = graph.shape(ofInput: idsInput), idsShape.count == 2,
            let outputName = graph.outputNames.first
        else {
            throw VisionError.bundleLayout(
                "unexpected classifier graph contract: inputs \(graph.inputNames), "
                    + "outputs \(graph.outputNames)")
        }
        self.idsInput = idsInput
        self.maskInput = maskInput
        self.sequenceLength = idsShape[1]
        self.outputName = outputName
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .shieldstral3B,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        template: Template = .shieldstral,
        threshold: Float = 0.5,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(
            bundleAt: url, computeUnits: computeUnits, template: template, threshold: threshold)
    }

    // MARK: - Classification

    /// P(the content violates `policy`) in [0, 1], plus the call made at ``threshold``.
    public func check(_ content: String, policy: Policy = .violence) async throws -> Verdict {
        // prefix + body + suffix, encoded without extra special tokens (the flattened template
        // already carries `<s>` as text, and this tokenizer's post-processor adds none). Keep
        // prefix AND suffix; truncate only the document — the suffix is the position scored.
        let pre = tokenizer.encode(text: template.prefix, addSpecialTokens: false)
        let suf = tokenizer.encode(text: template.suffix, addSpecialTokens: false)
        var body = tokenizer.encode(
            text: template.body(policy: policy, document: content), addSpecialTokens: false)
        let budget = sequenceLength - pre.count - suf.count
        guard budget >= 0 else {
            throw VisionError.bundleLayout(
                "classifier grid \(sequenceLength) too small for the policy scaffolding")
        }
        if body.count > budget { body = Array(body.prefix(budget)) }

        var ids = (pre + body + suf).map(Int32.init)
        var mask = [Int32](repeating: 1, count: ids.count)
        while ids.count < sequenceLength {
            ids.append(padToken)        // right-pad; the graph gathers the last real token
            mask.append(0)
        }
        let outputs = try await graph.run([
            idsInput: .int32(ids, shape: [1, sequenceLength]),
            maskInput: .int32(mask, shape: [1, sequenceLength]),
        ])
        guard let probs = outputs[outputName] else {
            throw VisionError.missingOutput(outputName)
        }
        let p = probs.floats()          // [P(no), P(yes)]
        let violation = p.count >= 2 ? p[1] : (p.first ?? 0)
        return Verdict(violates: violation >= threshold, probability: violation)
    }

    /// Checks one piece of content against several policies — the usual shape of a real filter,
    /// where "is this allowed" is more than one rule. Costs one forward per policy.
    public func check(
        _ content: String, policies: [Policy]
    ) async throws -> [(policy: Policy, verdict: Verdict)] {
        var out: [(policy: Policy, verdict: Verdict)] = []
        out.reserveCapacity(policies.count)
        for policy in policies {
            out.append((policy, try await check(content, policy: policy)))
        }
        return out
    }

    /// True if `content` violates any of `policies`. Stops at the first violation, so an early
    /// hit costs one forward rather than all of them.
    public func violatesAny(_ content: String, policies: [Policy]) async throws -> Bool {
        for policy in policies where try await check(content, policy: policy).violates {
            return true
        }
        return false
    }
}
