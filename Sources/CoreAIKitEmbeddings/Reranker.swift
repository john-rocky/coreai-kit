// Reranker.swift — on-device cross-encoder reranking for retrieval/RAG, over a static
// reranker graph (Qwen3-Reranker export: Qwen3 backbone + LM head; the scoring tail —
// gather the last real token, head on that one position, softmax over {no, yes} — is baked
// in-graph, so one forward returns the relevance probability P(yes)).
//
// Pairs with `TextEmbedder`: the embedder does fast first-stage retrieval, the reranker
// re-scores the shortlist for precision. `embed -> rerank -> generate`, all on device.

@_exported import CoreAIKitCore
@_exported import CoreAIKitVision

import Foundation
import Tokenizers

public final class Reranker: @unchecked Sendable {
    /// The prompt scaffolding the pair is wrapped in before scoring. Defaults to Qwen3-Reranker's
    /// official template; the instruction is task-tunable (the model is instruction-aware).
    public struct Template: Sendable {
        public var prefix: String
        public var suffix: String
        public var instruction: String

        public init(prefix: String, suffix: String, instruction: String) {
            self.prefix = prefix
            self.suffix = suffix
            self.instruction = instruction
        }

        /// Qwen3-Reranker's official scaffolding (from its model card).
        public static let qwen3 = Template(
            prefix: "<|im_start|>system\nJudge whether the Document meets the requirements based "
                + "on the Query and the Instruct provided. Note that the answer can only be "
                + "\"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
            suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            instruction: "Given a web search query, retrieve relevant passages that answer the query")

        func body(query: String, document: String) -> String {
            "<Instruct>: \(instruction)\n<Query>: \(query)\n<Document>: \(document)"
        }
    }

    private let graph: GraphModel
    private let tokenizer: any Tokenizer
    private let template: Template
    private let idsInput: String
    private let maskInput: String
    private let outputName: String
    private let padToken: Int32
    public let sequenceLength: Int

    /// Loads a bundle directory holding one `*.aimodel` plus a `tokenizer/` folder.
    public init(
        bundleAt url: URL,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        template: Template = .qwen3,
        padToken: Int32 = 151643
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
                "unexpected reranker graph contract: inputs \(graph.inputNames), "
                    + "outputs \(graph.outputNames)")
        }
        self.idsInput = idsInput
        self.maskInput = maskInput
        self.sequenceLength = idsShape[1]
        self.outputName = outputName
    }

    /// Downloads the bundle from the Hugging Face Hub if needed, then loads it.
    public convenience init(
        model: ModelID = .qwen3Reranker0_6B,
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        template: Template = .qwen3,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, computeUnits: computeUnits, template: template)
    }

    // MARK: - Scoring

    /// Relevance of `document` to `query` in [0, 1] = P("yes"). Higher is more relevant.
    public func score(query: String, document: String) async throws -> Float {
        // prefix + body + suffix, encoded without extra special tokens (the scaffolding already
        // carries the chat specials). Keep prefix AND suffix; truncate only the body to fit —
        // the suffix is what the model scores the next token of.
        let pre = tokenizer.encode(text: template.prefix, addSpecialTokens: false)
        let suf = tokenizer.encode(text: template.suffix, addSpecialTokens: false)
        var body = tokenizer.encode(
            text: template.body(query: query, document: document), addSpecialTokens: false)
        let budget = sequenceLength - pre.count - suf.count
        guard budget >= 0 else {
            throw VisionError.bundleLayout(
                "reranker grid \(sequenceLength) too small for the prompt scaffolding")
        }
        if body.count > budget { body = Array(body.prefix(budget)) }

        var ids = (pre + body + suf).map(Int32.init)
        let real = ids.count
        var mask = [Int32](repeating: 1, count: real)
        while ids.count < sequenceLength {
            ids.append(padToken)        // right-pad; graph gathers the last real token via the mask
            mask.append(0)
        }
        let outputs = try await graph.run([
            idsInput: .int32(ids, shape: [1, sequenceLength]),
            maskInput: .int32(mask, shape: [1, sequenceLength]),
        ])
        guard let probs = outputs[outputName] else {
            throw VisionError.missingOutput(outputName)
        }
        let p = probs.floats()        // [P(no), P(yes)]
        return p.count >= 2 ? p[1] : (p.first ?? 0)
    }

    /// Scores every document against the query and returns them sorted most-relevant first.
    public func rerank(
        query: String, documents: [String]
    ) async throws -> [(index: Int, document: String, score: Float)] {
        var scored: [(index: Int, document: String, score: Float)] = []
        scored.reserveCapacity(documents.count)
        for (i, doc) in documents.enumerated() {
            scored.append((i, doc, try await score(query: query, document: doc)))
        }
        return scored.sorted { $0.score > $1.score }
    }
}
