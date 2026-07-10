// KitDiffusionLM.swift — a masked-diffusion LM (LLaDA / d3LLM) behind one `reply(to:)`.
//
// Unlike the AR models (an InferenceEngine streaming one token at a time), a diffusion LM
// is a HOST loop over a single static, BIDIRECTIONAL forward:
//
//   bundle fn  main(input_ids[1,S] Int32) -> logits[1,S,VOCAB]   (full canvas, NO KV cache)
//   canvas = prompt ++ [MASK]*gen; each step: forward -> commit the lowest-ENTROPY masked
//   positions in the active semi-AR block (always ≥1, plus any below `threshold`) -> repeat
//   block by block.
//
// This is the validated loop from conversion/dllm/generate_llada.py, first shipped in
// CoreAIChatMac's LLaDAEngine — tokens pop in IN PARALLEL, out of reading order, so `onStep`
// streams whole-canvas snapshots (still-masked positions render as ░), not append-only
// deltas. That is why this is its own surface and not a `ChatSession`.

import CoreAI
import CoreAIKitVision
import Foundation
import Tokenizers

/// A masked-diffusion language model (`dllm` catalog kind): prompt in, denoised reply out,
/// with live canvas snapshots while it denoises. One generation at a time per instance.
public final class KitDiffusionLM: @unchecked Sendable {
    /// Diffusion geometry, read from the bundle's `metadata.json`.
    public struct Parameters: Sendable {
        public var sequenceLength = 128
        public var blockSize = 32
        public var threshold = 0.5
        public var maskTokenID: Int32 = 126_336
        public var eosTokenID: Int32 = 126_081
    }

    /// One denoising step: the live canvas text (masked positions as ░), how many tokens
    /// are committed, and the number of forwards so far.
    public struct Step: Sendable {
        public let text: String
        public let committed: Int
        public let forwards: Int
    }

    public let parameters: Parameters

    private let model: AIModel
    private let function: InferenceFunction
    private let descriptor: InferenceFunctionDescriptor
    private let tokenizer: any Tokenizer
    private let vocab: Int
    // Stable storage for the output view (`MutableViews` borrows its arrays up to
    // `function.run`; locals trip the lifetime checker).
    private var logitsArray: NDArray

    /// Loads a diffusion LM by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let dlm = try await KitDiffusionLM(catalog: "llada-8b")
    /// ```
    public convenience init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .dllm)
        guard let variant = entry.variant else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        let root = try await store.download(
            entry.modelID(path: variant.path), progress: downloadProgress)
        try await self.init(bundleAt: root)
    }

    /// Loads a local dLLM bundle dir (`metadata.json` + `*.aimodel` + `tokenizer/`).
    public init(bundleAt root: URL) async throws {
        let metadata =
            (try? JSONSerialization.jsonObject(
                with: Data(contentsOf: root.appendingPathComponent("metadata.json"))))
            as? [String: Any] ?? [:]
        let assetName = ((metadata["assets"] as? [String: Any])?["main"] as? String)
            ?? "main.aimodel"
        var parameters = Parameters()
        if let diffusion = metadata["diffusion"] as? [String: Any] {
            if let v = diffusion["seq"] as? Int { parameters.sequenceLength = v }
            if let v = diffusion["block_size"] as? Int { parameters.blockSize = v }
            if let v = diffusion["threshold"] as? Double { parameters.threshold = v }
            if let v = diffusion["mask_token_id"] as? Int { parameters.maskTokenID = Int32(v) }
            if let v = diffusion["eos_token_id"] as? Int { parameters.eosTokenID = Int32(v) }
        }
        self.parameters = parameters

        model = try await AIModel(
            contentsOf: root.appendingPathComponent(assetName),
            options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard
            let descriptor = model.functionDescriptor(for: "main"),
            let function = try model.loadFunction(named: "main")
        else {
            throw VisionError.functionNotFound("main")
        }
        self.descriptor = descriptor
        self.function = function
        guard case .ndArray(let ld)? = descriptor.outputDescriptor(of: "logits") else {
            throw VisionError.missingOutput("logits")
        }
        vocab = ld.shape.last.map { $0 < 0 ? 0 : $0 } ?? 0
        logitsArray = NDArray(
            descriptor: ld.resolvingDynamicDimensions(ld.shape.map { max($0, 1) }))
        tokenizer = try await AutoTokenizer.from(
            modelFolder: root.appendingPathComponent("tokenizer"))
    }

    /// Denoise one reply. `onStep` fires after every forward with the live canvas
    /// (masked positions as ░ — the parallel fill-in UX a diffusion LM actually has).
    public func reply(
        to prompt: String, onStep: (@Sendable (Step) -> Void)? = nil
    ) async throws -> String {
        try await reply(messages: [["role": "user", "content": prompt]], onStep: onStep)
    }

    /// Denoise a reply to a chat history (`role`/`content` dictionaries, oldest first).
    /// The canvas has no KV cache — prompt + answer share the fixed sequence — so the
    /// oldest turns are dropped first when the history outgrows half the canvas.
    public func reply(
        messages: [[String: String]], onStep: (@Sendable (Step) -> Void)? = nil
    ) async throws -> String {
        let S = parameters.sequenceLength
        let BLOCK = parameters.blockSize
        let MASK = parameters.maskTokenID

        // Fit the prompt while reserving generation room; never let generation collapse.
        let promptCap = max(8, S / 2)
        var history = messages
        var ids = try promptIDs(history)
        while ids.count > promptCap && history.count > 1 {
            history.removeFirst()
            ids = try promptIDs(history)
        }
        if ids.count > S - 1 { ids = Array(ids.suffix((S * 3) / 4)) }
        let promptLen = ids.count

        var canvas = [Int32](repeating: MASK, count: S)
        for i in 0..<promptLen { canvas[i] = ids[i] }

        let blocks = (S - promptLen + BLOCK - 1) / BLOCK
        var forwards = 0

        blockLoop: for block in 0..<blocks {
            let blockLow = promptLen + block * BLOCK
            let blockHigh = min(promptLen + (block + 1) * BLOCK, S)
            while true {
                forwards += 1
                let logits = try await forward(canvas)

                var candidates: [(position: Int, token: Int32, entropy: Double)] = []
                for i in 0..<blockHigh where canvas[i] == MASK {
                    let (token, entropy) = rowEntropyArgmax(logits, i * vocab)
                    candidates.append((i, token, entropy))
                }
                if candidates.isEmpty { break }
                candidates.sort { $0.entropy < $1.entropy }
                for (rank, candidate) in candidates.enumerated()
                where rank == 0 || candidate.entropy <= parameters.threshold {
                    canvas[candidate.position] = candidate.token
                }

                if let onStep {
                    let committed = (promptLen..<S).reduce(0) {
                        canvas[$1] != MASK ? $0 + 1 : $0
                    }
                    onStep(
                        Step(
                            text: renderCanvas(
                                canvas, promptLen: promptLen, frontier: blockHigh),
                            committed: committed, forwards: forwards))
                }
                await Task.yield()

                if !(blockLow..<blockHigh).contains(where: { canvas[$0] == MASK }) { break }
                if forwards > 4096 { break blockLoop }
            }
            if (promptLen..<blockHigh).contains(where: { canvas[$0] == parameters.eosTokenID }) {
                break
            }
        }
        return decodeGenerated(canvas, promptLen: promptLen)
    }

    // MARK: - forward

    private func forward(_ canvas: [Int32]) async throws -> [Float] {
        let S = parameters.sequenceLength
        guard case .ndArray(let inputDescriptor)? = descriptor.inputDescriptor(of: "input_ids")
        else { throw VisionError.unknownInput("input_ids") }
        var ids = NDArray(descriptor: inputDescriptor.resolvingDynamicDimensions([1, S]))
        var idsView = ids.mutableView(as: Int32.self)
        idsView.copyElements(fromContentsOf: canvas)

        guard case .ndArray(let outputDescriptor)? = descriptor.outputDescriptor(of: "logits")
        else { throw VisionError.missingOutput("logits") }
        logitsArray = NDArray(
            descriptor: outputDescriptor.resolvingDynamicDimensions([1, S, vocab]))
        var outputViews = InferenceFunction.MutableViews()
        outputViews.insert(&logitsArray, for: "logits")
        _ = try await function.run(
            inputs: ["input_ids": ids], states: InferenceFunction.MutableViews(),
            outputViews: consume outputViews)

        // Read back through a local binding (scoped views chained off the stored property
        // trip the lifetime checker). Logits may be fp16 or fp32 depending on the export.
        let produced = logitsArray
        let count = S * vocab
        switch produced.scalarType {
        case .float32:
            var out = [Float](repeating: 0, count: count)
            produced.view(as: Float.self).withUnsafePointer { pointer, _, _ in
                out.withUnsafeMutableBufferPointer {
                    $0.baseAddress!.update(from: pointer, count: count)
                }
            }
            return out
        default:
            var out = [Float16](repeating: 0, count: count)
            produced.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
                out.withUnsafeMutableBufferPointer {
                    $0.baseAddress!.update(from: pointer, count: count)
                }
            }
            return out.map { Float($0) }
        }
    }

    // (argmax token, entropy in nats) over one logits row. fp64 entropy, matches Python.
    private func rowEntropyArgmax(_ logits: [Float], _ base: Int) -> (Int32, Double) {
        var maxValue = -Float.greatestFiniteMagnitude
        var argmax = 0
        for j in 0..<vocab {
            let v = logits[base + j]
            if v > maxValue {
                maxValue = v
                argmax = j
            }
        }
        var sumExp = 0.0
        for j in 0..<vocab { sumExp += Foundation.exp(Double(logits[base + j] - maxValue)) }
        let logZ = Foundation.log(sumExp)
        var entropy = 0.0
        for j in 0..<vocab {
            let z = Double(logits[base + j] - maxValue)
            let p = Foundation.exp(z) / sumExp
            if p > 0 { entropy -= p * (z - logZ) }
        }
        return (Int32(argmax), entropy)
    }

    // MARK: - text

    private func promptIDs(_ messages: [[String: String]]) throws -> [Int32] {
        try tokenizer.applyChatTemplate(messages: messages).map(Int32.init)
    }

    // The committed (non-mask) generated tokens as clean readable text (the final answer).
    private func decodeGenerated(_ canvas: [Int32], promptLen: Int) -> String {
        var tokens: [Int] = []
        for i in promptLen..<canvas.count {
            let t = canvas[i]
            if t == parameters.eosTokenID { break }
            if t == parameters.maskTokenID { continue }
            tokens.append(Int(t))
        }
        return tokenizer.decode(tokens: tokens)
    }

    // Live diffusion view: the active region with ░ for still-masked positions, so the
    // caller watches tokens pop in out of reading order. Committed runs decode as text.
    private func renderCanvas(_ canvas: [Int32], promptLen: Int, frontier: Int) -> String {
        var parts: [String] = []
        var run: [Int] = []
        func flush() {
            if !run.isEmpty {
                parts.append(tokenizer.decode(tokens: run))
                run.removeAll(keepingCapacity: true)
            }
        }
        for i in promptLen..<min(frontier, canvas.count) {
            let t = canvas[i]
            if t == parameters.eosTokenID { break }
            if t == parameters.maskTokenID {
                flush()
                parts.append("░")
            } else {
                run.append(Int(t))
            }
        }
        flush()
        return parts.joined()
    }
}
