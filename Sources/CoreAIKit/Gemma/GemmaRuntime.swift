// GemmaRuntime.swift — the loaded pieces of a Gemma 4 decode bundle: a pipelined text engine
// wired with the two STATIC per-layer-embedding (PLE) table inputs, the tokenizer, and the
// owned host buffers those inputs are bound to.
//
// Gemma 4's `…_tbl` decode bundle declares inputs beyond input_ids/position_ids — `ple_table`
// and `ple_scale` — that the stock load path (`ModelRuntime` → `CoreAIRunner`) leaves unbound,
// so the engine rejects the bundle. Here we create the engine through
// `EngineFactory.createEngine(…, options:)` with those two bound as `StaticInputBuffer`s. Unlike
// the VL path's image buffers (rewritten per attached image), these are CONSTANT for the whole
// model: each table file is read once into an owned `MTLBuffer` and bound unchanged on every
// encode. The decoder graph does the PLE gather in-graph (gather row `input_ids` from the table,
// scale, reshape), and the head + final softcap are fused into the same graph — so the engine
// returns sampled tokens directly, exactly like a plain text bundle. Adapted from the
// device-verified `PipelinedBackend` (see NOTICE.txt).

import CoreAILanguageModels
import Foundation
import Metal
import Tokenizers

/// Failures specific to the Gemma path.
public enum KitGemmaError: Error, LocalizedError {
    case noMetalDevice
    case bufferAllocationFailed(String)
    case missingTableFile(String)

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice:
            return "No Metal device available for the Gemma runtime."
        case .bufferAllocationFailed(let file):
            return "Could not allocate the Gemma static-input buffer for '\(file)'."
        case .missingTableFile(let file):
            return "Gemma PLE table file '\(file)' is missing from the tables directory."
        }
    }
}

/// Owns a Gemma 4 model's engine + tokenizer + the two static PLE table buffers, for the
/// engine's lifetime. One runtime assumes serial use (one `LanguageModelSession` at a time) —
/// the underlying engine traps on concurrent generate calls, same as the text path.
public final class GemmaRuntime: @unchecked Sendable {
    public let arch: GemmaArchitecture
    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelName: String
    let maxContextLength: Int
    let vocabSize: Int

    /// Owned static-input buffers (the PLE tables), kept alive for the engine's lifetime.
    /// Bound unchanged on every encode — never rewritten after load.
    private let tableBuffers: [String: StaticInputBuffer]

    /// Loads a Gemma 4 `…_tbl` decode bundle and binds its two PLE tables from `tablesURL`
    /// (the paired `gemma4_gather_raw` directory holding `embed_per_layer.i8` +
    /// `embed_per_layer.scale.f32`). A QAT bundle must be paired with the QAT tables.
    public init(
        decoderBundleAt decoderURL: URL,
        tablesAt tablesURL: URL,
        arch: GemmaArchitecture = .gemma4,
        engineVariant: EngineVariant = .pipelined
    ) async throws {
        // The `…_tbl` graph is S=1 (input_ids is static [1,1]); chunked prefill feeds a
        // multi-token query the S=1 graph rejects, so chunking must stay off.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        self.arch = arch

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw KitGemmaError.noMetalDevice
        }

        // Read each PLE table file once into an owned storageModeShared buffer. (A device-proven
        // foreground config; a copy-on-write mmap was tried to shrink the background footprint but
        // background GPU is blocked by iOS anyway, and the mmap path regressed the foreground path —
        // reverted to owned.) ~2.35 GB for E2B.
        var buffers: [String: StaticInputBuffer] = [:]
        for (input, file) in arch.staticInputFiles {
            let url = tablesURL.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw KitGemmaError.missingTableFile(file)
            }
            buffers[input] = StaticInputBuffer(try Self.ownedBuffer(url: url, device: device))
        }
        self.tableBuffers = buffers

        let bundle = try LanguageBundle(at: decoderURL)
        self.modelName = bundle.name
        self.maxContextLength = bundle.maxContextLength
        self.vocabSize = bundle.vocabSize

        // Build the engine through the factory so we can bind the static inputs (CoreAIRunner
        // does not expose EngineOptions). `"main"` avoids importing CoreAIShared's ComponentKey.
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: "main")
        let modelURL = try bundle.requireModelURL(for: "main")
        self.engine = try await EngineFactory.createEngine(
            config: try JSONEncoder().encode(config),
            modelURL: modelURL,
            options: EngineOptions(
                variant: engineVariant.factoryOverride,
                staticInputBuffers: buffers))

        self.tokenizer = try await bundle.loadTokenizer()
    }

    // MARK: - Warmup

    /// One real 1-token generate + reset: compiles the sampler graph and touches the weights.
    /// `engine.warmup()` is deliberately not used — its default query length builds a step
    /// shape the S=1 graph rejects.
    func warmup() async throws {
        let seed = tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
        let stream = try engine.generate(
            with: [seed], samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1))
        for try await _ in stream {}
        try await engine.reset()
    }

    // MARK: - Table loading

    /// Reads a whole file into an owned storageModeShared `MTLBuffer`. Chunked reads keep the
    /// multi-GB PLE table within a single `read(2)` size limit. (Device-proven for the foreground
    /// path; reads fine from a file in the read-only signed app bundle when the model is side-loaded.)
    private static func ownedBuffer(url: URL, device: any MTLDevice) throws -> any MTLBuffer {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw KitGemmaError.missingTableFile(url.lastPathComponent)
        }
        defer { close(fd) }
        let size = Int(lseek(fd, 0, SEEK_END))
        _ = lseek(fd, 0, SEEK_SET)
        guard size > 0, let buffer = device.makeBuffer(length: size, options: .storageModeShared)
        else {
            throw KitGemmaError.bufferAllocationFailed(url.lastPathComponent)
        }
        var done = 0
        while done < size {
            let n = read(fd, buffer.contents() + done, min(1 << 27, size - done))
            guard n > 0 else {
                throw KitGemmaError.bufferAllocationFailed(url.lastPathComponent)
            }
            done += n
        }
        return buffer
    }
}
