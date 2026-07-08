// GemmaRuntime.swift — the loaded pieces of a Gemma 4 decode bundle: a pipelined text engine
// wired with the two STATIC per-layer-embedding (PLE) table inputs, the tokenizer, and the
// owned host buffers those inputs are bound to.
//
// Gemma 4's `…_tbl` decode bundle declares inputs beyond input_ids/position_ids — `ple_table`
// and `ple_scale` — that the stock load path (`ModelRuntime` → `CoreAIRunner`) leaves unbound,
// so the engine rejects the bundle. Here we create the engine through
// `EngineFactory.createEngine(…, options:)` with those two bound as `StaticInputBuffer`s. Unlike
// the VL path's image buffers (rewritten per attached image), these are CONSTANT for the whole
// model: each table is loaded once (owned copy when the memory headroom allows, COW file-backed
// mapping otherwise — see the init) and bound unchanged on every encode. The decoder graph does
// the PLE gather in-graph (gather row `input_ids` from the table,
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

        // OWNED vs FILE-BACKED tables is a speed/footprint trade measured on device
        // (iPhone 17 Pro, E2B): owned `MTLBuffer`s decode ~30 tok/s; COW-mmap decodes
        // ~9 tok/s because file-backed no-copy buffers pay a per-encode residency tax
        // (~ms/GB) — but their clean pages don't count against jetsam, which is what
        // lets an unentitled process or a backgrounded host (SiriAsk's gate) survive.
        // So pick per launch: owned when the jetsam headroom fits the copy plus the
        // generation peak, file-backed otherwise. COREAI_GEMMA_TABLES=owned|mapped
        // overrides. IMPORTANT for the mapped path: the file must live in a WRITABLE
        // location — ModelHost stages a side-loaded model out of the read-only signed
        // .app into a writable cache before load, because COW-mmap of a file inside
        // the signed bundle regressed.
        var tableFiles: [(input: String, url: URL)] = []
        for (input, file) in arch.staticInputFiles {
            let url = tablesURL.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw KitGemmaError.missingTableFile(file)
            }
            tableFiles.append((input, url))
        }
        let totalBytes = tableFiles.reduce(0) { sum, entry in
            sum + ((try? entry.url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        var buffers: [String: StaticInputBuffer] = [:]
        if Self.shouldOwnTables(totalBytes: totalBytes) {
            do {
                for (input, url) in tableFiles {
                    buffers[input] = StaticInputBuffer(try Self.ownedBuffer(url: url, device: device))
                }
            } catch {
                // Allocation failed under pressure — drop the copies and fall back to
                // the file-backed mapping (slower, but loads).
                buffers = [:]
            }
        }
        if buffers.isEmpty {
            for (input, url) in tableFiles {
                buffers[input] = StaticInputBuffer(try Self.mappedBuffer(url: url, device: device))
            }
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

    /// Owned tables need jetsam headroom for the copy itself plus the rest of a
    /// generation peak (~2.1 GB beyond the tables for E2B: file-backed weights + KV +
    /// engine scratch, measured on device). `os_proc_available_memory` is 0 where no
    /// jetsam accounting applies (macOS) — owned is always safe there.
    private static func shouldOwnTables(totalBytes: Int) -> Bool {
        switch ProcessInfo.processInfo.environment["COREAI_GEMMA_TABLES"] {
        case "owned": return true
        case "mapped": return false
        default: break
        }
        let available = os_proc_available_memory()
        if available == 0 { return true }
        return Int64(available) - Int64(totalBytes) > 2_600_000_000
    }

    /// Reads a PLE table file into an owned `storageModeShared` `MTLBuffer`. Dirty
    /// memory against the jetsam limit, but no per-encode residency tax — the fast
    /// path when the process has the headroom (see `shouldOwnTables`).
    private static func ownedBuffer(url: URL, device: any MTLDevice) throws -> any MTLBuffer {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw KitGemmaError.missingTableFile(url.lastPathComponent)
        }
        defer { close(fd) }
        let size = Int(lseek(fd, 0, SEEK_END))
        _ = lseek(fd, 0, SEEK_SET)
        guard size > 0,
            let buffer = device.makeBuffer(length: size, options: .storageModeShared)
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

    /// Maps a PLE table file copy-on-write (`MAP_PRIVATE`), file-backed, and wraps it as a
    /// `bytesNoCopy` `MTLBuffer` — no owned copy. The tables are read-only (gathered, never
    /// written), so the COW pages stay CLEAN and the kernel can evict them under pressure rather
    /// than counting the full ~2.35 GB against the process's jetsam footprint — this is what lets a
    /// backgrounded Gemma survive (USER confirmed the background answer worked with this). The file
    /// MUST be in a writable location (ModelHost stages a side-loaded model out of the signed .app
    /// into a writable cache first); COW-mmap of a file inside the read-only signed bundle regressed.
    /// `PROT_WRITE`+`MAP_PRIVATE` is deliberate: a pure `PROT_READ` mapping pays a ~208 ms/encode
    /// residency tax on the GPU delegate; COW does not (and we never write, so pages stay clean).
    private static func mappedBuffer(url: URL, device: any MTLDevice) throws -> any MTLBuffer {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw KitGemmaError.missingTableFile(url.lastPathComponent)
        }
        defer { close(fd) }  // the mapping outlives the descriptor
        let size = Int(lseek(fd, 0, SEEK_END))
        guard size > 0 else {
            throw KitGemmaError.bufferAllocationFailed(url.lastPathComponent)
        }
        // bytesNoCopy needs a page-aligned pointer AND length; mmap is page-aligned, round length up.
        let pageSize = Int(getpagesize())
        let mapLen = (size + pageSize - 1) & ~(pageSize - 1)
        let mapped = mmap(nil, mapLen, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0)
        guard let mapped, mapped != MAP_FAILED else {
            throw KitGemmaError.bufferAllocationFailed(url.lastPathComponent)
        }
        guard let buffer = device.makeBuffer(
            bytesNoCopy: mapped, length: mapLen, options: .storageModeShared,
            deallocator: { pointer, length in _ = munmap(pointer, length) })
        else {
            munmap(mapped, mapLen)
            throw KitGemmaError.bufferAllocationFailed(url.lastPathComponent)
        }
        return buffer
    }
}
