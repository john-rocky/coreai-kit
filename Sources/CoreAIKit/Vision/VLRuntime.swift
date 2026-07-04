// VLRuntime.swift — the loaded pieces of a vision-language model: a text decoder engine
// wired with the four static multimodal inputs, the paired vision tower, and the owned host
// buffers the decoder gathers from.
//
// This is the kit-core change DUAL_PROFILE_STATE #8 asked for. A VL decode bundle declares
// inputs beyond input_ids/position_ids (`image_embeds`, `deepstack_embeds`, `rope_shift_start`,
// `rope_shift_amount`); the stock load path leaves them unbound and the engine rejects the
// bundle. Here we create the engine through `EngineFactory.createEngine(…, options:)` with
// those four bound as `StaticInputBuffer`s — owned `MTLBuffer`s this class rewrites per
// attached image. The decoder graph does the gather in-graph (image tokens carry extension
// ids `vocab + slot`; interleaved M-RoPE is derived from two i32 scalars). Adapted from the
// device-verified `Qwen3VLBackend` (see NOTICE.txt).

import CoreAILanguageModels
import CoreAIKitVision
import CoreGraphics
import Foundation
import ImageIO
import Metal
import Synchronization
import Tokenizers

/// Failures specific to the VL path.
public enum KitVisionError: Error, LocalizedError {
    case noMetalDevice
    case bufferAllocationFailed
    case visionOutputMissing(String)
    case bundleMissingMain

    public var errorDescription: String? {
        switch self {
        case .noMetalDevice: return "No Metal device available for the VL runtime."
        case .bufferAllocationFailed: return "Could not allocate the VL static-input buffers."
        case .visionOutputMissing(let name):
            return "Vision tower did not produce the expected output '\(name)'."
        case .bundleMissingMain: return "VL decoder bundle has no 'main' model component."
        }
    }
}

/// Owns a VL model's engine + vision tower + the static multimodal buffers, for the engine's
/// lifetime. One runtime assumes serial use (one `LanguageModelSession` at a time) — the
/// underlying engine traps on concurrent generate calls, same as the text path.
public final class VLRuntime: @unchecked Sendable {
    public let arch: VLArchitecture
    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelName: String
    let maxContextLength: Int
    let vocabSize: Int

    private let vision: GraphModel
    // Owned static-input buffers, alive for the engine's lifetime; rewritten per attach.
    // Only the inputs the arch's decoder graph declares exist (MiniCPM-V: image embeds only).
    private let imageBuffer: any MTLBuffer
    private let deepstackBuffer: (any MTLBuffer)?
    private let shiftStartBuffer: (any MTLBuffer)?
    private let shiftAmountBuffer: (any MTLBuffer)?

    /// The FM segment id of the image currently in the buffers (nil = text-only). Guards the
    /// expensive vision encode: re-run only when the attached image changes.
    private let attachedSegmentID = Mutex<String?>(nil)

    /// Loads the decoder + vision bundles and wires the four static inputs.
    public init(
        decoderBundleAt decoderURL: URL,
        visionModelAt visionURL: URL,
        arch: VLArchitecture,
        engineVariant: EngineVariant = .pipelined
    ) async throws {
        // S=1 VL bundles ship with the static-query twin; chunking must stay off (a known
        // trap — chunked prefill feeds a multi-token query the S=1 graph rejects).
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        self.arch = arch

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw KitVisionError.noMetalDevice
        }
        func owned(_ length: Int) throws -> any MTLBuffer {
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared)
            else { throw KitVisionError.bufferAllocationFailed }
            memset(buffer.contents(), 0, length)
            return buffer
        }
        let imageBuffer = try owned(arch.imageEmbedCount * MemoryLayout<Float16>.size)
        var staticInputs = ["image_embeds": StaticInputBuffer(imageBuffer)]
        if arch.deepstackPerToken > 0 {
            let deepstackBuffer = try owned(
                arch.deepstackEmbedCount * MemoryLayout<Float16>.size)
            staticInputs["deepstack_embeds"] = StaticInputBuffer(deepstackBuffer)
            self.deepstackBuffer = deepstackBuffer
        } else {
            self.deepstackBuffer = nil
        }
        if arch.ropeShifted {
            let shiftStartBuffer = try owned(64)  // engine pads [1] i32 static inputs to 64 B
            let shiftAmountBuffer = try owned(64)
            staticInputs["rope_shift_start"] = StaticInputBuffer(shiftStartBuffer)
            staticInputs["rope_shift_amount"] = StaticInputBuffer(shiftAmountBuffer)
            self.shiftStartBuffer = shiftStartBuffer
            self.shiftAmountBuffer = shiftAmountBuffer
        } else {
            self.shiftStartBuffer = nil
            self.shiftAmountBuffer = nil
        }
        self.imageBuffer = imageBuffer

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
                staticInputBuffers: staticInputs))

        self.tokenizer = try await bundle.loadTokenizer()
        self.vision = try await GraphModel(contentsOf: visionURL, computeUnits: .gpu)

        setTextOnlyShift()
    }

    /// Whether image embeds are currently resident in the buffers.
    public var imageAttached: Bool { attachedSegmentID.withLock { $0 != nil } }

    // MARK: - Image attach

    /// Vision-encode `cgImage` and write its embeds into the decoder's static buffers, unless
    /// `segmentID` already names the resident image (then this is a no-op). `segmentID` is the
    /// FM transcript segment id; pass `nil` to always re-encode.
    func attach(
        cgImage: CGImage, orientation: CGImagePropertyOrientation = .up, segmentID: String?
    ) async throws {
        if let segmentID, attachedSegmentID.withLock({ $0 == segmentID }) { return }

        let upright = cgImage.upright(orientation)
        let outputs: [String: TensorValue]
        switch arch.visionInput {
        case .patches:
            let patches = VLImagePreprocessor.patches(from: upright, arch: arch)
            outputs = try await vision.run([
                "patches": .float16(patches, shape: [arch.patches, arch.patchDim])
            ])
        case .pixels:
            let pixels = VLImagePreprocessor.pixelValues(from: upright, arch: arch)
            outputs = try await vision.run([
                "pixel_values": .float16(
                    pixels, shape: [1, 3, arch.imageSide, arch.imageSide])
            ])
        }
        guard let embeds = outputs[arch.visionOutput] else {
            throw KitVisionError.visionOutputMissing(arch.visionOutput)
        }
        write(embeds.floats(), into: imageBuffer, capacity: arch.imageEmbedCount)
        if let deepstackBuffer {
            guard let deepstack = outputs["deepstack_embeds"] else {
                throw KitVisionError.visionOutputMissing("deepstack_embeds")
            }
            write(deepstack.floats(), into: deepstackBuffer, capacity: arch.deepstackEmbedCount)
        }
        attachedSegmentID.withLock { $0 = segmentID ?? "<anonymous>" }
    }

    /// Clears the resident image and reverts to the text-only rope shift.
    func detach() {
        memset(imageBuffer.contents(), 0, imageBuffer.length)
        if let deepstackBuffer {
            memset(deepstackBuffer.contents(), 0, deepstackBuffer.length)
        }
        setTextOnlyShift()
        attachedSegmentID.withLock { $0 = nil }
    }

    private func write(_ values: [Float], into buffer: any MTLBuffer, capacity: Int) {
        let pointer = buffer.contents().assumingMemoryBound(to: Float16.self)
        let count = min(values.count, capacity)
        for i in 0..<count { pointer[i] = Float16(values[i]) }
    }

    // MARK: - Rope shift

    /// Binds the rope shift for an image whose first vision token sits at `imageStart`
    /// (start = imageStart + mergedTokens; amount = mergedTokens - grid). No-op on
    /// plain-1D-position archs (no shift inputs on the graph).
    func setImageShift(imageStart: Int) {
        guard let shiftStartBuffer, let shiftAmountBuffer else { return }
        shiftStartBuffer.contents().assumingMemoryBound(to: Int32.self)[0] =
            Int32(imageStart + arch.mergedTokens)
        shiftAmountBuffer.contents().assumingMemoryBound(to: Int32.self)[0] = arch.ropeShiftAmount
    }

    /// Degenerates the graph to a plain Qwen3 LLM (start = 1<<30 so no position is shifted).
    func setTextOnlyShift() {
        guard let shiftStartBuffer, let shiftAmountBuffer else { return }
        shiftStartBuffer.contents().assumingMemoryBound(to: Int32.self)[0] = 1 << 30
        shiftAmountBuffer.contents().assumingMemoryBound(to: Int32.self)[0] = 0
    }

    // MARK: - Warmup

    /// One real 1-token generate + reset: compiles the sampler graph and touches the weights
    /// (the text-only path, so it does not depend on an attached image).
    func warmup() async throws {
        let seed = tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
        let stream = try engine.generate(
            with: [seed], samplingConfiguration: .greedy,
            inferenceOptions: InferenceOptions(maxTokens: 1))
        for try await _ in stream {}
        try await engine.reset()
    }
}
