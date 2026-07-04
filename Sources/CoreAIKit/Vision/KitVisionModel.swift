// KitVisionModel.swift — a Core AI vision-language bundle behind FoundationModels'
// `LanguageModel` protocol, so a local VLM answers image prompts through a plain
// `LanguageModelSession` (and routes inside a `DynamicProfile`, closing DUAL_PROFILE G6).
//
// ```swift
// let model = try await KitVisionModel(model: .qwen3VL2B)
// let session = LanguageModelSession(model: model)
// let answer = try await session.respond(to: Prompt {
//     "What is in this photo?"
//     Attachment(cgImage)
// })
// ```
//
// The vision path is its own executor (`KitVisionExecutor`) and runtime (`VLRuntime`): the
// text `KitLanguageModel` is untouched. Tool calling / guided generation are not advertised
// for the VL model in v1.

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

/// A downloadable VL model = a decoder bundle + its paired vision tower in one repo, plus the
/// architecture geometry. Both sub-bundles are addressed as paths inside the same HF repo.
public struct VLModelID: Sendable, Hashable {
    public let decoder: ModelID
    public let vision: ModelID
    public let arch: VLArchitecture

    public init(decoder: ModelID, vision: ModelID, arch: VLArchitecture) {
        self.decoder = decoder
        self.vision = vision
        self.arch = arch
    }

    /// Qwen3-VL-2B (Apache-2.0). iPhone-class — the zoo's first on-device VLM.
    public static let qwen3VL2B = VLModelID(
        decoder: ModelID(
            "mlboydaisuke/Qwen3-VL-2B-CoreAI",
            path: "gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1"),
        vision: ModelID(
            "mlboydaisuke/Qwen3-VL-2B-CoreAI",
            path: "gpu-pipelined/qwen3_vl_2b_instruct_vision"),
        arch: .qwen3VL2B)

    /// Qwen3-VL-4B (Apache-2.0). iPhone-class (thermally limited), Mac-comfortable.
    public static let qwen3VL4B = VLModelID(
        decoder: ModelID(
            "mlboydaisuke/Qwen3-VL-4B-CoreAI",
            path: "gpu-pipelined/qwen3_vl_4b_instruct_decode_int8hu_s1"),
        vision: ModelID(
            "mlboydaisuke/Qwen3-VL-4B-CoreAI",
            path: "gpu-pipelined/qwen3_vl_4b_instruct_vision"),
        arch: .qwen3VL4B)

    /// Qwen3-VL-8B (Apache-2.0). Mac-class (8.7 GB decoder).
    public static let qwen3VL8B = VLModelID(
        decoder: ModelID(
            "mlboydaisuke/Qwen3-VL-8B-CoreAI",
            path: "gpu-pipelined/qwen3_vl_8b_instruct_decode_int8hu_s1"),
        vision: ModelID(
            "mlboydaisuke/Qwen3-VL-8B-CoreAI",
            path: "gpu-pipelined/qwen3_vl_8b_instruct_vision"),
        arch: .qwen3VL8B)

    /// Holo2-4B (Apache-2.0). H Company's GUI-grounding / computer-use VLM — give it a
    /// screenshot and an instruction and it localizes the UI element / click point. Fine-tuned
    /// from Qwen3-VL-4B with identical graph geometry, so it rides the same architecture.
    public static let holo2_4B = VLModelID(
        decoder: ModelID(
            "mlboydaisuke/Holo2-4B-CoreAI",
            path: "gpu-pipelined/holo2_4b_decode_int8lin_s1"),
        vision: ModelID(
            "mlboydaisuke/Holo2-4B-CoreAI",
            path: "gpu-pipelined/holo2_4b_vision"),
        arch: .qwen3VL4B)

    /// MiniCPM-V 4.6 (OpenBMB). Single-slice 448px VLM: SigLIP tower fed raw pixels →
    /// `image_features` [64, 1024], one `image_embeds` static input on the qwen3_5-hybrid
    /// decoder, plain 1D positions. The device-verified ship combo: fp16 vision + int8lin
    /// decode.
    public static let miniCPMV46 = VLModelID(
        decoder: ModelID(
            "mlboydaisuke/MiniCPM-V-4.6-CoreAI",
            path: "gpu-pipelined/minicpmv46_vlm_decode_int8lin"),
        vision: ModelID(
            "mlboydaisuke/MiniCPM-V-4.6-CoreAI",
            path: "gpu-pipelined/minicpmv46_vision"),
        arch: .miniCPMV46)

    /// Presets by catalog id. A VL model is two bundles (decoder + vision tower) plus graph
    /// geometry — none of which ride catalog.json — so every `vlm` catalog entry pairs with
    /// a preset here; the id is the one the model's card shows.
    static let byCatalogID: [String: VLModelID] = [
        "qwen3-vl-2b": .qwen3VL2B,
        "qwen3-vl-4b": .qwen3VL4B,
        "qwen3-vl-8b": .qwen3VL8B,
        "holo2-4b": .holo2_4B,
        "minicpm-v-4.6": .miniCPMV46,
    ]
}

/// A Core AI VL bundle as a `LanguageModelSession` provider.
public struct KitVisionModel: LanguageModel {
    public typealias Executor = KitVisionExecutor

    let runtime: VLRuntime
    let modelID: String
    let profile: OutputProfile

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        // Qwen3-VL is a thinking model; reasoning streams as `.reasoning`. Tool calling and
        // guided generation are not wired for the VL path in v1.
        if profile.rules.contains(where: { $0.kind == .thinking }) {
            capabilities.append(.reasoning)
        }
        return LanguageModelCapabilities(capabilities: capabilities)
    }

    public var executorConfiguration: KitVisionExecutor.Configuration {
        KitVisionExecutor.Configuration(runtime: runtime, modelID: modelID, profile: profile)
    }

    /// Loads a vision-language model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let vlm = try await KitVisionModel(catalog: "qwen3-vl-2b")
    /// ```
    ///
    /// Resolves the platform variant from the live catalog (built-in snapshot offline) and
    /// the decoder + vision bundle pair internally — consumers never touch bundle paths.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .vlm)
        guard let model = VLModelID.byCatalogID[entry.id] else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(model: model, store: store, downloadProgress: downloadProgress)
    }

    /// Downloads the decoder + vision bundles from the Hub (if needed) and loads them.
    public init(
        model: VLModelID,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        let visionRoot = try await store.download(model.vision, progress: downloadProgress)
        try await self.init(
            decoderBundleAt: decoderURL,
            visionModelAt: Self.resolveVisionModel(in: visionRoot),
            arch: model.arch)
    }

    /// Loads a local decoder bundle directory + a local vision `.aimodel`.
    public init(
        decoderBundleAt decoderURL: URL, visionModelAt visionURL: URL, arch: VLArchitecture,
        modelID: String? = nil
    ) async throws {
        let runtime = try await VLRuntime(
            decoderBundleAt: decoderURL, visionModelAt: visionURL, arch: arch)
        self.init(runtime: runtime, modelID: modelID ?? decoderURL.standardizedFileURL.path)
    }

    /// Wraps an already-created runtime. `modelID` keys the session's executor cache.
    public init(runtime: VLRuntime, modelID: String) {
        self.runtime = runtime
        self.modelID = modelID
        self.profile = OutputProfile.detect(probing: runtime.tokenizer)
    }

    /// The `.aimodel` inside a downloaded vision bundle root (the single `*.aimodel` entry, or
    /// the conventional `<root-name>.aimodel`).
    private static func resolveVisionModel(in root: URL) throws -> URL {
        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil),
            let aimodel = entries.first(where: { $0.pathExtension == "aimodel" })
        {
            return aimodel
        }
        return root.appendingPathComponent("\(root.lastPathComponent).aimodel")
    }
}
