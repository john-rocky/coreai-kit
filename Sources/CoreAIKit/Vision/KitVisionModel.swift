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

    /// LFM2.5-VL-450M: a SigLIP2-NaFlex tower (host-patchified 512x512 → `image_embeds`
    /// [256, 1024]) + the LFM2 hybrid decoder. The zoo's smallest VLM: 658 MB for the pair,
    /// device-gated on an iPhone 17 Pro — 112 tok/s decode with the image bound, 33.6 ms per
    /// encode.
    ///
    /// iOS takes the **AOT** subtree, which is the configuration that was gated: a JIT
    /// `.aimodel` specializes on device instead, and that path has not been measured here.
    /// (`ModelID`'s own platform default is the `ios`/`macos` layout of the starter repos,
    /// which this repo does not use, so the choice is made here.)
    public static let lfm2VL450M = {
        let repo = "mlboydaisuke/LFM2.5-VL-450M-CoreAI"
        #if os(iOS)
        let subtree = "ios-h18p"
        #else
        let subtree = "gpu-pipelined"
        #endif
        return VLModelID(
            decoder: ModelID(repo, path: "\(subtree)/lfm2_5_vl_450m_decode_int8lin"),
            vision: ModelID(repo, path: "\(subtree)/lfm2_5_vl_450m_vision_fp16"),
            arch: .lfm2VL450M)
    }()

    /// LFM2.5-VL-3B: the same two-bundle shape as the 450M with a wider tower and a 128k
    /// vocab. Pick it over the 450M when the answer has to hold detail; the 450M when the
    /// budget has to hold an app.
    /// Different QUANTIZATION per platform, not just a different subtree: the int8 decoder's
    /// AOT `resources.bin` is 3.13 GiB and does not load on iOS, while int4's 2.03 GiB does —
    /// and on this model int4 costs nothing (7/9 on the suite, the same cases as fp16). Mac
    /// takes int8 because it has no wall to clear. Both are device- or Mac-gated as shipped.
    public static let lfm2VL3B = {
        let repo = "mlboydaisuke/LFM2.5-VL-3B-CoreAI"
        #if os(iOS)
        let decoder = "ios-h18p/lfm2_5_vl_3b_decode_int4lin"
        let vision = "ios-h18p/lfm2_5_vl_3b_vision_fp16"
        #else
        let decoder = "gpu-pipelined/lfm2_5_vl_3b_decode_int8lin"
        let vision = "gpu-pipelined/lfm2_5_vl_3b_vision_fp16"
        #endif
        return VLModelID(
            decoder: ModelID(repo, path: decoder),
            vision: ModelID(repo, path: vision),
            arch: .lfm2VL3B)
    }()

    /// North-Micro-Vision (Cohere, 2.4B, Apache-2.0): 11 languages, and token-exact against
    /// fp32 on an iPhone 17 Pro. int8 on both platforms — int4 craters on this model.
    public static let northMicroVision = {
        let repo = "mlboydaisuke/North-Micro-Vision-CoreAI"
        #if os(iOS)
        let subtree = "ios-h18p"
        #else
        let subtree = "gpu-pipelined"
        #endif
        return VLModelID(
            decoder: ModelID(repo, path: "\(subtree)/north_micro_vision_instruct_decode_int8lin"),
            vision: ModelID(repo, path: "\(subtree)/north_micro_vision_instruct_vision_fp16"),
            arch: .northMicroVision)
    }()

    /// Presets by catalog id. A VL model is two bundles (decoder + vision tower) plus graph
    /// geometry — none of which ride catalog.json — so every `vlm` catalog entry pairs with
    /// a preset here; the id is the one the model's card shows.
    static let byCatalogID: [String: VLModelID] = [
        "qwen3-vl-2b": .qwen3VL2B,
        "qwen3-vl-4b": .qwen3VL4B,
        "qwen3-vl-8b": .qwen3VL8B,
        "holo2-4b": .holo2_4B,
        "minicpm-v-4.6": .miniCPMV46,
        "lfm2.5-vl-450m": .lfm2VL450M,
        "lfm2.5-vl-3b": .lfm2VL3B,
        "north-micro-vision": .northMicroVision,
    ]

    /// A copy with both sub-bundle ids pinned to a Hub revision (nil = unchanged), so a
    /// catalog entry's pin covers the decoder and its paired vision tower alike.
    public func pinned(_ revision: String?) -> VLModelID {
        guard let revision else { return self }
        return VLModelID(
            decoder: decoder.pinned(revision), vision: vision.pinned(revision), arch: arch)
    }
}

/// A Core AI VL bundle as a `LanguageModelSession` provider.
public struct KitVisionModel: LanguageModel {
    public typealias Executor = KitVisionExecutor

    let runtime: VLRuntime
    let modelID: String
    let profile: OutputProfile

    public var capabilities: LanguageModelCapabilities {
        // Advertise image input; the framework may gate attachments on this in future betas.
        var capabilities: [LanguageModelCapabilities.Capability] = [.vision]
        // Qwen3-VL is a thinking model; reasoning streams as `.reasoning`. Tool calling and
        // guided generation are not wired for the VL path in v1.
        if profile.rules.contains(where: { $0.kind == .thinking }) {
            capabilities.append(.reasoning)
        }
        return LanguageModelCapabilities(capabilities)
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
        try await self.init(
            model: model.pinned(entry.revision), store: store, downloadProgress: downloadProgress)
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
