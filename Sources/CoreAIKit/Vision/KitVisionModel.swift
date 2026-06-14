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
