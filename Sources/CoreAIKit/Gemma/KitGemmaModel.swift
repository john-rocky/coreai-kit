// KitGemmaModel.swift — a Gemma 4 (E2B/E4B) Core AI bundle behind FoundationModels'
// `LanguageModel` protocol, so a local Gemma 4 answers through a plain `LanguageModelSession`
// (and "Ask Gemma 4" from Siri / App Intents falls out for free).
//
// ```swift
// let model = try await KitGemmaModel(model: .gemma4E2B)
// let session = LanguageModelSession(model: model)
// let answer = try await session.respond(to: "What is the capital of France?")
// ```
//
// The Gemma path is its own executor (`KitGemmaExecutor`) and runtime (`GemmaRuntime`): the
// text `KitLanguageModel` and the VL path are untouched. v1 is plain Q&A — tool calling and
// guided generation are not advertised (Gemma emits a non-Hermes call syntax, and the
// pipelined engine samples on-GPU with no per-step logits); thinking is off (the renderer
// closes an empty thought channel) so the model answers directly.

import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

/// A downloadable Gemma 4 model = a `…_tbl` decode bundle + its paired PLE tables directory,
/// addressed as two paths inside the same Hugging Face repo, plus the chat format.
public struct GemmaModelID: Sendable, Hashable {
    public let decoder: ModelID
    public let tables: ModelID
    public let arch: GemmaArchitecture

    public init(decoder: ModelID, tables: ModelID, arch: GemmaArchitecture = .gemma4) {
        self.decoder = decoder
        self.tables = tables
        self.arch = arch
    }

    /// Gemma 4 E2B, official-QAT int4 (license: gemma). The shippable on-device size — its
    /// 2.0 GB pipelined `…_tbl` bundle + 2.4 GB PLE tables run on both Mac and iPhone 17 Pro
    /// (device builds use the paired AOT `…_tbl_aotc_h18p` bundle). int4 ≈ bf16 by design
    /// (Google's QAT). The QAT bundle is paired with the QAT tables.
    public static let gemma4E2B = GemmaModelID(
        decoder: ModelID(
            "mlboydaisuke/gemma-4-E2B-CoreAI",
            path: "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl"),
        tables: ModelID(
            "mlboydaisuke/gemma-4-E2B-CoreAI",
            path: "ios-frontend/gemma4_qat_gather_raw"),
        arch: .gemma4)

    /// Gemma 4 E4B, official-QAT int4 (license: gemma). A 4B-class Gemma — Mac-comfortable
    /// (3.7 GB bundle + 2.7 GB tables); on iPhone the static-table form exceeds the memory
    /// budget (the provider-mode bundle is the device path — a follow-up).
    public static let gemma4E4B = GemmaModelID(
        decoder: ModelID(
            "mlboydaisuke/gemma-4-E4B-CoreAI",
            path: "gpu-pipelined/gemma4_e4b_qat_decode_int4lin_tbl"),
        tables: ModelID(
            "mlboydaisuke/gemma-4-E4B-CoreAI",
            path: "ios-frontend/gemma4_e4b_qat_gather_raw"),
        arch: .gemma4)
}

/// A Core AI Gemma 4 bundle as a `LanguageModelSession` provider.
public struct KitGemmaModel: LanguageModel {
    public typealias Executor = KitGemmaExecutor

    let runtime: GemmaRuntime
    let modelID: String
    let profile: OutputProfile

    /// v1 is plain Q&A: no tool calling, no guided generation. Reasoning is advertised because
    /// Gemma 4 may open a `<|channel>thought … <channel|>` span, which streams as `.reasoning`.
    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = []
        if profile.rules.contains(where: { $0.kind == .thinking }) {
            capabilities.append(.reasoning)
        }
        return LanguageModelCapabilities(capabilities: capabilities)
    }

    public var executorConfiguration: KitGemmaExecutor.Configuration {
        KitGemmaExecutor.Configuration(runtime: runtime, modelID: modelID, profile: profile)
    }

    /// Downloads the decode bundle + its paired PLE tables from the Hub (if needed) and loads
    /// them.
    public init(
        model: GemmaModelID,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        let tablesURL = try await store.download(model.tables, progress: downloadProgress)
        try await self.init(
            decoderBundleAt: decoderURL, tablesAt: tablesURL, arch: model.arch)
    }

    /// Loads a local decode bundle directory + a local PLE tables directory.
    public init(
        decoderBundleAt decoderURL: URL,
        tablesAt tablesURL: URL,
        arch: GemmaArchitecture = .gemma4,
        modelID: String? = nil
    ) async throws {
        let runtime = try await GemmaRuntime(
            decoderBundleAt: decoderURL, tablesAt: tablesURL, arch: arch)
        self.init(runtime: runtime, modelID: modelID ?? decoderURL.standardizedFileURL.path)
    }

    /// Wraps an already-created runtime. `modelID` keys the session's executor cache.
    public init(runtime: GemmaRuntime, modelID: String) {
        self.runtime = runtime
        self.modelID = modelID
        // Gemma 4 frames an optional reasoning span as `<|channel>thought\n … <channel|>`. With
        // `.stream` default, a direct answer flows straight out as response; when the model does
        // think, the span is routed to `.reasoning` and the answer after `<channel|>` follows as
        // response. The `<turn|>` / `<eos>` stop tokens are filtered in the executor before the
        // parser, so they never reach the text.
        self.profile = OutputProfile(
            rules: [
                OutputRule(
                    open: runtime.arch.thoughtOpen, close: [runtime.arch.thoughtClose],
                    kind: .thinking, streamsBody: true)
            ],
            defaultText: .stream)
    }
}
