// ModelID.swift — identifies a hosted Core AI model bundle on the Hugging Face Hub.
//
// A model is addressed as repo + path + revision, where `path` is a variant subtree inside
// the repo holding one complete bundle (metadata.json + *.aimodel/ + tokenizer/). When `path`
// is nil the platform default is used: "macos" on macOS, "ios" on iOS — the layout of the
// `*-CoreAI-official` starter repos.

import Foundation

public struct ModelID: Hashable, Sendable {
    public let repo: String
    public let path: String?
    public let revision: String

    public init(_ repo: String, path: String? = nil, revision: String = "main") {
        self.repo = repo
        self.path = path
        self.revision = revision
    }

    /// The repo subtree downloaded for this platform.
    public var resolvedPath: String {
        if let path { return path }
        #if os(iOS)
        return "ios"
        #else
        return "macos"
        #endif
    }

    /// Directory under a store root where this model is cached.
    var cacheSubpath: String { "\(repo)/\(revision)/\(resolvedPath)" }
}

// MARK: - Starter catalog
//
// Official-recipe exports of upstream models, hosted ready to download. The qwen3 repos carry
// both "macos" (4-bit GPU, dynamic shapes) and "ios" (mixed 4/8-bit, static ANE) variants;
// the larger models are macOS-only and fail with a clear error when requested on iOS.
extension ModelID {
    /// Qwen3 0.6B (Apache-2.0). Smallest starter; macOS + iOS variants.
    public static let qwen3_0_6B = ModelID("mlboydaisuke/qwen3-0.6b-CoreAI-official")
    /// Qwen3 4B (Apache-2.0). macOS + iOS variants.
    public static let qwen3_4B = ModelID("mlboydaisuke/qwen3-4b-CoreAI-official")
    /// Mistral 7B v0.3 (Apache-2.0). macOS only.
    public static let mistral_7B = ModelID("mlboydaisuke/mistral-7b-v0.3-CoreAI-official")
    /// Gemma 3 4B instruction-tuned. macOS only.
    public static let gemma3_4B = ModelID("mlboydaisuke/gemma-3-4b-it-CoreAI-official")
    /// CLIP ViT-B/32 joint image+text encoder (fp16). Same bundle on both platforms.
    public static let clipViTB32 = ModelID(
        "mlboydaisuke/clip-vit-base-patch32-CoreAI-official", path: "model")
    /// Depth Anything 3 (small, fp32). Same bundle on both platforms.
    public static let depthAnything3Small = ModelID(
        "mlboydaisuke/depth-anything-3-small-CoreAI-official", path: "model")
}
