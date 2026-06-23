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
    /// EmbeddingGemma 300m text embeddings (fp16, 768-d). Same bundle on both platforms.
    public static let embeddingGemma300m = ModelID(
        "mlboydaisuke/embeddinggemma-300m-CoreAI", path: "model")
    /// Qwen3-Embedding 0.6B multilingual text embeddings (Apache-2.0, fp16, 1024-d, MRL).
    /// Flat repo (bundle at root). Same bundle on both platforms.
    public static let qwen3Embedding0_6B = ModelID(
        "mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI", path: "")
    /// Qwen3-Reranker 0.6B cross-encoder reranker (Apache-2.0, fp16; yes/no logit score).
    /// Flat repo (bundle at root). Same bundle on both platforms.
    public static let qwen3Reranker0_6B = ModelID(
        "mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI", path: "")
    /// RF-DETR object detection, nano 384px (Apache-2.0, fp32, no NMS). Same bundle on both platforms.
    public static let rfdetrNano = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "rfdetr-nano_float32.aimodel")
    /// RF-DETR object detection, small 512px (Apache-2.0, fp32, no NMS).
    public static let rfdetrSmall = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "rfdetr-small_float32.aimodel")
    /// RF-DETR object detection, medium 576px (Apache-2.0, fp32, no NMS).
    public static let rfdetrMedium = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "rfdetr-medium_float32.aimodel")
    /// RF-DETR object detection, large 704px (Apache-2.0, fp32, no NMS).
    public static let rfdetrLarge = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "rfdetr-large_float32.aimodel")
    /// RF-DETR nano split deployment: ViT backbone (pair with `rfdetrNanoHead`).
    public static let rfdetrNanoBackbone = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "split/rfdetr-nano_backbone.aimodel")
    /// RF-DETR nano split deployment: deformable head.
    public static let rfdetrNanoHead = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "split/rfdetr-nano_head.aimodel")
    /// RF-DETR medium split deployment: ViT backbone (pair with `rfdetrMediumHead`).
    public static let rfdetrMediumBackbone = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "split/rfdetr-medium_backbone.aimodel")
    /// RF-DETR medium split deployment: deformable head.
    public static let rfdetrMediumHead = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "split/rfdetr-medium_head.aimodel")
    /// RF-DETR-Seg nano 312px instance segmentation (Apache-2.0, fp32, no NMS).
    public static let rfdetrSegNano = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "rfdetr-seg-nano_float32.aimodel")
    /// RF-DETR-Seg medium 432px instance segmentation (Apache-2.0, fp32, no NMS).
    public static let rfdetrSegMedium = ModelID(
        "mlboydaisuke/RF-DETR-CoreAI", path: "rfdetr-seg-medium_float32.aimodel")
    /// AdcSR ×4 one-step diffusion-GAN super-resolution (CVPR 2025). Apache-2.0 code; weights
    /// derived from Stable Diffusion 2.1 (CreativeML OpenRAIL++-M — commercial use OK, the same
    /// license under which Apple ships SD CoreML). One 128→512 tile (`SuperResolver` tiles any-size
    /// input); image→image, no noise/prompt, RAW output (SuperResolver applies the per-image
    /// color-match host-side). fp32 (~1.7 GB) — the pruned SD-2.1 UNet's attention/group-norm
    /// overflow in fp16 (NaN on smooth tiles), so fp32 ships.
    public static let adcsrX4 = ModelID(
        "mlboydaisuke/AdcSR-CoreAI", path: "adcsr_x4_float32.aimodel")
}
