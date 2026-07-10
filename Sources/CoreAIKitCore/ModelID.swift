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

    /// A copy of this id pinned to `revision` (a Hub commit hash) — the same bundle,
    /// addressed immutably. `nil` leaves the id unchanged, so callers can pass a catalog
    /// entry's optional pin straight through.
    public func pinned(_ revision: String?) -> ModelID {
        guard let revision else { return self }
        return ModelID(repo, path: path, revision: revision)
    }
}

// MARK: - Starter catalog
//
// Official-recipe exports of upstream models, hosted ready to download. The qwen3 repos carry
// both "macos" (4-bit GPU, dynamic shapes) and "ios" (mixed 4/8-bit, static ANE) variants;
// the larger models are macOS-only and fail with a clear error when requested on iOS.
extension ModelID {
    /// MiniCPM5-1B (OpenBMB, Apache-2.0). 1.08B on-device LLM, hybrid Think/No-Think, 128K.
    /// Ship = weight-only int8 (lossless, iPhone 17 Pro 66.8 tok/s decode). Dynamic-shape
    /// bundle on the pipelined engine; same bundle runs on macOS and iOS.
    public static let miniCPM5_1B = ModelID(
        "mlboydaisuke/MiniCPM5-1B-CoreAI", path: "int8")
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
    /// Depth Anything 3 (small, fp16 — 54 MB monocular depth). Same bundle on both platforms.
    public static let depthAnything3Small = ModelID(
        "mlboydaisuke/Depth-Anything-3-CoreAI", path: "small/da3-small_float16.aimodel")
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
    /// YOLOX-S 640px single-stage anchor-free detector (Megvii, Apache-2.0, fp32). Unlike the
    /// DETR family this is a dense detector — `YOLOXDetector` does the obj·cls threshold +
    /// per-class NMS host-side. Same bundle on both platforms. (Upload pending: until the repo
    /// is live, sideload `yolox-s_float32.aimodel` into Documents/Models/.)
    public static let yoloxS = ModelID(
        "mlboydaisuke/YOLOX-CoreAI", path: "yolox-s_float32.aimodel")
    /// AdcSR ×4 one-step diffusion-GAN super-resolution (CVPR 2025). Apache-2.0 code; weights
    /// derived from Stable Diffusion 2.1 (CreativeML OpenRAIL++-M — commercial use OK, the same
    /// license under which Apple ships SD CoreML). One 128→512 tile (`SuperResolver` tiles any-size
    /// input); image→image, no noise/prompt, RAW output (SuperResolver applies the per-image
    /// color-match host-side). fp32 (~1.7 GB) — the pruned SD-2.1 UNet's attention/group-norm
    /// overflow in fp16 (NaN on smooth tiles), so fp32 ships.
    public static let adcsrX4 = ModelID(
        "mlboydaisuke/AdcSR-CoreAI", path: "adcsr_x4_float32.aimodel")
    /// Whisper large-v3-turbo (OpenAI, MIT) — the zoo's first official ASR. fp16 fixed-128-window
    /// encoder-decoder transcription graph + HF tokenizer. Platform subtrees: `macos/` is the stock
    /// JIT `.aimodel` (Mac JIT-compiles fine); `ios/` is the AOT-compiled `.aimodelc` (h18p — the
    /// on-device JIT compiler aborts on the 1.6 GB graph). `path` is nil so `resolvedPath` picks the
    /// right one. Driven by `KitWhisperModel`. ≤30 s clips, 100 languages.
    public static let whisperLargeV3Turbo = ModelID(
        "mlboydaisuke/whisper-large-v3-turbo-CoreAI-official")
    /// ColModernVBERT visual document retriever (MIT) — query/text encoder (fp16, 298 MB).
    /// The `query/` subtree holds the `.aimodel` + `tokenizer/`; pair with `colModernVBERTDoc`.
    /// Drives `VisualDocumentRetriever` (late-interaction / MaxSim, OCR-free page retrieval).
    public static let colModernVBERTQuery = ModelID(
        "mlboydaisuke/ColModernVBERT-CoreAI", path: "query")
    /// ColModernVBERT visual document retriever (MIT) — document/image encoder (fp16, 407 MB,
    /// single 512px tile). The `doc/` subtree holds the `.aimodel`. Pair with `colModernVBERTQuery`.
    public static let colModernVBERTDoc = ModelID(
        "mlboydaisuke/ColModernVBERT-CoreAI", path: "doc")
    /// NVIDIA Parakeet-TDT-0.6B-v3 (cc-by-4.0) — the zoo's first transducer / TDT (RNN-T family)
    /// ASR. FastConformer encoder (fp16) + LSTM predictor + joint (fp32), 25 EU languages, ≤~29 s
    /// clips. Flat repo (bundle at root): the three `.aimodel` graphs + `tokenizer.json`. Driven by
    /// `KitParakeetModel`. (Upload pending — until live, sideload the bundle into Documents/Models/.)
    public static let parakeetTDT = ModelID(
        "mlboydaisuke/Parakeet-TDT-0.6B-CoreAI", path: "")
    /// NVIDIA Nemotron 3.5 ASR Streaming 0.6B (OpenMDW-1.1) — the zoo's first STREAMING ASR.
    /// Cache-aware FastConformer (fp16, explicit KV/conv caches, 320 ms chunks) + pure-RNNT
    /// LSTM predictor + joint (fp32). 40 locales in one checkpoint (language one-hot input),
    /// punctuation + capitalization built in, any-length audio. Platform subtrees like Whisper:
    /// `macos/` = six JIT `.aimodel` graphs; `ios/` = the two conformer halves AOT-compiled to
    /// h18p `.aimodelc` (a single >2 GB AOT bundle fails to load on-device) + the four small JIT
    /// graphs. `path` is nil so `resolvedPath` picks the right one. Driven by `KitNemotronModel`.
    public static let nemotronASRStreaming = ModelID(
        "mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI")
    /// NVIDIA Streaming Sortformer 4-spk v2 (cc-by-4.0, 117M) — the zoo's first speaker
    /// diarization model: "who spoke when", up to 4 speakers, 80 ms frames. Only the neural core
    /// is a Core AI graph (fixed-buffer forward, fp16); the NeMo 128-mel frontend, streaming
    /// chunk loop, and AOSC speaker-cache compression run in the Swift host. Flat repo, one graph
    /// per platform — the path picks the JIT `.aimodel` on macOS / the AOT h18p `.aimodelc` on
    /// iOS (the device JIT is avoided) so each platform downloads only its own form. The mel
    /// filterbank ships inside CoreAIKit (the Parakeet resource — same NeMo family). Driven by
    /// `KitDiarizer`.
    #if os(iOS)
    public static let sortformerDiarV2 = ModelID(
        "mlboydaisuke/Streaming-Sortformer-Diar-CoreAI", path: "sortformer_float16.h18p.aimodelc")
    #else
    public static let sortformerDiarV2 = ModelID(
        "mlboydaisuke/Streaming-Sortformer-Diar-CoreAI", path: "sortformer_float16.aimodel")
    #endif
    /// GLiNER2-PII (fastino, Apache-2.0) — the zoo's first NER / schema-driven information-extraction
    /// model and its first DeBERTa-v3 (disentangled-attention) port. mDeBERTa-v3 + SpanMarker +
    /// CountLSTM fused into one static graph (fp16); pass any label set at call time (zero-shot,
    /// ≤MMAX). Flagship use is on-device PII redaction. Platform subtrees like Whisper: `macos/` is
    /// the JIT `.aimodel` (~582 MB), `ios/` is the AOT-compiled h18p bundle (~823 MB — the device JIT
    /// is avoided). `path` is nil so `resolvedPath` picks the right one. Each subtree holds the
    /// `.aimodel` + `tokenizer/` + `extractor.json`. Driven by `InformationExtractor`.
    public static let gliner2PII = ModelID(
        "mlboydaisuke/GLiNER2-PII-CoreAI")
}
