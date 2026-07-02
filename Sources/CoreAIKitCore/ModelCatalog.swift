// ModelCatalog.swift — a remote, updatable list of known-good hosted models, so new
// models reach apps without a package update. Fetched from the repo's catalog.json with
// a built-in snapshot as offline fallback.

import Foundation

public struct CatalogEntry: Sendable, Identifiable, Codable, Hashable {
    public enum Kind: String, Sendable, Codable {
        case chat
        case imageText
        case depth
        case textEmbedding
        case superResolution
        /// Speech-to-text (Whisper / Qwen3-ASR / Parakeet).
        case asr
        /// Object detection (RF-DETR / YOLOX).
        case detection
        /// Vision-language chat (Qwen3-VL): image + prompt → answer.
        case vlm
        /// Text-to-speech (VoxCPM).
        case tts
        /// Forward-compat: a kind this build doesn't know (e.g. a newer catalog.json entry).
        /// Such entries decode cleanly and are simply filtered out of `available(_:)`.
        case unknown

        public init(from decoder: any Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Kind(rawValue: raw) ?? .unknown
        }
    }

    public struct Variant: Sendable, Codable, Hashable {
        /// Subtree inside the repo holding the bundle.
        public let path: String
        /// Approximate download size, for UI.
        public let sizeMB: Int?

        public init(path: String, sizeMB: Int? = nil) {
            self.path = path
            self.sizeMB = sizeMB
        }
    }

    public let id: String
    public let name: String
    public let repo: String
    public let kind: Kind
    /// Keyed by platform: "macos" / "ios". A missing key = not published there.
    public let variants: [String: Variant]
    public let thinking: Bool?
    /// Engine override hint: "sequential" / "pipelined" / "static-shape"; nil = auto-detect.
    /// Zoo decode-only ports hint "pipelined": their S=1 graphs need S=1 prefill (the
    /// runtime's chunk threshold drops to 1 on load) and the hybrid ones (extra conv/SSM
    /// states) only load on the pipelined engine — "sequential" validates exactly 2 states
    /// and rejects them. Official-recipe bundles (dynamic shapes) leave this nil (auto).
    public let engine: String?

    public init(
        id: String, name: String, repo: String, kind: Kind,
        variants: [String: Variant], thinking: Bool? = nil, engine: String? = nil
    ) {
        self.id = id
        self.name = name
        self.repo = repo
        self.kind = kind
        self.variants = variants
        self.thinking = thinking
        self.engine = engine
    }

    static var platformKey: String {
        #if os(iOS)
        return "ios"
        #else
        return "macos"
        #endif
    }

    /// The variant for this platform, or nil if the model is not published here.
    public var variant: Variant? { variants[Self.platformKey] }

    /// `ModelID` for this platform, or nil if the model is not published here.
    public var modelID: ModelID? {
        variant.map { ModelID(repo, path: $0.path) }
    }
}

public struct ModelCatalog: Sendable, Codable {
    public let version: Int
    public let models: [CatalogEntry]

    public init(version: Int, models: [CatalogEntry]) {
        self.version = version
        self.models = models
    }

    /// Entries available on this platform, optionally filtered by kind.
    public func available(_ kind: CatalogEntry.Kind? = nil) -> [CatalogEntry] {
        models.filter { entry in
            entry.modelID != nil && entry.kind != .unknown
                && (kind == nil || entry.kind == kind)
        }
    }

    public static let defaultURL = URL(
        string: "https://raw.githubusercontent.com/john-rocky/coreai-kit/main/catalog.json")!

    public static func fetch(from url: URL = defaultURL) async throws -> ModelCatalog {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw CoreAIKitError.httpError(statusCode: -1, file: url.lastPathComponent)
        }
        return try JSONDecoder().decode(ModelCatalog.self, from: data)
    }

    /// Fetches the live catalog, falling back to the built-in snapshot — never throws.
    public static func load(from url: URL = defaultURL) async -> ModelCatalog {
        (try? await fetch(from: url)) ?? .builtin
    }

    /// The entry with this catalog id, or nil.
    public func entry(id: String) -> CatalogEntry? {
        models.first { $0.id == id }
    }

    /// Resolves a catalog id against the live catalog (built-in snapshot offline) to an entry
    /// available on this platform. This is what the `catalog:` model initializers ride, so a
    /// card's id works verbatim: unknown id and wrong-platform failures throw with the id in
    /// the message instead of surfacing as a download error.
    public static func entry(
        forID id: String, expecting kind: CatalogEntry.Kind? = nil, from url: URL = defaultURL
    ) async throws -> CatalogEntry {
        let entry = await load(from: url).entry(id: id) ?? ModelCatalog.builtin.entry(id: id)
        guard let entry else { throw CoreAIKitError.modelNotInCatalog(id: id) }
        if let kind, entry.kind != kind {
            throw CoreAIKitError.catalogKindMismatch(
                id: id, expected: kind.rawValue, found: entry.kind.rawValue)
        }
        guard entry.modelID != nil else {
            throw CoreAIKitError.modelNotAvailableOnPlatform(id: id)
        }
        return entry
    }

    /// Snapshot of the hosted models at packaging time (mirrors catalog.json).
    public static let builtin = ModelCatalog(
        version: 1,
        models: [
            CatalogEntry(
                id: "qwen3-0.6b", name: "Qwen3 0.6B",
                repo: "mlboydaisuke/qwen3-0.6b-CoreAI-official", kind: .chat,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 352),
                    "ios": .init(path: "ios", sizeMB: 454),
                ],
                thinking: true),
            CatalogEntry(
                id: "qwen3-4b", name: "Qwen3 4B",
                repo: "mlboydaisuke/qwen3-4b-CoreAI-official", kind: .chat,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 2280),
                    "ios": .init(path: "ios", sizeMB: 2498),
                ],
                thinking: true),
            CatalogEntry(
                id: "mistral-7b-v0.3", name: "Mistral 7B v0.3",
                repo: "mlboydaisuke/mistral-7b-v0.3-CoreAI-official", kind: .chat,
                variants: ["macos": .init(path: "macos", sizeMB: 4078)]),
            CatalogEntry(
                id: "gemma-3-4b-it", name: "Gemma 3 4B",
                repo: "mlboydaisuke/gemma-3-4b-it-CoreAI-official", kind: .chat,
                variants: ["macos": .init(path: "macos", sizeMB: 2223)]),
            // ── Zoo ports that also run on iPhone (≤4B, JIT bundles) — S=1 decode-only
            //    graphs, hint "pipelined". The same gpu-pipelined bundle drives macOS + iOS. ──
            CatalogEntry(
                id: "qwen3.5-0.8b", name: "Qwen3.5 0.8B",
                repo: "mlboydaisuke/qwen3.5-0.8B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym", sizeMB: 1300),
                    "ios": .init(
                        path: "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym", sizeMB: 1300),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "qwen3.5-2b", name: "Qwen3.5 2B",
                repo: "mlboydaisuke/qwen3.5-2B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/qwen3_5_2b_decode_int8hu_perchan_sym", sizeMB: 2900),
                    "ios": .init(
                        path: "gpu-pipelined/qwen3_5_2b_decode_int8hu_perchan_sym", sizeMB: 2900),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "lfm2.5-1.2b", name: "LFM2.5 1.2B",
                repo: "mlboydaisuke/LFM2.5-1.2B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8hu_block32_sym",
                        sizeMB: 1702),
                    "ios": .init(
                        path: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8hu_block32_sym",
                        sizeMB: 1702),
                ],
                engine: "pipelined"),
            CatalogEntry(
                id: "granite-4.0-h-1b", name: "Granite 4.0-H 1B",
                repo: "mlboydaisuke/granite-4.0-h-CoreAI", kind: .chat,
                variants: [
                    // Split ship: Mac ships int8lin (136.5 tok/s), the device ships the
                    // untied-int8-head variant (+17–21% decode on iPhone).
                    "macos": .init(
                        path: "gpu-pipelined/granite_4_0_h_1b_decode_int8lin", sizeMB: 1706),
                    "ios": .init(
                        path: "gpu-pipelined/granite_4_0_h_1b_decode_int8hu_block32_sym",
                        sizeMB: 1872),
                ],
                engine: "pipelined"),
            CatalogEntry(
                id: "minicpm5-1b", name: "MiniCPM5 1B",
                repo: "mlboydaisuke/MiniCPM5-1B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(path: "int8", sizeMB: 2000),
                    "ios": .init(path: "int8", sizeMB: 2000),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "nanbeige4.1-3b", name: "Nanbeige4.1 3B",
                repo: "mlboydaisuke/Nanbeige4.1-3B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/nanbeige4_1_3b_decode_int8hu_block32_sym_s1",
                        sizeMB: 3900),
                    "ios": .init(
                        path: "gpu-pipelined/nanbeige4_1_3b_decode_int8hu_block32_sym_s1",
                        sizeMB: 3900),
                ],
                thinking: true, engine: "pipelined"),
            // ── More official-recipe chat (stock runtime, macOS) ──
            CatalogEntry(
                id: "qwen3-8b", name: "Qwen3 8B",
                repo: "mlboydaisuke/qwen3-8b-CoreAI-official", kind: .chat,
                variants: ["macos": .init(path: "macos", sizeMB: 4400)],
                thinking: true),
            CatalogEntry(
                id: "gemma-3-12b-it", name: "Gemma 3 12B",
                repo: "mlboydaisuke/gemma-3-12b-it-CoreAI-official", kind: .chat,
                variants: ["macos": .init(path: "macos", sizeMB: 6000)]),
            // ── Zoo community ports (decode-pipelined, some with custom Metal kernels) —
            //    S=1 decode-only graphs, hint "pipelined". macOS-only: they run well past a
            //    12 GB iPhone's per-process limit. Proven by the CoreAIChatMac dmg. ──
            CatalogEntry(
                id: "qwen3.6-35b-a3b", name: "Qwen3.6-35B-A3B (MoE)",
                repo: "mlboydaisuke/Qwen3.6-35B-A3B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/qwen3_6_35b_a3b_decode_sym8_gather", sizeMB: 35000)],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "qwen3.6-27b", name: "Qwen3.6-27B (dense)",
                repo: "mlboydaisuke/Qwen3.6-27B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/qwen3_6_27b_decode_int8hu_block32_sym", sizeMB: 28000)],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "glm-4.7-flash", name: "GLM-4.7-Flash (MoE+MLA)",
                repo: "mlboydaisuke/GLM-4.7-Flash-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/glm_4_7_flash_decode_sym8_gather", sizeMB: 30000)],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "lfm2.5-8b-a1b", name: "LFM2.5-8B-A1B (MoE)",
                repo: "mlboydaisuke/LFM2.5-8B-A1B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/lfm2_5_8b_a1b_decode_sym8_gather", sizeMB: 9000)],
                engine: "pipelined"),
            CatalogEntry(
                id: "gemma-4-12b", name: "Gemma 4 12B",
                repo: "mlboydaisuke/Gemma-4-12B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/gemma4_12b_qat_decode_int8lin_msdpa_g8", sizeMB: 13000)],
                engine: "pipelined"),
            CatalogEntry(
                id: "gemma-4-31b", name: "Gemma 4 31B",
                repo: "mlboydaisuke/Gemma-4-31B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/gemma4_31b_qat_decode_int4linsym_msdpa_g8", sizeMB: 18000)],
                engine: "pipelined"),
            // ── Vision-language (image + prompt → answer). A VL model is TWO bundles
            //    (decoder + vision tower); `path` names the decoder (the primary artifact,
            //    used for platform gating), sizeMB is the decoder + vision total the first
            //    run actually downloads. KitVisionModel(catalog:) resolves both bundle
            //    paths + graph geometry via VLModelID.byCatalogID. 8B is Mac-only: its
            //    decoder exceeds the iPhone jetsam ceiling. ──
            CatalogEntry(
                id: "qwen3-vl-2b", name: "Qwen3-VL 2B",
                repo: "mlboydaisuke/Qwen3-VL-2B-CoreAI", kind: .vlm,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1", sizeMB: 3278),
                    "ios": .init(
                        path: "gpu-pipelined/qwen3_vl_2b_instruct_decode_int8hu_s1", sizeMB: 3278),
                ]),
            CatalogEntry(
                id: "qwen3-vl-4b", name: "Qwen3-VL 4B",
                repo: "mlboydaisuke/Qwen3-VL-4B-CoreAI", kind: .vlm,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/qwen3_vl_4b_instruct_decode_int8hu_s1", sizeMB: 5897),
                    "ios": .init(
                        path: "gpu-pipelined/qwen3_vl_4b_instruct_decode_int8hu_s1", sizeMB: 5897),
                ]),
            CatalogEntry(
                id: "qwen3-vl-8b", name: "Qwen3-VL 8B",
                repo: "mlboydaisuke/Qwen3-VL-8B-CoreAI", kind: .vlm,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/qwen3_vl_8b_instruct_decode_int8hu_s1", sizeMB: 10453)
                ]),
            // ── Text-to-speech. A VoxCPM voice is a family of graphs (base/res LM,
            //    diffusion, VAE, vocoder) plus tokenizer + host-glue tables; the variant
            //    path names the platform bundle dir and KitSpeaker resolves the rest.
            //    sizeMB is the platform + tokenizer + glue total the first run downloads. ──
            CatalogEntry(
                id: "voxcpm-0.5b", name: "VoxCPM 0.5B",
                repo: "mlboydaisuke/VoxCPM-0.5B-CoreAI", kind: .tts,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 1373),
                    "ios": .init(path: "ios", sizeMB: 1679),
                ]),
            // ── Speech-to-text ──
            CatalogEntry(
                id: "whisper-large-v3-turbo", name: "Whisper large-v3-turbo",
                repo: "mlboydaisuke/whisper-large-v3-turbo-CoreAI-official", kind: .asr,
                variants: [
                    // macos = stock JIT .aimodel; ios = AOT-compiled (h18p) — the on-device JIT
                    // aborts on the 1.6 GB graph. Driven by `KitWhisperModel`.
                    "macos": .init(path: "macos", sizeMB: 1620),
                    "ios": .init(path: "ios", sizeMB: 3240),
                ]),
            CatalogEntry(
                id: "qwen3-asr-1.7b", name: "Qwen3-ASR 1.7B",
                repo: "mlboydaisuke/Qwen3-ASR-1.7B-CoreAI", kind: .asr,
                // The variant path is the decoder bundle; `KitASRModel(catalog:)` resolves the
                // paired AuT encoder internally, and sizeMB covers both downloads. macOS-only:
                // the JIT decoder graph has no device-verified iOS path.
                variants: ["macos": .init(
                    path: "gpu-pipelined/qwen3_asr_1.7b_decode_int8hu_n390_s1", sizeMB: 3100)]),
            CatalogEntry(
                id: "parakeet-tdt-0.6b-v3", name: "Parakeet-TDT 0.6B v3",
                repo: "mlboydaisuke/Parakeet-TDT-0.6B-CoreAI", kind: .asr,
                // Flat repo (path ""): three JIT .aimodel graphs + tokenizer.json, driven by
                // `KitParakeetModel`. macOS-only: the published iPhone numbers rode an AOT
                // encoder this repo doesn't carry yet.
                variants: ["macos": .init(path: "", sizeMB: 1290)]),
            CatalogEntry(
                id: "clip-vit-b32", name: "CLIP ViT-B/32",
                repo: "mlboydaisuke/clip-vit-base-patch32-CoreAI-official", kind: .imageText,
                variants: [
                    "macos": .init(path: "model", sizeMB: 291),
                    "ios": .init(path: "model", sizeMB: 291),
                ]),
            CatalogEntry(
                id: "depth-anything-3-small", name: "Depth Anything 3 Small",
                repo: "mlboydaisuke/Depth-Anything-3-CoreAI", kind: .depth,
                variants: [
                    "macos": .init(path: "small/da3-small_float16.aimodel", sizeMB: 54),
                    "ios": .init(path: "small/da3-small_float16.aimodel", sizeMB: 54),
                ]),
            CatalogEntry(
                id: "embeddinggemma-300m", name: "EmbeddingGemma 300m",
                repo: "mlboydaisuke/embeddinggemma-300m-CoreAI", kind: .textEmbedding,
                variants: [
                    "macos": .init(path: "model", sizeMB: 1230),
                    "ios": .init(path: "model", sizeMB: 1230),
                ]),
            CatalogEntry(
                id: "rf-detr", name: "RF-DETR (nano)",
                repo: "mlboydaisuke/RF-DETR-CoreAI", kind: .detection,
                // One bundle per platform is the catalog contract, so the entry carries the
                // nano tier (DetectCamera's default); the other tiers stay reachable through
                // the `ModelID.rfdetr*` presets.
                variants: [
                    "macos": .init(path: "rfdetr-nano_float32.aimodel", sizeMB: 103),
                    "ios": .init(path: "rfdetr-nano_float32.aimodel", sizeMB: 103),
                ]),
            CatalogEntry(
                id: "adcsr-x4", name: "AdcSR ×4 Super-Resolution",
                repo: "mlboydaisuke/AdcSR-CoreAI", kind: .superResolution,
                variants: [
                    "macos": .init(path: "adcsr_x4_float32.aimodel", sizeMB: 1740),
                    "ios": .init(path: "adcsr_x4_float32.aimodel", sizeMB: 1740),
                ]),
        ])
}
