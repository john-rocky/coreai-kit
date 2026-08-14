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
        /// Speaker diarization (Streaming Sortformer): clip → who spoke when.
        case diarization
        /// Object detection (RF-DETR / YOLOX).
        case detection
        /// Vision-language chat (Qwen3-VL): image + prompt → answer.
        case vlm
        /// Text-to-speech (VoxCPM).
        case tts
        /// Text-to-music / audio generation (Stable Audio).
        case music
        /// Audio understanding (Qwen2.5-Omni): clip + question → answer.
        case audio
        /// Video understanding (V-JEPA 2): clip → action classification.
        case video
        /// Document OCR (Unlimited-OCR): image → structured markdown.
        case ocr
        /// Masked-diffusion language model (LLaDA): parallel canvas denoising, not AR chat.
        case dllm
        /// Visual document retrieval (ColModernVBERT): text query + page images → MaxSim
        /// ranking, no OCR.
        case retrieval
        /// Time-series forecasting (TimesFM 2.5): univariate series → point + quantile forecast.
        case forecasting
        /// Music source separation (Mel-Band RoFormer): song → vocals + instrumental stems.
        case separation
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
    /// Hub commit hash this entry is pinned to. When set, every bundle the entry resolves
    /// (including multi-subtree models) downloads from exactly this revision — the artifact
    /// that was verified — so a later push to the repo can't change what apps receive until
    /// the pin is deliberately bumped. `nil` (older catalogs) falls back to `main`.
    public let revision: String?
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
        id: String, name: String, repo: String, revision: String? = nil, kind: Kind,
        variants: [String: Variant], thinking: Bool? = nil, engine: String? = nil
    ) {
        self.id = id
        self.name = name
        self.repo = repo
        self.revision = revision
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
        variant.map { modelID(path: $0.path) }
    }

    /// A `ModelID` for an arbitrary subtree of this entry's repo, carrying the entry's
    /// revision pin. Multi-bundle models (VL towers, TTS glue, OCR assets) resolve their
    /// sibling paths through this so every part downloads from the same pinned revision.
    public func modelID(path: String) -> ModelID {
        ModelID(repo, path: path, revision: revision ?? "main")
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
    /// A copy of this catalog with revision pins filled in by id.
    ///
    /// An entry that already carries a pin keeps it; only `nil` revisions are filled. Used to
    /// pin `builtin` from `catalog.json` without hand-maintaining the hashes in two places.
    public func pinned(with pins: [String: String]) -> ModelCatalog {
        ModelCatalog(
            version: version,
            models: models.map { e in
                guard e.revision == nil, let rev = pins[e.id] else { return e }
                return CatalogEntry(
                    id: e.id, name: e.name, repo: e.repo, revision: rev, kind: e.kind,
                    variants: e.variants, thinking: e.thinking, engine: e.engine)
            })
    }

    /// The compiled-in fallback, used when the network fetch of `catalog.json` fails.
    ///
    /// Pins come from `BuiltinPins` (generated from `catalog.json`), because a fallback happens
    /// exactly when the network is unavailable — the moment a shipped app must *not* silently
    /// resolve `main` instead of the reviewed bytes. `CoreAIKitTests.builtinIsFullyPinned`
    /// fails the build if any entry here loses its pin.
    public static let builtin = builtinLiteral.pinned(with: BuiltinPins.byID)

    /// Everything about the built-in entries except their revisions: paths, sizes, engine
    /// hints, and the per-model notes that explain them.
    static let builtinLiteral = ModelCatalog(
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
                        path: "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym", sizeMB: 1276),
                    "ios": .init(
                        path: "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym", sizeMB: 1276),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "qwen3.5-2b", name: "Qwen3.5 2B",
                repo: "mlboydaisuke/qwen3.5-2B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/qwen3_5_2b_decode_int8hu_block32_sym", sizeMB: 2905),
                    "ios": .init(
                        path: "gpu-pipelined/qwen3_5_2b_decode_int8hu_block32_sym", sizeMB: 2905),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "youtu-llm-2b", name: "Youtu-LLM 2B",
                repo: "mlboydaisuke/Youtu-LLM-2B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/youtu_llm_2b_decode_absorbed_int8_msdpa", sizeMB: 2058),
                    "ios": .init(
                        path: "gpu-pipelined/youtu_llm_2b_decode_absorbed_int8_msdpa", sizeMB: 2058),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "lfm2.5-1.2b", name: "LFM2.5 1.2B",
                repo: "mlboydaisuke/LFM2.5-1.2B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8hu_block32_sym",
                        sizeMB: 1623),
                    "ios": .init(
                        path: "gpu-pipelined/lfm2_5_1_2b_instruct_decode_int8hu_block32_sym",
                        sizeMB: 1623),
                ],
                engine: "pipelined"),
            // macOS only on purpose: int8hu is 3.4 GB and even int4lin is 2.0 GB, right at the
            // iOS wall, and no device measurement has been taken. Add "ios" when one has.
            CatalogEntry(
                id: "lfm2.5-2.6b", name: "LFM2.5 2.6B",
                repo: "mlboydaisuke/LFM2.5-2.6B-CoreAI",
                revision: "79704a10ac98e49dabf7051543f151d6f5da3a83", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/lfm2_5_2_6b_decode_int8hu_block32_sym",
                        sizeMB: 3474),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "granite-4.0-h-1b", name: "Granite 4.0-H 1B",
                repo: "mlboydaisuke/granite-4.0-h-CoreAI", kind: .chat,
                variants: [
                    // Split ship: Mac ships int8lin (136.5 tok/s), the device ships the
                    // untied-int8-head variant (+17–21% decode on iPhone).
                    "macos": .init(
                        path: "gpu-pipelined/granite_4_0_h_1b_decode_int8lin", sizeMB: 1630),
                    "ios": .init(
                        path: "gpu-pipelined/granite_4_0_h_1b_decode_int8hu_block32_sym",
                        sizeMB: 1786),
                ],
                engine: "pipelined"),
            CatalogEntry(
                id: "nemotron-3-nano-4b", name: "Nemotron-3-Nano 4B",
                repo: "mlboydaisuke/Nemotron-3-Nano-4B-CoreAI", kind: .chat,
                variants: [
                    // A 4B graph cannot specialize on-device, so iOS ships the AOT h18p
                    // bundle; the Mac takes the JIT one. Same int8hu weights either way.
                    "macos": .init(
                        path: "gpu-pipelined/nemotron_3_nano_4b_decode_int8hu", sizeMB: 4626),
                    "ios": .init(
                        path: "ios-h18p/nemotron_3_nano_4b_decode_int8hu", sizeMB: 4626),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "minicpm5-1b", name: "MiniCPM5 1B",
                repo: "mlboydaisuke/MiniCPM5-1B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(path: "int8", sizeMB: 1092),
                    "ios": .init(path: "int8", sizeMB: 1092),
                ],
                thinking: true, engine: "pipelined"),
            CatalogEntry(
                id: "nanbeige4.1-3b", name: "Nanbeige4.1 3B",
                repo: "mlboydaisuke/Nanbeige4.1-3B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/nanbeige4_1_3b_decode_int8hu_block32_sym_s1",
                        sizeMB: 4387),
                    "ios": .init(
                        path: "gpu-pipelined/nanbeige4_1_3b_decode_int8hu_block32_sym_s1",
                        sizeMB: 4387),
                ],
                thinking: true, engine: "pipelined"),
            // First community-contributed model (zoo PR #6, @ukint-vs) — recurrent
            // two-pass Llama (22 physical blocks x2, 44 KV layers), contributor's HF repo.
            CatalogEntry(
                id: "nanbeige4.2-3b", name: "Nanbeige4.2 3B",
                repo: "ukint-vs/Nanbeige4.2-3B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/nanbeige4_2_3b_decode_int8hu_block32_sym_s1",
                        sizeMB: 4702),
                    "ios": .init(
                        path: "gpu-pipelined/nanbeige4_2_3b_decode_int8hu_block32_sym_s1",
                        sizeMB: 4702),
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
                variants: ["macos": .init(path: "macos", sizeMB: 6660)]),
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
                    path: "gpu-pipelined/gemma4_12b_qat_decode_int8lin_msdpa_g8", sizeMB: 14699)],
                engine: "pipelined"),
            CatalogEntry(
                id: "gemma-4-31b", name: "Gemma 4 31B",
                repo: "mlboydaisuke/Gemma-4-31B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/gemma4_31b_qat_decode_int4linsym_msdpa_g8", sizeMB: 20121)],
                engine: "pipelined"),
            // ── Gemma 4 E2B/E4B (per-layer embeddings, official-QAT int4): a `…_tbl`
            //    decode bundle + paired PLE tables bound as static graph inputs. The
            //    variant path names the decoder; ChatSession(catalog:) resolves both
            //    subtrees via GemmaModelID.byCatalogID and loads through GemmaRuntime.
            //    sizeMB is the decoder + tables total the first run downloads. E2B's ios
            //    variant is the AOT `…_aotc_h18p` decoder (the plain bundle crashes the
            //    on-device specializer; ~4.45 GB peak, so the driving app needs the
            //    increased-memory entitlement). E4B stays macOS-only — the static-table
            //    form exceeds the device memory budget. ──
            CatalogEntry(
                id: "gemma-4-e2b", name: "Gemma 4 E2B",
                repo: "mlboydaisuke/gemma-4-E2B-CoreAI", kind: .chat,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl", sizeMB: 4929),
                    "ios": .init(
                        path: "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p",
                        sizeMB: 4931),
                ],
                engine: "pipelined"),
            // (gemma-4-e2b-metal removed here to match catalog.json — the raw-Metal pack's
            // repo is not published yet; see 7d4600f. Re-add BOTH sides when it ships.)
            CatalogEntry(
                id: "gemma-4-e4b", name: "Gemma 4 E4B",
                repo: "mlboydaisuke/gemma-4-E4B-CoreAI", kind: .chat,
                variants: ["macos": .init(
                    path: "gpu-pipelined/gemma4_e4b_qat_decode_int4lin_tbl", sizeMB: 7589)],
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
            CatalogEntry(
                id: "holo2-4b", name: "Holo2 4B",
                repo: "mlboydaisuke/Holo2-4B-CoreAI", kind: .vlm,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/holo2_4b_decode_int8lin_s1", sizeMB: 5484),
                    "ios": .init(
                        path: "gpu-pipelined/holo2_4b_decode_int8lin_s1", sizeMB: 5484),
                ]),
            // MiniCPM-V 4.6 rides its own VLArchitecture shape (raw-pixel SigLIP tower,
            // single image_embeds static input, no deepstack/rope shift); the iPhone path
            // is the same JIT bundle the device-verified zoo apps run.
            CatalogEntry(
                id: "minicpm-v-4.6", name: "MiniCPM-V 4.6",
                repo: "mlboydaisuke/MiniCPM-V-4.6-CoreAI", kind: .vlm,
                variants: [
                    "macos": .init(
                        path: "gpu-pipelined/minicpmv46_vlm_decode_int8lin", sizeMB: 2145),
                    "ios": .init(
                        path: "gpu-pipelined/minicpmv46_vlm_decode_int8lin", sizeMB: 2145),
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
            // Kokoro is three stateless graph bundles + host glue at the repo root (no
            // platform dir); the variant path is empty and KitSpeaker resolves the subtrees.
            // sizeMB is the three bundles + glue (lexicons + voice packs) total.
            CatalogEntry(
                id: "kokoro-82m", name: "Kokoro 82M",
                repo: "mlboydaisuke/Kokoro-82M-CoreAI", kind: .tts,
                variants: [
                    "macos": .init(path: "", sizeMB: 341)
                ]),
            CatalogEntry(
                id: "voxcpm2-2b", name: "VoxCPM2 2B",
                repo: "mlboydaisuke/VoxCPM2-CoreAI", kind: .tts,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 4719),
                    "ios": .init(path: "ios", sizeMB: 5658),
                ]),
            // ── Text-to-music: prompt → 44.1 kHz audio (T5 cond + DiT + VAE, one subtree). ──
            CatalogEntry(
                id: "stable-audio-open-small", name: "Stable Audio Open Small",
                repo: "mlboydaisuke/Stable-Audio-Open-Small-CoreAI", kind: .music,
                variants: ["macos": .init(path: "macos", sizeMB: 1060)]),
            // ── Audio understanding: clip + question → answer (thinker decoder + audio
            //    encoder pair; KitAudioModel resolves both via AudioModelID.byCatalogID). ──
            CatalogEntry(
                id: "qwen2.5-omni-3b-audio", name: "Qwen2.5-Omni 3B Audio",
                repo: "mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI", kind: .audio,
                variants: ["macos": .init(path: "gpu-pipelined", sizeMB: 5485)]),
            // ── Video understanding: 16-frame clip → action class (ActionRecognizer).
            //    The platform dir holds the graph (JIT .aimodel / AOT .aimodelc) + labels. ──
            CatalogEntry(
                id: "vjepa2-vitl-ssv2", name: "V-JEPA 2 ViT-L (SSv2)",
                repo: "mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI", kind: .video,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 708),
                    "ios": .init(path: "ios", sizeMB: 710),
                ]),
            // ── Document OCR: image → structured markdown (vision + unified prefill/decode
            //    decoder + host constant tables; KitDocReader resolves the four subtrees). ──
            CatalogEntry(
                id: "unlimited-ocr", name: "Unlimited-OCR",
                repo: "mlboydaisuke/Unlimited-OCR-CoreAI", kind: .ocr,
                variants: [
                    "macos": .init(path: "", sizeMB: 4532)
                ]),
            // Whole-page document parsing (stock Qwen2-VL) behind KitMineruReader — the VL
            // rope-shift rider, fixed portrait 32×24 grid, letterbox + CLIP norm.
            CatalogEntry(
                id: "mineru2.5-pro", name: "MinerU2.5-Pro",
                repo: "mlboydaisuke/MinerU2.5-Pro-CoreAI", kind: .ocr,
                // iOS loads the h18p bundles sideloaded into Documents/mineru_pf/ (KitMineruReader
                // is pointed there, not the catalog download), so the ios variant just makes the
                // entry selectable in the picker.
                variants: [
                    "macos": .init(path: "", sizeMB: 1980),
                    "ios": .init(path: "", sizeMB: 1980)
                ]),
            // GLM-OCR (GLM-4.V small, MIT) document OCR behind KitGlmOcrReader — the VL rope-shift
            // rider, fixed portrait 32×24 grid, letterbox + CLIP norm. Bundles sideloaded into
            // Documents/glm_ocr_pf/ (the reader is pointed there, not the catalog download).
            CatalogEntry(
                id: "glm-ocr", name: "GLM-OCR",
                repo: "mlboydaisuke/GLM-OCR-CoreAI", kind: .ocr,
                variants: [
                    "macos": .init(path: "", sizeMB: 1600),
                    "ios": .init(path: "", sizeMB: 1600)
                ]),
            // ── Diffusion LM: parallel canvas denoising behind KitDiffusionLM (its own
            //    surface — snapshots, not append-only streams — so not a ChatSession). ──
            CatalogEntry(
                id: "llada-8b", name: "LLaDA-8B (diffusion)",
                repo: "mlboydaisuke/LLaDA-8B-dLLM-CoreAI", kind: .dllm,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 5265)
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
            // Streaming ASR (cache-aware FastConformer + pure-RNNT, 40 locales, any-length
            // audio) driven by `KitNemotronModel`; platform subtrees like Whisper — `ios/`
            // carries the AOT h18p conformer halves (the device JIT is avoided).
            CatalogEntry(
                id: "nemotron-3.5-asr-streaming-0.6b", name: "Nemotron 3.5 ASR Streaming 0.6B",
                repo: "mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI", kind: .asr,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 1340),
                    "ios": .init(path: "ios", sizeMB: 1336),
                ]),
            // ── Speaker diarization: clip → who spoke when (up to 4 speakers, 80 ms frames).
            //    Flat repo, one graph per platform: the variant path names the JIT .aimodel
            //    (macOS) / AOT h18p .aimodelc (iOS) directly, so each platform downloads only
            //    its own form. The 128-mel frontend filterbank ships inside CoreAIKit (the
            //    Parakeet resource — same NeMo family). Driven by `KitDiarizer`. ──
            CatalogEntry(
                id: "sortformer-diar-v2", name: "Streaming Sortformer v2 (4-spk)",
                repo: "mlboydaisuke/Streaming-Sortformer-Diar-CoreAI", kind: .diarization,
                variants: [
                    "macos": .init(path: "sortformer_float16.aimodel", sizeMB: 226),
                    "ios": .init(path: "sortformer_float16.h18p.aimodelc", sizeMB: 238),
                ]),
            // ── Multi-speaker / dialogue TTS (VibeVoice-Realtime-0.5B): five fp16 graphs in the
            //    platform dir + `coreai_host/` (voice prefill caches, glue, tokenizer) + the fp16
            //    embedding table at the repo root. Driven by `KitDialogue`. ──
            CatalogEntry(
                id: "vibevoice-realtime-0.5b", name: "VibeVoice-Realtime 0.5B (multi-speaker)",
                repo: "mlboydaisuke/VibeVoice-Realtime-0.5B-CoreAI", kind: .tts,
                variants: [
                    "macos": .init(path: "macos", sizeMB: 1420),
                    "ios": .init(path: "ios", sizeMB: 1420),
                ]),
            // ── Music source separation (Mel-Band RoFormer, Kim Vocal). One graph with STFT +
            //    iSTFT folded in; the host only frames and overlap-adds. Driven by `KitSeparator`. ──
            CatalogEntry(
                id: "melband-roformer-vocal", name: "Mel-Band RoFormer (Kim Vocal)",
                repo: "mlboydaisuke/MelBandRoformer-Vocal-CoreAI", kind: .separation,
                variants: [
                    "macos": .init(path: "mbr_full_fp16.aimodel", sizeMB: 470),
                    "ios": .init(path: "mbr_full_fp16.h18p.aimodelc", sizeMB: 470),
                ]),
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
            // ── Visual document retrieval: two fp16 graph bundles (query encoder +
            //    tokenizer, doc encoder). The variant path names the doc encoder;
            //    VisualDocumentRetriever(catalog:) resolves the pair, and sizeMB covers
            //    both downloads. Same bundles on both platforms. ──
            CatalogEntry(
                id: "colmodernvbert", name: "ColModernVBERT",
                repo: "mlboydaisuke/ColModernVBERT-CoreAI", kind: .retrieval,
                variants: [
                    "macos": .init(path: "doc", sizeMB: 741),
                    "ios": .init(path: "doc", sizeMB: 741),
                ]),
            CatalogEntry(
                id: "yolox-s", name: "YOLOX-S",
                repo: "mlboydaisuke/YOLOX-CoreAI", kind: .detection,
                variants: [
                    "macos": .init(path: "yolox-s_float32.aimodel", sizeMB: 36),
                    "ios": .init(path: "yolox-s_float32.aimodel", sizeMB: 36),
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
            // ── Time-series forecasting: one stateless transformer graph (fixed context 2048;
            //    KitForecaster front-pads + masks shorter series host-side) + host RevIN/flip DSP. ──
            CatalogEntry(
                id: "timesfm-2.5-200m", name: "TimesFM 2.5 200M",
                repo: "mlboydaisuke/TimesFM-2.5-200M-CoreAI", kind: .forecasting,
                variants: [
                    "macos": .init(path: "timesfm_2p5_200m_ctx2048_fp16.aimodel", sizeMB: 463),
                    "ios": .init(path: "ios", sizeMB: 464),   // AOT .aimodelc (h18p; weights + MPSGraph)
                ], engine: "static-shape"),
        ])
}
