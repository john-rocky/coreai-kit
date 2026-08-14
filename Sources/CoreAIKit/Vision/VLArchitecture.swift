// VLArchitecture.swift — fixed-grid vision-language geometry for a Qwen3-VL-style decoder.
//
// A VL decoder bundle exported with the id-space recipe (see NOTICE.txt) keeps the text
// path a plain Qwen3 LLM and rides multimodal state on four static graph inputs. Everything
// the host must know to feed those inputs is here: the merged-token count, the embedding
// width, the patch geometry of the paired ViT, and the special-token strings used to splice
// the vision block into a ChatML prompt.
//
// Only `hidden` (the decoder/ViT embedding width) differs across the published sizes; the
// 448×448 grid and its derived counts are identical. Construct one explicitly for a custom
// export, or use the `qwen3VL2B/4B/8B` presets.

import Foundation

/// Geometry and token vocabulary for one VL decoder + its paired vision tower.
public struct VLArchitecture: Sendable, Hashable {
    /// How the paired vision tower ingests an image.
    public enum VisionInput: Sendable, Hashable {
        /// Flattened ViT patches `[patches, patchDim]` fed as `patches` (Qwen3-VL).
        case patches
        /// Raw normalized pixels `[1, 3, imageSide, imageSide]` fed as `pixel_values`
        /// (MiniCPM-V: resize to the square side, x/127.5 − 1, CHW).
        case pixels
    }

    /// Pixel normalization applied before patchifying.
    public enum Normalization: Sendable, Hashable {
        /// `x/127.5 − 1` → [−1, 1] (Qwen3-VL / MiniCPM-V).
        case symmetric
        /// `(x/255 − mean)/std` with CLIP mean/std (Qwen2-VL / MinerU).
        case clip
    }

    /// Byte order INSIDE one flattened patch. Both produce `[patches, patchDim]`; getting it
    /// wrong is silent — the tower still runs and still describes something.
    public enum PatchLayout: Sendable, Hashable {
        /// `[C][T][py][px]` — channel-major with the temporal duplicate (Qwen2/3-VL).
        case channelMajor
        /// `[py][px][C]` — channel fastest, row-major within the patch (SigLIP2 NaFlex:
        /// HF permutes `(b,C,ph,P,pw,P) -> (b,ph,pw,P,P,C)`).
        case channelLast
    }

    /// How the source image is fit into the fixed export canvas.
    public enum Resize: Sendable, Hashable {
        /// Stretch to fill the canvas, ignoring aspect (Qwen3-VL square grid).
        case stretch
        /// Letterbox: preserve aspect, pad the remainder white (document OCR).
        case aspectFitPad
    }

    /// Text vocabulary size. Image tokens are encoded as extension ids `vocab + slot`, which
    /// the decoder graph gathers from `image_embeds` by `id - vocab`.
    public let vocab: Int32
    /// Merged vision tokens spliced into the prompt (14×14 = 196 for the 448px grid).
    public let mergedTokens: Int
    /// Merged-grid side length (14); the rope-shift amount is `mergedTokens - grid`.
    public let grid: Int
    /// Decoder hidden width and ViT output width (2B 2048 / 4B 2560 / 8B 4096).
    public let hidden: Int
    /// Pre-merge ViT patches (28×28 = 784 for the 448px grid).
    public let patches: Int
    /// Flattened per-patch dimension fed to the ViT: channels·temporal·patch·patch
    /// (3·2·16·16 = 1536).
    public let patchDim: Int
    /// `deepstack_embeds` carries this many rows per merged token (3 deepstack layers).
    public let deepstackPerToken: Int
    /// Square side the image is resized to before patchifying (448). For a non-square export
    /// this is the height; see `imageWidth`.
    public let imageSide: Int
    /// Canvas width the image is fit into before patchifying. Defaults to `imageSide` (square);
    /// a non-square grid (Qwen2-VL / MinerU) sets it independently. `imageSide` is the height.
    public let imageWidth: Int
    /// ViT patch side in pixels (16).
    public let patchSize: Int
    /// Temporal patch size; a still frame is duplicated this many times (2).
    public let temporalPatchSize: Int
    /// Pixel normalization before patchify (`.symmetric` Qwen3-VL / `.clip` Qwen2-VL·MinerU).
    public let normalization: Normalization
    /// How the source image is fit into the canvas (`.stretch` / `.aspectFitPad`).
    public let resize: Resize
    /// Byte order inside one patch (`.channelMajor` Qwen-VL / `.channelLast` NaFlex).
    public let patchLayout: PatchLayout

    // Special tokens (ChatML + optional vision framing). Strings, not ids: the bundle
    // tokenizer maps them, and `imagePad` must be a single token for the splice to work.
    public let imStart: String
    public let imEnd: String
    /// Role markers and the separator between a marker and its content. ChatML writes
    /// `<|im_start|>user\n … <|im_end|>\n`; Cohere writes
    /// `<|START_OF_TURN_TOKEN|><|USER_TOKEN|> … <|END_OF_TURN_TOKEN|>` with no newlines,
    /// so the turn shape is data rather than a hardcoded string.
    public let userRole: String
    public let assistantRole: String
    public let systemRole: String
    public let roleSeparator: String
    public let visionStart: String
    public let visionEnd: String
    public let imagePad: String

    /// How the vision tower ingests an image (`patches` for Qwen3-VL, `pixels` for MiniCPM-V).
    public let visionInput: VisionInput
    /// The vision tower's embeds output name (`image_embeds` Qwen3-VL / `image_features`
    /// MiniCPM-V).
    public let visionOutput: String
    /// Whether the decoder graph declares the interleaved-M-RoPE shift inputs
    /// (`rope_shift_start`/`rope_shift_amount`). MiniCPM-V uses plain 1D positions — no
    /// shift inputs exist on its graph, so nothing is bound.
    public let ropeShifted: Bool

    public init(
        vocab: Int32, mergedTokens: Int, grid: Int, hidden: Int, patches: Int, patchDim: Int,
        deepstackPerToken: Int, imageSide: Int, patchSize: Int, temporalPatchSize: Int,
        imageWidth: Int? = nil, normalization: Normalization = .symmetric,
        resize: Resize = .stretch, patchLayout: PatchLayout = .channelMajor,
        imStart: String = "<|im_start|>", imEnd: String = "<|im_end|>",
        userRole: String = "user", assistantRole: String = "assistant",
        systemRole: String = "system", roleSeparator: String = "\n",
        visionStart: String = "<|vision_start|>", visionEnd: String = "<|vision_end|>",
        imagePad: String = "<|image_pad|>",
        visionInput: VisionInput = .patches, visionOutput: String = "image_embeds",
        ropeShifted: Bool = true
    ) {
        self.vocab = vocab
        self.mergedTokens = mergedTokens
        self.grid = grid
        self.hidden = hidden
        self.patches = patches
        self.patchDim = patchDim
        self.deepstackPerToken = deepstackPerToken
        self.imageSide = imageSide
        self.imageWidth = imageWidth ?? imageSide
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.normalization = normalization
        self.resize = resize
        self.patchLayout = patchLayout
        self.imStart = imStart
        self.imEnd = imEnd
        self.userRole = userRole
        self.assistantRole = assistantRole
        self.systemRole = systemRole
        self.roleSeparator = roleSeparator
        self.visionStart = visionStart
        self.visionEnd = visionEnd
        self.imagePad = imagePad
        self.visionInput = visionInput
        self.visionOutput = visionOutput
        self.ropeShifted = ropeShifted
    }

    /// Canvas height (== `imageSide`).
    public var imageHeight: Int { imageSide }
    /// `image_embeds` element count (`mergedTokens · hidden`).
    public var imageEmbedCount: Int { mergedTokens * hidden }
    /// `deepstack_embeds` element count (`deepstackPerToken · mergedTokens · hidden`).
    public var deepstackEmbedCount: Int { deepstackPerToken * mergedTokens * hidden }
    /// The rope-shift amount for an attached image (`mergedTokens - grid`; for a non-square
    /// grid `grid` is set to `max(gridTall, gridWide)`, matching the decoder's baked geometry).
    public var ropeShiftAmount: Int32 { Int32(mergedTokens - grid) }

    /// The Qwen3-VL 448px grid shared by every published size. `hidden` selects the variant.
    private static func qwen3VL(hidden: Int) -> VLArchitecture {
        VLArchitecture(
            vocab: 151_936, mergedTokens: 196, grid: 14, hidden: hidden, patches: 784,
            patchDim: 1536, deepstackPerToken: 3, imageSide: 448, patchSize: 16,
            temporalPatchSize: 2)
    }

    /// Qwen3-VL-2B (decoder/ViT hidden 2048). iPhone-class.
    public static let qwen3VL2B = qwen3VL(hidden: 2048)
    /// Qwen3-VL-4B (decoder/ViT hidden 2560).
    public static let qwen3VL4B = qwen3VL(hidden: 2560)
    /// Qwen3-VL-8B (decoder/ViT hidden 4096). Mac-class.
    public static let qwen3VL8B = qwen3VL(hidden: 4096)

    /// MiniCPM-V 4.6: single-slice 448px SigLIP tower fed RAW pixels (the tower patchifies
    /// in-graph) → `image_features` [64, 1024], one static `image_embeds` input on the
    /// decoder, no deepstack, plain 1D positions (no rope shift). The 64 image pads ride
    /// bare in the user turn (no vision framing tokens); a newline separates them from the
    /// prompt text, matching the gated reference prompt.
    public static let miniCPMV46 = VLArchitecture(
        vocab: 248_094, mergedTokens: 64, grid: 8, hidden: 1024,
        patches: 0, patchDim: 0, deepstackPerToken: 0,
        imageSide: 448, patchSize: 14, temporalPatchSize: 1,
        visionStart: "", visionEnd: "\n",
        visionInput: .pixels, visionOutput: "image_features",
        ropeShifted: false)

    /// LFM2.5-VL-450M: a SigLIP2-**NaFlex** tower fed FLATTENED patches (the host patchifies)
    /// at a baked 32x32 grid — one 512x512 tile → 2x pixel-unshuffle → 256 tokens, which is the
    /// checkpoint's own `max_image_tokens` — into one static `image_embeds` [256, 1024] on the
    /// LFM2 hybrid decoder. No deepstack, plain 1D positions (no rope shift).
    ///
    /// Two things differ from every other `.patches` entry here and both are silent when wrong:
    /// the patch bytes are **channel-last** (`[py][px][C]`, HF's NaFlex permute), and there is
    /// no temporal duplicate (`temporalPatchSize: 1`, which also makes the patch order plain
    /// row-major rather than block-major). `patchDim` = 3·16·16 = 768.
    ///
    /// Framing is the checkpoint's own: `<|image_start|>` + 256 `<image>` + `<|image_end|>`.
    public static let lfm2VL450M = VLArchitecture(
        vocab: 65_536, mergedTokens: 256, grid: 16, hidden: 1024,
        patches: 1024, patchDim: 768, deepstackPerToken: 0,
        imageSide: 512, patchSize: 16, temporalPatchSize: 1,
        patchLayout: .channelLast,
        visionStart: "<|image_start|>", visionEnd: "<|image_end|>", imagePad: "<image>",
        visionInput: .patches, visionOutput: "image_embeds",
        ropeShifted: false)

    /// LFM2.5-VL-3B: the 450M's bigger sibling — same graph shape end to end (baked 32x32
    /// grid, 256 tokens, one `image_embeds` static input, plain 1D positions), a wider tower
    /// (hidden 1152, 27 layers) and a 128k-vocab decoder at hidden 2048.
    ///
    /// One thing does NOT carry over from the 450M and is invisible here: the checkpoints
    /// declare different `resample` filters (450M PIL BILINEAR, 3B PIL BICUBIC). This host
    /// resizes with CoreGraphics `.high` for every model — an approximation of both, verified
    /// token-exact through each model rather than assumed. If a port ever needs the exact
    /// filter, that is where to add it.
    public static let lfm2VL3B = VLArchitecture(
        vocab: 128_000, mergedTokens: 256, grid: 16, hidden: 2048,
        patches: 1024, patchDim: 768, deepstackPerToken: 0,
        imageSide: 512, patchSize: 16, temporalPatchSize: 1,
        patchLayout: .channelLast,
        visionStart: "<|image_start|>", visionEnd: "<|image_end|>", imagePad: "<image>",
        visionInput: .patches, visionOutput: "image_embeds",
        ropeShifted: false)

    /// North-Micro-Vision (Cohere `cohere_compass`, 2.4B): a Qwen3-VL visual encoder at
    /// SigLIP2-SO400M dimensions — deepstack and all — in front of a parallel-block Cohere
    /// decoder, so the graph contract is Qwen3-VL's: `.patches`, three deepstack rows per
    /// merged token, rope-shift inputs. A 512x512 canvas at patch 16 with a 2x2 merge gives
    /// 16x16 = 256 tokens; `patchDim` = 3 * 2 * 16 * 16 = 1536 (temporal duplicate).
    ///
    /// The turn shape is NOT ChatML: Cohere marks roles with single tokens and ends a turn
    /// with `<|END_OF_TURN_TOKEN|>`, no newlines anywhere.
    public static let northMicroVision = VLArchitecture(
        vocab: 262_144, mergedTokens: 256, grid: 16, hidden: 2048,
        patches: 1024, patchDim: 1536, deepstackPerToken: 3,
        imageSide: 512, patchSize: 16, temporalPatchSize: 2,
        imStart: "<|START_OF_TURN_TOKEN|>", imEnd: "<|END_OF_TURN_TOKEN|>",
        userRole: "<|USER_TOKEN|>", assistantRole: "<|CHATBOT_TOKEN|>",
        systemRole: "<|SYSTEM_TOKEN|>", roleSeparator: "",
        visionStart: "<|VISION_START|>", visionEnd: "<|VISION_END|>",
        imagePad: "<|IMAGE_PAD|>",
        visionInput: .patches, visionOutput: "image_embeds",
        ropeShifted: true)

    /// MinerU2.5-Pro (stock Qwen2-VL): a Qwen2-VL ViT (`.patches`, no deepstack) + Qwen2-0.5B
    /// decoder on the rope-shift rider. Fixed **portrait non-square** grid 32×24 merged (768
    /// tokens) at a 672×896 canvas — the document is letterboxed (aspect-fit + white pad) and
    /// CLIP-normalized (not `x/127.5−1`). `grid = max(32,24) = 32` gives the rope-shift amount
    /// `768 − 32 = 736`, matching the decoder's baked `grid_w = 24`. patchDim = 3·2·14·14 = 1176,
    /// patches = 64·48 = 3072. Prompt with `"Text Recognition:"` (whole-page single pass). The
    /// S=1 prefill of 768 image tokens is sped up on-device by the `pf64` multifunction bundle
    /// (static S=64 chunked prefill + S=1 decode), not by cutting tokens.
    public static let mineru = VLArchitecture(
        vocab: 151_936, mergedTokens: 768, grid: 32, hidden: 896,
        patches: 3072, patchDim: 1176, deepstackPerToken: 0,
        imageSide: 896, patchSize: 14, temporalPatchSize: 2,
        imageWidth: 672, normalization: .clip, resize: .aspectFitPad,
        visionInput: .patches, visionOutput: "image_embeds", ropeShifted: true)

    /// MinerU2.5 **layout** grid: a 1036×1036 **square** (37×37 merged = 1369 tokens) with the page
    /// **stretched** to fill it (matches the reference `layout_image_size`) — the resolution the
    /// `Layout Detection:` head needs (a 768 portrait grid mis-detects). Boxes come back 0–1 of the
    /// stretched square, so they map linearly onto the original page (no letterbox inverse). Paired
    /// with `.mineru` (768) for per-region recognition in the 2-stage `KitMineruReader.readStructured`.
    public static let mineruLayout = VLArchitecture(
        vocab: 151_936, mergedTokens: 1369, grid: 37, hidden: 896,
        patches: 5476, patchDim: 1176, deepstackPerToken: 0,
        imageSide: 1036, patchSize: 14, temporalPatchSize: 2,
        imageWidth: 1036, normalization: .clip, resize: .stretch,
        visionInput: .patches, visionOutput: "image_embeds", ropeShifted: true)

    /// GLM-OCR (zai-org, GLM-4.V small, MIT): a CogViT tower (patch 14, spatial-merge 2, out 1536) +
    /// a GLM text decoder (hidden 1536, vocab 59 392, sectioned M-RoPE) on the same rope-shift rider.
    /// Fixed **portrait** grid 32×24 merged (768 tokens) at a 672×896 canvas — letterbox (aspect-fit +
    /// white pad) + CLIP-normalize, same vision geometry as `.mineru`. The image block is framed with
    /// GLM's `<|begin_of_image|>`/`<|end_of_image|>` (pad `<|image|>`); the surrounding ChatML differs
    /// (`[gMASK]<sop><|user|>…<|assistant|>`), so `KitGlmOcrReader` builds the prompt itself.
    public static let glmOcr = VLArchitecture(
        vocab: 59_392, mergedTokens: 768, grid: 32, hidden: 1536,
        patches: 3072, patchDim: 1176, deepstackPerToken: 0,
        imageSide: 896, patchSize: 14, temporalPatchSize: 2,
        imageWidth: 672, normalization: .clip, resize: .aspectFitPad,
        visionStart: "<|begin_of_image|>", visionEnd: "<|end_of_image|>", imagePad: "<|image|>",
        visionInput: .patches, visionOutput: "image_embeds", ropeShifted: true)
}
