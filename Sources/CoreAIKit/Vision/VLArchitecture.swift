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
    /// Square side the image is resized to before patchifying (448).
    public let imageSide: Int
    /// ViT patch side in pixels (16).
    public let patchSize: Int
    /// Temporal patch size; a still frame is duplicated this many times (2).
    public let temporalPatchSize: Int

    // Special tokens (ChatML + optional vision framing). Strings, not ids: the bundle
    // tokenizer maps them, and `imagePad` must be a single token for the splice to work.
    public let imStart: String
    public let imEnd: String
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
        imStart: String = "<|im_start|>", imEnd: String = "<|im_end|>",
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
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.imStart = imStart
        self.imEnd = imEnd
        self.visionStart = visionStart
        self.visionEnd = visionEnd
        self.imagePad = imagePad
        self.visionInput = visionInput
        self.visionOutput = visionOutput
        self.ropeShifted = ropeShifted
    }

    /// `image_embeds` element count (`mergedTokens · hidden`).
    public var imageEmbedCount: Int { mergedTokens * hidden }
    /// `deepstack_embeds` element count (`deepstackPerToken · mergedTokens · hidden`).
    public var deepstackEmbedCount: Int { deepstackPerToken * mergedTokens * hidden }
    /// The rope-shift amount for an attached image (`mergedTokens - grid`).
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
}
