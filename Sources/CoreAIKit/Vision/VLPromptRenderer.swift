// VLPromptRenderer.swift — renders a FoundationModels `Transcript` into VL decoder tokens.
//
// Unlike the text renderer (which drops `.attachment` segments), this one splices the latest
// image's vision block into the ChatML prompt: `<|vision_start|>` + N×`<|image_pad|>` +
// `<|vision_end|>` at the image's position. After tokenizing, the N image-pad ids are
// rewritten to extension ids `vocab + slot`, which the decoder graph gathers from
// `image_embeds`. The first rewritten id's index (`imageStart`) drives the rope shift.
//
// v1 scope: one image per session (a single embeds buffer). When several image segments are
// present the LATEST wins; earlier ones render as a short text marker so the conversation
// still reads coherently.

import CoreGraphics
import Foundation
import FoundationModels
import ImageIO
import Tokenizers

enum VLPromptRenderer {
    /// The image chosen to occupy the embeds buffer this turn.
    struct ChosenImage {
        let cgImage: CGImage
        let orientation: CGImagePropertyOrientation
        let segmentID: String
    }

    struct Rendered {
        let tokens: [Int32]
        /// Index of the first vision token, or nil for a text-only prompt.
        let imageStart: Int?
        /// The image to vision-encode this turn, or nil for text-only.
        let image: ChosenImage?
    }

    static func render(
        transcript: Transcript, tokenizer: any Tokenizer, arch: VLArchitecture
    ) throws -> Rendered {
        let chosen = latestImage(in: transcript)

        // Build the ChatML text. Instructions fold into a system turn; the chosen image's
        // segment becomes the vision block; other images become a short marker.
        var system = ""
        var body = ""
        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                let text = plainText(instructions.segments)
                if !text.isEmpty { system = text }
            case .prompt(let prompt):
                body += "\(arch.imStart)user\n"
                    + userContent(prompt.segments, chosen: chosen, arch: arch)
                    + "\(arch.imEnd)\n"
            case .response(let response):
                body += "\(arch.imStart)assistant\n\(plainText(response.segments))\(arch.imEnd)\n"
            default:
                continue  // reasoning/tool entries are not part of the VL chat path
            }
        }

        let head = system.isEmpty ? "" : "\(arch.imStart)system\n\(system)\(arch.imEnd)\n"
        let text = head + body + "\(arch.imStart)assistant\n"

        var tokens = tokenizer.encode(text: text).map(Int32.init)
        guard chosen != nil else {
            return Rendered(tokens: tokens, imageStart: nil, image: nil)
        }

        // Rewrite the N consecutive image-pad ids to extension ids `vocab + slot`.
        guard let padID = singleTokenID(arch.imagePad, tokenizer: tokenizer) else {
            throw KitVisionError.visionOutputMissing(arch.imagePad)
        }
        var imageStart = -1
        var slot: Int32 = 0
        for i in tokens.indices where tokens[i] == padID && slot < Int32(arch.mergedTokens) {
            if imageStart < 0 { imageStart = i }
            tokens[i] = arch.vocab + slot
            slot += 1
        }
        guard slot == Int32(arch.mergedTokens), imageStart >= 0 else {
            // The pad string did not tokenize to exactly `mergedTokens` ids — render is wrong.
            throw KitVisionError.visionOutputMissing(
                "\(arch.imagePad)×\(arch.mergedTokens) (found \(slot))")
        }
        return Rendered(tokens: tokens, imageStart: imageStart, image: chosen)
    }

    // MARK: - Helpers

    /// The last image attachment anywhere in the transcript (latest wins for the buffer).
    private static func latestImage(in transcript: Transcript) -> ChosenImage? {
        var chosen: ChosenImage?
        for entry in transcript {
            guard case .prompt(let prompt) = entry else { continue }
            for segment in prompt.segments {
                if case .attachment(let attachment) = segment,
                    case .image(let image) = attachment.content
                {
                    chosen = ChosenImage(
                        cgImage: image.cgImage, orientation: image.orientation,
                        segmentID: attachment.id)
                }
            }
        }
        return chosen
    }

    /// Renders a user turn's segments, replacing the chosen image with the vision block and
    /// any other image with a short marker.
    private static func userContent(
        _ segments: [Transcript.Segment], chosen: ChosenImage?, arch: VLArchitecture
    ) -> String {
        var parts: [String] = []
        for segment in segments {
            switch segment {
            case .text(let text):
                parts.append(text.content)
            case .structure(let structure):
                parts.append(structure.content.jsonString)
            case .attachment(let attachment):
                guard case .image = attachment.content else { continue }
                if attachment.id == chosen?.segmentID {
                    parts.append(
                        arch.visionStart
                            + String(repeating: arch.imagePad, count: arch.mergedTokens)
                            + arch.visionEnd)
                } else {
                    parts.append("[image]")
                }
            default:
                continue
            }
        }
        return parts.joined()
    }

    private static func plainText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }

    /// The token id for `token` if it encodes to exactly one id, else nil.
    private static func singleTokenID(_ token: String, tokenizer: any Tokenizer) -> Int32? {
        if let id = tokenizer.convertTokenToId(token) { return Int32(id) }
        let ids = tokenizer.encode(text: token)
        return ids.count == 1 ? Int32(ids[0]) : nil
    }
}
