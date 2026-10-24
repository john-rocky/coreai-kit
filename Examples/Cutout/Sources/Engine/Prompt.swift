// Prompt — the words a person types → the decoder's id row, the package's contract (Tencent/WeVisOne-VL 40f735b8,
// `prompts.py build_prompt` and `engine.py _messages`, through the checkpoint's `chat_template.jinja`):
//
//   cut out   "Locate the instances that match the following description and provide the alpha mattes: " + phrase
//             (the package's segmentation / matting prompt; phrase = the words, with "." added unless they end in . ? !)
//   boxes     "Locate all the instances that match the following description: " + labels joined by ". " + "."
//             (the detection prompt; one label per line)
//   row       <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>
//             <|image_pad|> × 196 <|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n
//
// The tokenizer is swift-transformers' over the checkpoint's tokenizer.json (BPE, NFC, byte-level); its added-token
// split puts the special tokens in place, as Hugging Face tokenizers do. The 196 image rows are the 448 tower's
// (14 × 14 merged patches). The exporter's host contract maps each <|image_pad|> to V + k (row k of the tower
// output) and starts the rope shift after the image by N − 14 (conversion/wevisone/gate_common.py `host_inputs`).

import Foundation
import Tokenizers

public enum CutoutTask: String, Sendable, Codable, CaseIterable {
    /// Boxes and a matte per instance (the package's segmentation prompt).
    case cutout
    /// Boxes only (the package's detection prompt).
    case boxes
}

public struct PromptRow: Sendable, Equatable {
    /// The decoder's ids: <|image_pad|> mapped to V + k.
    public let ids: [Int]
    /// The same row with <|image_pad|> (the processor's form; what the oracle's input_ids hold).
    public let processorIDs: [Int]
    public let ropeShiftStart: Int32
    public let ropeShiftAmount: Int32
    public let prompt: String
}

public struct PromptBuilder: Sendable {
    public static let vocab = 152_680
    public static let imStart = 151_644
    public static let imEnd = 151_645
    public static let endOfText = 151_643
    public static let visionStart = 151_652
    public static let visionEnd = 151_653
    public static let imagePad = 151_655
    public static let obj = 152_678
    public static let stopIDs: Set<Int> = [151_645, 151_643]
    public static let systemPrompt = "You are a helpful assistant."
    public static let grid = 14
    public static let imageTokens = 196

    public let tokenizer: any Tokenizer

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public static func load(tokenizerFolder: URL) async throws -> PromptBuilder {
        PromptBuilder(tokenizer: try await AutoTokenizer.from(modelFolder: tokenizerFolder))
    }

    /// The package's `build_prompt` for the two tasks this app asks.
    public static func instruction(task: CutoutTask, words: String) -> String {
        switch task {
        case .cutout:
            let text = words.trimmingCharacters(in: .whitespacesAndNewlines)
            let phrase = text.isEmpty || ".?!".contains(text.last!) ? text : text + "."
            return "Locate the instances that match the following description and provide the alpha mattes: " + phrase
        case .boxes:
            let labels = words.split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            return "Locate all the instances that match the following description: " + labels.joined(separator: ". ") + "."
        }
    }

    /// The decoder row for one image and one instruction.
    public func row(prompt: String) throws -> PromptRow {
        let text = "<|im_start|>system\n\(Self.systemPrompt)<|im_end|>\n<|im_start|>user\n"
            + "<|vision_start|><|image_pad|><|vision_end|>\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        let one = tokenizer.encode(text: text, addSpecialTokens: false)
        guard let pad = one.firstIndex(of: Self.imagePad), one.filter({ $0 == Self.imagePad }).count == 1 else {
            throw CutoutError.prompt("the template did not encode to one <|image_pad|> (\(one.prefix(24)) …)")
        }
        let processor = Array(one[..<pad]) + [Int](repeating: Self.imagePad, count: Self.imageTokens) + Array(one[(pad + 1)...])
        var ids = processor
        for k in 0..<Self.imageTokens { ids[pad + k] = Self.vocab + k }
        return PromptRow(ids: ids, processorIDs: processor, ropeShiftStart: Int32(pad + Self.imageTokens),
                         ropeShiftAmount: Int32(Self.imageTokens - Self.grid), prompt: prompt)
    }

    /// The generated ids as text (special tokens kept), cut at <|im_end|> / <|endoftext|> and trimmed, as the zoo's
    /// gates read a response (conversion/wevisone/mask/mask_e2e.py).
    public func response(_ generated: [Int]) -> String {
        let raw = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
        let cut = raw.components(separatedBy: "<|im_end|>")[0].components(separatedBy: "<|endoftext|>")[0]
        return cut.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
