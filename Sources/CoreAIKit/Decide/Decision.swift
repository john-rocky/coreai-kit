// Decision.swift — the value types of a typed decision: a state, a question with a fixed
// answer shape, and an answer read as probabilities over that shape.
//
// A decision is not a generation. The model never writes text: one prompt is scored once,
// and the answer is the probability the model assigns to each listed option at the answer
// slot. Three shapes cover the ways an app branches on a piece of text:
//
//   choice  — which of these options (2–maxOptions)    → the option, plus every option's probability
//   score   — where on this ordered scale (2–10 levels) → the expected level, plus the distribution
//   noul    — yes or no                                  → P(yes)
//
// Every question carries free-text instructions and, optionally, a description per option.
// The same request shape is what `TypedDecisions` scores and what `CoreAI.decide` resolves a
// catalog model behind.

import Foundation

/// Namespace for the typed-decision value types.
public enum Decision {
    /// How a state and a question become the prompt that is scored.
    public enum Format: String, Sendable, Codable {
        /// One JSON request in a user turn under the model's chat template, the assistant
        /// turn opened with its thinking closed; the answer is the letter at the next token.
        /// The rendering an instruction-tuned chat model answers zero-shot, and the one the
        /// published fixture numbers use.
        case chat
        /// The plain-text form a decision model (`decider-0.8b`) is trained on — `Context:`,
        /// the state, `Question:`, `Options:` as `(A) …` lines, `Answer: (` — with no chat
        /// template and no special tokens. A score question is judged one level per row, each
        /// level alone as a yes/no, and the levels' P(yes) normalised into the distribution;
        /// that is the form the model's own API uses, and its temperature is 1.03.
        case decider
        /// The control-token form of a slot-head decision model (OpenThai-SystemOne): the
        /// state and one question laid out with `<|ts_…|>` tokens, the hidden state at
        /// `<|ts_answer|>` projected by a small head whose slot i is option i and whose last
        /// slot abstains. The bundle declares it (`decision.head == "slot"` in its
        /// metadata.json) together with a temperature per question type; a score is one row
        /// over its levels. `SlotPrompt.swift`.
        case slot
        /// The `Shared state:` + JSON task form (APUS-OpenJev-v1): one user turn under the chat
        /// template holding the state, then a JSON object whose criteria carry the letters
        /// A–P, then `Answer:`; the answer is the letter at the next token, read from the
        /// ordinary LM head with no temperature. The model's own primitives are `choice` and a
        /// yes/no on a proposition; a score is rendered as a choice over its levels.
        /// `SharedStatePrompt.swift`.
        case sharedState
        /// The plain-text "decision function" form (Jev-Style-Qwen3.5-2B-Decision): a fixed
        /// header, `[State]`, `[Question]`, `[Options]` as `A. …` lines and `Answer:`, no chat
        /// template; the answer is the next token among the space-prefixed letters ` A`, ` B`, …
        /// (up to 26), read at temperature 1 because the model's calibration is folded into its
        /// weights. A bool is the choice `yes` / `no`, a score the choice over its levels.
        /// `DecisionFunctionPrompt.swift`.
        case decisionFunction
        /// The per-option scalar form (pngwn's System One scorer): one row per option —
        /// `State:`, the state cut to fit, `Question:`, `Option:` — read by a scalar head at
        /// the row's last token, the rows softmaxed together at the calibration temperature
        /// the bundle declares (`decision.head == "scalar"` with `temperature` and `max_len`
        /// in its metadata.json). No chat template. A yes/no is the rows `yes` / `no`, a score
        /// one row per level. `ScalarPrompt.swift`.
        case scalar
        /// The lettered option list under the chat template (OpenJev): one user turn —
        /// `State:`, the state, `Question:`, `Options:` as `[A] key: description` lines,
        /// "Answer with the letter of the best option only." — read at the bare letters A–Z
        /// then a–z (up to 52) from the LM head at the temperature the bundle declares
        /// (`decision.readout == "letters"` with `temperature` and the yes/no calibration
        /// `noul` in its metadata.json). A score lists its levels as `0: level` …; a yes/no
        /// lists `yes` / `no` with what each means and is calibrated the helper's way.
        /// `LetterListPrompt.swift`.
        case letterList
        /// The encoder form (laya): one forward pass over `[CLS] <type> question: <instructions>
        /// [SEP] [MASK] option [MASK] option … [SEP] <state> [SEP]`, the model giving every
        /// position a logit and each option read at its mask marker, softmaxed within the
        /// question at the temperature the bundle declares for its type and option count. No
        /// chat template and no answer slot; the bundle is not a language bundle and declares
        /// itself (`decision.head == "encoder"` in its metadata.json, with its window, head
        /// budget, special ids and temperatures). `EncoderPrompt.swift`, `EncoderDecider.swift`.
        case encoder
    }

    /// One listed answer for a `choice` question. `id` is what the answer reports;
    /// `description` is what the model reads (the id when no description is given).
    public struct Option: Sendable, Hashable, Codable {
        public let id: String
        public let description: String

        public init(id: String, description: String) {
            self.id = id
            self.description = description
        }

        /// An option whose id and description are the same text.
        public init(_ text: String) {
            self.init(id: text, description: text)
        }
    }

    /// A typed question about a state.
    public struct Question: Sendable, Hashable {
        public enum Kind: Sendable, Hashable {
            /// Pick one of the options.
            case choice([Option])
            /// Place the state on an ordered scale; each string describes one level, lowest first.
            case score(levels: [String])
            /// Yes or no, with an optional description of what each side means.
            case noul(yes: String?, no: String?)
        }

        /// What to decide about the state — the criterion the model applies.
        public var instructions: String
        public var kind: Kind

        public init(_ instructions: String, kind: Kind) {
            self.instructions = instructions
            self.kind = kind
        }

        /// Pick one of `options`; each string is both the reported id and the description.
        public static func choice(_ instructions: String, _ options: [String]) -> Question {
            Question(instructions, kind: .choice(options.map(Option.init)))
        }

        /// Pick one of `options`, with separate ids and descriptions.
        public static func choice(_ instructions: String, options: [Option]) -> Question {
            Question(instructions, kind: .choice(options))
        }

        /// Place the state on the ordered `levels` (lowest first).
        public static func score(_ instructions: String, levels: [String]) -> Question {
            Question(instructions, kind: .score(levels: levels))
        }

        /// Yes or no. `yes` / `no` describe what each side means, when the instructions alone
        /// leave it open.
        public static func noul(_ instructions: String, yes: String? = nil, no: String? = nil) -> Question {
            Question(instructions, kind: .noul(yes: yes, no: no))
        }

        /// The options as the model reads them, in answer order.
        var optionDescriptions: [String] {
            switch kind {
            case .choice(let options):
                return options.map(\.description)
            case .score(let levels):
                return levels.enumerated().map { "\($0.offset): \($0.element)" }
            case .noul(let yes, let no):
                return [
                    no.map { "no: \($0)" } ?? "no",
                    yes.map { "yes: \($0)" } ?? "yes",
                ]
            }
        }

        /// The ids the answer reports, in answer order.
        var optionIDs: [String] {
            switch kind {
            case .choice(let options): return options.map(\.id)
            case .score(let levels): return levels.indices.map(String.init)
            case .noul: return ["no", "yes"]
            }
        }
    }

    /// The answer to a `choice` question.
    public struct Choice: Sendable, Equatable {
        /// The option with the highest probability.
        public let id: String
        /// Probability of the chosen option.
        public let confidence: Double
        /// 1 − normalised entropy of the distribution: 1 when all mass is on one option, 0 when flat.
        public let certainty: Double
        /// Probability per option id.
        public let probabilities: [String: Double]
        /// Option ids, most probable first.
        public let ranking: [String]
        /// Option ids in the order they were asked.
        public let options: [String]
    }

    /// The answer to a `score` question.
    public struct Score: Sendable, Equatable {
        /// Expected level: Σ level × probability. A 3-level scale answers in [0, 2].
        public let value: Double
        /// The level with the highest probability.
        public let level: Int
        /// Probability of that level.
        public let confidence: Double
        /// 1 − normalised entropy of the distribution.
        public let certainty: Double
        /// Probability per level, lowest level first.
        public let probabilities: [Double]
        /// P(this level fits) per level, before normalisation, when every level was judged
        /// alone (`Format.decider`); nil when the levels were scored together in one row.
        /// Their sum is near 1 when exactly one level fits, low when none does, high when
        /// several do.
        public let fit: [Double]?
    }

    /// Where the tokens of one decision went.
    public struct Timing: Sendable, Equatable {
        /// Tokens in the rendered prompt.
        public let promptTokens: Int
        /// Leading tokens the engine already held from the previous decision on the same state.
        public let reusedTokens: Int
        /// Wall-clock seconds from the rewind to the logits.
        public let seconds: Double

        public init(promptTokens: Int, reusedTokens: Int, seconds: Double) {
            self.promptTokens = promptTokens
            self.reusedTokens = reusedTokens
            self.seconds = seconds
        }

        /// Tokens the engine had to process for this decision.
        public var processedTokens: Int { promptTokens - reusedTokens }
        public var milliseconds: Double { seconds * 1000 }
    }

    /// One decision, with where its tokens went.
    public struct Answer: Sendable, Equatable {
        public enum Value: Sendable, Equatable {
            case choice(Choice)
            case score(Score)
            /// P(yes).
            case noul(Double)
        }

        public let value: Value
        public let timing: Timing
        /// The probability the model puts on "none of the listed options", when its head has
        /// a slot for that (`Format.slot`); nil for the other formats. The option
        /// probabilities are renormalised without it, as the model's own API reports them.
        public let abstain: Double?

        public init(value: Value, timing: Timing, abstain: Double? = nil) {
            self.value = value
            self.timing = timing
            self.abstain = abstain
        }

        /// P(yes) of a `noul` question; nil for the other shapes.
        public var noul: Double? {
            if case .noul(let p) = value { return p }
            return nil
        }

        /// The chosen option id of a `choice` question; nil for the other shapes.
        public var choice: String? {
            if case .choice(let c) = value { return c.id }
            return nil
        }

        /// The expected level of a `score` question; nil for the other shapes.
        public var score: Double? {
            if case .score(let s) = value { return s.value }
            return nil
        }

        /// Probability of the reported answer (P(yes) or P(no) for `noul`, whichever is larger).
        public var confidence: Double {
            switch value {
            case .choice(let c): return c.confidence
            case .score(let s): return s.confidence
            case .noul(let p): return max(p, 1 - p)
            }
        }

        /// Probabilities over the answer shape, in option order.
        public var probabilities: [Double] {
            switch value {
            case .choice(let c): return c.options.map { c.probabilities[$0] ?? 0 }
            case .score(let s): return s.probabilities
            case .noul(let p): return [1 - p, p]
            }
        }
    }
}

/// Failures specific to typed decisions.
public enum DecisionError: Error, LocalizedError, Equatable {
    /// The engine the model loaded with samples on the GPU and cannot expose logits.
    case engineWithoutLogits(model: String)
    /// A model the decision runtime cannot drive (the Gemma 4 pairs and raw-Metal packs).
    case unsupportedModel(id: String, reason: String)
    case emptyInstructions
    case tooFewOptions(count: Int)
    case tooManyOptions(count: Int, max: Int)
    /// The tokenizer has no single-token answer slot for this letter.
    case answerSlotNotSingleToken(letter: String)
    /// A slot-head bundle's tokenizer does not carry this control token as one token.
    case controlTokenNotSingleToken(token: String)
    case promptTooLong(tokens: Int, max: Int)
    /// The engine returned no logits for the prompt.
    case noLogits

    public var errorDescription: String? {
        switch self {
        case .engineWithoutLogits(let model):
            return "'\(model)' loaded on an engine that cannot expose logits; typed decisions need "
                + "the sequential or static-shape engine."
        case .unsupportedModel(let id, let reason):
            return "'\(id)' cannot be used for typed decisions: \(reason)"
        case .emptyInstructions:
            return "A question needs instructions."
        case .tooFewOptions(let count):
            return "A question needs at least 2 options, got \(count)."
        case .tooManyOptions(let count, let max):
            return "A question can list at most \(max) options, got \(count)."
        case .answerSlotNotSingleToken(let letter):
            return "The tokenizer does not encode answer slot '\(letter)' as one token."
        case .controlTokenNotSingleToken(let token):
            return "The tokenizer does not encode control token '\(token)' as one token; "
                + "this bundle is not the slot-head decision model its metadata declares."
        case .promptTooLong(let tokens, let max):
            return "The rendered prompt is \(tokens) tokens; this model takes at most \(max)."
        case .noLogits:
            return "The engine returned no logits for the prompt."
        }
    }
}
