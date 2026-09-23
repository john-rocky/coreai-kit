// LabelTable.swift — the answer labels of a readout at one token per option, so a choice can list
// as many options as the hosted API takes (255) where the tokenizer allows it.
//
// A table keeps, in order, the names from a candidate list that the tokenizer writes as one
// token, and the token each is read at. Three candidate lists:
//
//   - `candidates` — A–Z, then the two-letter uppercase strings AA, AB, … ZZ (702 names): the
//     labels decider-0.8b's author gives a wide question. A name that is not one token is
//     skipped, so which two-letter names a table holds depends on the tokenizer (the decider's
//     skips BQ, BZ, CJ, … and ends at JT; MiniCPM5's keeps every name up to IU), and an option's
//     label is looked up by its index in the table built for the loaded tokenizer.
//   - `letters` — A–Z: a chat model's labels up to 26 options. Where A–P are single tokens the
//     first sixteen are A–P, so every question the sixteen-letter readout rendered renders as it
//     did.
//   - `numbers` — "1", "2", … "255": a chat model's labels past 26 options. A run with no gaps:
//     the table ends at the first number that is not one token (MiniCPM5 keeps all 255; the
//     Qwen tokenizers stop at 9, so a chat model on them lists 26 options).
//
// Why a chat model is not read at two-letter names: asked to answer with a label such as `DX`,
// MiniCPM5 answers with one of its letters (`D`) — fp32 and int8 alike — and picks the right
// option out of 255 about one time in six; with numbers it does on 5 of 6.
//
// One rule per prompt form:
//   - `.decider`: the name encodes to one token (the author's `label_table`).
//   - `.chat`: also, the token decodes back to the name, and the name does not merge with the
//     "\n\n" before the answer slot, where a chat model writes its answer after the closed
//     think block.
// A name whose token is already in the table is skipped (or ends a run), so no two options
// share a slot.
//
// Built once when a model loads (`TypedDecisions`) and handed to the renderers. The builder
// takes the tokenizer's encode and decode as closures so a table can be checked without one.

import Foundation
import Tokenizers

struct LabelTable: Sendable, Equatable {
    enum Rule: Sendable {
        /// One token: the plain-text form of a decision model (`DeciderPrompt`).
        case decider
        /// One token that round-trips and stands alone after "\n\n": the answer slot under a
        /// chat template (`DecisionPrompt`).
        case chat
    }

    /// A–Z.
    static let letters: [String] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".map(String.init)
    /// A–Z, then AA, AB, … ZZ.
    static let candidates: [String] = letters + letters.flatMap { first in letters.map { first + $0 } }
    /// "1" … "255".
    static let numbers: [String] = (1...limit).map(String.init)

    /// The widest choice a table labels: the hosted API's 255 options.
    static let limit = 255

    /// The label of each option, in option order.
    let names: [String]
    /// The answer-slot token of each label.
    let ids: [Int32]

    var count: Int { names.count }

    /// The first `limit` of `names` that are one token under `rule`. A name that is not is
    /// skipped — or, for a `run`, ends the table, so the labels have no gaps. `encode` and
    /// `decode` are the tokenizer's, without special tokens.
    static func build(
        _ names: [String] = candidates, rule: Rule, limit: Int = limit, run: Bool = false,
        encode: (String) -> [Int], decode: ([Int]) -> String
    ) -> LabelTable {
        let newline = encode("\n\n")
        var kept: [String] = []
        var ids: [Int32] = []
        var used = Set<Int>()
        for name in names where kept.count < limit {
            guard let id = slotID(name, rule: rule, newline: newline, encode: encode, decode: decode),
                used.insert(id).inserted
            else {
                if run { break }
                continue
            }
            kept.append(name)
            ids.append(Int32(id))
        }
        return LabelTable(names: kept, ids: ids)
    }

    static func build(
        _ names: [String] = candidates, rule: Rule, limit: Int = limit, run: Bool = false, tokenizer: any Tokenizer
    ) -> LabelTable {
        build(
            names, rule: rule, limit: limit, run: run,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) },
            decode: { tokenizer.decode(tokens: $0, skipSpecialTokens: false) })
    }

    /// The token `name` is read at under `rule`, or nil when it is not a single token there.
    /// `newline` is the encoding of "\n\n".
    static func slotID(
        _ name: String, rule: Rule, newline: [Int], encode: (String) -> [Int], decode: ([Int]) -> String
    ) -> Int? {
        let ids = encode(name)
        guard ids.count == 1 else { return nil }
        if rule == .chat, decode(ids) != name || encode("\n\n" + name) != newline + ids { return nil }
        return ids[0]
    }

    /// The answer-slot tokens of the first `count` options.
    func slots(count: Int) throws -> [Int32] {
        guard count <= ids.count else { throw DecisionError.tooManyOptions(count: count, max: ids.count) }
        return Array(ids.prefix(count))
    }
}
