// LabelTable.swift — the answer labels of a letter readout: one single-token name per option,
// so a choice can list as many options as the hosted API takes (255).
//
// The names are the ones decider-0.8b's author labels a wide question with: A–Z, then the
// two-letter uppercase strings AA, AB, … ZZ in that order, 702 candidates. A table keeps the
// first `limit` of them the tokenizer writes as one token, so which two-letter names it holds
// depends on the tokenizer — MiniCPM5 keeps every candidate up to IU, the Qwen tokenizers skip
// BQ, BZ, CJ, … and end at JT — and an option's label is looked up by its index in the table
// built for the loaded tokenizer, never derived from the index: option 100 is `CV` on MiniCPM5
// and `DA` on Qwen. Where A–P are single tokens the first sixteen names are A–P, so every
// question the sixteen-letter readout could render renders exactly as it did.
//
// One rule per prompt form:
//   - `.decider`: the name encodes to one token (the author's `label_table`).
//   - `.chat`: also, the token decodes back to the name, and the name does not merge with the
//     "\n\n" before the answer slot, where a chat model writes its answer after the closed
//     think block.
// A name whose token is already in the table is skipped under both, so no two options share
// a slot.
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

    /// A–Z, then AA, AB, … ZZ.
    static let candidates: [String] = {
        let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".map(String.init)
        return letters + letters.flatMap { first in letters.map { first + $0 } }
    }()

    /// The widest choice a table labels: the hosted API's 255 options.
    static let limit = 255

    /// The label of each option, in option order.
    let names: [String]
    /// The answer-slot token of each label.
    let ids: [Int32]

    var count: Int { names.count }

    /// The first `limit` candidates that are one token under `rule`. `encode` and `decode`
    /// are the tokenizer's, without special tokens.
    static func build(
        rule: Rule, limit: Int = limit, encode: (String) -> [Int], decode: ([Int]) -> String
    ) -> LabelTable {
        let newline = encode("\n\n")
        var names: [String] = []
        var ids: [Int32] = []
        var used = Set<Int>()
        for name in candidates where names.count < limit {
            guard let id = slotID(name, rule: rule, newline: newline, encode: encode, decode: decode),
                used.insert(id).inserted
            else { continue }
            names.append(name)
            ids.append(Int32(id))
        }
        return LabelTable(names: names, ids: ids)
    }

    static func build(rule: Rule, limit: Int = limit, tokenizer: any Tokenizer) -> LabelTable {
        build(
            rule: rule, limit: limit,
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
