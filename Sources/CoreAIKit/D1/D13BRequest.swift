// From the model zoo's apps/D1/Sources/D1/Request.swift (e36ad15, sha256 f308ab6ec170), identifiers prefixed D13B for the kit.
// Request — the System One request and the provider's text for it (conversion/d1/host.py §1–2: `validate_question`,
// `validate_request`, `state_block`, `prefix_text`, `option_codes`, `question_block`, `suffix_text`; the provider's
// prompt.py / api.py at LiquidAI/d1-3B@da1fe36a):
//
//   {"state": <JSON> | null, "questions": {name: question, ...}}               names in request order
//   question = {"type": "noul",   "instructions": str, "criteria"?: {"true"?: str | null, "false"?: str | null} | null}
//            | {"type": "choice", "instructions": str, "criteria": {label: str | null, ...}}   1+ options
//            | {"type": "score",  "instructions": str, "criteria": [str, ...]}               1..10 levels
//   `type` absent is "choice"; any other string but "noul" / "score" is a choice; `instructions` is required (the
//   provider indexes it); anything outside these shapes is refused with host.py's ValueError text, character for
//   character (the first failing question in request order). A state is any JSON value; null = no state block.
//
//   prefix  = BOS + "<|im_start|>user\n" + images + state_block + "\nQUESTION:\n"      (state null: no state part)
//   state_block = a string + "\n\n"; any other value json.dumps(state, ensure_ascii=False, indent=2) + "\n\n"
//   suffix  = question_block + "<|im_end|>\n<|im_start|>assistant\n"
//     choice  instructions + "\n\nOptions:\n" + "\n".join(code + " " + (desc or label with "_" -> " "))
//             + "\n\nReply with the option code only."                    (codes = the aliases' codes, Encoder.swift)
//     noul    instructions + ("\nYes: " + str(criteria.get("true")) + "\nNo: " + str(criteria.get("false")) for a
//             non-empty criteria object; a missing or null side prints "None") + "\n\nReply with yes or no only."
//     score   instructions + "\n\n" + "\n".join(str(i) + " " + level) + "\n\nReply with a single digit 0-" + (K - 1)
//             + " only."
//   option_codes(labels) = the labels stripped (str.strip) when every one is one code point and alphabetic
//   (str.isalpha); else "A", "B", .. for at most 26 labels; else "00", "01", .. (f"{i:02d}", so "100" past 99).
//   keys (the order the probabilities come in): noul ["yes", "no"]; choice the labels; score "0" .. "K-1".
//
// Text is compared and edited on code points (Python's str), never on Swift's grapheme clusters.

import Foundation

@available(macOS 27, iOS 27, *)
struct D13BQuestion: Sendable {
    enum Kind: String, Sendable {
        case noul, choice, score
    }

    let name: String
    let kind: Kind
    let instructions: String
    /// noul: the criteria object's members (nil for an absent or null criteria); only "true" and "false" are read
    let noulCriteria: [D13BJSONMember]?
    /// choice: (label, description) in request order; a null description is nil
    let options: [(label: String, desc: String?)]
    /// score: the levels
    let levels: [String]

    var labels: [String] { options.map(\.label) }

    /// `host.option_keys`: noul yes / no, choice the labels, score "0".."K-1".
    var keys: [String] {
        switch kind {
        case .noul: return ["yes", "no"]
        case .choice: return labels
        case .score: return levels.indices.map(String.init)
        }
    }
}

@available(macOS 27, iOS 27, *)
struct D13BRequest: Sendable {
    let state: D13BJSONValue
    let questions: [D13BQuestion]

    init(data: Data) throws { try self.init(json: try D13BJSONParser.parse(data)) }

    /// `host.validate_request`.
    init(json: D13BJSONValue) throws {
        guard let top = json.members else { throw D13BError.request("the request must be a JSON object") }
        guard top.contains(where: { $0.key == "state" }), let state = json["state"] else {
            throw D13BError.request("state: field required (null for no state)")
        }
        guard let qs = json["questions"]?.members, !qs.isEmpty else {
            throw D13BError.request("questions: must be an object of name -> question with at least one entry")
        }
        self.state = state
        questions = try qs.map { try Self.validateQuestion(name: $0.key, $0.value) }
    }

    /// `host.validate_question`.
    static func validateQuestion(name: String, _ q: D13BJSONValue) throws -> D13BQuestion {
        guard let members = q.members else { throw D13BError.request("questions.\(name): must be an object") }
        func has(_ k: String) -> Bool { members.contains(where: { $0.key == k }) }
        var kind = D13BQuestion.Kind.choice
        if has("type") {
            guard case .string(let t)? = q["type"] else { throw D13BError.request("questions.\(name).type: must be a string") }
            kind = t == "noul" ? .noul : t == "score" ? .score : .choice
        }
        guard has("instructions") else { throw D13BError.request("questions.\(name).instructions: field required") }
        guard case .string(let instructions)? = q["instructions"] else {
            throw D13BError.request("questions.\(name).instructions: must be a string")
        }
        let criteria = q["criteria"] ?? .null
        switch kind {
        case .noul:
            var noul: [D13BJSONMember]? = nil
            if !criteria.isNull {
                guard let m = criteria.members else {
                    throw D13BError.request("questions.\(name).criteria: a noul question takes an object or null")
                }
                for side in ["true", "false"] {
                    if let v = criteria[side], !(v.isNull || v.string != nil) {
                        throw D13BError.request("questions.\(name).criteria.\(side): must be a string or null")
                    }
                }
                noul = m
            }
            return D13BQuestion(name: name, kind: .noul, instructions: instructions, noulCriteria: noul, options: [], levels: [])
        case .choice:
            guard has("criteria") else { throw D13BError.request("questions.\(name).criteria: field required") }
            guard let m = criteria.members, !m.isEmpty else {
                throw D13BError.request("questions.\(name).criteria: a choice question takes an object of 1+ options")
            }
            var options: [(label: String, desc: String?)] = []
            for member in m {
                switch member.value {
                case .null: options.append((member.key, nil))
                case .string(let d): options.append((member.key, d))
                default: throw D13BError.request("questions.\(name).criteria.\(member.key): a description is a string or null")
                }
            }
            return D13BQuestion(name: name, kind: .choice, instructions: instructions, noulCriteria: nil, options: options,
                              levels: [])
        case .score:
            guard has("criteria") else { throw D13BError.request("questions.\(name).criteria: field required") }
            guard let a = criteria.array, (1...D13BText.maxScoreLevels).contains(a.count) else {
                throw D13BError.request("questions.\(name).criteria: a score question takes a list of 1..\(D13BText.maxScoreLevels) levels")
            }
            let levels = a.compactMap(\.string)
            guard levels.count == a.count else { throw D13BError.request("questions.\(name).criteria: every level is a string") }
            return D13BQuestion(name: name, kind: .score, instructions: instructions, noulCriteria: nil, options: [], levels: levels)
        }
    }
}

@available(macOS 27, iOS 27, *)
enum D13BText {
    static let bos = "<|startoftext|>"
    static let imStart = "<|im_start|>"
    static let imEnd = "<|im_end|>"
    /// the card's 2..10 (the prompt asks for a single digit); the provider's code would read 0..999
    static let maxScoreLevels = 10

    /// `prompt.state_block(state, "json_only")`.
    static func stateBlock(_ state: D13BJSONValue) -> String {
        if case .string(let s) = state { return s + "\n\n" }
        return D13BPythonFormat.dumps(state, indent: 2, asciiOnly: false) + "\n\n"
    }

    /// `prompt.prefix_text(tok, state, BOS, "json_only", "none", images)`.
    static func prefix(_ state: D13BJSONValue, images: String = "") -> String {
        let body = state.isNull ? "" : stateBlock(state) + "\nQUESTION:\n"
        return bos + imStart + "user\n" + images + body
    }

    /// `prompt.option_codes`.
    static func optionCodes(_ labels: [String]) -> [String] {
        let labs = labels.map(D13BPythonFormat.strip)
        if !labs.isEmpty, labs.allSatisfy({ $0.unicodeScalars.count == 1 && D13BPythonFormat.isAlpha($0) }) { return labs }
        if labs.count <= 26 { return labs.indices.map { String(UnicodeScalar(UInt8(65 + $0))) } }
        return labs.indices.map { $0 < 10 ? "0\($0)" : String($0) }
    }

    /// `str.replace("_", " ")` on code points.
    static func underscoresToSpaces(_ s: String) -> String {
        String(String.UnicodeScalarView(s.unicodeScalars.map { $0 == "_" ? " " : $0 }))
    }

    /// `prompt.question_block(tok, q, "desc")`; `codes` = the aliases' codes (choice only).
    static func questionBlock(_ q: D13BQuestion, codes: [String]?) -> String {
        switch q.kind {
        case .choice:
            let codes = codes ?? optionCodes(q.labels)
            let lines = q.options.enumerated().map { k, o -> String in
                let text = (o.desc?.isEmpty == false) ? o.desc! : underscoresToSpaces(o.label)
                return codes[k] + " " + text
            }.joined(separator: "\n")
            return q.instructions + "\n\nOptions:\n" + lines + "\n\nReply with the option code only."
        case .noul:
            var extra = ""
            if let c = q.noulCriteria, !c.isEmpty {
                let crit = D13BJSONValue.object(c)
                extra = "\nYes: " + D13BPythonFormat.strOrNone(crit["true"]) + "\nNo: " + D13BPythonFormat.strOrNone(crit["false"])
            }
            return q.instructions + extra + "\n\nReply with yes or no only."
        case .score:
            let legend = q.levels.enumerated().map { "\($0.offset) \($0.element)" }.joined(separator: "\n")
            return q.instructions + "\n\n" + legend + "\n\nReply with a single digit 0-\(q.levels.count - 1) only."
        }
    }

    /// `prompt.suffix_text(tok, q, lead="", "desc")`.
    static func suffix(_ q: D13BQuestion, codes: [String]?) -> String {
        questionBlock(q, codes: codes) + imEnd + "\n" + imStart + "assistant\n"
    }
}
