// From the model zoo's apps/Kev/Sources/Kev/Request.swift (9e06b5a, sha256 7f667356851f), identifiers prefixed Kev for the kit.
// Request — the SystemOne request and the author's text for it (conversion/kev/host.py §1–2: `validate_request`,
// `render`, `option_text`, `question_keys`, `to_record`; the author's kev/api.py at tag kev-1.0):
//
//   {"model": str = "kev-latest", "state": <JSON>, "questions": {qid: question, ...}}
//   question = {"type": "noul",   "instructions"?: <JSON>, "criteria"?: {str: <JSON>} | null}
//            | {"type": "choice", "instructions"?: <JSON>, "criteria":  {str: <JSON>}}      1..255 entries
//            | {"type": "score",  "instructions"?: <JSON>, "criteria":  [<JSON>, ...]}      1..255 items
//   `state` is required (null allowed); `questions` needs one entry; `model`, when present, is a string; a noul
//   `criteria` object may carry any keys (only "true" and "false" are read); unknown fields are ignored.
//
//   render(null) = ""; render(str | number | bool) = Python's str() ("True", an int's digits, a float's repr)
//   render(array, i) = "\n".join(pad + "- " + render(item, i + 1).lstrip())
//   render(object, i) = "\n".join(pad + key + ":\n" + render(value, i + 1)   for an object or array value
//                                 pad + key + ": " + render(value)         otherwise)          pad = 2 spaces per level
//   option_text(name, desc) = name when desc is null / absent / "", else name + ": " + render(desc)
//   options: noul [option_text("no", criteria.false), option_text("yes", criteria.true)]; choice the criteria in
//   request order; score render(level) per level. keys: noul ["false", "true"]; choice the names; score "0".."n-1".

import Foundation

struct KevRequest: Sendable {
    struct Question: Sendable {
        let id: String
        /// "noul", "choice" or "score"
        let type: String
        /// .null when absent
        let instructions: KevJSON
        /// nil when absent
        let criteria: KevJSON?
    }

    static let defaultModel = "kev-latest"
    static let questionTypes: Set<String> = ["noul", "choice", "score"]
    static let maxOptions = 255

    let model: String
    let state: KevJSON
    let questions: [Question]

    init(data: Data) throws { try self.init(json: try KevJSONParser.parse(data)) }

    /// `host.validate_request`: the accept / reject decisions of the author's pydantic models.
    init(json: KevJSON) throws {
        guard let top = json.members else { throw KevError.request("the request must be a JSON object") }
        guard let state = json["state"] else { throw KevError.request("state: field required") }
        var model = Self.defaultModel
        if top.contains(where: { $0.key == "model" }) {
            guard case .string(let m)? = json["model"] else { throw KevError.request("model: must be a string") }
            model = m
        }
        guard let qs = json["questions"]?.members else {
            throw KevError.request("questions: must be an object of question id -> question")
        }
        guard !qs.isEmpty else { throw KevError.request("questions: at least one question is required") }
        var out: [Question] = []
        for m in qs {
            let qid = m.key
            guard let q = m.value.members else { throw KevError.request("questions.\(qid): must be an object") }
            guard case .string(let kind)? = m.value["type"], Self.questionTypes.contains(kind) else {
                throw KevError.request("questions.\(qid).type: must be one of 'noul', 'choice', 'score'")
            }
            let hasCriteria = q.contains(where: { $0.key == "criteria" })
            let criteria = m.value["criteria"]
            switch kind {
            case "noul":
                if let c = criteria, !c.isNull, c.members == nil {
                    throw KevError.request("questions.\(qid).criteria: a noul question takes an object or null")
                }
            case "choice":
                guard hasCriteria else { throw KevError.request("questions.\(qid).criteria: field required") }
                guard let c = criteria?.members else {
                    throw KevError.request("questions.\(qid).criteria: a choice question takes an object")
                }
                guard (1...Self.maxOptions).contains(c.count) else {
                    throw KevError.request("questions.\(qid).criteria: must have 1..\(Self.maxOptions) options")
                }
            default:
                guard hasCriteria else { throw KevError.request("questions.\(qid).criteria: field required") }
                guard let c = criteria?.array else {
                    throw KevError.request("questions.\(qid).criteria: a score question takes an array")
                }
                guard (1...Self.maxOptions).contains(c.count) else {
                    throw KevError.request("questions.\(qid).criteria: must have 1..\(Self.maxOptions) levels")
                }
            }
            out.append(Question(id: qid, type: kind, instructions: m.value["instructions"] ?? .null,
                                criteria: hasCriteria ? criteria : nil))
        }
        self.model = model
        self.state = state
        self.questions = out
    }
}

/// One question's place in the answers: its id, type, keys (the order its probabilities are reported in) and, for a
/// score question, the legend {"0": render(level 0), ...}.
struct KevQuestionMeta: Sendable, Equatable {
    let id: String
    let type: String
    let keys: [String]
    let legend: [(String, String)]?

    static func == (a: KevQuestionMeta, b: KevQuestionMeta) -> Bool {
        a.id == b.id && a.type == b.type && a.keys == b.keys
            && a.legend?.map { [$0.0, $0.1] } == b.legend?.map { [$0.0, $0.1] }
    }
}

/// `kev.api.to_record`'s record: the rendered state and, per question, the rendered instructions and option texts.
struct KevRecord: Sendable {
    struct Question: Sendable {
        let instr: String
        let options: [String]
    }

    let state: String
    let questions: [Question]
}

enum KevText {
    /// `kev.api.render`.
    static func render(_ v: KevJSON, indent: Int = 0) -> String {
        let pad = String(repeating: "  ", count: indent)
        switch v {
        case .null: return ""
        case .bool(let b): return b ? "True" : "False"
        case .number(let s): return KevPythonFormat.numberStr(s)
        case .string(let s): return s
        case .array(let a):
            return a.map { pad + "- " + KevPythonFormat.lstrip(render($0, indent: indent + 1)) }.joined(separator: "\n")
        case .object(let m):
            return m.map { member -> String in
                switch member.value {
                case .object, .array: return pad + member.key + ":\n" + render(member.value, indent: indent + 1)
                default: return pad + member.key + ": " + render(member.value)
                }
            }.joined(separator: "\n")
        }
    }

    /// `kev.api.option_text`: the name alone when the description is absent, null or "".
    static func optionText(_ name: String, _ desc: KevJSON?) -> String {
        guard let d = desc, !d.isNull, d != .string("") else { return name }
        return name + ": " + render(d)
    }

    /// `kev.api.question_keys`.
    static func questionKeys(_ q: KevRequest.Question) -> [String] {
        switch q.type {
        case "choice": return q.criteria?.members?.map(\.key) ?? []
        case "noul": return ["false", "true"]
        default: return (0..<(q.criteria?.array?.count ?? 0)).map(String.init)
        }
    }

    /// `kev.api.to_record` on a validated request.
    static func record(_ r: KevRequest) -> (KevRecord, [KevQuestionMeta]) {
        var qs: [KevRecord.Question] = []
        var meta: [KevQuestionMeta] = []
        for q in r.questions {
            let keys = questionKeys(q)
            let opts: [String]
            var legend: [(String, String)]? = nil
            switch q.type {
            case "noul":
                let c = q.criteria
                opts = [optionText("no", c?["false"]), optionText("yes", c?["true"])]
            case "choice":
                opts = (q.criteria?.members ?? []).map { optionText($0.key, $0.value) }
            default:
                opts = (q.criteria?.array ?? []).map { render($0) }
                legend = Array(zip(keys, opts))
            }
            qs.append(KevRecord.Question(instr: render(q.instructions), options: opts))
            meta.append(KevQuestionMeta(id: q.id, type: q.type, keys: keys, legend: legend))
        }
        return (KevRecord(state: render(r.state), questions: qs), meta)
    }
}
