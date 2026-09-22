// decide-cli — the argument shell over `decide(state:questions:model:)` (Sources/QuickStart.swift),
// plus the two measurements the README quotes. Progress goes to stderr so stdout stays
// machine-checkable (agents: assert on stdout).
//
//   swift run -c release decide-cli ask --state "…" --noul "Does the speaker want a reply?" \
//       --choice "What is it about?|billing|delivery|other" --score "How urgent?|can wait|this week|today"
//   swift run -c release decide-cli bench --model minicpm5-2b --repeat 3
//   swift run -c release decide-cli oracle --model qwen3-0.6b --fixture authored144.jsonl \
//       --prompts prompts.jsonl --reference predictions.jsonl --limit 48 --out predictions.out.jsonl
//   swift run -c release decide-cli parity --fixture fixtures-decider-0.8b.json --model decider-0.8b
//   (a slot-head model's fixture, coreai-slot-fixtures/1, reads the same way; JSON states need no --states)
//   printf 'line\nline\n' | swift run -c release decide-cli filter --noul "Is this a bug report?"
//   swift run -c release decide-cli --list-models

import CoreAIOps
import Foundation

let usage = """
    usage: decide-cli ask   --state <text> [--model <catalog-id>]
                            (--noul <q> | --choice "<q>|<opt>|<opt>…" | --score "<q>|<level>|<level>…")…
           decide-cli bench [--model <catalog-id>] [--state-file <path>] [--repeat <n>]
           decide-cli oracle --fixture <rows.jsonl> [--prompts <rows.jsonl>] [--reference <rows.jsonl>]
                            [--model <catalog-id>] [--limit <n>] [--out <predictions.jsonl>]
           decide-cli parity --fixture <decider-fixtures.json> [--states <id-to-text.json>] [--model <catalog-id>] [--verbose]
                            (--bundle <dir> loads a local bundle directory instead of a catalog id, for any command)
           decide-cli filter (--noul <q> [--threshold <p>] | --choice "<q>|<opt>|<opt>…") [--all] [--model <catalog-id>]
                            (one text per line on stdin; passing lines on stdout, tab-separated with the answer)
           decide-cli serve [--model <catalog-id>] [--host 127.0.0.1] [--port 8090]
                            (a /v1/systemone endpoint over the loaded model, in the hosted request and answer forms)
           decide-cli --list-models
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String) -> Never {
    stderrPrint(message)
    exit(1)
}

func fmt(_ value: Double, _ digits: Int = 3) -> String {
    String(format: "%.\(digits)f", value)
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else { return 0 }
    return sorted.count % 2 == 1
        ? sorted[sorted.count / 2] : (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
}

let progress: @Sendable (DownloadProgress) -> Void = { progress in
    stderrPrint(
        String(format: "\rdownloading  %3.0f%%", progress.fraction * 100),
        terminator: progress.fraction < 1 ? "" : "\n")
}

var args = CommandLine.arguments.dropFirst()
guard let command = args.popFirst() else { fail(usage) }

if command == "--list-models" {
    for entry in ModelCatalog.builtin.available(.chat) + ModelCatalog.builtin.available(.decision) {
        print("\(entry.id)  —  \(entry.name)  [\(entry.kind.rawValue)]")
    }
    exit(0)
}

var modelID = CoreAI.defaultDecisionModel
var state: String?
var stateFile: String?
var questions: [(String, Decision.Question)] = []
var repeatCount = 3
var fixture: String?
var promptsPath: String?
var referencePath: String?
var limit = Int.max
var outPath: String?
var verbose = false
var threshold = 0.5
var printAll = false
var statesPath: String?
var host = "127.0.0.1"
var port: UInt16 = 8090
/// A local bundle directory instead of a catalog id — a port gated before it is published.
var bundlePath: String?

/// The decider every command loads: the `--bundle` directory when given, else the catalog id.
@MainActor func loadDecider(configuration: TypedDecisions.Configuration = .init()) async throws -> TypedDecisions {
    if let bundlePath {
        return try await TypedDecisions(bundleAt: URL(fileURLWithPath: bundlePath), configuration: configuration)
    }
    return try await TypedDecisions(catalog: modelID, configuration: configuration, downloadProgress: progress)
}

func parts(_ spec: String) -> (String, [String]) {
    let pieces = spec.split(separator: "|", omittingEmptySubsequences: false).map(String.init)
    return (pieces[0], Array(pieces.dropFirst()))
}

while let arg = args.popFirst() {
    switch arg {
    case "--model": modelID = args.popFirst() ?? modelID
    case "--state": state = args.popFirst()
    case "--state-file": stateFile = args.popFirst()
    case "--noul":
        guard let q = args.popFirst() else { fail(usage) }
        questions.append(("q\(questions.count + 1)", .noul(q)))
    case "--choice":
        guard let spec = args.popFirst() else { fail(usage) }
        let (q, options) = parts(spec)
        questions.append(("q\(questions.count + 1)", .choice(q, options)))
    case "--score":
        guard let spec = args.popFirst() else { fail(usage) }
        let (q, levels) = parts(spec)
        questions.append(("q\(questions.count + 1)", .score(q, levels: levels)))
    case "--repeat": repeatCount = Int(args.popFirst() ?? "") ?? repeatCount
    case "--fixture": fixture = args.popFirst()
    case "--prompts": promptsPath = args.popFirst()
    case "--reference": referencePath = args.popFirst()
    case "--limit": limit = Int(args.popFirst() ?? "") ?? limit
    case "--out": outPath = args.popFirst()
    case "--verbose": verbose = true
    case "--threshold": threshold = Double(args.popFirst() ?? "") ?? threshold
    case "--states": statesPath = args.popFirst()
    case "--all": printAll = true
    case "--host": host = args.popFirst() ?? host
    case "--port": port = UInt16(args.popFirst() ?? "") ?? port
    case "--bundle":
        bundlePath = args.popFirst()
        if let bundlePath { modelID = URL(fileURLWithPath: bundlePath).lastPathComponent }
    default: fail(usage)
    }
}

func describe(_ answer: Decision.Answer) -> String {
    let t = answer.timing
    let where_ = "\(fmt(t.milliseconds, 1)) ms, \(t.promptTokens) tokens, \(t.reusedTokens) reused"
    switch answer.value {
    case .noul(let p):
        return "noul  P(yes)=\(fmt(p))  [\(where_)]"
    case .choice(let c):
        let dist = c.options.map { "\($0)=\(fmt(c.probabilities[$0] ?? 0))" }.joined(separator: " ")
        return "choice \(c.id)  confidence=\(fmt(c.confidence)) certainty=\(fmt(c.certainty))  {\(dist)}  [\(where_)]"
    case .score(let s):
        let dist = s.probabilities.enumerated().map { "\($0.offset)=\(fmt($0.element))" }.joined(separator: " ")
        return "score \(fmt(s.value, 2))  level=\(s.level) confidence=\(fmt(s.confidence))  {\(dist)}  [\(where_)]"
    }
}

let id = modelID

// MARK: - ask

@MainActor func runAsk() async throws {
    guard let state, !questions.isEmpty else { fail(usage) }
    let asked = Dictionary(uniqueKeysWithValues: questions)
    // `--bundle` must reach every command: the QuickStart snippet only knows catalog ids.
    let answers: [String: Decision.Answer]
    if bundlePath != nil {
        let decider = try await loadDecider()
        stderrPrint("model: \(id) (\(await decider.modelName))   format: \(decider.format.rawValue)")
        answers = try await decider.decide(state, asked)
    } else {
        answers = try await decide(state: state, questions: asked, model: id, downloadProgress: progress)
    }
    for (key, _) in questions {
        print("\(key): \(describe(answers[key]!))")
    }
}

// MARK: - bench

/// A state long enough to make prefix reuse visible (about 120 tokens), and the mix of
/// questions an app would ask about it.
let benchState = """
    Support ticket #4821. Customer wrote: "The espresso machine I ordered on the 3rd arrived today \
    with the box crushed on one side. The machine powers on but the steam wand is bent and \
    leaks at the joint. I need this fixed before the weekend because we open the café on \
    Saturday. I would prefer a replacement unit over a repair. Please advise on next steps and \
    whether you can ship overnight." Order value: 1,240. Customer since 2023. Previous tickets: 0.
    """

let benchQuestions: [Decision.Question] = [
    .noul("Does the customer want a replacement rather than a repair?"),
    .noul("Is the customer asking for a response today?"),
    .choice("What is the ticket mainly about?", ["billing", "damaged delivery", "how-to question", "cancellation"]),
    .choice("Which team should handle it?", ["sales", "logistics", "technical support", "finance"]),
    .score("How urgent is the request?", levels: ["can wait", "this week", "today"]),
    .score("How frustrated is the customer?", levels: ["calm", "concerned", "upset", "angry"]),
    .noul("Does the ticket mention a deadline?"),
    .choice("What does the customer ask for first?", ["refund", "replacement", "repair", "information"]),
]

@MainActor func bench(decider: TypedDecisions, state: String, label: String) async throws -> (perDecision: Double, total: Double, reused: Int) {
    var totals: [Double] = []
    var per: [Double] = []
    var reused = 0
    for _ in 0..<repeatCount {
        try await decider.reset()
        let start = SuspendingClock.now
        let prefilled = try await decider.prefill(state)
        var answers: [Decision.Answer] = []
        for question in benchQuestions {
            answers.append(try await prefilled.decide(question))
        }
        let elapsed = SuspendingClock.now - start
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        totals.append(seconds * 1000)
        per.append(contentsOf: answers.map(\.timing.milliseconds))
        reused = answers.map(\.timing.reusedTokens).min() ?? 0
        stderrPrint("  \(label): \(fmt(seconds * 1000, 0)) ms for prefill + \(benchQuestions.count) decisions")
    }
    return (median(per), median(totals), reused)
}

@MainActor func runBench() async throws {
    let text = try stateFile.map { try String(contentsOfFile: $0, encoding: .utf8) } ?? benchState
    var shared = TypedDecisions.Configuration()
    shared.sharePrefix = true
    let decider = try await loadDecider(configuration: shared)
    let name = await decider.modelName
    // Warm: the first prompt at a new length pays the engine's specialization.
    _ = try await decider.decide(text, benchQuestions[0])
    let s = try await bench(decider: decider, state: text, label: "shared")
    var direct = TypedDecisions.Configuration()
    direct.sharePrefix = false
    let directDecider = try await loadDecider(configuration: direct)
    _ = try await directDecider.decide(text, benchQuestions[0])
    let d = try await bench(decider: directDecider, state: text, label: "direct")
    let stateTokens = try await decider.prefill(text).tokens
    print("model: \(id) (\(name))")
    print("state tokens: \(stateTokens)   questions: \(benchQuestions.count)   repeat: \(repeatCount)")
    print("| mode | ms per decision (median) | ms per state, prefill + \(benchQuestions.count) decisions (median) | reused tokens |")
    print("|---|---:|---:|---:|")
    print("| shared | \(fmt(s.perDecision, 1)) | \(fmt(s.total, 0)) | \(s.reused) |")
    print("| direct | \(fmt(d.perDecision, 1)) | \(fmt(d.total, 0)) | \(d.reused) |")
    print("shared / direct: \(fmt(d.total / s.total, 2))× per state, \(fmt(d.perDecision / s.perDecision, 2))× per decision")
}

// MARK: - oracle

struct FixtureRow: Decodable {
    struct Option: Decodable {
        let id: String
        let description: String
    }
    let id: String
    let state: String
    let question: String
    let options: [Option]
    let label: Int?
}

struct PromptRow: Decodable {
    let id: String
    let ids: [Int32]
    let answer_token_ids: [Int32]
}

struct ReferenceRow: Decodable {
    let id: String
    let probabilities: [Double]
    let option_logits: [Double]?
}

func readRows<Row: Decodable>(_ path: String, as type: Row.Type) throws -> [Row] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let decoder = JSONDecoder()
    return try text.split(separator: "\n").filter { !$0.isEmpty }.map { line in
        try decoder.decode(Row.self, from: Data(line.utf8))
    }
}

@MainActor func runOracle() async throws {
    guard let fixture else { fail(usage) }
    let rows = try readRows(fixture, as: FixtureRow.self)
    let prompts = try promptsPath.map { try readRows($0, as: PromptRow.self) } ?? []
    let promptByID = Dictionary(uniqueKeysWithValues: prompts.map { ($0.id, $0) })
    let references = try referencePath.map { try readRows($0, as: ReferenceRow.self) } ?? []
    let referenceByID = Dictionary(uniqueKeysWithValues: references.map { ($0.id, $0) })

    let decider = try await loadDecider()
    let name = await decider.modelName
    var out: [String] = []
    var scored = 0, tokensExact = 0, tokensChecked = 0, argmaxAgree = 0, labelCorrect = 0, labelled = 0
    var deltas: [Double] = []
    var milliseconds: [Double] = []
    var skippedStates = 0
    for row in rows.prefix(limit) {
        let question = Decision.Question.choice(
            row.question, options: row.options.map { .init(id: $0.id, description: $0.description) })
        let answer = try await decider.decide(row.state, question)
        guard case .choice(let choice) = answer.value else { continue }
        scored += 1
        milliseconds.append(answer.timing.milliseconds)
        let p = choice.options.map { choice.probabilities[$0] ?? 0 }
        let best = p.indices.max { p[$0] < p[$1] } ?? 0
        if let label = row.label {
            labelled += 1
            if best == label { labelCorrect += 1 }
        }
        if let prompt = promptByID[row.id] {
            tokensChecked += 1
            let rendered = try decider.promptRows(row.state, question)[0]
            if rendered.tokens == prompt.ids, rendered.slots == prompt.answer_token_ids {
                tokensExact += 1
            } else if verbose {
                let firstDiff = zip(rendered.tokens, prompt.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                stderrPrint("  \(row.id): tokens differ (kit \(rendered.tokens.count) vs ref \(prompt.ids.count), first diff at \(firstDiff.map(String.init) ?? "-"); slots kit \(rendered.slots) ref \(prompt.answer_token_ids))")
            }
        }
        if let reference = referenceByID[row.id] {
            let refBest = reference.probabilities.indices.max { reference.probabilities[$0] < reference.probabilities[$1] } ?? 0
            if refBest == best { argmaxAgree += 1 }
            deltas.append(zip(p, reference.probabilities).map { abs($0 - $1) }.max() ?? 0)
            if verbose {
                stderrPrint("  \(row.id) label=\(row.label.map(String.init) ?? "-") kit=\(best) \(p.map { fmt($0) })  ref=\(refBest) \(reference.probabilities.map { fmt($0) })")
            }
        }
        let probs = p.map { fmt($0, 6) }.joined(separator: ", ")
        let optionIDs = row.options.map { "\"\($0.id)\"" }.joined(separator: ", ")
        out.append(
            "{\"id\": \"\(row.id)\", \"option_ids\": [\(optionIDs)], \"probabilities\": [\(probs)], "
                + "\"input_tokens\": \(answer.timing.promptTokens), \"forward_seconds\": \(fmt(answer.timing.seconds, 4)), "
                + "\"model\": {\"catalog\": \"\(id)\", \"name\": \"\(name)\"}}")
    }
    if let outPath {
        try out.joined(separator: "\n").appending("\n").write(toFile: outPath, atomically: true, encoding: .utf8)
    }
    print("model: \(id) (\(name))   rows scored: \(scored)   skipped (non-string state): \(skippedStates)")
    if labelled > 0 {
        print("accuracy vs fixture label: \(labelCorrect)/\(labelled) = \(fmt(Double(labelCorrect) / Double(labelled), 4))")
    }
    if tokensChecked > 0 {
        print("prompt token count == reference rendering: \(tokensExact)/\(tokensChecked)")
    }
    if !deltas.isEmpty {
        print("argmax agreement with reference: \(argmaxAgree)/\(deltas.count) = \(fmt(Double(argmaxAgree) / Double(deltas.count), 4))")
        print("max |Δp| vs reference: max \(fmt(deltas.max() ?? 0, 4)), mean \(fmt(deltas.reduce(0, +) / Double(deltas.count), 4))")
    }
    print("ms per decision: median \(fmt(median(milliseconds), 1)), max \(fmt(milliseconds.max() ?? 0, 1))")
}

// MARK: - parity (a decision model's own fixture: every row's token ids, slot and probabilities)

/// The fixture a decision-model port ships (`coreai-decider-fixtures/1`): requests in the
/// model's wire shape, and the rows they were planned into with the author's fp32 readout.
struct DeciderFixture: Decodable {
    struct Request: Decodable {
        let id: String
        /// The state as text; nil when the fixture carries it as a JSON value (the model's
        /// API serialises those itself — `--states` supplies that rendering).
        let state: String?

        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            state = try? c.decode(String.self, forKey: .state)
        }

        enum CodingKeys: String, CodingKey { case id, state }
    }
    struct Row: Decodable {
        let id: String
        let request_id: String
        let question_id: String
        let kind: String  // "list" (one row) or "iso" (one row per score level); "slot" (a slot-head row)
        let type: String  // choice / score / noul
        let level_index: Int?
        let question: String
        let options: [String]
        let ids: [Int32]
        let slot: Int
        let nopts: Int
        let label_ids: [Int32]
        let p_oracle: [Double]
        /// A slot-head fixture (`coreai-slot-fixtures/1`): the abstain mass and this row's
        /// temperature beside the renormalised option probabilities.
        let abstain: Double?
        let temperature: Double?
    }
    let schema: String
    /// One temperature for the letter readout; a slot-head fixture carries one per type instead.
    let temperature: Double?
    let temperature_by_type: [String: Double]?
    let requests: [Request]
    let rows: [Row]
}

/// Rebuilds the typed question a group of fixture rows was planned from, so the kit's own
/// rendering (`Decision.Question` → rows → tokens) is what gets compared, not the fixture's
/// pre-rendered strings.
func fixtureQuestion(_ rows: [DeciderFixture.Row]) -> Decision.Question? {
    guard let first = rows.first else { return nil }
    switch first.type {
    case "choice":
        let options = first.options.map { text -> Decision.Option in
            guard let range = text.range(of: ": ") else { return .init(text) }
            // A slot row keeps the wire codec's composed form (`name: description` as the
            // description), so a description equal to its name survives; the decider rows
            // were produced from separate fields.
            if first.kind == "slot" { return .init(id: String(text[..<range.lowerBound]), description: text) }
            return .init(id: String(text[..<range.lowerBound]), description: String(text[range.upperBound...]))
        }
        return .choice(first.question, options: options)
    case "noul":
        func tail(_ text: String, _ head: String) -> String? {
            text.hasPrefix(head + ": ") ? String(text.dropFirst(head.count + 2)) : nil
        }
        return .noul(first.question, yes: tail(first.options[1], "yes"), no: tail(first.options[0], "no"))
    case "score":
        if first.kind == "slot" {  // one row, the levels as "i: level"
            let levels = first.options.enumerated().compactMap { index, text -> String? in
                let prefix = "\(index): "
                return text.hasPrefix(prefix) ? String(text.dropFirst(prefix.count)) : nil
            }
            return levels.count == first.options.count ? .score(first.question, levels: levels) : nil
        }
        let marker = "\nProposed answer: "
        guard let head = first.question.range(of: marker) else { return nil }
        let instructions = String(first.question[..<head.lowerBound])
        let levels = rows.sorted { ($0.level_index ?? 0) < ($1.level_index ?? 0) }.compactMap { row -> String? in
            guard let start = row.question.range(of: marker),
                let end = row.question.range(of: "\nDoes the proposed answer fit?")
            else { return nil }
            return String(row.question[start.upperBound..<end.lowerBound])
        }
        return levels.count == rows.count ? .score(instructions, levels: levels) : nil
    default:
        return nil
    }
}

/// A letter-readout fixture (`coreai-letter-fixtures/1`, APUS-OpenJev-v1): one row per
/// request in the author's own request shape, the compiled token sequence (chat template
/// included), the letter token per criterion and the fp32 probabilities in label order.
struct LetterFixture: Decodable {
    struct Criterion: Decodable {
        let id: String
        let description: String
    }
    struct Request: Decodable {
        let state: String
        let instructions: String
        let primitive: String
        let criteria: [Criterion]
    }
    struct Row: Decodable {
        let id: String
        let request: Request
        let ids: [Int32]
        let slot: Int
        let label_ids: [Int32]
        let p_oracle: [Double]
        let zoo_only: Bool?
    }
    let schema: String
    let rows: [Row]
}

@MainActor func runLetterParity(_ data: Data) async throws {
    let fx = try JSONDecoder().decode(LetterFixture.self, from: data)
    let decider = try await loadDecider()
    let name = await decider.modelName
    print("model: \(id) (\(name))   format: \(decider.format.rawValue)   temperature: \(decider.temperature)   fixture: \(fx.schema)")
    var checked = 0, tokensExact = 0, slotExact = 0, argmaxAgree = 0, skipped: [String] = []
    var deltas: [Double] = []
    var milliseconds: [Double] = []
    var lines: [String] = []
    for row in fx.rows {
        let request = row.request
        let question: Decision.Question
        switch request.primitive {
        case "choice":
            question = .choice(request.instructions, options: request.criteria.map { .init(id: $0.id, description: $0.description) })
        case "noul":
            question = .noul(request.instructions)
        default:
            // score_level is the author's yes/no on one proposition; the kit's questions have no such kind.
            skipped.append("\(row.id) (primitive \(request.primitive); the kit renders choice and noul)")
            continue
        }
        if request.criteria.count > decider.maxOptions {
            skipped.append("\(row.id) (\(request.criteria.count) criteria; this model lists at most \(decider.maxOptions))")
            continue
        }
        let rendered = try decider.promptRows(request.state, question)[0]
        let answer = try await decider.decide(request.state, question)
        milliseconds.append(answer.timing.milliseconds)
        checked += 1
        let tokensOK = rendered.tokens == row.ids
        let slotsOK = rendered.slots == row.label_ids && rendered.tokens.count - 1 == row.slot
        if tokensOK { tokensExact += 1 }
        if slotsOK { slotExact += 1 }
        // The fixture's probabilities are in label order: yes then no for a noul.
        let p: [Double]
        if case .noul(let yes) = answer.value { p = [yes, 1 - yes] } else { p = answer.probabilities }
        let best = p.indices.max { p[$0] < p[$1] } ?? 0
        let refBest = row.p_oracle.indices.max { row.p_oracle[$0] < row.p_oracle[$1] } ?? 0
        if best == refBest { argmaxAgree += 1 }
        let delta = zip(p, row.p_oracle).map { abs($0 - $1) }.max() ?? 0
        deltas.append(delta)
        let flag = (tokensOK && slotsOK && best == refBest) ? "ok" : "DIFF"
        lines.append("| \(row.id) | \(request.primitive) | \(request.criteria.count) | \(tokensOK ? "=" : "≠") | \(slotsOK ? "=" : "≠") | \(best == refBest ? "=" : "≠") | \(fmt(delta, 4)) | \(flag) |")
        if verbose || flag == "DIFF" {
            let firstDiff = zip(rendered.tokens, row.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            stderrPrint("  \(row.id): tokens kit \(rendered.tokens.count) ref \(row.ids.count) first diff \(firstDiff.map(String.init) ?? "-"); kit p \(p.map { fmt($0) }) ref \(row.p_oracle.map { fmt($0) })")
            if verbose, !tokensOK {
                stderrPrint("  \(row.id): kit tokens \(rendered.tokens.map(String.init).joined(separator: ","))")
            }
        }
    }
    print("| row | primitive | criteria | tokens | slot | argmax | max \\|Δp\\| | |")
    print("|---|---|---:|:-:|:-:|:-:|---:|---|")
    lines.forEach { print($0) }
    print("rows checked: \(checked)   tokens identical: \(tokensExact)/\(checked)   slot + labels identical: \(slotExact)/\(checked)")
    print("argmax agreement with the fp32 readout: \(argmaxAgree)/\(checked)")
    print("|Δp| vs the fp32 readout: max \(fmt(deltas.max() ?? 0, 4)), mean \(fmt(deltas.reduce(0, +) / Double(max(1, deltas.count)), 4))")
    print("ms per question: median \(fmt(median(milliseconds), 1)), max \(fmt(milliseconds.max() ?? 0, 1))")
    if !skipped.isEmpty { print("skipped: " + skipped.joined(separator: "; ")) }
}

@MainActor func runParity() async throws {
    guard let fixture else { fail(usage) }
    let data = try Data(contentsOf: URL(fileURLWithPath: fixture))
    if let schema = try JSONValue.parse(data)["schema"]?.stringValue, schema.hasPrefix("coreai-letter-fixtures") {
        try await runLetterParity(data)
        return
    }
    let fx = try JSONDecoder().decode(DeciderFixture.self, from: data)
    var stateByRequest: [String: String] = [:]
    for request in fx.requests { if let state = request.state { stateByRequest[request.id] = state } }
    // A structured state is what the model's API serialises itself; `JSONValue.dumps` writes
    // the reference bytes (Python's json.dumps, ensure_ascii=False), so no --states is needed.
    if let requests = try JSONValue.parse(data)["requests"]?.elements {
        for request in requests {
            guard let rid = request["id"]?.stringValue, stateByRequest[rid] == nil, let state = request["state"] else { continue }
            if state.stringValue == nil, state != .null { stateByRequest[rid] = state.dumps() }
        }
    }
    if let statesPath {
        let rendered = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: URL(fileURLWithPath: statesPath)))
        for (id, state) in rendered where stateByRequest[id] == nil { stateByRequest[id] = state }
    }
    let decider = try await loadDecider()
    let name = await decider.modelName
    let format = decider.format
    let temperature = decider.temperature
    let fixtureTemperature = fx.temperature.map { String($0) }
        ?? fx.temperature_by_type.map { t in t.keys.sorted().map { "\($0) \(t[$0]!)" }.joined(separator: ", ") }
        ?? "-"
    print("model: \(id) (\(name))   format: \(format.rawValue)   temperature: \(temperature) (fixture \(fixtureTemperature))")

    // Group rows by (request, question), keeping request order.
    var groups: [(key: String, rows: [DeciderFixture.Row])] = []
    var index: [String: Int] = [:]
    for row in fx.rows {
        let key = row.request_id + "/" + row.question_id
        if let i = index[key] {
            groups[i].rows.append(row)
        } else {
            index[key] = groups.count
            groups.append((key, [row]))
        }
    }

    var checked = 0, tokensExact = 0, slotExact = 0, argmaxAgree = 0, skipped: [String] = []
    var deltas: [Double] = []
    var abstainDeltas: [Double] = []
    var milliseconds: [Double] = []
    var lines: [String] = []
    for group in groups {
        let rows = group.rows.sorted { ($0.level_index ?? 0) < ($1.level_index ?? 0) }
        guard let state = stateByRequest[rows[0].request_id] else {
            skipped.append("\(group.key) (JSON state; pass --states)")
            continue
        }
        guard let question = fixtureQuestion(rows) else {
            skipped.append("\(group.key) (could not rebuild the question)")
            continue
        }
        if rows[0].nopts > decider.maxOptions {  // 16 for a letter readout, the slot count for a slot head
            skipped.append("\(group.key) (\(rows[0].nopts) options; this model lists at most \(decider.maxOptions))")
            continue
        }
        let rendered = try decider.promptRows(state, question)
        guard rendered.count == rows.count else {
            skipped.append("\(group.key) (kit planned \(rendered.count) rows, fixture has \(rows.count))")
            continue
        }
        let answer = try await decider.decide(state, question)
        milliseconds.append(answer.timing.milliseconds)
        // Per-row probabilities the kit produced, in the fixture's row order.
        let kitRows: [[Double]]
        if case .score(let score) = answer.value, let fit = score.fit {
            kitRows = fit.map { [1 - $0, $0] }
        } else {
            kitRows = [answer.probabilities]
        }
        for (i, row) in rows.enumerated() {
            checked += 1
            let r = rendered[i]
            let tokensOK = r.tokens == row.ids
            let slotsOK = Array(r.slots.prefix(row.nopts)) == Array(row.label_ids.prefix(row.nopts)) && r.tokens.count - 1 == row.slot
            if tokensOK { tokensExact += 1 }
            if slotsOK { slotExact += 1 }
            let p = kitRows[i]
            let best = p.indices.max { p[$0] < p[$1] } ?? 0
            let refBest = row.p_oracle.indices.max { row.p_oracle[$0] < row.p_oracle[$1] } ?? 0
            if best == refBest { argmaxAgree += 1 }
            let delta = zip(p, row.p_oracle).map { abs($0 - $1) }.max() ?? 0
            deltas.append(delta)
            if let reference = row.abstain, let abstain = answer.abstain { abstainDeltas.append(abs(abstain - reference)) }
            let flag = (tokensOK && slotsOK && best == refBest) ? "ok" : "DIFF"
            lines.append("| \(row.id) | \(row.type) | \(row.nopts) | \(tokensOK ? "=" : "≠") | \(slotsOK ? "=" : "≠") | \(best == refBest ? "=" : "≠") | \(fmt(delta, 4)) | \(flag) |")
            if verbose || flag == "DIFF" {
                let firstDiff = zip(r.tokens, row.ids).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                stderrPrint("  \(row.id): tokens kit \(r.tokens.count) ref \(row.ids.count) first diff \(firstDiff.map(String.init) ?? "-"); kit p \(p.map { fmt($0) }) ref \(row.p_oracle.map { fmt($0) })")
                if verbose, !tokensOK {
                    stderrPrint("  \(row.id): kit tokens \(r.tokens.map(String.init).joined(separator: ","))")
                }
            }
        }
    }
    print("| row | type | options | tokens | slot | argmax | max \\|Δp\\| | |")
    print("|---|---|---:|:-:|:-:|:-:|---:|---|")
    lines.forEach { print($0) }
    print("rows checked: \(checked)   tokens identical: \(tokensExact)/\(checked)   slot + labels identical: \(slotExact)/\(checked)")
    print("argmax agreement with the fp32 readout: \(argmaxAgree)/\(checked)")
    print("|Δp| vs the fp32 readout: max \(fmt(deltas.max() ?? 0, 4)), mean \(fmt(deltas.reduce(0, +) / Double(max(1, deltas.count)), 4))")
    if !abstainDeltas.isEmpty {
        print("|Δabstain| vs the fp32 readout: max \(fmt(abstainDeltas.max() ?? 0, 4)), mean \(fmt(abstainDeltas.reduce(0, +) / Double(abstainDeltas.count), 4)) over \(abstainDeltas.count) rows")
    }
    print("ms per question (all rows of a score question summed): median \(fmt(median(milliseconds), 1)), max \(fmt(milliseconds.max() ?? 0, 1))")
    if !skipped.isEmpty { print("skipped: " + skipped.joined(separator: "; ")) }
}

// MARK: - filter (a semantic `grep`: one decision per stdin line)

@MainActor func runFilter() async throws {
    guard questions.count == 1, let (_, question) = questions.first else { fail(usage) }
    var lines: [String] = []
    while let line = readLine() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty { lines.append(trimmed) }
    }
    let decider = try await loadDecider()
    var passed = 0
    var milliseconds: [Double] = []
    for line in lines {
        let answer = try await decider.decide(line, question)
        milliseconds.append(answer.timing.milliseconds)
        switch answer.value {
        case .noul(let p):
            let pass = p >= threshold
            if pass { passed += 1 }
            if pass || printAll { print("\(fmt(p, 2))\t\(line)") }
        case .choice(let c):
            passed += 1
            print("\(c.id)\t\(fmt(c.confidence, 2))\t\(line)")
        case .score(let s):
            passed += 1
            print("\(fmt(s.value, 2))\t\(line)")
        }
    }
    stderrPrint("\(passed)/\(lines.count) lines · median \(fmt(median(milliseconds), 1)) ms per decision · \(id)")
}

// MARK: - serve (a /v1/systemone endpoint over the loaded model — `SystemOneServer` in the kit;
// `systemone serve`, the Homebrew-installed binary, is the same server without a toolchain)

@MainActor func runServe() async throws {
    let decider = try await loadDecider()
    let models: JSONValue
    if bundlePath == nil {
        let entry = try await ModelCatalog.entry(forID: id)
        models = SystemOne.modelsValue(
            id: id,
            description: "\(entry.name), CoreAIKit catalog kind \(entry.kind.rawValue), bundle \(await decider.modelName), on this machine",
            revision: entry.revision)
    } else {
        models = SystemOne.modelsValue(
            id: id, description: "local bundle \(await decider.modelName), on this machine", revision: nil)
    }
    stderrPrint("loaded \(id) (\(await decider.modelName)); one request at a time, questions share the state's prefill")
    let server = SystemOneServer(host: host, port: port, modelID: id, models: models, decider: decider) { line in
        stderrPrint("decide-cli serve: \(line)  (Ctrl-C stops)")
    }
    try await server.run()
}

do {
    switch command {
    case "ask": try await runAsk()
    case "bench": try await runBench()
    case "oracle": try await runOracle()
    case "parity": try await runParity()
    case "filter": try await runFilter()
    case "serve": try await runServe()
    default: fail(usage)
    }
} catch {
    fail("error: \(error.localizedDescription)")
}
