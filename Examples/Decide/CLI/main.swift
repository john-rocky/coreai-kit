// decide-cli — the argument shell over `decide(state:questions:model:)` (Sources/QuickStart.swift),
// plus the two measurements the README quotes. Progress goes to stderr so stdout stays
// machine-checkable (agents: assert on stdout).
//
//   swift run -c release decide-cli ask --state "…" --noul "Does the speaker want a reply?" \
//       --choice "What is it about?|billing|delivery|other" --score "How urgent?|can wait|this week|today"
//   swift run -c release decide-cli bench --model minicpm5-2b --repeat 3
//   swift run -c release decide-cli oracle --model qwen3-0.6b --fixture authored144.jsonl \
//       --prompts prompts.jsonl --reference predictions.jsonl --limit 48 --out predictions.out.jsonl
//   swift run -c release decide-cli --list-models

import CoreAIOps
import Foundation

let usage = """
    usage: decide-cli ask   --state <text> [--model <catalog-id>]
                            (--noul <q> | --choice "<q>|<opt>|<opt>…" | --score "<q>|<level>|<level>…")…
           decide-cli bench [--model <catalog-id>] [--state-file <path>] [--repeat <n>]
           decide-cli oracle --fixture <rows.jsonl> [--prompts <rows.jsonl>] [--reference <rows.jsonl>]
                            [--model <catalog-id>] [--limit <n>] [--out <predictions.jsonl>]
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
    for entry in ModelCatalog.builtin.available(.chat) {
        print("\(entry.id)  —  \(entry.name)")
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
    let answers = try await decide(state: state, questions: asked, model: id, downloadProgress: progress)
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
    let decider = try await TypedDecisions(catalog: id, configuration: shared, downloadProgress: progress)
    let name = await decider.modelName
    // Warm: the first prompt at a new length pays the engine's specialization.
    _ = try await decider.decide(text, benchQuestions[0])
    let s = try await bench(decider: decider, state: text, label: "shared")
    var direct = TypedDecisions.Configuration()
    direct.sharePrefix = false
    let directDecider = try await TypedDecisions(catalog: id, configuration: direct)
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

    let decider = try await TypedDecisions(catalog: id, downloadProgress: progress)
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
            let rendered = try decider.promptTokens(row.state, question)
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

do {
    switch command {
    case "ask": try await runAsk()
    case "bench": try await runBench()
    case "oracle": try await runOracle()
    default: fail(usage)
    }
} catch {
    fail("error: \(error.localizedDescription)")
}
