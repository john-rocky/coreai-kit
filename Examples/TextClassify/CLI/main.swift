// textclassify-cli — argument shell over `TextClassifier.classify(_:tasks:)`: zero-shot text
// classification on-device, several tasks in one forward. Progress goes to stderr so stdout stays
// machine-checkable (agents: assert on stdout).
//
//   swift run textclassify-cli --bundle <dir> --text "Can I get that charge refunded?" \
//       --task intent=order_status,refund_request,cancel_subscription
//   swift run textclassify-cli --bundle <dir> --text "Battery dies before lunch." \
//       --task aspects=battery,keyboard,screen --multi --threshold 0.4 --task sentiment=positive,negative
//   swift run textclassify-cli --bundle <dir> --readme <readme21.json>     # the model card's examples
//   swift run textclassify-cli --bundle <dir> --gate <fixture.json>... [--pygpu <gate_s<S>_gpu.json>...]
//
// --bundle is a directory holding classifier.json, tokenizer/ and the .aimodel files classifier.json
// names. --multi / --threshold / --prompt / --describe apply to the --task before them. --gate compares
// the collated input with the gliner2 oracle's (input_ids, pieces, [P]/[L] positions, shape), then runs
// the graph and compares decisions and logits (--collate-only stops after the first part); --pygpu
// adds the zoo's Python engine gate on the same bundle, matched by case id and S. Exit 0 = all equal,
// 3 = a difference, 1 = usage or load error.

import CoreAIKitEmbeddings
import Foundation

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }
func fail(_ s: String) -> Never { err(s); exit(1) }

let usage = """
    usage: textclassify-cli --bundle <dir> --text <str> --task name=l1,l2,... [--multi] [--threshold 0.4]
                            [--prompt <str>] [--describe label=<description>]... [--task ...]
           textclassify-cli --bundle <dir> --readme <readme21.json>
           textclassify-cli --bundle <dir> --gate <fixture.json>... [--pygpu <gate.json>...] [--collate-only]
    """

var bundle: String?
var text: String?
var tasks: [ClassificationTask] = []
var readme: String?
var gateFiles: [String] = []
var pygpuFiles: [String] = []
var collateOnly = false

@MainActor func lastTask(_ flag: String) -> Int {
    guard !tasks.isEmpty else { fail("\(flag) must follow a --task\n\(usage)") }
    return tasks.count - 1
}

var args = CommandLine.arguments.dropFirst()
while let a = args.popFirst() {
    switch a {
    case "--bundle": bundle = args.popFirst()
    case "--text": text = args.popFirst()
    case "--task":
        guard let spec = args.popFirst(), let eq = spec.firstIndex(of: "=") else { fail(usage) }
        let labels = spec[spec.index(after: eq)...].split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        tasks.append(ClassificationTask(String(spec[..<eq]), labels: labels))
    case "--multi": tasks[lastTask(a)].multiLabel = true
    case "--threshold":
        guard let t = args.popFirst().flatMap(Float.init) else { fail(usage) }
        tasks[lastTask(a)].threshold = t
    case "--prompt": tasks[lastTask(a)].prompt = args.popFirst()
    case "--describe":
        guard let spec = args.popFirst(), let eq = spec.firstIndex(of: "=") else { fail(usage) }
        tasks[lastTask(a)].descriptions[String(spec[..<eq])] = String(spec[spec.index(after: eq)...])
    case "--readme": readme = args.popFirst()
    case "--gate":
        while let f = args.first, !f.hasPrefix("--") { gateFiles.append(f); args.removeFirst() }
    case "--pygpu":
        while let f = args.first, !f.hasPrefix("--") { pygpuFiles.append(f); args.removeFirst() }
    case "--collate-only": collateOnly = true
    case "-h", "--help": print(usage); exit(0)
    default: fail("unknown arg \(a)\n\(usage)")
    }
}
guard let bundlePath = bundle else { fail(usage) }

// MARK: - output

/// gliner2's `classify_text(..., include_confidence=True)` shape: {"task": {"label", "confidence"}} for a
/// single-label task, {"task": [{"label", "confidence"}, ...]} for a multi-label one. Confidence to 4 places.
struct Picked: Encodable { let label: String; let confidence: Float }
enum Answer: Encodable {
    case one(Picked), many([Picked])
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .one(let p): try c.encode(p)
        case .many(let ps): try c.encode(ps)
        }
    }
}

func jsonLine(_ results: [String: ClassificationResult], multi: Set<String>) -> String {
    var obj: [String: Answer] = [:]
    for (name, r) in results {
        let picked = r.labels.map { l in
            let p = r.probabilities.first { $0.label == l }?.probability ?? .nan
            return Picked(label: l, confidence: (p * 10_000).rounded() / 10_000)
        }
        obj[name] = multi.contains(name) ? .many(picked) : .one(picked[0])
    }
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return String(data: try! enc.encode(obj), encoding: .utf8)!
}

// MARK: - oracle fixtures (conversion/gliner25_decide_oracle.py in the zoo)

/// One entry of a case's `tasks`: gliner2's classify_text input, `[labels]` or
/// `{"labels": [..] | {label: description}, "multi_label", "cls_threshold", "prompt"}`.
struct TaskSpec: Decodable {
    var labels: [String]?
    var descriptions: [String: String] = [:]
    var multiLabel: Bool?
    var threshold: Double?
    var prompt: String?

    enum K: String, CodingKey { case labels, multi_label, cls_threshold, prompt }

    init(from decoder: Decoder) throws {
        if let list = try? decoder.singleValueContainer().decode([String].self) {
            labels = list
            return
        }
        let c = try decoder.container(keyedBy: K.self)
        if let list = try? c.decode([String].self, forKey: .labels) {
            labels = list
        } else {
            descriptions = try c.decode([String: String].self, forKey: .labels)
        }
        multiLabel = try c.decodeIfPresent(Bool.self, forKey: .multi_label)
        threshold = try c.decodeIfPresent(Double.self, forKey: .cls_threshold)
        prompt = try c.decodeIfPresent(String.self, forKey: .prompt)
    }
}

struct TaskResult: Decodable {
    let task: String
    let labels: [String]
    let multiLabel: Bool
    let clsThreshold: Double
    let logits: [Double]
    /// The oracle's decision as a list (a single-label decision is one string in the JSON).
    let decision: [String]

    enum K: String, CodingKey { case task, labels, logits, decision, multi_label, cls_threshold }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: K.self)
        task = try c.decode(String.self, forKey: .task)
        labels = try c.decode([String].self, forKey: .labels)
        multiLabel = try c.decode(Bool.self, forKey: .multi_label)
        clsThreshold = try c.decode(Double.self, forKey: .cls_threshold)
        logits = try c.decode([Double].self, forKey: .logits)
        decision = multiLabel
            ? try c.decode([String].self, forKey: .decision) : [try c.decode(String.self, forKey: .decision)]
    }
}

/// A "Potential output" value of the model card: one label, or a list for a multi-label task.
struct CardValue: Decodable {
    let labels: [String]
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        labels = try (try? c.decode([String].self)) ?? [c.decode(String.self)]
    }
}

struct Case: Decodable {
    let id: String
    let text_raw: String
    let readme_output: [String: CardValue]?
    /// A case built with gliner2's max_len at the word count the kit's budget keeps (kit_r3 probes).
    let expect_truncated: Bool?
    let tasks: [String: TaskSpec]
    let seq_len: Int
    let n_words: Int
    let input_ids: [Int]
    let schema_tokens_list: [[String]]
    let subword_list: [String]
    let schema_special_indices: [[Int]]
    let task_results: [TaskResult]
}

struct Fixture: Decodable {
    struct Header: Decodable { let set: String; let model_rev: String }
    let header: Header
    let cases: [Case]
}

/// The case's tasks as `ClassificationTask`s. JSON objects lose their key order in Foundation, so the
/// order (tasks, and labels given as {label: description}) comes from `task_results`, which the oracle
/// wrote in gliner2's schema order; everything else comes from `tasks` and is cross-checked.
func classificationTasks(_ c: Case) throws -> [ClassificationTask] {
    guard Set(c.tasks.keys) == Set(c.task_results.map(\.task)), c.tasks.count == c.task_results.count else {
        throw NSError(domain: "fixture", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(c.id): tasks vs task_results"])
    }
    return try c.task_results.map { tr in
        let spec = c.tasks[tr.task]!
        let t = ClassificationTask(
            tr.task, labels: tr.labels, multiLabel: spec.multiLabel ?? false,
            threshold: Float(spec.threshold ?? 0.5), prompt: spec.prompt, descriptions: spec.descriptions)
        let labelsAgree = spec.labels.map { $0 == tr.labels } ?? (Set(spec.descriptions.keys) == Set(tr.labels))
        guard labelsAgree, t.multiLabel == tr.multiLabel, (spec.threshold ?? 0.5) == tr.clsThreshold else {
            throw NSError(domain: "fixture", code: 2, userInfo: [NSLocalizedDescriptionKey: "\(c.id) \(tr.task): spec vs task_results"])
        }
        return t
    }
}

func loadFixture(_ path: String) -> Fixture {
    do {
        let f = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        return f
    } catch { fail("fixture \(path): \(error)") }
}

/// Scalar for scalar: Swift's String == is canonical equivalence, Python's is code-point equality.
func same(_ a: String, _ b: String) -> Bool { a.unicodeScalars.elementsEqual(b.unicodeScalars) }
func same(_ a: [String], _ b: [String]) -> Bool { a.count == b.count && zip(a, b).allSatisfy { same($0, $1) } }

func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }

// MARK: - main

err("[textclassify] loading \(bundlePath) …")
let classifier: TextClassifier
do {
    classifier = try await TextClassifier(bundleAt: URL(fileURLWithPath: bundlePath))
} catch { fail("[textclassify] load failed: \(error)") }
err("[textclassify] ready: S \(classifier.sequenceLengths), MMAX \(classifier.maxLabels) (graphs load on first use)")

if !gateFiles.isEmpty {
    var cases: [Case] = []
    for f in gateFiles {
        let fx = loadFixture(f)
        err("[gate] \(fx.header.set): \(fx.cases.count) cases (\(f))")
        cases += fx.cases
    }
    // Python engine gates: S -> case id -> logits (the first n_labels slots of the row).
    struct PyGate: Decodable {
        struct Header: Decodable { let S: Int }
        struct Row: Decodable { let id: String; let logits: [Double] }
        struct Pass: Decodable { let cases: [Row] }
        struct Engine: Decodable { let clean: Pass }
        let header: Header
        let engine: Engine
    }
    var pygpu: [Int: [String: [Double]]] = [:]
    for f in pygpuFiles {
        do {
            let g = try JSONDecoder().decode(PyGate.self, from: Data(contentsOf: URL(fileURLWithPath: f)))
            pygpu[g.header.S, default: [:]].merge(g.engine.clean.cases.map { ($0.id, $0.logits) }) { a, _ in a }
            err("[gate] Python GPU gate S=\(g.header.S): \(g.engine.clean.cases.count) cases (\(f))")
        } catch { fail("pygpu \(f): \(error)") }
    }

    // 1. collate: the input the graph gets, against gliner2's
    var collated: [(Case, [ClassificationTask], TextClassifier.Collated)] = []
    var collateDiffs = 0
    for c in cases {
        let tasks: [ClassificationTask]
        let got: TextClassifier.Collated
        do {
            tasks = try classificationTasks(c)
            got = try classifier.collate(c.text_raw, tasks: tasks)
        } catch {
            err("COLLATE ERROR \(c.id): \(error)")
            collateDiffs += 1
            continue
        }
        var problems: [String] = []
        if !same(got.schemaTokens.flatMap { $0 }, c.schema_tokens_list.flatMap { $0 }) { problems.append("schema_tokens_list") }
        if got.inputIds.map(Int.init) != c.input_ids { problems.append("input_ids") }
        if !same(got.pieces, c.subword_list) { problems.append("subword_list") }
        let ssi = zip(got.promptPositions, got.labelPositions).map { [$0] + $1 }
        if ssi != c.schema_special_indices { problems.append("schema_special_indices") }
        if got.words.count != c.n_words { problems.append("n_words \(got.words.count) vs \(c.n_words)") }
        let wantS = classifier.sequenceLengths.first { c.seq_len <= $0 }
        if got.sequenceLength != wantS || got.truncated != (c.expect_truncated ?? false) {
            problems.append("S \(got.sequenceLength) truncated \(got.truncated)")
        }
        if problems.isEmpty {
            collated.append((c, tasks, got))
            continue
        }
        collateDiffs += 1
        err("COLLATE DIFF \(c.id): \(problems.joined(separator: ", "))")
        let n = max(got.pieces.count, c.subword_list.count)
        if let k = (0..<n).first(where: { k in
            k >= got.pieces.count || k >= c.subword_list.count || !same(got.pieces[k], c.subword_list[k])
                || Int(got.inputIds[k]) != c.input_ids[k]
        }) {
            let lo = max(0, k - 3)
            let mine = (lo..<min(k + 4, got.pieces.count)).map { "\(got.pieces[$0])/\(got.inputIds[$0])" }
            let theirs = (lo..<min(k + 4, c.subword_list.count)).map { "\(c.subword_list[$0])/\(c.input_ids[$0])" }
            err("   first difference at piece \(k): kit \(mine) | gliner2 \(theirs)")
            let mineScalars = k < got.pieces.count ? got.pieces[k].unicodeScalars.map { String($0.value, radix: 16) } : []
            let theirScalars = k < c.subword_list.count ? c.subword_list[k].unicodeScalars.map { String($0.value, radix: 16) } : []
            err("   scalars kit \(mineScalars) gliner2 \(theirScalars)")
        }
    }
    let tasksTotal = cases.reduce(0) { $0 + $1.task_results.count }
    err("[gate] collate: \(cases.count - collateDiffs)/\(cases.count) cases equal (input_ids, subword pieces, "
        + "schema_tokens_list, [P]/[L] positions, word count, S)")
    if collateOnly {
        print("collate \(cases.count - collateDiffs)/\(cases.count)")
        exit(collateDiffs == 0 ? 0 : 3)
    }

    // 2. graph + decisions, on the S the host picks
    var decisionsEqual = 0, tasksRun = 0
    var maxVsOracle = 0.0, maxVsOracleId = ""
    var pyCases = 0, pyBitEqual = 0, pyElements = 0, maxVsPy = 0.0, pyMissing = 0
    var perS: [Int: (cases: Int, tasks: Int, equal: Int, maxd: Double)] = [:]
    for (i, (c, tasks, got)) in collated.enumerated() {
        let rows: [[Float]]
        do { rows = try await classifier.logits(for: got) } catch { fail("[gate] \(c.id): \(error)") }
        var caseEqual = 0, caseMax = 0.0
        for (j, (tr, row)) in zip(c.task_results, rows).enumerated() {
            let r = TextClassifier.decide(row, task: tasks[j])
            tasksRun += 1
            if r.labels.count == tr.decision.count && zip(r.labels, tr.decision).allSatisfy({ same($0, $1) }) {
                caseEqual += 1
            } else {
                err("   DIFF \(c.id) task \(tr.task): kit \(r.labels) vs oracle \(tr.decision)")
            }
            for (a, b) in zip(row, tr.logits) { caseMax = max(caseMax, abs(Double(a) - b)) }
        }
        decisionsEqual += caseEqual
        if caseMax > maxVsOracle { maxVsOracle = caseMax; maxVsOracleId = c.id }
        var s = perS[got.sequenceLength] ?? (0, 0, 0, 0)
        s.cases += 1; s.tasks += rows.count; s.equal += caseEqual; s.maxd = max(s.maxd, caseMax)
        perS[got.sequenceLength] = s
        var pyNote = ""
        if !pygpu.isEmpty {
            if let py = pygpu[got.sequenceLength]?[c.id] {
                let flat = rows.flatMap { $0 }
                if py.count != flat.count { fail("[gate] \(c.id): Python row has \(py.count) logits, kit \(flat.count)") }
                pyCases += 1
                pyElements += flat.count
                var m = 0.0
                for (a, b) in zip(flat, py) {
                    if a.bitPattern == Float(b).bitPattern { pyBitEqual += 1 }
                    m = max(m, abs(Double(a) - b))
                }
                maxVsPy = max(maxVsPy, m)
                pyNote = String(format: " | vs Python GPU max|d| %.3g", m)
            } else {
                pyMissing += 1
            }
        }
        if caseEqual < rows.count || i < 3 || i % 50 == 0 {
            err(String(format: "[%3d] %@ %@ S=%d len=%3d max|dlogit| vs oracle %.4f", i, caseEqual == rows.count ? "OK  " : "DIFF",
                       pad(c.id, 34), got.sequenceLength, got.inputIds.count, caseMax) + pyNote)
        }
    }
    for S in perS.keys.sorted() {
        let s = perS[S]!
        err(String(format: "[gate] S=%d: %d cases, decisions %d/%d, max|dlogit| vs oracle %.5f", S, s.cases, s.equal, s.tasks, s.maxd))
    }
    var summary = "collate \(cases.count - collateDiffs)/\(cases.count) cases | decisions \(decisionsEqual)/\(tasksTotal) tasks"
        + String(format: " | max|dlogit| vs oracle %.5f (%@)", maxVsOracle, maxVsOracleId)
    if !pygpu.isEmpty {
        summary += String(format: " | vs Python GPU: %d cases, bit-equal %d/%d, max|d| %.3g", pyCases, pyBitEqual, pyElements, maxVsPy)
            + (pyMissing > 0 ? ", \(pyMissing) cases without a Python row" : "")
    }
    print(summary)
    let pass = collateDiffs == 0 && decisionsEqual == tasksTotal && tasksRun == tasksTotal
    err("\n=== KIT TEXTCLASSIFY GATE: \(pass ? "PASS" : "FAIL") ===")
    exit(pass ? 0 : 3)
}

if let readmePath = readme {
    // Each example through the public API; the gate is the gliner2 oracle's decisions. The card's
    // "Potential output" is printed beside it: gliner2 itself does not reproduce all of it.
    let fx = loadFixture(readmePath)
    var equal = 0, cardEqual = 0, withCard = 0
    for c in fx.cases {
        do {
            let tasks = try classificationTasks(c)
            let got = try await classifier.classify(c.text_raw, tasks: tasks)
            func decided(_ task: String, _ want: [String]) -> Bool {
                guard let r = got[task] else { return false }
                return r.labels.count == want.count && zip(r.labels, want).allSatisfy { same($0, $1) }
            }
            let ok = c.task_results.allSatisfy { decided($0.task, $0.decision) }
            if ok { equal += 1 }
            var cardNote = ""
            if let card = c.readme_output {
                withCard += 1
                // the card lists a multi-label answer as a set
                let cardOK = card.allSatisfy { task, v in
                    guard let r = got[task] else { return false }
                    return Set(r.labels.map { Array($0.unicodeScalars) }) == Set(v.labels.map { Array($0.unicodeScalars) })
                        && r.labels.count == v.labels.count
                }
                if cardOK { cardEqual += 1 } else { cardNote = "  (card: \(card.mapValues(\.labels)))" }
            }
            print(jsonLine(got, multi: Set(tasks.filter(\.multiLabel).map(\.name))))
            err("\(ok ? "PASS" : "FAIL") | \(c.id)" + (ok ? "" : "  oracle: \(c.task_results.map { "\($0.task)=\($0.decision)" })")
                + cardNote)
        } catch { fail("[readme] \(c.id): \(error)") }
    }
    err("\n=== README: \(equal)/\(fx.cases.count) cases decide as the gliner2 oracle"
        + (withCard > 0 ? "; \(cardEqual)/\(withCard) equal the card's Potential output ===" : " ==="))
    exit(equal == fx.cases.count ? 0 : 3)
}

guard let t = text, !tasks.isEmpty else { fail(usage) }
do {
    let got = try await classifier.classify(t, tasks: tasks)
    if got.values.contains(where: \.truncated) { err("[textclassify] note: the text was truncated to fit the largest bundle") }
    print(jsonLine(got, multi: Set(tasks.filter(\.multiLabel).map(\.name))))
} catch { fail("[textclassify] \(error.localizedDescription)") }
