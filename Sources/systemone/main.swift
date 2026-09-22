// systemone — System One typed decisions on this machine, as one signed binary.
//
//   brew install john-rocky/tap/systemone
//   systemone serve                         # http://127.0.0.1:8090/v1/systemone over MiniCPM5 2B
//   brew services start systemone           # the same, kept running by launchd
//   systemone ask --state "…" --noul "Does the customer want a refund?"
//   systemone models                        # what can decide, what is downloaded
//
// The server is `SystemOneServer` in the kit (Sources/CoreAIKit/Decide); this file is the
// argument shell. Progress and log lines go to stderr, answers to stdout.

import CoreAIOps
import Foundation

let usage = """
    usage: systemone serve  [--model <catalog-id>] [--host 127.0.0.1] [--port 8090]
                            (POST /v1/systemone in the hosted request and answer forms; GET /v1/models, GET /health)
           systemone ask    --state <text> | --state-file <path> | <text on stdin>
                            (--noul <q> | --choice "<q>|<opt>|<opt>…" | --score "<q>|<level>|<level>…")…
                            [--model <catalog-id>] [--json]
           systemone models (catalog models that can decide; * = default, cached = already downloaded)
           systemone --version

    The first run of a model downloads it (MiniCPM5 2B: 2.7 GB) into
    ~/Library/Application Support/CoreAIKit/Models; later runs load from there.
    """

func stderrPrint(_ message: String, terminator: String = "\n") {
    FileHandle.standardError.write(Data((message + terminator).utf8))
}

func fail(_ message: String, status: Int32 = 1) -> Never {
    stderrPrint(message)
    exit(status)
}

func fmt(_ value: Double, _ digits: Int = 3) -> String {
    String(format: "%.\(digits)f", value)
}

/// Download progress: a redrawn percentage on a terminal, one line per tenth in a log
/// (`brew services` keeps stderr in a file).
final class ProgressPrinter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastTenth = -1
    private let interactive = isatty(STDERR_FILENO) != 0

    func report(_ progress: DownloadProgress) {
        if interactive {
            stderrPrint(
                String(format: "\rdownloading  %3.0f%%", progress.fraction * 100),
                terminator: progress.fraction < 1 ? "" : "\n")
            return
        }
        let tenth = Int(progress.fraction * 10)
        lock.lock()
        defer { lock.unlock() }
        guard tenth > lastTenth else { return }
        lastTenth = tenth
        stderrPrint(String(
            format: "downloading %3d%%  %.1f of %.1f GB", tenth * 10,
            Double(progress.completedBytes) / 1e9, Double(progress.totalBytes) / 1e9))
    }
}

let printer = ProgressPrinter()
let progress: @Sendable (DownloadProgress) -> Void = { printer.report($0) }

let launched = SuspendingClock.now

func secondsSinceLaunch() -> Double {
    let elapsed = SuspendingClock.now - launched
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

var args = CommandLine.arguments.dropFirst()
guard let command = args.popFirst() else { fail(usage, status: 2) }

switch command {
case "--version", "version":
    print("systemone \(systemoneVersion)")
    exit(0)
case "--help", "-h", "help":
    print(usage)
    exit(0)
default:
    break
}

var modelID = CoreAI.defaultDecisionModel
var state: String?
var stateFile: String?
var questions: [(String, Decision.Question)] = []
var host = "127.0.0.1"
var port: UInt16 = 8090
var json = false

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
        guard let q = args.popFirst() else { fail(usage, status: 2) }
        questions.append(("q\(questions.count + 1)", .noul(q)))
    case "--choice":
        guard let spec = args.popFirst() else { fail(usage, status: 2) }
        let (q, options) = parts(spec)
        questions.append(("q\(questions.count + 1)", .choice(q, options)))
    case "--score":
        guard let spec = args.popFirst() else { fail(usage, status: 2) }
        let (q, levels) = parts(spec)
        questions.append(("q\(questions.count + 1)", .score(q, levels: levels)))
    case "--host": host = args.popFirst() ?? host
    case "--port":
        guard let p = UInt16(args.popFirst() ?? "") else { fail("--port takes a number 1–65535", status: 2) }
        port = p
    case "--json": json = true
    case "--help", "-h":
        print(usage)
        exit(0)
    default: fail("unknown argument \(arg)\n\n" + usage, status: 2)
    }
}

let id = modelID

// MARK: - models

@MainActor func runModels() async {
    let catalog = await ModelCatalog.load()
    let store = ModelStore.default
    print("id\tname\tkind\tdownload\tstatus")
    for entry in catalog.available() where TypedDecisions.supports(entry) {
        let size = entry.variant?.sizeMB.map { String(format: "%.1f GB", Double($0) / 1000) } ?? "-"
        let cached = entry.modelID.map(store.isCached) ?? false
        let marker = entry.id == CoreAI.defaultDecisionModel ? " *" : ""
        print("\(entry.id)\(marker)\t\(entry.name)\t\(entry.kind.rawValue)\t\(size)\t\(cached ? "cached" : "not downloaded")")
    }
}

// MARK: - ask

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

@MainActor func runAsk() async throws {
    var text = state
    if text == nil, let stateFile {
        text = try String(contentsOfFile: stateFile, encoding: .utf8)
    }
    if text == nil, isatty(STDIN_FILENO) == 0 {
        // `pbpaste | systemone ask --noul "…"`
        let piped = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
        text = piped.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    guard let text, !text.isEmpty, !questions.isEmpty else { fail(usage, status: 2) }
    let decider = try await TypedDecisions(catalog: id, downloadProgress: progress)
    let prefilled = try await decider.prefill(text)
    var answers: [(id: String, question: Decision.Question, answer: Decision.Answer)] = []
    for (key, question) in questions {
        answers.append((key, question, try await prefilled.decide(question)))
    }
    if json {
        print(SystemOne.response(model: id, answers: answers).dumps())
    } else {
        for (key, _, answer) in answers {
            print("\(key): \(describe(answer))")
        }
    }
}

// MARK: - serve

@MainActor func runServe() async throws {
    let decider = try await TypedDecisions(catalog: id, downloadProgress: progress)
    let entry = try await ModelCatalog.entry(forID: id)
    // GET /v1/models in the hosted list form: the TypeSafe SDK's models.list() reads this one.
    let models = SystemOne.modelsValue(
        id: id,
        description: "\(entry.name), CoreAIKit catalog kind \(entry.kind.rawValue), bundle \(await decider.modelName), on this machine",
        revision: entry.revision)
    let loaded = secondsSinceLaunch()
    stderrPrint("systemone \(systemoneVersion): loaded \(id) (\(await decider.modelName)) \(fmt(loaded, 1)) s after launch; one request at a time, questions share the state's prefill")
    let server = SystemOneServer(host: host, port: port, modelID: id, models: models, decider: decider) { line in
        if line.hasPrefix("listening") {
            stderrPrint("systemone serve: \(line); ready \(fmt(secondsSinceLaunch(), 1)) s after launch, Ctrl-C stops")
        } else {
            stderrPrint("systemone serve: \(line)")
        }
    }
    try await server.run()
}

do {
    switch command {
    case "serve": try await runServe()
    case "ask": try await runAsk()
    case "models": await runModels()
    default: fail("unknown command \(command)\n\n" + usage, status: 2)
    }
} catch {
    fail("error: \(error.localizedDescription)")
}
