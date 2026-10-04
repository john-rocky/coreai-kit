// KitKevDecider.swift — typed decisions with Kev-0.8B and Kev-4B (Jared Palmer, Apache-2.0: a rank-16 LoRA adapter and a
// pointer head on Qwen3.5-0.8B-Base / Qwen3.5-4B-Base): a state (text or JSON) and typed questions in, a probability
// for every option of every question out. Nothing is generated.
//
// ```swift
// let decider = try await KitKevDecider(catalog: "kev-0.8b")
// let response = try await decider.systemOne(try SystemOne.request(from: body))   // the /v1/systemone forms
// response["team"]?.choice
//
// let prepared = try await decider.prepare(state: ticket)                          // questions that arrive later
// let later = try await decider.decide(prepared: prepared, questionsJSON: questions)
// ```
//
// One Core AI graph driven on the low-level runtime by the files beside this one, ported from the model zoo's
// `apps/Kev` (9e06b5a) with every numeric path unchanged: one row per question in the author's form (the state rendered
// the author's way, then the question, each option and the decide token, every delimiter written by the host and user
// text unable to produce one), the decoder (fp16, no vocabulary head, a static 128 tokens per call from zeroed states)
// returning the hidden state at every position, and the author's pointer head on the host in float64 (the hidden state
// at the decide token against each option's closing token, softmaxed per question at the author's temperature, each p
// rounded to float32 once). With the shared prefix (on by default) the state's whole 128-token calls run once and every
// question continues from a copy of the four states; on this static graph its hidden rows equal the direct run's bit
// for bit.
//
// Not a `TypedDecisions`: no engine of the kit returns hidden states. Kev-0.8B ships for the Mac and the iPhone, Kev-4B
// for the Mac; the first load specializes the graph (3.3 s and 2.5 GB of the runtime's cache for 0.8B, 16.6 s and
// 15.55 GB for 4B in the zoo's runs on an M4 Max).

import CoreAI
import CryptoKit
import Foundation

@available(macOS 27, iOS 27, *)
public actor KitKevDecider: DecisionBackend {
    /// The catalog `format` of a Kev entry (`kind: rowDecision`).
    public static let format = "pointerHead"
    /// The widest choice the author's request accepts.
    public static let maxOptionsPerQuestion = 255

    /// A state run once and kept (`prepare(state:)`): its whole calls (the first ⌊Ls / 128⌋ · 128 of its tokens) and the
    /// four states after them, one copy (about 60 MB for Kev-0.8B, 160 MB for Kev-4B). The caller holds it; nothing is
    /// cached behind it.
    public final class PreparedState: @unchecked Sendable {
        /// The state as the request would carry it.
        public let state: JSONValue
        /// The state's tokens (`<state>` included), and how many of them the kept states hold.
        public let stateTokens: Int
        public let keptTokens: Int
        /// What preparing it took.
        public let seconds: Double
        /// The decider that made it: its states fit no other model.
        let owner: ObjectIdentifier
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let prepared: KevPipeline.Prepared

        init(state: JSONValue, prepared: KevPipeline.Prepared, owner: ObjectIdentifier) {
            self.state = state
            self.stateTokens = prepared.stateIDs.count
            self.keptTokens = prepared.plan.tokens
            self.seconds = prepared.seconds
            self.owner = owner
            self.prepared = prepared
        }
        #else
        init(state: JSONValue, owner: ObjectIdentifier) {
            self.state = state
            self.stateTokens = 0
            self.keptTokens = 0
            self.seconds = 0
            self.owner = owner
        }
        #endif
    }

    /// Everything one request read, for a gate: the rows, the hidden rows' digests, every option's logit and probability,
    /// and the response in the author's form.
    public struct Readout: Sendable {
        public struct Row: Sendable, Equatable {
            public let questionID: String
            public let type: String
            /// The author's option keys, in the head's order: choice names as given, noul `false` / `true`, score levels.
            public let keys: [String]
            /// The row: the state's ids, then the question's.
            public let ids: [Int]
            /// The index of the decide token and of each option's closing token in `ids`.
            public let decide: Int
            public let options: [Int]
            /// sha256 of the decoder's fp16 hidden rows [T, d] as stored.
            public let hiddenSHA256: String
            /// z per option before the temperature (float64), and p (float32), in `keys` order.
            public let logits: [Double]
            public let probabilities: [Float]
        }

        /// The request's tokens in the author's packed form (the state once, then every question's branch).
        public let packedIDs: [Int]
        public let stateLength: Int
        public let rows: [Row]
        /// Tokens of the state run once for every question; nil for a direct run.
        public let sharedPrefixTokens: Int?
        /// The answers as `json.dumps` writes them (the text `output_tokens` counts), and the author's usage.
        public let answersJSON: String
        public let inputTokens: Int
        public let outputTokens: Int
        /// The author's response body, compact (its `latency_ms` is this run's).
        public let response: String
        /// Every graph call's length, its pad included, in order.
        public let callLengths: [Int]
        /// Seconds per stage: "rows", "graph", "head", "answers", "latency", "wall".
        public let seconds: [String: Double]
    }

    /// The catalog id, or the bundle directory's name for local files.
    public nonisolated let id: String
    /// The bundle's name, from its metadata.
    public nonisolated let modelName: String
    public nonisolated var maxOptions: Int { Self.maxOptionsPerQuestion }
    /// Whether a request's state runs once for all its questions (on by default; the answers are the same either way).
    public nonisolated let sharePrefix: Bool
    /// What every response says beside its answers (a response adds the author's own token count and the shared
    /// prefix).
    public nonisolated var metadata: JSONValue { .object(metadataMembers) }

    private nonisolated let metadataMembers: [JSONValue.Member]
    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    private let pipeline: KevPipeline
    #endif
    /// Held for a whole call: a graph `await` lets another call into the actor, and two calls would share one set of
    /// decoder states.
    private let lock = AsyncMutex()

    /// Whether `entry` is a Kev model this type loads.
    public static func supports(_ entry: CatalogEntry) -> Bool {
        entry.kind == .rowDecision && entry.format == format && entry.modelID != nil
    }

    /// Loads Kev by its catalog id (`kind: rowDecision`, `format: pointerHead`), downloading the bundle on first use.
    public init(
        catalog id: String,
        sharePrefix: Bool = true,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .rowDecision)
        guard Self.supports(entry), let model = entry.modelID else {
            throw DecisionError.unsupportedModel(
                id: id, reason: "its catalog format is \(entry.format.map { "'\($0)'" } ?? "not given"), not \(Self.format)")
        }
        let url = try await store.download(model, progress: downloadProgress)
        try await self.init(bundleAt: url, id: id, sharePrefix: sharePrefix)
    }

    /// Loads a local bundle directory (metadata.json, the graph, head/, tokenizer/).
    public init(bundleAt url: URL, sharePrefix: Bool = true) async throws {
        try await self.init(bundleAt: url, id: url.lastPathComponent, sharePrefix: sharePrefix)
    }

    private init(bundleAt url: URL, id: String, sharePrefix: Bool) async throws {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let metadata = try KevPipeline.Metadata(bundle: url)
        // The graph that takes any call length up to its cap was measured and does not ship: a process that keeps
        // changing its call length keeps growing its memory until the AIModel is created again (zoo card, Kev-0.8B).
        guard !metadata.shape.dynamic else {
            throw DecisionError.unsupportedModel(
                id: id,
                reason: "its graph takes a dynamic query length (language.query_len_range), a form that does not ship: "
                    + "a process that keeps changing its call length grows its memory; load a static bundle "
                    + "(language.prefill_chunk)")
        }
        let stem = (metadata.asset as NSString).deletingPathExtension
        guard let graph = try GraphBundle.graph(named: stem, in: url) else {
            throw DecisionError.unsupportedModel(id: id, reason: "\(url.lastPathComponent) holds no \(metadata.asset)")
        }
        // An `.aimodel` is specialized here with the zoo's flags (the GPU, frequent reshapes); a compiled graph loads as
        // compiled.
        let pipeline = try await KevPipeline(bundle: url, asset: graph)
        self.pipeline = pipeline
        self.modelName = metadata.name
        self.metadataMembers = [
            .init("backend", .string("kev pointer head")),
            .init("model", .string(id)),
            .init("bundle", .string(metadata.name)),
            .init("temperature", .double(pipeline.head.temperature)),
            .init("calibration", .string("the author's, in head/kev_head.json")),
        ]
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "Kev runs on Apple silicon only")
        #endif
        self.id = id
        self.sharePrefix = sharePrefix
    }

    // MARK: - Decide

    /// One question on one state: a request of one question, read the same way.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        let response = try await systemOne(SystemOne.Request(state: state, questions: [(id: "q", question: question)]))
        return response.answers[0].answer
    }

    /// A whole request, one row per question, the answers in request order. A request parsed from the wire is read from
    /// its own values (numbers as written, structured values rendered the author's way); one built in Swift from its
    /// typed questions.
    public func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        if let refusal = imageRefusal(request) { throw refusal }
        try SystemOne.validateIDs(request.questions.map(\.id))
        for (_, question) in request.questions {
            try DecisionPrompt.validate(question, maxOptions: maxOptions)
        }
        let kev = try Self.kevRequest(Self.kevRoot(request, id: id))
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run { try await pipeline.trace(request: kev, shared: sharePrefix) }
        }
        return try response(request.questions, trace)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "Kev runs on Apple silicon only")
        #endif
    }

    /// Runs the state once and keeps it: the questions of `decide(prepared:questionsJSON:)` then skip the state's calls.
    /// A string is the plain state; an object or an array a structured one.
    public func prepare(state: JSONValue) async throws -> PreparedState {
        guard state != .null else { throw SystemOne.WireError("'state' must be a string, an object or an array") }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let prepared = try await lock.withLock {
            try await run { try await pipeline.prepare(state: Self.kevJSON(state)) }
        }
        return PreparedState(state: state, prepared: prepared, owner: ObjectIdentifier(self))
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "Kev runs on Apple silicon only")
        #endif
    }

    /// The same for a plain-text state.
    public func prepare(state: String) async throws -> PreparedState {
        try await prepare(state: .string(state))
    }

    /// Answers questions on a prepared state: `questionsJSON` is a request's `questions` object (question id → question,
    /// in the `/v1/systemone` forms). The hidden rows and probabilities equal a shared request's bit for bit.
    public func decide(prepared: PreparedState, questionsJSON: Data) async throws -> SystemOne.Response {
        guard prepared.owner == ObjectIdentifier(self) else {
            throw SystemOne.WireError("this prepared state was made by another decider")
        }
        let questions: JSONValue
        do {
            questions = try JSONValue.parse(questionsJSON)
        } catch {
            throw SystemOne.WireError("invalid JSON: \(error.localizedDescription)")
        }
        let request = try SystemOne.request(
            from: .object([.init("state", prepared.state), .init("questions", questions)]), maxOptions: maxOptions)
        for (_, question) in request.questions {
            try DecisionPrompt.validate(question, maxOptions: maxOptions)
        }
        let root = Self.kevRoot(request, id: id)
        _ = try Self.kevRequest(root)
        guard let questionsValue = root["questions"] else { throw SystemOne.WireError("'questions' is required") }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run { try await pipeline.trace(prepared: prepared.prepared, questions: questionsValue) }
        }
        return try response(request.questions, trace)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "Kev runs on Apple silicon only")
        #endif
    }

    /// Every call length of the graph once, so the first request does not pay the process's first call: on a static
    /// graph one length, and optional. Each length's seconds.
    public func warmUp() async throws -> [(length: Int, seconds: Double)] {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        return try await lock.withLock { try await pipeline.warmUp() }
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "Kev runs on Apple silicon only")
        #endif
    }

    /// The whole call in the author's terms, for a gate: `requestJSON` is a request in the author's form (`model`,
    /// `state`, `questions`) read byte for byte; `shared` runs the state's whole calls once.
    public func readout(requestJSON: Data, shared: Bool) async throws -> Readout {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run { try await pipeline.trace(request: try KevRequest(data: requestJSON), shared: shared) }
        }
        return Readout(
            packedIDs: trace.rows.packed, stateLength: trace.rows.stateLength,
            rows: trace.rows.rows.indices.map { k in
                let row = trace.rows.rows[k]
                return Readout.Row(
                    questionID: row.qid, type: row.type, keys: row.keys, ids: row.ids, decide: row.decide,
                    options: row.opts, hiddenSHA256: Self.sha256(of: trace.hidden[k]), logits: trace.logits[k],
                    probabilities: trace.probabilities[k])
            },
            sharedPrefixTokens: trace.shared.map(\.tokens), answersJSON: trace.answersJSON,
            inputTokens: trace.rows.inputTokens, outputTokens: trace.response["usage"]?["output_tokens"]?.intValue ?? -1,
            response: KevPythonFormat.dumps(trace.response, asciiOnly: false), callLengths: trace.callLengths,
            seconds: trace.seconds)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "Kev runs on Apple silicon only")
        #endif
    }

    // MARK: - Internals

    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    /// The kit's response from a trace: each question's probabilities in the kit's option order (choice as asked, score
    /// by level, noul no / yes — the author's key order), the state's shared prefix as the prefill, and each question's
    /// own calls and its share of the head as its time.
    private func response(
        _ questions: [(id: String, question: Decision.Question)], _ trace: KevPipeline.Trace
    ) throws -> SystemOne.Response {
        let rows = trace.rows.rows
        guard rows.count == questions.count, trace.probabilities.count == rows.count else {
            throw KevError.contract("\(rows.count) rows for \(questions.count) questions")
        }
        let shape = pipeline.metadata.shape
        let prefixTokens = trace.shared?.tokens ?? 0
        // A prepared state ran its prefix before this call.
        let prepared = trace.mode == "prepared"
        // The calls in the order they ran: the prefix's (when shared), then each row's (its part after the prefix).
        var rowCalls: [Int] = []
        for row in rows { rowCalls.append(try shape.plan(row.ids.count - prefixTokens).count) }
        let prefixCalls = prefixTokens > 0 && !prepared ? try shape.plan(prefixTokens).count : 0
        guard prefixCalls + rowCalls.reduce(0, +) == trace.callSeconds.count else {
            throw KevError.contract("\(trace.callSeconds.count) calls for a plan of \(prefixCalls + rowCalls.reduce(0, +))")
        }
        let prefixSeconds = trace.callSeconds.prefix(prefixCalls).reduce(0, +)
        var rowSeconds: [Double] = []
        var at = prefixCalls
        for n in rowCalls {
            rowSeconds.append(trace.callSeconds[at..<(at + n)].reduce(0, +))
            at += n
        }
        // The rest of the latency (state resets and copies, the head) goes to the questions in proportion to their calls.
        let latency = trace.seconds["latency"] ?? 0
        let rest = max(0, latency - prefixSeconds - rowSeconds.reduce(0, +))
        let callTotal = rowSeconds.reduce(0, +)
        var answers: [SystemOne.Answer] = []
        for (k, (key, question)) in questions.enumerated() {
            let row = rows[k]
            let share = callTotal > 0 ? rest * rowSeconds[k] / callTotal : rest / Double(rows.count)
            let timing = Decision.Timing(
                promptTokens: row.ids.count, reusedTokens: prefixTokens, seconds: rowSeconds[k] + share)
            let p = try Self.kitOrder(question, keys: row.keys, trace.probabilities[k], id: key)
            answers.append(SystemOne.Answer(
                id: key, question: question, answer: DecisionPrompt.answer(for: question, probabilities: p, timing: timing)))
        }
        return SystemOne.Response(
            model: id, answers: answers, stateTokens: prefixTokens,
            prefill: Decision.Timing(
                promptTokens: prefixTokens, reusedTokens: prepared ? prefixTokens : 0, seconds: prefixSeconds),
            metadata: .object(metadataMembers + [
                .init("packed_tokens", .int(trace.rows.inputTokens)), .init("shared_prefix_tokens", .int(prefixTokens)),
            ]))
    }
    #endif

    /// The pipeline's request and length failures as the 422s they are; a contract or bundle failure stays itself.
    private func run<T>(_ body: () async throws -> T) async throws -> T {
        do {
            return try await body()
        } catch let error as KevError {
            switch error {
            case .request(let message), .json(let message):
                throw SystemOne.WireError(message)
            case .contextOverflow, .graphLimit:
                throw SystemOne.WireError(error.description)
            case .contract, .bundle:
                throw error
            }
        }
    }

    static func sha256<T>(of values: [T]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }

    /// The request in the author's form: the wire's own values when it came over the wire, else built from the typed
    /// questions. A choice whose options came as a list reads them as names without descriptions, each the option id the
    /// kit's wire gave it; `model` is `id` unless the request names one.
    static func kevRoot(_ request: SystemOne.Request, id: String) -> KevJSON {
        var members: [KevJSONMember]
        if var json = request.json?.members {
            if let at = json.firstIndex(where: { $0.key == "questions" }), let questions = json[at].value.members {
                json[at] = .init("questions", .object(questions.map { .init($0.key, listedChoiceAsNames($0.value)) }))
            }
            members = kevJSON(.object(json)).members ?? []
        } else {
            var state = KevJSON.string(request.state)
            if request.structuredState, let parsed = try? KevJSONParser.parse(request.state) { state = parsed }
            members = [
                KevJSONMember("state", state),
                KevJSONMember(
                    "questions", .object(request.questions.map { KevJSONMember($0.id, kevQuestion($0.question)) })),
            ]
        }
        if let at = members.firstIndex(where: { $0.key == "model" }) {
            if members[at].value.string == nil { members[at].value = .string(id) }
        } else {
            members.insert(KevJSONMember("model", .string(id)), at: 0)
        }
        return .object(members)
    }

    /// The author's checks on the request (`validate_request`), their refusals as 422s.
    static func kevRequest(_ root: KevJSON) throws -> KevRequest {
        do {
            return try KevRequest(json: root)
        } catch let error as KevError {
            throw SystemOne.WireError(error.description)
        }
    }

    /// A choice question whose criteria came as a list, its options as names without descriptions (`{name: null}`), each
    /// name the id the kit's wire parser gave the option; any other question as it came.
    static func listedChoiceAsNames(_ question: JSONValue) -> JSONValue {
        guard let fields = question.members, question["type"] == .string("choice"),
            let elements = question["criteria"]?.elements
        else { return question }
        return .object(fields.map { field in
            guard field.key == "criteria" else { return field }
            return .init("criteria", .object(elements.map { .init($0.stringValue ?? $0.dumps(), .null) }))
        })
    }

    /// The kit's parsed JSON as the zoo host's: members in order, a repeated key in its first place with its last value
    /// (`json.loads`), numbers as written.
    static func kevJSON(_ value: JSONValue) -> KevJSON {
        switch value {
        case .null: return .null
        case .bool(let b): return .bool(b)
        case .number(let text): return .number(text)
        case .string(let s): return .string(s)
        case .array(let elements): return .array(elements.map(kevJSON))
        case .object(let members):
            var out: [KevJSONMember] = []
            var index: [String: Int] = [:]
            for member in members {
                // Python compares keys by code point; Swift's String == by canonical equivalence.
                let key = member.key.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: ",")
                if let at = index[key] {
                    out[at].value = kevJSON(member.value)
                } else {
                    index[key] = out.count
                    out.append(KevJSONMember(member.key, kevJSON(member.value)))
                }
            }
            return .object(out)
        }
    }

    /// A typed question in the author's form. An option whose description is its id reads as a name alone.
    static func kevQuestion(_ question: Decision.Question) -> KevJSON {
        var fields = [KevJSONMember("instructions", .string(question.instructions))]
        switch question.kind {
        case .choice(let options):
            fields.insert(KevJSONMember("type", .string("choice")), at: 0)
            fields.append(KevJSONMember("criteria", .object(options.map {
                KevJSONMember($0.id, $0.description == $0.id ? .null : .string($0.description))
            })))
        case .score(let levels):
            fields.insert(KevJSONMember("type", .string("score")), at: 0)
            fields.append(KevJSONMember("criteria", .array(levels.map { .string($0) })))
        case .noul(let yes, let no):
            fields.insert(KevJSONMember("type", .string("noul")), at: 0)
            var criteria: [KevJSONMember] = []
            if let yes { criteria.append(KevJSONMember("true", .string(yes))) }
            if let no { criteria.append(KevJSONMember("false", .string(no))) }
            if !criteria.isEmpty { fields.append(KevJSONMember("criteria", .object(criteria))) }
        }
        return .object(fields)
    }

    /// A question's probabilities in the kit's option order (choice: as asked; score: by level; noul: no, yes) from the
    /// head's, whose order is the author's keys.
    static func kitOrder(_ question: Decision.Question, keys: [String], _ p: [Float], id: String) throws -> [Double] {
        let want: [String]
        switch question.kind {
        case .noul: want = ["false", "true"]
        default: want = question.optionIDs
        }
        guard keys == want, p.count == keys.count else {
            throw SystemOne.WireError("question '\(id)': the model's options \(keys) are not the question's \(want)")
        }
        return p.map(Double.init)
    }
}
