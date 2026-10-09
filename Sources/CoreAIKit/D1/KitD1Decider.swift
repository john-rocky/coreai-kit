// KitD1Decider.swift — typed decisions with d1-3B (Liquid AI, LFM Open License v1.0: LFM2.5-VL-3B post-trained as a
// decision model): a state (text or JSON), pictures, and typed questions in, a probability for every option of every
// question out. Nothing is generated.
//
// ```swift
// let decider = try await KitD1Decider(catalog: "d1-3b")
// let response = try await decider.systemOne(try SystemOne.request(from: body))     // the /v1/systemone forms
// response["team"]?.choice
//
// let seen = try await decider.systemOne(
//     state: nil, questions: [(id: "circle", question: .noul("Is there a circle in the picture?"))],
//     images: [.file(picture)])
// let prepared = try await decider.prepare(state: ticket)                         // questions that arrive later
// let later = try await decider.decide(prepared: prepared, questionsJSON: questions)
// ```
//
// Two Core AI graphs on the low-level runtime, driven by the files beside this one: the model zoo's `apps/D1` (e36ad15)
// ported with every numeric path unchanged. One row per question in the provider's form (prompt.py / runner.py: the
// state block, the question block, `<|im_end|>`, the assistant turn), its ids from the checkpoint's tokenizer through
// swift-transformers and the three steps it does differently (the added tokens cut out before the regex, the Split regex on code points,
// `ignore_merges`). The decoder (no vocabulary head, a static 64 tokens a call from zeroed states) returns the
// final-norm hidden state at every position, and the provider's readout runs on the host: the last position's hidden
// row against the tied embedding rows of the question's option tokens (`head/option_rows`) in float64, through the BLAS
// calls NumPy makes (`CoreAIKitD1BLAS`), each option's highest logit, a softmax over the options. A picture is decoded,
// capped at one megapixel, cut into the processor's crops (torch's uint8 bicubic resize) and read by the vision tower
// once per crop; the crops' rows fill the decoder's image buffer, which the prompt's `<image>` runs read. With the
// shared prefix (on by default) the state's whole 64-token calls run once and every question continues from a copy of
// the three states; on this static graph its hidden rows equal the direct run's bit for bit.
//
// Not a `TypedDecisions`: no engine of the kit returns hidden states. The catalog entry gives the Mac the fp16 decoder
// and the iPhone the one whose MLP linears are int8; the tower downloads when a request carries a picture.
// An iPhone app needs `com.apple.developer.kernel.increased-memory-limit` in its entitlements: without it the decoder
// does not load (the zoo's iPhone 18 Pro runs).

import CoreAI
import CryptoKit
import Foundation

@available(macOS 27, iOS 27, *)
public actor KitD1Decider: DecisionBackend {
    /// The catalog `format` of a d1-3B entry (`kind: tokenDecision`).
    public static let format = "optionRows"
    /// The widest choice the hosted API accepts. The provider's own limit is its alias pool, 1,639 options, reached
    /// through `readout(requestJSON:)`.
    public static let maxOptionsPerQuestion = 255

    /// A state run once and kept (`prepare(state:)`): its whole 64-token calls (its leading ⌊Ls / 64⌋ · 64 stable
    /// tokens) and the three states after them, one copy. The caller holds it; nothing is cached behind it.
    public final class PreparedState: @unchecked Sendable {
        /// The state as the request would carry it (nil: the provider's None, no state block).
        public let state: JSONValue?
        /// The state's stable tokens (the prefix every row starts with), and how many of them the kept states hold.
        public let stateTokens: Int
        public let keptTokens: Int
        /// What preparing it took.
        public let seconds: Double
        /// The decider that made it: its states fit no other model.
        let owner: ObjectIdentifier
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let prepared: D13BPrepared

        init(state: JSONValue?, prepared: D13BPrepared, owner: ObjectIdentifier) {
            self.state = state
            self.stateTokens = prepared.stateTokens
            self.keptTokens = prepared.k
            self.seconds = prepared.seconds
            self.owner = owner
            self.prepared = prepared
        }
        #else
        init(state: JSONValue?, owner: ObjectIdentifier) {
            self.state = state
            self.stateTokens = 0
            self.keptTokens = 0
            self.seconds = 0
            self.owner = owner
        }
        #endif
    }

    /// Everything one request read, for a gate: each question's row, the hidden rows' digest, every read id's logit,
    /// the probabilities, and the response in the provider's form.
    public struct Readout: Sendable {
        public struct Row: Sendable, Equatable {
            public let questionID: String
            public let type: String
            /// The provider's option keys, in its order: noul `yes`, `no`; choice the labels; score `0` … `K-1`.
            public let keys: [String]
            /// The row's ids (with pictures: the k-th `<image>` as 128,000 + k) and the answer slot, its last position.
            public let ids: [Int]
            public let slot: Int
            /// The ids the readout reads, one group per key.
            public let groups: [[Int]]
            /// sha256 of the decoder's fp16 hidden rows [T, 2048] as stored (NumPy's `tobytes()`).
            public let hiddenSHA256: String
            /// z = h · E[id] in float64 for every id of `groups`, each once in the groups' order (`logitIDs`).
            public let logitIDs: [Int]
            public let logits: [Double]
            /// p per key (float64), the provider's order.
            public let probabilities: [Double]
        }

        /// "direct", "shared" or "prepared".
        public let mode: String
        /// Tokens run once for every question (0: each row ran whole), and the state's stable tokens.
        public let sharedTokens: Int
        public let stateTokens: Int
        /// The provider's `usage.input_tokens` (several questions: the prefix once, then each question's suffix).
        public let inputTokens: Int
        public let imageTokens: Int
        public let rows: [Row]
        /// sha256 of each crop's whole tower output (float32 [256, 2048]), crops in order.
        public let towerOutputSHA256: [String]
        /// The provider's response, `json.dumps(response, indent=2, ensure_ascii=False)`.
        public let response: String
        /// The decoder's calls this request made.
        public let calls: Int
        /// Seconds per stage: "plan", "images", "graph", "readout", "wall".
        public let seconds: [String: Double]
    }

    /// The catalog id, or the bundle directory's name for local files.
    public nonisolated let id: String
    /// The decoder bundle's name, from its metadata.
    public nonisolated let modelName: String
    public nonisolated var maxOptions: Int { Self.maxOptionsPerQuestion }
    public nonisolated var readsImages: Bool { true }
    /// Whether a request's state runs once for all its questions (on by default; the answers are the same either way).
    public nonisolated let sharePrefix: Bool

    /// Where the vision tower comes from: a local bundle directory, or the catalog's `assets.tower` at the pin.
    private enum TowerSource: Sendable {
        case local(URL)
        case catalog(ModelID, ModelStore, (@Sendable (DownloadProgress) -> Void)?)
    }

    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    private let pipeline: D13BPipeline
    #endif
    private let towerSource: TowerSource?
    /// Held for a whole call: a graph `await` lets another call into the actor, and two calls would share one set of
    /// decoder states and one image buffer.
    private let lock = AsyncMutex()

    /// Whether `entry` is a d1-3B model this type loads.
    public static func supports(_ entry: CatalogEntry) -> Bool {
        entry.kind == .tokenDecision && entry.format == format && entry.modelID != nil
    }

    /// Loads d1-3B by its catalog id (`kind: tokenDecision`, `format: optionRows`), downloading this platform's decoder
    /// bundle when the store does not hold it (the Mac's fp16, the iPhone's int8mlp). The tower (`assets.tower`) downloads when a request
    /// carries a picture, or at `loadTower()`.
    public init(
        catalog id: String,
        sharePrefix: Bool = true,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .tokenDecision)
        guard Self.supports(entry), let model = entry.modelID else {
            throw DecisionError.unsupportedModel(
                id: id, reason: "its catalog format is \(entry.format.map { "'\($0)'" } ?? "not given"), not \(Self.format)")
        }
        let url = try await store.download(model, progress: downloadProgress)
        let tower = entry.assets?.tower.map { TowerSource.catalog(entry.modelID(path: $0), store, downloadProgress) }
        try await self.init(bundleAt: url, tower: tower, id: id, sharePrefix: sharePrefix)
    }

    /// Loads a local decoder bundle directory (metadata.json, the graph, tokenizer/, head/) and, for pictures, a tower
    /// bundle directory (metadata.json, the graph, host/), loaded when a request carries a picture.
    public init(bundleAt url: URL, towerAt tower: URL? = nil, sharePrefix: Bool = true) async throws {
        try await self.init(bundleAt: url, tower: tower.map { .local($0) }, id: url.lastPathComponent, sharePrefix: sharePrefix)
    }

    private init(bundleAt url: URL, tower: TowerSource?, id: String, sharePrefix: Bool) async throws {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let pipeline = try await D13BPipeline(bundle: url)
        let stem = (pipeline.metadata.asset as NSString).deletingPathExtension
        guard let graph = try GraphBundle.graph(named: stem, in: url) else {
            throw DecisionError.unsupportedModel(id: id, reason: "\(url.lastPathComponent) holds no \(pipeline.metadata.asset)")
        }
        // An `.aimodel` is specialized here with the zoo's flags (the GPU, frequent reshapes); a compiled graph loads as
        // compiled.
        try await pipeline.loadGraph(assetURL: graph)
        self.pipeline = pipeline
        self.modelName = pipeline.metadata.name
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
        self.id = id
        self.sharePrefix = sharePrefix
        self.towerSource = tower
    }

    // MARK: - Decide

    /// One question on one state: a request of one question, read the same way.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        let response = try await systemOne(SystemOne.Request(state: state, questions: [(id: "q", question: question)]))
        return response.answers[0].answer
    }

    /// A whole request, one row per question, the answers in request order. A request parsed from the wire is read from
    /// its own values (numbers as written, structured values as the provider's `json.dumps` writes them); one built in
    /// Swift from its typed questions. `images` are the request's pictures (the wire carries one), read at the
    /// provider's own crops: `grid` is refused.
    public func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        if let refusal = audioRefusal(request) { throw refusal }
        if request.grid != nil {
            throw SystemOne.WireError("'\(id)' reads a picture at the provider's own crops; 'grid' is clef-flash's tile side, leave it out")
        }
        try Self.validate(request.questions, id: id)
        return try await answer(request.questions, root: Self.providerRequest(request), images: request.images)
    }

    /// The same request in Swift: `state` nil is the provider's None (no state block), a string the plain state, an
    /// object or an array a structured one; `images` are read in order, the k-th for the k-th picture.
    public func systemOne(
        state: JSONValue?, questions: [(id: String, question: Decision.Question)], images: [SystemOne.Image] = []
    ) async throws -> SystemOne.Response {
        try Self.validate(questions, id: id)
        let root = D13BJSONValue.object([
            D13BJSONMember("state", state.map(Self.d13bJSON) ?? .null),
            D13BJSONMember("questions", .object(questions.map { D13BJSONMember($0.id, Self.d13bQuestion($0.question)) })),
        ])
        return try await answer(questions, root: root, images: images)
    }

    /// Runs the state once and keeps it: the questions of `decide(prepared:questionsJSON:)` then skip the state's calls.
    /// Text states only (nil is the provider's None).
    public func prepare(state: JSONValue?) async throws -> PreparedState {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let prepared = try await lock.withLock {
            try await run { try await pipeline.prepare(state: state.map(Self.d13bJSON) ?? .null) }
        }
        return PreparedState(state: state, prepared: prepared, owner: ObjectIdentifier(self))
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
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
        let (questions, asked) = try Self.preparedQuestions(questionsJSON)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run { try await pipeline.trace(prepared: prepared.prepared, questions: questions) }
        }
        return try response(asked, trace, keptIDs: prepared.prepared.ids)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
    }

    /// Loads the vision tower now (downloading it if needed), so no request with a picture waits for it.
    public func loadTower() async throws {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        _ = try await lock.withLock { try await run { try await towerLoaded() } }
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
    }

    /// One call of pad ids, so no request pays the process's opening call. Seconds.
    public func warmUp() async throws -> Double {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        return try await lock.withLock { try await pipeline.requireGraph().decoder.warmUp() }
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
    }

    /// The decoder's and the tower's calls since the load, for a gate around a refused request (no call).
    func graphCalls() throws -> (decoder: Int, tower: Int?) {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let g = try pipeline.requireGraph()
        return (g.decoder.callCount, g.tower?.callCount)
        #else
        return (0, nil)
        #endif
    }

    /// The whole call in the provider's terms, for a gate: `requestJSON` is `{"state": …, "questions": {…}}` as the
    /// provider's `system_one` takes it (null for no state), read byte for byte as `json.loads` reads it, with its
    /// refusals in the provider's words; `images` are its picture files in order; `shared` runs the state's whole calls
    /// once.
    public func readout(requestJSON: Data, images: [URL] = [], shared: Bool = false) async throws -> Readout {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run {
                let request = try D13BRequest(data: requestJSON)
                let pictures = images.isEmpty ? nil : try await self.pictures(images.map { .file($0) })
                return try await pipeline.trace(request: request, mode: shared ? .shared : .direct, pictures: pictures)
            }
        }
        return Self.readout(trace)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
    }

    /// The same on a prepared state: `questionsJSON` in the provider's form.
    public func readout(prepared: PreparedState, questionsJSON: Data) async throws -> Readout {
        guard prepared.owner == ObjectIdentifier(self) else {
            throw SystemOne.WireError("this prepared state was made by another decider")
        }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run {
                try await pipeline.trace(prepared: prepared.prepared, questions: try D13BJSONParser.parse(questionsJSON))
            }
        }
        return Self.readout(trace)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
    }

    // MARK: - Internals

    /// One request through the pipeline, its refusals as 422s, and the kit's response.
    private func answer(
        _ asked: [(id: String, question: Decision.Question)], root: D13BJSONValue, images: [SystemOne.Image]
    ) async throws -> SystemOne.Response {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            try await run {
                let request = try D13BRequest(json: root)
                let pictures = images.isEmpty ? nil : try await self.pictures(images)
                return try await pipeline.trace(request: request, mode: sharePrefix ? .shared : .direct, pictures: pictures)
            }
        }
        return try response(asked, trace)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "d1-3B runs on Apple silicon only")
        #endif
    }

    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

    /// The loaded tower, downloading and loading it when it is not: the decoder stays as loaded, the graph gains
    /// the tower (the zoo's `loadGraph(tower:)` check: its width is the decoder's hidden size).
    @discardableResult
    private func towerLoaded() async throws -> D13BTower {
        let g = try pipeline.requireGraph()
        if let tower = g.tower { return tower }
        let url: URL
        switch towerSource {
        case .local(let local)?:
            url = local
        case .catalog(let model, let store, let progress)?:
            url = try await store.download(model, progress: progress)
        case nil:
            throw SystemOne.WireError("'\(id)' was loaded without its vision tower and cannot read a picture")
        }
        let stem = (try D13BTower.modelAsset(bundle: url).lastPathComponent as NSString).deletingPathExtension
        guard let graph = try GraphBundle.graph(named: stem, in: url) else {
            throw DecisionError.unsupportedModel(id: id, reason: "\(url.lastPathComponent) holds no \(stem) graph")
        }
        let tower = try await D13BTower(bundle: url, asset: graph)
        guard tower.width == g.metadata.hidden else {
            throw D13BError.contract("the tower's width \(tower.width) != the decoder's \(g.metadata.hidden)")
        }
        pipeline.graph = D13BGraph(
            metadata: g.metadata, decoder: g.decoder, tower: tower, asset: g.asset, assetURL: g.assetURL,
            loadSeconds: g.loadSeconds, warmUpSeconds: g.warmUpSeconds)
        return tower
    }

    /// The request's pictures, in order, cut into every crop's four tower inputs: a file through the zoo's
    /// `D13BTowerInputs.pictures(files:table:)`, bytes through the same steps (`D13BPixels.decode(data:)`).
    private func pictures(_ images: [SystemOne.Image]) async throws -> D13BTowerInputs {
        let table = try await towerLoaded().positionTable
        var out: [D13BTowerInputs.Picture] = []
        for (k, image) in images.enumerated() {
            switch image {
            case .file(let url):
                out += try D13BTowerInputs.pictures(files: [url], table: table).pictures
            case .data(let data):
                let pic = D13BPixels.picture(try D13BPixels.decode(data: data))
                let w = pic.decoded.rgb.width, h = pic.decoded.rgb.height
                guard D13BVision.plan(pictureWidth: w, pictureHeight: h).crops == pic.plan.crops else {
                    throw D13BError.contract("image \(k): the plan of the capped pixels differs from the plan of \(w) x \(h)")
                }
                let inputs = try D13BPixels.towerInputs(pic, table: table)
                let crops = zip(pic.plan.crops, inputs).map { c, t in
                    D13BTower.CropInputs(
                        patches: t.patches, posTable: t.posTable, keyBias: t.keyBias, unshuffle: t.unshuffleIndex,
                        grid: (t.gridHeight, t.gridWidth), tokens: c.tokens)
                }
                out.append(D13BTowerInputs.Picture(id: "image \(k)", width: w, height: h, plan: pic.plan, crops: crops))
            }
        }
        return D13BTowerInputs(pictures: out)
    }

    /// The kit's response from a trace: each question's probabilities in the kit's option order (choice as asked, score
    /// by level, noul no / yes), the shared prefix (and the pictures) as the prefill, each question's own calls as its
    /// time. `metadata.input_tokens` is the provider's count; the wire's `usage` counts each question's row.
    private func response(
        _ asked: [(id: String, question: Decision.Question)], _ trace: D13BTrace, keptIDs: [Int]? = nil
    ) throws -> SystemOne.Response {
        let rows = trace.plan.rows
        guard rows.count == asked.count, trace.probabilities.count == rows.count else {
            throw D13BError.contract("\(rows.count) rows for \(asked.count) questions")
        }
        let S = try pipeline.requireGraph().metadata.chunk
        let k = trace.sharedK
        let prepared = trace.mode == "prepared"
        // The calls in the order they ran: the prefix's (shared), then each row's part after the prefix (a prepared
        // state's row that does not start with the kept ids runs whole).
        let prefixCalls = k > 0 && !prepared ? k / S : 0
        // Whether a row continued from the k shared or kept ids, or ran whole.
        let continued = rows.map { row in k > 0 && (keptIDs.map { Array(row.ids.prefix(k)) == $0 } ?? true) }
        var rowCalls: [Int] = []
        for (i, row) in rows.enumerated() {
            rowCalls.append(((continued[i] ? row.ids.count - k : row.ids.count) + S - 1) / S)
        }
        guard prefixCalls + rowCalls.reduce(0, +) == trace.callSeconds.count else {
            throw D13BError.contract("\(trace.callSeconds.count) calls for a plan of \(prefixCalls + rowCalls.reduce(0, +))")
        }
        let prefixSeconds = trace.callSeconds.prefix(prefixCalls).reduce(0, +)
        var at = prefixCalls
        let readoutShare = (trace.seconds["readout"] ?? 0) / Double(rows.count)
        var answers: [SystemOne.Answer] = []
        for (i, (key, question)) in asked.enumerated() {
            let seconds = trace.callSeconds[at..<(at + rowCalls[i])].reduce(0, +) + readoutShare
            at += rowCalls[i]
            let timing = Decision.Timing(
                promptTokens: rows[i].ids.count, reusedTokens: continued[i] ? k : 0, seconds: seconds)
            let p = try Self.kitOrder(question, keys: rows[i].row.keys, trace.probabilities[i], id: key)
            answers.append(SystemOne.Answer(
                id: key, question: question, answer: DecisionPrompt.answer(for: question, probabilities: p, timing: timing)))
        }
        return SystemOne.Response(
            model: id, answers: answers, stateTokens: k,
            prefill: Decision.Timing(
                promptTokens: k, reusedTokens: prepared ? k : 0, seconds: prefixSeconds + (trace.seconds["images"] ?? 0)),
            metadata: .object([
                .init("backend", .string("d1-3b option rows")), .init("model", .string(id)),
                .init("bundle", .string(modelName)),
                .init("calibration", .string(
                    "the provider's readout: each option's highest token logit at the last position, a softmax over the options")),
                .init("input_tokens", .int(trace.plan.inputTokens)), .init("image_tokens", .int(trace.plan.imageTokens)),
                .init("shared_prefix_tokens", .int(k)),
            ]))
    }

    /// A trace in the gate's terms.
    static func readout(_ trace: D13BTrace) -> Readout {
        let rows = trace.plan.rows.indices.map { i -> Readout.Row in
            let r = trace.plan.rows[i]
            let ids = D13BReadout.groupIDs(r.readGroups)
            return Readout.Row(
                questionID: r.row.name, type: r.row.kind.rawValue, keys: r.row.keys, ids: r.ids, slot: r.slot,
                groups: r.readGroups, hiddenSHA256: D13BDecoder.sha256(trace.hidden[i]), logitIDs: ids,
                logits: ids.map { trace.logits[i][$0] ?? .nan }, probabilities: trace.probabilities[i])
        }
        return Readout(
            mode: trace.mode, sharedTokens: trace.sharedK, stateTokens: trace.plan.stateTokens,
            inputTokens: trace.plan.inputTokens, imageTokens: trace.plan.imageTokens, rows: rows,
            towerOutputSHA256: trace.towerOutputs.map { sha256(of: $0) },
            response: D13BPythonFormat.dumps(trace.response, indent: 2, asciiOnly: false),
            calls: trace.callSeconds.count, seconds: trace.seconds)
    }
    #endif

    /// The pipeline's request, length and JSON failures as the 422s they are, in the provider's words; a contract or
    /// bundle failure stays itself.
    private func run<T>(_ body: () async throws -> T) async throws -> T {
        do {
            return try await body()
        } catch let error as D13BError {
            switch error {
            case .request(let message), .graphLimit(let message), .json(let message):
                throw SystemOne.WireError(message)
            case .contract, .bundle, .graphNotWired:
                throw error
            }
        }
    }

    static func sha256<T>(of values: [T]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }

    /// The question ids once, and each question within the kit's shapes (a choice of 2–255 options, a score of 2–10
    /// levels).
    static func validate(_ questions: [(id: String, question: Decision.Question)], id: String) throws {
        guard !questions.isEmpty else { throw SystemOne.WireError("'questions' is empty") }
        try SystemOne.validateIDs(questions.map(\.id))
        for (_, question) in questions {
            try DecisionPrompt.validate(question, maxOptions: maxOptionsPerQuestion)
        }
    }

    /// A `questions` object for a prepared state: checked in the wire's forms, then the provider's form of each
    /// question as it came (a choice listed as names, its names alone).
    static func preparedQuestions(_ data: Data) throws -> (D13BJSONValue, [(id: String, question: Decision.Question)]) {
        let questions: JSONValue
        do {
            questions = try JSONValue.parse(data)
        } catch {
            throw SystemOne.WireError("invalid JSON: \(error.localizedDescription)")
        }
        let request = try SystemOne.request(
            from: .object([.init("state", .string("")), .init("questions", questions)]), maxOptions: maxOptionsPerQuestion)
        try validate(request.questions, id: "")
        guard let members = questions.members else { throw SystemOne.WireError("'questions' must be an object keyed by question id") }
        return (.object(members.map { D13BJSONMember($0.key, d13bJSON(textFields(listedChoiceAsNames($0.value)))) }),
                request.questions)
    }

    /// The request in the provider's form: the wire's own values when it came over the wire (`state` and `questions`),
    /// else built from the typed questions. A choice whose options came as a list reads them as names without
    /// descriptions, each the option id the kit's wire gave it; text given as a JSON value reads as its JSON text
    /// (`textFields`).
    static func providerRequest(_ request: SystemOne.Request) -> D13BJSONValue {
        if let json = request.json, let questions = json["questions"]?.members {
            return .object([
                D13BJSONMember("state", json["state"].map(d13bJSON) ?? .null),
                D13BJSONMember("questions", .object(questions.map {
                    D13BJSONMember($0.key, d13bJSON(textFields(listedChoiceAsNames($0.value))))
                })),
            ])
        }
        var state = D13BJSONValue.string(request.state)
        if request.structuredState, let parsed = try? D13BJSONParser.parse(request.state) { state = parsed }
        return .object([
            D13BJSONMember("state", state),
            D13BJSONMember("questions", .object(request.questions.map { D13BJSONMember($0.id, d13bQuestion($0.question)) })),
        ])
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

    /// A wire question's text as the provider's code takes it, text only: instructions, a score's levels, a choice's
    /// descriptions and a noul's `true` / `false` given as a JSON value (an object, a list, a number) read as that value's
    /// JSON text, the kit's wire rule for every model (`SystemOne.question(from:)`). Text and null stay as they came.
    static func textFields(_ question: JSONValue) -> JSONValue {
        guard let fields = question.members else { return question }
        func text(_ value: JSONValue) -> JSONValue {
            switch value {
            case .string, .null: return value
            default: return .string(value.dumps())
            }
        }
        return .object(fields.map { field in
            switch field.key {
            case "instructions":
                return .init(field.key, text(field.value))
            case "criteria":
                if let levels = field.value.elements { return .init(field.key, .array(levels.map(text))) }
                if let members = field.value.members { return .init(field.key, .object(members.map { .init($0.key, text($0.value)) })) }
                return field
            default:
                return field
            }
        })
    }

    /// The kit's parsed JSON as the provider's `json.loads` makes it: members in order, a repeated key in the place it
    /// had with its last value, numbers as written.
    static func d13bJSON(_ value: JSONValue) -> D13BJSONValue {
        switch value {
        case .null: return .null
        case .bool(let b): return .bool(b)
        case .number(let text): return .number(text)
        case .string(let s): return .string(s)
        case .array(let elements): return .array(elements.map(d13bJSON))
        case .object(let members):
            var out: [D13BJSONMember] = []
            var index: [String: Int] = [:]
            for member in members {
                // Python compares keys by code point; Swift's String == by canonical equivalence.
                let key = member.key.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: ",")
                if let at = index[key] {
                    out[at].value = d13bJSON(member.value)
                } else {
                    index[key] = out.count
                    out.append(D13BJSONMember(member.key, d13bJSON(member.value)))
                }
            }
            return .object(out)
        }
    }

    /// A typed question in the provider's form. An option whose description is its id reads as a name alone.
    static func d13bQuestion(_ question: Decision.Question) -> D13BJSONValue {
        var fields: [D13BJSONMember] = []
        switch question.kind {
        case .choice(let options):
            fields = [
                D13BJSONMember("type", .string("choice")), D13BJSONMember("instructions", .string(question.instructions)),
                D13BJSONMember("criteria", .object(options.map {
                    D13BJSONMember($0.id, $0.description == $0.id ? .null : .string($0.description))
                })),
            ]
        case .score(let levels):
            fields = [
                D13BJSONMember("type", .string("score")), D13BJSONMember("instructions", .string(question.instructions)),
                D13BJSONMember("criteria", .array(levels.map { .string($0) })),
            ]
        case .noul(let yes, let no):
            fields = [D13BJSONMember("type", .string("noul")), D13BJSONMember("instructions", .string(question.instructions))]
            var criteria: [D13BJSONMember] = []
            if let yes { criteria.append(D13BJSONMember("true", .string(yes))) }
            if let no { criteria.append(D13BJSONMember("false", .string(no))) }
            if !criteria.isEmpty { fields.append(D13BJSONMember("criteria", .object(criteria))) }
        }
        return .object(fields)
    }

    /// A question's probabilities in the kit's option order (choice: as asked; score: by level; noul: no, yes) from the
    /// provider's, whose order is its keys (noul: yes, no).
    static func kitOrder(_ question: Decision.Question, keys: [String], _ p: [Double], id: String) throws -> [Double] {
        let want: [String]
        switch question.kind {
        case .noul: want = ["yes", "no"]
        case .choice: want = question.optionIDs
        case .score(let levels): want = levels.indices.map(String.init)
        }
        guard keys == want, p.count == keys.count else {
            throw SystemOne.WireError("question '\(id)': the model's options \(keys) are not the question's \(want)")
        }
        if case .noul = question.kind { return [p[1], p[0]] }
        return p
    }
}
