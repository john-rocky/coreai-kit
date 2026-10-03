// KitClefDecider.swift — typed decisions with clef-flash (Cloudflare, Apache-2.0, from Qwen3.5-9B): a state (text,
// JSON, and at most one image) and typed questions in, a probability for every option of every question out, all
// read in one pass. Nothing is generated.
//
// ```swift
// let decider = try await KitClefDecider(catalog: "clef-flash")
// let response = try await decider.systemOne(try SystemOne.request(from: body))   // the /v1/systemone forms
// response["department"]?.choice
// ```
//
// Three Core AI graphs and one host table, driven on the low-level runtime by the files beside this one, ported from
// the model zoo's `apps/ClefFlash` (5ef2247) with every numeric path unchanged: the request rendered as the author's
// `encode_record()` (the state and each question's instructions and options as canonical JSON, every piece
// tokenized alone), the image resized by Pillow's own integer bicubic to the grid's square and read by a tower baked
// at that grid, the decoder (no vocabulary head) run 64 tokens per call from zeroed states for the hidden state at
// every position, the joint schema head over span means, the last token and each option's rows of the lm_head
// table (2.03 GB fp16, memory-mapped on the host), and a float32 softmax per question at temperature 1.
//
// Not a `TypedDecisions`: no engine of the kit returns hidden states, and the head reads every question at once.
// The Mac only: the fp16 decoder alone is 15.9 GB. The first load specializes it (56 s and 31.8 GB of the runtime's
// cache in the zoo's run); later loads read that cache.

import CoreAI
import CoreGraphics
import CryptoKit
import Foundation
import ImageIO

/// clef-flash's parts as downloads at one Hub revision: the decoder bundle (the catalog variant), the head bundle,
/// the folder holding the lm_head table (the Hub lists folders, not files) and a vision tower per grid.
@available(macOS 27, iOS 27, *)
public struct ClefFlashModelID: Sendable, Hashable {
    public let decoder: ModelID
    public let head: ModelID
    public let tableFolder: ModelID
    /// The table's file name inside `tableFolder`.
    public let tableFile: String
    public let towers: [KitClefDecider.Grid: ModelID]

    public init(
        decoder: ModelID, head: ModelID, tableFolder: ModelID, tableFile: String,
        towers: [KitClefDecider.Grid: ModelID]
    ) {
        self.decoder = decoder
        self.head = head
        self.tableFolder = tableFolder
        self.tableFile = tableFile
        self.towers = towers
    }

    /// The parts a catalog entry names: its variant path for the decoder and `assets` for the rest, at its pin.
    public init(entry: CatalogEntry) throws {
        guard let decoder = entry.modelID else { throw CoreAIKitError.modelNotAvailableOnPlatform(id: entry.id) }
        guard let assets = entry.assets, let head = assets.head, let table = assets.table,
            let tableFile = table.split(separator: "/").last.map(String.init), !tableFile.isEmpty
        else {
            throw DecisionError.unsupportedModel(
                id: entry.id,
                reason: "its catalog entry names no 'assets' head and table, which Format.jointHead needs")
        }
        var towers: [KitClefDecider.Grid: ModelID] = [:]
        for (name, path) in assets.towers ?? [:] {
            guard let grid = KitClefDecider.Grid.allCases.first(where: { $0.description == name }) else { continue }
            towers[grid] = entry.modelID(path: path)
        }
        self.init(
            decoder: decoder, head: entry.modelID(path: head),
            tableFolder: entry.modelID(path: (table as NSString).deletingLastPathComponent), tableFile: tableFile,
            towers: towers)
    }
}

/// clef-flash behind `systemOne(_:)` and `decide(_:_:)`. Calls on one instance serialize: the decoder's states are
/// the instance's.
@available(macOS 27, iOS 27, *)
public actor KitClefDecider: DecisionBackend {
    /// The vision tower's fixed grid.
    public enum Grid: Int, Sendable, Hashable, CaseIterable, CustomStringConvertible {
        /// A 256×256 tile, 8×8 merged = 64 image tokens.
        case g256 = 8
        /// A 448×448 tile, 14×14 merged = 196 image tokens: the default, the grid whose answers the author's own
        /// code at its native resolution agrees with more often (the zoo card's grid-price table).
        case g448 = 14

        /// The merged grid's side.
        public var side: Int { rawValue }
        /// Pixels per side of the square the image is resized to (the aspect ratio is not kept).
        public var tile: Int { 32 * rawValue }
        /// Image tokens in the row.
        public var imageTokens: Int { rawValue * rawValue }
        public var description: String { "g\(tile)" }

        /// The grid whose tile is `tile` pixels (256 or 448).
        public init?(tile: Int) {
            guard let grid = Grid.allCases.first(where: { $0.tile == tile }) else { return nil }
            self = grid
        }
    }

    /// Everything one call read, for a gate: the row, every option's logit and probability, the hidden state's
    /// digest and the author's response form.
    public struct Readout: Sendable {
        public struct Question: Sendable, Equatable {
            public let id: String
            public let type: String
            /// [start, end) of the question's instructions in `ids`.
            public let questionSpan: [Int]
            /// [start, end) of each option's rendering in `ids`, in the head's option order.
            public let optionSpans: [[Int]]
            /// The head's option order: noul true, false; choice ids by code point; score levels 0 ..< n.
            public let optionIDs: [String]
        }

        /// The row in the author's processor form (each image token as `<|image_pad|>`).
        public let ids: [Int]
        public let questions: [Question]
        /// The tower's grid; nil for a text-only row.
        public let grid: Grid?
        /// The head function that read the row (`t512` … `t4096`) and the decoder calls it took.
        public let bucket: String
        public let calls: Int
        /// One logit per option, every question's options in turn, in the head's option order.
        public let logits: [Float]
        /// Per question, the float32 softmax of its options' logits, in the head's option order.
        public let probabilities: [[Float]]
        /// sha256 of the decoder's fp16 hidden rows [T, 4096] as stored.
        public let hiddenSHA256: String
        /// sha256 of the tower's float32 rows; nil for a text-only row.
        public let imageRowsSHA256: String?
        /// The author's response form (`systemone_answer`, 4 decimals by Python's `round`), as compact JSON.
        public let response: String
        /// Seconds per stage: "tokenize", "decoder", "head", "wall", and for an image "decode_rgb", "resize",
        /// "patches", "tower".
        public let seconds: [String: Double]
    }

    /// The head's limits: questions per request and options over all of them.
    public static let maxQuestions = 16
    public static let maxOptionsPerRequest = 128

    /// The catalog id, or the decoder bundle's directory name for local files.
    public nonisolated let id: String
    /// The decoder bundle's name, from its metadata.
    public nonisolated let modelName: String
    /// A choice may list as many options as the head reads in one request.
    public nonisolated var maxOptions: Int { Self.maxOptionsPerRequest }
    public nonisolated var readsImages: Bool { true }
    /// What every response says beside its answers: the read-out and its temperature (a response adds `row_tokens`,
    /// the row it read once; `usage` counts that row once per answer, as the kit counts every model's).
    public nonisolated var metadata: JSONValue { .object(metadataMembers) }

    private nonisolated var metadataMembers: [JSONValue.Member] {
        [
            .init("backend", .string("clef-flash joint schema head")),
            .init("model", .string(id)),
            .init("bundle", .string(modelName)),
            .init("temperature", .int(1)),
            .init("calibration", .string("the author's, in the weights")),
        ]
    }

    /// Where a tower that was not loaded up front comes from.
    private struct Source: Sendable {
        let model: ClefFlashModelID
        let store: ModelStore
        let progress: (@Sendable (DownloadProgress) -> Void)?
    }

    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    private let pipeline: ClefPipeline
    #endif
    private let source: Source?
    private var towerURLs: [Grid: URL]
    /// Held for a whole call: a graph `await` lets another call into the actor, and two calls would share one set of
    /// decoder states.
    private let lock = AsyncMutex()

    /// Whether `entry` is a joint-head model this type loads.
    public static func supports(_ entry: CatalogEntry) -> Bool {
        entry.kind == .jointDecision && entry.format == Decision.Format.jointHead.rawValue && entry.modelID != nil
    }

    /// Loads clef-flash by its catalog id (`kind: jointDecision`, `format: jointHead`), downloading on first use:
    /// the decoder, the head and the lm_head table. A grid's tower downloads the first time an image asks for it,
    /// unless `grids` names it here.
    public init(
        catalog id: String,
        grids: Set<Grid> = [],
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .jointDecision)
        guard Self.supports(entry) else {
            throw DecisionError.unsupportedModel(
                id: id, reason: "its catalog format is \(entry.format.map { "'\($0)'" } ?? "not given"), not jointHead")
        }
        let model = try ClefFlashModelID(entry: entry)
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        let headURL = try await store.download(model.head, progress: downloadProgress)
        let tableURL = try await store.download(model.tableFolder, progress: downloadProgress)
            .appendingPathComponent(model.tableFile)
        var towers: [Grid: URL] = [:]
        for grid in grids.sorted(by: { $0.rawValue < $1.rawValue }) {
            guard let tower = model.towers[grid] else {
                throw DecisionError.unsupportedModel(id: id, reason: "its catalog entry names no \(grid) tower")
            }
            towers[grid] = try await store.download(tower, progress: downloadProgress)
        }
        try await self.init(
            decoderAt: decoderURL, headAt: headURL, tableAt: tableURL, towersAt: towers, id: id,
            source: Source(model: model, store: store, progress: downloadProgress))
    }

    /// Loads local files: the decoder bundle directory (metadata.json, the graph, tokenizer/), the head bundle
    /// directory, the lm_head table file and a tower per grid (the graph, or the directory holding it).
    public init(
        decoderAt decoderURL: URL, headAt headURL: URL, tableAt tableURL: URL, towersAt towers: [Grid: URL] = [:]
    ) async throws {
        try await self.init(
            decoderAt: decoderURL, headAt: headURL, tableAt: tableURL, towersAt: towers,
            id: decoderURL.lastPathComponent, source: nil)
    }

    private init(
        decoderAt decoderURL: URL, headAt headURL: URL, tableAt tableURL: URL, towersAt towers: [Grid: URL],
        id: String, source: Source?
    ) async throws {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let metadata = try ClefPipeline.Metadata(bundle: decoderURL, head: headURL)
        let decoder = try Self.graph(named: metadata.asset, in: decoderURL)
        let head = try Self.graph(named: metadata.headAsset, in: headURL)
        let pipeline = try await ClefPipeline(
            assets: .init(decoderBundle: decoderURL, decoder: decoder, head: headURL, headAsset: head, table: tableURL),
            decoderOptions: Self.decoderOptions(for: decoder), headOptions: Self.graphOptions(for: head))
        for (grid, url) in towers.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            let graph = try GraphBundle.resolve(in: url)
            try await pipeline.loadTower(
                Self.internalGrid(grid), contentsOf: graph, options: Self.graphOptions(for: graph))
        }
        self.pipeline = pipeline
        self.modelName = metadata.name
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "clef-flash runs on Apple silicon only")
        #endif
        self.id = id
        self.source = source
        self.towerURLs = towers
    }

    /// The graph a bundle's metadata names (`<name>.aimodel`), or one compiled for this device beside it.
    private static func graph(named asset: String, in dir: URL) throws -> URL {
        let stem = (asset as NSString).deletingPathExtension
        guard let graph = try GraphBundle.graph(named: stem, in: dir) else {
            throw ClefFlashError.bundle("\(dir.lastPathComponent) holds no \(asset)")
        }
        return graph
    }

    /// The runtime specializes a JIT `.aimodel` with the settings the zoo's Swift gate ran it with: the GPU, and
    /// frequent reshapes for the decoder (its `position_ids` grows by 64 per call). A compiled graph loads as compiled.
    static func decoderOptions(for graph: URL) -> SpecializationOptions {
        guard graph.pathExtension == "aimodel" else { return .default }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        return options
    }

    static func graphOptions(for graph: URL) -> SpecializationOptions {
        graph.pathExtension == "aimodel" ? SpecializationOptions(preferredComputeUnitKind: .gpu) : .default
    }

    /// Loads the towers for `grids` now (downloading them if needed), so the first image at each does not pay for it.
    public func prepare(_ grids: Grid...) async throws {
        try await lock.withLock {
            for grid in grids { try await loadTower(grid) }
        }
    }

    // MARK: - Decide

    /// One question on one state: a request of one question, read the same way.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        let response = try await systemOne(SystemOne.Request(state: state, questions: [(id: "q", question: question)]))
        return response.answers[0].answer
    }

    /// A whole request, every question read in one pass, the answers in request order. A request parsed from the
    /// wire is rendered from its own values (numbers as written, structured values as canonical JSON); one built in
    /// Swift from its typed questions. `images` holds at most one image, read at `grid` (448 by default).
    public func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        try SystemOne.validateIDs(request.questions.map(\.id))
        guard request.questions.count <= Self.maxQuestions else {
            throw SystemOne.WireError(
                "'\(id)' reads at most \(Self.maxQuestions) questions per request, got \(request.questions.count)")
        }
        var options = 0
        for (_, question) in request.questions {
            try DecisionPrompt.validate(question, maxOptions: Self.maxOptionsPerRequest)
            options += question.optionIDs.count
        }
        guard options <= Self.maxOptionsPerRequest else {
            throw SystemOne.WireError(
                "'\(id)' reads at most \(Self.maxOptionsPerRequest) options over all of a request's questions, "
                    + "got \(options)")
        }
        guard request.images.count <= 1 else {
            throw SystemOne.WireError("a request carries at most one image, got \(request.images.count)")
        }
        var grid = Grid.g448
        if let side = request.grid {
            guard let chosen = Grid(tile: side) else {
                throw SystemOne.WireError("'grid' must be 256 or 448, got \(side)")
            }
            grid = chosen
        }
        let image = try request.images.first.map(Self.cgImage)
        let clef = try Self.clefRequest(request, id: id)
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let trace = try await lock.withLock {
            if image != nil { try await loadTower(grid) }
            return try await run {
                try await pipeline.trace(request: clef, image: image, grid: Self.internalGrid(grid))
            }
        }
        let tokens = trace.row.ids.count
        let imageSeconds = image == nil
            ? nil : ["decode_rgb", "resize", "patches", "tower"].compactMap { trace.seconds[$0] }.reduce(0, +)
        let timing = Decision.Timing(
            promptTokens: tokens, reusedTokens: 0, seconds: trace.seconds["wall"] ?? 0, imageSeconds: imageSeconds,
            decoderSeconds: trace.pass.seconds)
        var answers: [SystemOne.Answer] = []
        for (k, (key, question)) in request.questions.enumerated() {
            let layout = trace.row.questions[k]
            let byID = Dictionary(uniqueKeysWithValues: zip(layout.optionIDs, trace.probabilities[k].map(Double.init)))
            let p = try Self.kitOrder(question, byID, id: key)
            answers.append(SystemOne.Answer(
                id: key, question: question,
                answer: DecisionPrompt.answer(for: question, probabilities: p, timing: timing)))
        }
        return SystemOne.Response(
            model: id, answers: answers, stateTokens: tokens, prefill: timing,
            metadata: .object(metadataMembers + [.init("row_tokens", .int(tokens))]))
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "clef-flash runs on Apple silicon only")
        #endif
    }

    /// The whole call in the author's terms, for a gate: `requestJSON` is a request in the author's form (`model`,
    /// `state`, `questions`) read byte for byte, `image` the decoded image (nil = text only).
    public func readout(requestJSON: Data, image: CGImage?, grid: Grid = .g448) async throws -> Readout {
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        let request = try ClefRequest(data: requestJSON)
        let trace = try await lock.withLock {
            if image != nil { try await loadTower(grid) }
            return try await run {
                try await pipeline.trace(request: request, image: image, grid: Self.internalGrid(grid))
            }
        }
        let row = trace.row
        return Readout(
            ids: row.ids,
            questions: row.questions.map {
                Readout.Question(
                    id: $0.questionID, type: $0.type, questionSpan: $0.questionSpan, optionSpans: $0.optionSpans,
                    optionIDs: $0.optionIDs)
            },
            grid: row.grid == nil ? nil : grid, bucket: trace.bucket.function, calls: trace.pass.calls,
            logits: trace.logits, probabilities: trace.probabilities, hiddenSHA256: Self.sha256(of: trace.pass.hidden),
            imageRowsSHA256: trace.imageRows.map(Self.sha256(of:)),
            response: ClefPythonJSON.dumps(trace.response, sortKeys: false), seconds: trace.seconds)
        #else
        throw DecisionError.unsupportedModel(id: id, reason: "clef-flash runs on Apple silicon only")
        #endif
    }

    /// Decodes an image the way the zoo's host does (ImageIO, the first frame).
    public static func cgImage(_ image: SystemOne.Image) throws -> CGImage {
        let source: CGImageSource?
        switch image {
        case .data(let data): source = CGImageSourceCreateWithData(data as CFData, nil)
        case .file(let url): source = CGImageSourceCreateWithURL(url as CFURL, nil)
        }
        guard let source, let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            let what: String
            switch image {
            case .data(let data): what = "\(data.count) bytes"
            case .file(let url): what = url.path
            }
            throw SystemOne.WireError("the image could not be decoded (\(what))")
        }
        return decoded
    }

    // MARK: - Internals

    #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
    static func internalGrid(_ grid: Grid) -> ClefPipeline.Grid { grid == .g256 ? .g256 : .g448 }

    /// The loaded tower for `grid`, downloading and loading it first when this decider has a source for it.
    private func loadTower(_ grid: Grid) async throws {
        if pipeline.towers[Self.internalGrid(grid)] != nil { return }
        var url = towerURLs[grid]
        if url == nil, let source, let model = source.model.towers[grid] {
            url = try await source.store.download(model, progress: source.progress)
        }
        guard let url else {
            throw SystemOne.WireError(
                "no \(grid) vision tower is loaded, and this decider has no download source for one")
        }
        let graph = try GraphBundle.resolve(in: url)
        try await pipeline.loadTower(Self.internalGrid(grid), contentsOf: graph, options: Self.graphOptions(for: graph))
        towerURLs[grid] = url
    }

    /// The pipeline's request and prompt failures as the 422s they are; a contract or bundle failure stays itself.
    private func run<T>(_ body: () async throws -> T) async throws -> T {
        do {
            return try await body()
        } catch let error as ClefFlashError {
            switch error {
            case .request(let message), .prompt(let message), .image(let message), .json(let message):
                throw SystemOne.WireError(message)
            case .contract, .bundle:
                throw error
            }
        }
    }

    static func sha256<T>(of values: [T]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined() }
    }
    #else
    static func internalGrid(_ grid: Grid) -> Grid { grid }
    private func loadTower(_ grid: Grid) async throws {}
    #endif

    /// The request in the author's form: the wire's own values when it came over the wire, else built from the typed
    /// questions. A choice whose options came as a list reads them as options without descriptions; `model` is `id`
    /// when the request names none.
    static func clefRequest(_ request: SystemOne.Request, id: String) throws -> ClefRequest {
        var root: ClefJSON
        if let json = request.json {
            root = Self.clefJSON(json)
        } else {
            var state = ClefJSON.string(request.state)
            if request.structuredState, let parsed = try? ClefJSONParser.parse(request.state) { state = parsed }
            root = .object([
                ClefJSONMember("state", state),
                ClefJSONMember(
                    "questions", .object(request.questions.map { ClefJSONMember($0.id, Self.clefQuestion($0.question)) })),
            ])
        }
        guard case .object(var members) = root else { throw SystemOne.WireError("the request must be a JSON object") }
        if let at = members.firstIndex(where: { $0.key == "model" }) {
            if members[at].value.string == nil { members[at].value = .string(id) }
        } else {
            members.insert(ClefJSONMember("model", .string(id)), at: 0)
        }
        if let at = members.firstIndex(where: { $0.key == "questions" }),
            case .object(var questions) = members[at].value
        {
            for q in questions.indices {
                guard case .object(var fields) = questions[q].value,
                    fields.contains(where: { $0.key == "type" && $0.value == .string("choice") }),
                    let c = fields.firstIndex(where: { $0.key == "criteria" }), case .array(let list) = fields[c].value
                else { continue }
                fields[c].value = .object(list.map {
                    ClefJSONMember($0.string ?? ClefPythonJSON.dumps($0, sortKeys: false), .null)
                })
                questions[q].value = .object(fields)
            }
            members[at].value = .object(questions)
        }
        do {
            return try ClefRequest(json: .object(members))
        } catch let error as ClefFlashError {
            throw SystemOne.WireError(error.description)
        }
    }

    /// The kit's parsed JSON as the zoo host's: members in order, a repeated key in its first place with its last
    /// value (`json.loads`), numbers as written.
    static func clefJSON(_ value: JSONValue) -> ClefJSON {
        switch value {
        case .null: return .null
        case .bool(let b): return .bool(b)
        case .number(let text): return .number(text)
        case .string(let s): return .string(s)
        case .array(let elements): return .array(elements.map(clefJSON))
        case .object(let members):
            var out: [ClefJSONMember] = []
            var index: [String: Int] = [:]
            for member in members {
                // Python compares keys by code point; Swift's String == by canonical equivalence.
                let key = member.key.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: ",")
                if let at = index[key] {
                    out[at].value = clefJSON(member.value)
                } else {
                    index[key] = out.count
                    out.append(ClefJSONMember(member.key, clefJSON(member.value)))
                }
            }
            return .object(out)
        }
    }

    /// A typed question in the author's form. An option whose description is its id reads as an option alone.
    static func clefQuestion(_ question: Decision.Question) -> ClefJSON {
        var fields = [ClefJSONMember("instructions", .string(question.instructions))]
        switch question.kind {
        case .choice(let options):
            fields.insert(ClefJSONMember("type", .string("choice")), at: 0)
            fields.append(ClefJSONMember("criteria", .object(options.map {
                ClefJSONMember($0.id, $0.description == $0.id ? .null : .string($0.description))
            })))
        case .score(let levels):
            fields.insert(ClefJSONMember("type", .string("score")), at: 0)
            fields.append(ClefJSONMember("criteria", .array(levels.map { .string($0) })))
        case .noul(let yes, let no):
            fields.insert(ClefJSONMember("type", .string("noul")), at: 0)
            var criteria: [ClefJSONMember] = []
            if let yes { criteria.append(ClefJSONMember("true", .string(yes))) }
            if let no { criteria.append(ClefJSONMember("false", .string(no))) }
            if !criteria.isEmpty { fields.append(ClefJSONMember("criteria", .object(criteria))) }
        }
        return .object(fields)
    }

    /// A question's probabilities in the kit's option order (choice: as asked; score: by level; noul: no, yes) from
    /// the head's, keyed by option id.
    static func kitOrder(_ question: Decision.Question, _ byID: [String: Double], id: String) throws -> [Double] {
        let ids: [String]
        switch question.kind {
        case .noul: ids = ["false", "true"]
        default: ids = question.optionIDs
        }
        return try ids.map {
            guard let p = byID[$0] else {
                throw SystemOne.WireError("question '\(id)': no probability for option '\($0)'")
            }
            return p
        }
    }
}
