// KitD1OmniDecider.swift — typed decisions with d1-omni-600M (Liquid AI, LFM Open License v1.0: LFM2.5-Encoder-350M with
// a typed decision head, a SigLIP2 vision tower and a FastConformer audio tower): a state (text or JSON), images or a
// 16 kHz clip, and typed questions in, a probability for every option of every question out. Nothing is generated.
//
// ```swift
// let decider = try await KitD1OmniDecider(catalog: "d1-omni-600m")
// let response = try await decider.systemOne(try SystemOne.request(from: body))     // the /v1/systemone forms
// response["team"]?.choice
//
// let heard = try await decider.systemOne(
//     state: nil, questions: [(id: "refund", question: .noul("Does the caller want a refund?"))], audio: .file(wav))
// let seen = try await decider.systemOne(
//     state: nil, questions: [(id: "room", question: .choice("What room is this?", ["kitchen", "bedroom"]))],
//     images: [.file(photo)])
// ```
//
// Twelve Core AI graphs on the low-level runtime, driven by the files beside this one: the model zoo's `apps/D1Omni`
// (f9e0e09) ported with every numeric path unchanged. One row per question in the publisher's form (prompt.py's
// `encode`: the state, the question and each option between the publisher's delimiter tokens, a `<|mask|>` marker
// before each option, caller text escaped so that it cannot produce a delimiter). The decision graph (fp16, a static
// length of 64, 128, 256, 512, 1,024, 2,048 or 4,096 positions) runs the row at the smallest length that holds its media
// prefix and its ids, and each option is read at its marker: on a text row ÷ the publisher's temperature for the
// question's type and option count, then a float32 softmax with NumPy's pairwise sum; a noul is read as [false, true].
// An image is decoded as PIL reads it (a baseline JPEG whose components share one sampling factor by libjpeg's
// arithmetic, anything else by ImageIO), cut into vision.py's crops, resized by torchvision's float32 antialias kernel
// and run through the vision graph once per crop. A clip (16-bit PCM WAV, mono, 16 kHz) becomes the publisher's log-mel
// in float64 and runs through the audio graph of its 5 / 10 / 20 / 30 s bucket. The media rows go in front of each
// question's ids. A request's state is tokenized once for all its questions.
//
// Not a `TypedDecisions`: the decision graph is not a language model, and its rows, temperatures and media prefixes are
// not laya's encoder form, although the bundle's metadata says `decision.head = "encoder"`. The platform folder's twelve
// bundles download on first use (6.5 GB); a graph loads the first time a row or a medium needs it and stays loaded (the
// zoo's first loads on an M4 Max: 1.9 s at L256 and 2.4 s at L4096, the vision graph 0.6 s, a clip bucket 1.4 s).

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public actor KitD1OmniDecider: DecisionBackend {
    /// The catalog `format` of a d1-omni entry (`kind: omniDecision`).
    public static let format = "markerScores"
    /// The widest choice the hosted API accepts; the publisher sets no limit of its own, a row past 4,096 positions is
    /// refused instead.
    public static let maxOptionsPerQuestion = 255
    /// The decision graph's lengths, the vision graph and the clip buckets: the folders of the platform folder.
    public static let decisionLengths = [64, 128, 256, 512, 1024, 2048, 4096]
    public static let clipSeconds = [5, 10, 20, 30]
    public static let folders: [String] =
        decisionLengths.map { "decide-fp16-L\($0)" } + ["vision-fp16"] + clipSeconds.map { "audio-fp16-\($0)s" }

    /// Everything one request read, for a gate: each question's row, the marker logits and the probabilities, and the
    /// response in the publisher's form.
    public struct Readout: Sendable {
        public struct Row: Sendable, Equatable {
            public let questionID: String
            public let type: String
            /// The row's ids (the media prefix not included) and the marker positions in them.
            public let ids: [Int]
            public let markers: [Int]
            /// P, the media prefix rows in front of the ids; P + ids.count positions in all.
            public let prefixRows: Int
            /// The decision graph's length the row ran at.
            public let bucket: Int
            /// The scores at the markers (float32, the graph's), in the publisher's option order (noul: false, true).
            public let logits: [Float]
            /// The reported probabilities (float32): the option order, a noul as [yes, no].
            public let probabilities: [Float]
        }

        /// "text", "image" or "audio".
        public let mode: String
        public let rows: [Row]
        /// The publisher's response body, `json.dumps(response, ensure_ascii=False)`.
        public let response: String
        /// Seconds per stage: "media" (decode, preprocessing, the media graphs), "rows" (the tokenizer and the prompt),
        /// "graphs" (every row's inputs, call and readout), "wall".
        public let seconds: [String: Double]
    }

    /// The catalog id, or the folder's name for local files.
    public nonisolated let id: String
    /// The checkpoint the decision bundles were converted from, from their metadata (`LiquidAI/d1-omni-600M`).
    public nonisolated let modelName: String
    public nonisolated var maxOptions: Int { Self.maxOptionsPerQuestion }
    public nonisolated var readsImages: Bool { true }
    public nonisolated var readsAudio: Bool { true }

    private let pipeline: D1OmniPipeline
    /// Held for a whole call: a graph `await` lets another call into the actor, and the graphs and their caches serve
    /// one call at a time.
    private let lock = AsyncMutex()

    /// Whether `entry` is a d1-omni model this type loads.
    public static func supports(_ entry: CatalogEntry) -> Bool {
        entry.kind == .omniDecision && entry.format == format && entry.modelID != nil
    }

    /// Loads d1-omni by its catalog id (`kind: omniDecision`, `format: markerScores`), downloading the platform folder's
    /// bundles on first use. Each bundle is its own download at `<variant>/<folder>`, a path the store takes as given:
    /// an iPhone reads the JIT `.aimodel`s of `ios/`, never the repo's `ios-<arch>/` AOT set.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .omniDecision)
        guard Self.supports(entry), let variant = entry.variant else {
            throw DecisionError.unsupportedModel(
                id: id, reason: "its catalog format is \(entry.format.map { "'\($0)'" } ?? "not given"), not \(Self.format)")
        }
        var urls: [String: URL] = [:]
        for name in Self.folders {
            urls[name] = try await store.download(entry.modelID(path: "\(variant.path)/\(name)"), progress: downloadProgress)
        }
        var decisions: [Int: URL] = [:]
        for L in Self.decisionLengths { decisions[L] = urls["decide-fp16-L\(L)"] }
        var clips: [Int: URL] = [:]
        for sec in Self.clipSeconds { clips[sec] = urls["audio-fp16-\(sec)s"] }
        try await self.init(
            decisions: decisions, media: D1OmniPipeline.MediaFolders(vision: urls["vision-fp16"], audio: clips), id: id)
    }

    /// Loads a local platform folder: the repo's `macos/` or `ios/`, a `decide-fp16-L<L>/` folder per length beside
    /// `vision-fp16/` and `audio-fp16-<sec>s/`. A length or a medium whose folder is missing is refused when a request
    /// needs it.
    public init(folderAt url: URL) async throws {
        try await self.init(
            decisions: try D1OmniPipeline.folders(macos: url), media: D1OmniPipeline.MediaFolders.find(macos: url),
            id: url.lastPathComponent)
    }

    private init(decisions: [Int: URL], media: D1OmniPipeline.MediaFolders, id: String) async throws {
        let pipeline = try await D1OmniPipeline(folders: decisions, media: media)
        self.pipeline = pipeline
        self.id = id
        let smallest = pipeline.lengths.first.flatMap { pipeline.buckets[$0]?.folder }
        let metadata = smallest.flatMap { try? D1JSONParser.parse(Data(contentsOf: $0.appendingPathComponent("metadata.json"))) }
        self.modelName = metadata?["source"]?["hf_model_id"]?.string ?? id
    }

    // MARK: - Decide

    /// One question on one state: a request of one question, read the same way.
    public func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer {
        let response = try await systemOne(SystemOne.Request(state: state, questions: [(id: "q", question: question)]))
        return response.answers[0].answer
    }

    /// A whole request, one row per question, the answers in request order. A request parsed from the wire is read from
    /// its own values (numbers as written, structured values as the publisher's `json.dumps` writes them, `state` left
    /// out for a clip as the publisher's `None`); one built in Swift from its typed questions. `images` and `audio` are
    /// the request's media: one image or one clip, not both. `grid` is refused: the image is read at the publisher's
    /// own crops.
    public func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response {
        try Self.validate(request.questions, id: id)
        if request.grid != nil {
            throw SystemOne.WireError(
                "'\(id)' reads an image at its own crops; 'grid' is clef-flash's tile side, leave it out")
        }
        let (state, questions) = Self.publisherRequest(request)
        return try await answer(
            request.questions, state: state, questions: questions, images: request.images, audio: request.audio)
    }

    /// The same request in Swift: `state` nil is the publisher's `None` (an empty state; for a clip the `{}` the audio
    /// questions were trained with), a string the plain state, an object or an array a structured one. Several images
    /// are read in order, as the publisher reads them; an image and a clip together are refused.
    public func systemOne(
        state: JSONValue?, questions: [(id: String, question: Decision.Question)], images: [SystemOne.Image] = [],
        audio: SystemOne.Audio? = nil
    ) async throws -> SystemOne.Response {
        try Self.validate(questions, id: id)
        let d1State = state.map(Self.d1JSON)
        let d1Questions = D1JSONValue.object(questions.map { D1JSONMember($0.id, Self.d1Question($0.question)) })
        return try await answer(questions, state: d1State, questions: d1Questions, images: images, audio: audio)
    }

    /// The whole call in the publisher's terms, for a gate: `requestJSON` is `{"state": …, "questions": {…}}` as the
    /// publisher's `system_one` takes it (`state` left out or null for None), read byte for byte as `json.loads` reads it;
    /// `images` (files, in order) or `audio` (a WAV file) are its media.
    public func readout(requestJSON: Data, images: [URL] = [], audio: URL? = nil) async throws -> Readout {
        try await lock.withLock {
            try await run {
                let root = try D1JSONParser.parse(requestJSON)
                guard let questions = root["questions"] else { throw D1OmniError.request("the request has no 'questions'") }
                let media = try await self.media(images: images.map { .file($0) }, audio: audio.map { .file($0) })
                let trace = try await self.trace(state: root["state"], questions: questions, media: media)
                return Readout(
                    mode: media.mode.rawValue,
                    rows: zip(trace.rows, trace.results).map { row, result in
                        Readout.Row(
                            questionID: row.qid, type: row.question.type.rawValue, ids: row.ids, markers: row.markers,
                            prefixRows: row.prefixLength, bucket: result.bucket, logits: result.logits,
                            probabilities: result.probabilities)
                    },
                    response: D1PythonFormat.dumps(trace.response, asciiOnly: false), seconds: trace.seconds)
            }
        }
    }

    /// The decision lengths whose graphs are loaded: a length loads the first time a row needs it.
    public func loadedLengths() async throws -> [Int] {
        try await lock.withLock { pipeline.loaded.keys.sorted() }
    }

    // MARK: - Internals

    /// One request's media: the mode, the prefix rows [P * 1024] (nil for text), P, and the seconds it took.
    struct Media {
        let mode: D1Mode
        let prefix: [Float]?
        let rows: Int
        let seconds: Double
    }

    /// The rows, each row's decision and the publisher's response.
    struct Trace {
        let rows: [D1Row]
        let results: [D1OmniPipeline.RowResult]
        let response: D1JSONValue
        let seconds: [String: Double]
    }

    /// The kit's response from the publisher's rows: each question's probabilities in the kit's option order (choice as
    /// asked, score by level, noul no / yes), the media prefix as the prefill.
    private func answer(
        _ asked: [(id: String, question: Decision.Question)], state: D1JSONValue?, questions: D1JSONValue,
        images: [SystemOne.Image], audio: SystemOne.Audio?
    ) async throws -> SystemOne.Response {
        let (media, trace) = try await lock.withLock {
            try await run {
                let media = try await self.media(images: images, audio: audio)
                return (media, try await self.trace(state: state, questions: questions, media: media))
            }
        }
        guard trace.rows.count == asked.count else {
            throw D1OmniError.contract("\(trace.rows.count) rows for \(asked.count) questions")
        }
        var answers: [SystemOne.Answer] = []
        for (k, (key, question)) in asked.enumerated() {
            let row = trace.rows[k], result = trace.results[k]
            let timing = Decision.Timing(
                promptTokens: row.positions, reusedTokens: 0,
                seconds: result.seconds.inputs + result.seconds.graph + result.seconds.readout)
            let p = try Self.kitOrder(question, row: row, result.probabilities, id: key)
            answers.append(SystemOne.Answer(
                id: key, question: question, answer: DecisionPrompt.answer(for: question, probabilities: p, timing: timing)))
        }
        return SystemOne.Response(
            model: id, answers: answers, stateTokens: media.rows,
            prefill: Decision.Timing(promptTokens: media.rows, reusedTokens: 0, seconds: media.seconds),
            metadata: .object([
                .init("backend", .string("d1-omni marker scores")), .init("model", .string(id)),
                .init("bundle", .string(modelName)),
                .init("calibration", .string(
                    "the publisher's temperature by question type and option count on a text row; none on an image or audio row")),
                .init("mode", .string(media.mode.rawValue)), .init("prefix_rows", .int(media.rows)),
                .init("buckets", .array(trace.results.map { .int($0.bucket) })),
            ]))
    }

    /// The request's media through the vision graph (once per crop, images in order) or the audio graph of the clip's
    /// bucket; nothing for a text request.
    private func media(images: [SystemOne.Image], audio: SystemOne.Audio?) async throws -> Media {
        let t0 = ContinuousClock.now
        if !images.isEmpty, audio != nil {
            throw D1OmniError.request("a request carries images or a clip, not both")
        }
        if !images.isEmpty {
            var decoded: [D1RGBImage] = []
            for (k, image) in images.enumerated() {
                let (data, name) = try Self.bytes(image, name: "image \(k)")
                decoded.append(try D1ImagePreprocess.decode(data: data, name: name))
            }
            let p = try D1MediaLength.imagePrefixLength(decoded.map { ($0.width, $0.height) })
            let prefix = try await pipeline.imagePrefix(try pipeline.imageInputs(decoded))
            guard prefix.count == p * D1DecisionGraph.hidden else {
                throw D1OmniError.contract("an image prefix of \(prefix.count / D1DecisionGraph.hidden) rows, P = \(p)")
            }
            return Media(mode: .image, prefix: prefix, rows: p, seconds: d1SecondsSince(t0))
        }
        if let audio {
            let (data, name) = try Self.bytes(audio, name: "audio")
            let x = try pipeline.audioInputs(samples: try D1AudioPreprocess.samples(wav: data, name: name))
            let prefix = try await pipeline.audioPrefix(x)
            return Media(mode: .audio, prefix: prefix, rows: x.prefixRows, seconds: d1SecondsSince(t0))
        }
        return Media(mode: .text, prefix: nil, rows: 0, seconds: 0)
    }

    /// host.py's order: the rows (the state tokenized once), each row through the graph of its length, the response.
    private func trace(state: D1JSONValue?, questions: D1JSONValue, media: Media) async throws -> Trace {
        let t0 = ContinuousClock.now
        let rows = try pipeline.rows(state: state, questions: questions, mode: media.mode, prefixLength: media.rows)
        let t1 = ContinuousClock.now
        var results: [D1OmniPipeline.RowResult] = []
        for row in rows { results.append(try await pipeline.decide(row, prefix: media.prefix)) }
        let t2 = ContinuousClock.now
        let response = D1Readout.response(rows: rows, probabilities: results.map(\.probabilities))
        return Trace(
            rows: rows, results: results, response: response,
            seconds: ["media": media.seconds, "rows": D1OmniPipeline.seconds(t0, t1), "graphs": D1OmniPipeline.seconds(t1, t2),
                      "wall": media.seconds + d1SecondsSince(t0)])
    }

    /// The publisher's refusals as the 422s they are; a contract or bundle failure stays itself.
    private func run<T>(_ body: () async throws -> T) async throws -> T {
        do {
            return try await body()
        } catch let error as D1OmniError {
            switch error {
            case .request(let message), .json(let message), .graphLimit(let message):
                throw SystemOne.WireError(message)
            case .contract, .bundle:
                throw error
            }
        }
    }

    /// A medium's bytes and the name its refusals use; a file that cannot be read is a 422 that names it.
    static func bytes(_ image: SystemOne.Image, name: String) throws -> (Data, String) {
        switch image {
        case .data(let data): return (data, name)
        case .file(let url): return (try read(url), url.lastPathComponent)
        }
    }

    static func bytes(_ audio: SystemOne.Audio, name: String) throws -> (Data, String) {
        switch audio {
        case .data(let data): return (data, name)
        case .file(let url): return (try read(url), url.lastPathComponent)
        }
    }

    private static func read(_ url: URL) throws -> Data {
        do {
            return try Data(contentsOf: url)
        } catch {
            throw SystemOne.WireError("\(url.path) could not be read: \(error.localizedDescription)")
        }
    }

    /// The question ids once, and each question within the publisher's shapes (a choice of 2–255 options, a score of
    /// 2–10 levels).
    static func validate(_ questions: [(id: String, question: Decision.Question)], id: String) throws {
        guard !questions.isEmpty else { throw SystemOne.WireError("'questions' is empty") }
        try SystemOne.validateIDs(questions.map(\.id))
        for (_, question) in questions {
            try DecisionPrompt.validate(question, maxOptions: maxOptionsPerQuestion)
        }
    }

    /// The request in the publisher's form: the wire's own values when it came over the wire (a missing or null state
    /// is None), else built from the typed questions. A choice whose options came as a list reads them as names without
    /// descriptions, each the option id the kit's wire gave it.
    static func publisherRequest(_ request: SystemOne.Request) -> (state: D1JSONValue?, questions: D1JSONValue) {
        if let json = request.json, let questions = json["questions"]?.members {
            let state = json["state"].map(d1JSON)
            return (state, .object(questions.map { D1JSONMember($0.key, d1JSON(listedChoiceAsNames($0.value))) }))
        }
        var state = D1JSONValue.string(request.state)
        if request.structuredState, let parsed = try? D1JSONParser.parse(request.state) { state = parsed }
        return (state, .object(request.questions.map { D1JSONMember($0.id, d1Question($0.question)) }))
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

    /// The kit's parsed JSON as the publisher's `json.loads` makes it: members in order, a repeated key in its first
    /// place with its last value, numbers as written.
    static func d1JSON(_ value: JSONValue) -> D1JSONValue {
        switch value {
        case .null: return .null
        case .bool(let b): return .bool(b)
        case .number(let text): return .number(text)
        case .string(let s): return .string(s)
        case .array(let elements): return .array(elements.map(d1JSON))
        case .object(let members):
            var out: [D1JSONMember] = []
            var index: [String: Int] = [:]
            for member in members {
                // Python compares keys by code point; Swift's String == by canonical equivalence.
                let key = member.key.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: ",")
                if let at = index[key] {
                    out[at].value = d1JSON(member.value)
                } else {
                    index[key] = out.count
                    out.append(D1JSONMember(member.key, d1JSON(member.value)))
                }
            }
            return .object(out)
        }
    }

    /// A typed question in the publisher's form. An option whose description is its id reads as a name alone.
    static func d1Question(_ question: Decision.Question) -> D1JSONValue {
        var fields: [D1JSONMember] = []
        switch question.kind {
        case .choice(let options):
            fields = [
                D1JSONMember("type", .string("choice")), D1JSONMember("instructions", .string(question.instructions)),
                D1JSONMember("criteria", .object(options.map {
                    D1JSONMember($0.id, $0.description == $0.id ? .null : .string($0.description))
                })),
            ]
        case .score(let levels):
            fields = [
                D1JSONMember("type", .string("score")), D1JSONMember("instructions", .string(question.instructions)),
                D1JSONMember("criteria", .array(levels.map { .string($0) })),
            ]
        case .noul(let yes, let no):
            fields = [D1JSONMember("type", .string("noul")), D1JSONMember("instructions", .string(question.instructions))]
            var criteria: [D1JSONMember] = []
            if let yes { criteria.append(D1JSONMember("true", .string(yes))) }
            if let no { criteria.append(D1JSONMember("false", .string(no))) }
            if !criteria.isEmpty { fields.append(D1JSONMember("criteria", .object(criteria))) }
        }
        return .object(fields)
    }

    /// A question's probabilities in the kit's option order (choice: as asked; score: by level; noul: no, yes) from the
    /// publisher's reported order (a noul as [yes, no]).
    static func kitOrder(_ question: Decision.Question, row: D1Row, _ p: [Float], id: String) throws -> [Double] {
        let ordered: [Float]
        switch question.kind {
        case .noul:
            ordered = Array(p.reversed())
        case .choice:
            guard row.question.names == question.optionIDs else {
                throw SystemOne.WireError(
                    "question '\(id)': the model's options \(row.question.names) are not the question's \(question.optionIDs)")
            }
            ordered = p
        case .score:
            ordered = p
        }
        guard ordered.count == question.optionIDs.count else {
            throw SystemOne.WireError("question '\(id)': \(ordered.count) probabilities for \(question.optionIDs.count) options")
        }
        return ordered.map(Double.init)
    }
}
