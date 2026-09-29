// KitVisionDecider.swift — typed decisions about an image with decider-2b-vision (Mapika, Apache-2.0): an image, a
// state and typed questions in, a probability per option out, every question read in one pass at its own answer
// slot. Nothing is generated.
//
// ```swift
// let decider = try await KitVisionDecider(catalog: "decider-2b-vision")
// let answers = try await decider.decide(
//     image: frame, state: "The image shows the current game screen.",
//     questions: [.choice("Where should the paddle move?", ["up", "down", "stay"]), .noul("Is the ball visible?")])
// answers[0].choice   // "up"
// answers[1].noul     // P(yes)
// ```
//
// Two Core AI graphs and the host between them, ported from the model zoo's `apps/DeciderVision` library (2307ecf),
// the code its gates ran: the image resized in Pillow's pass order to the grid's square and cut into patches
// (`DeciderVisionPreprocessor`), a vision tower baked at that grid, the author's prompt with the tower's rows as its
// image block (`DeciderVisionPromptRenderer`), and a two-function decoder read in the bundle's chunk order
// (`DeciderVisionRuntime`); at each slot the letters A… over the question's options, softmaxed at the bundle's
// temperature (1).
//
// The grid: `.g256` (a 256×256 tile, 64 image tokens) for game frames and speed, `.g448` (448×448, 196 tokens) for
// photos, where the author's own code agrees with its native-resolution answer more often (Visual7W 0.980 against
// 0.947, the zoo's grid-price table). The aspect ratio is not kept. A tower loads the first time its grid is asked
// for, unless it was loaded up front.
//
// Not a `TypedDecisions`: the answer needs the image rows as a static input of the decoder and the logits at every
// slot of one pass, which no engine of the kit returns together, so this drives the low-level runtime itself. Not
// behind `/v1/systemone` either: the wire form has no image field.

import CoreAI
import CoreGraphics
import Foundation
import Tokenizers

/// A downloadable decider-2b-vision: the decoder bundle and one vision tower per grid, as paths inside one Hub repo.
@available(macOS 27, iOS 27, *)
public struct DeciderVisionModelID: Sendable, Hashable {
    /// The decoder's LanguageBundle directory (metadata.json, the `.aimodel`, tokenizer/).
    public let decoder: ModelID
    /// The tower directory for each grid.
    public let towers: [KitVisionDecider.Grid: ModelID]

    public init(decoder: ModelID, towers: [KitVisionDecider.Grid: ModelID]) {
        self.decoder = decoder
        self.towers = towers
    }

    /// decider-2b-vision at the revision the zoo gated: the decoder with int8 linears (fp16 in layers 0, 2 and 5,
    /// the tied head fp16), 2,664 MB, and the towers with fp16 weights and fp32 math, 660 MB (`g256`) and 663 MB
    /// (`g448`). The same `.aimodel` files serve the Mac and the iPhone; an iPhone app needs the
    /// increased-memory-limit entitlement to specialize the decoder. The pin moves with the catalog's.
    public static let decider2bVision: DeciderVisionModelID = {
        let repo = "mlboydaisuke/decider-2b-vision-CoreAI"
        let revision = "4948e3231035df85c7b03c90b766eb17a08d9451"
        return DeciderVisionModelID(
            decoder: ModelID(repo, path: "gpu-pipelined/decider_2b_vision_decode_int8mix_pf16", revision: revision),
            towers: [
                .g256: ModelID(repo, path: "gpu-pipelined/decider_2b_vision_g256_vision_fp16w32", revision: revision),
                .g448: ModelID(repo, path: "gpu-pipelined/decider_2b_vision_g448_vision_fp16w32", revision: revision),
            ])
    }()

    /// Presets by catalog id. The catalog carries one variant path per entry (the decoder), while this model is a
    /// decoder plus a tower per grid, so every `visionDecision` entry pairs with a preset here.
    static let byCatalogID: [String: DeciderVisionModelID] = ["decider-2b-vision": .decider2bVision]

    /// A copy with the decoder and every tower pinned to a Hub revision (nil = unchanged), so a catalog entry's pin
    /// covers all of them.
    public func pinned(_ revision: String?) -> DeciderVisionModelID {
        guard let revision else { return self }
        return DeciderVisionModelID(
            decoder: decoder.pinned(revision), towers: towers.mapValues { $0.pinned(revision) })
    }
}

/// decider-2b-vision behind `decide(image:state:questions:)`. Calls on one instance serialize: the decoder's states
/// are the instance's.
@available(macOS 27, iOS 27, *)
public actor KitVisionDecider {
    /// The vision tower's fixed grid.
    public enum Grid: Int, Sendable, Hashable, CaseIterable, CustomStringConvertible {
        /// A 256×256 tile, 8×8 merged = 64 image tokens: game frames, speed.
        case g256 = 8
        /// A 448×448 tile, 14×14 merged = 196 image tokens: photos.
        case g448 = 14

        /// The merged grid's side.
        public var side: Int { rawValue }
        /// Pixels per side of the square the image is resized to.
        public var tile: Int { DeciderVisionPreprocessor.tileSide(grid: rawValue) }
        /// Image tokens in the prompt.
        public var imageTokens: Int { rawValue * rawValue }
        public var description: String { "g\(tile)" }

        /// The grid whose tile is `tile` pixels (256 or 448).
        public init?(tile: Int) {
            guard let grid = Grid.allCases.first(where: { $0.tile == tile }) else { return nil }
            self = grid
        }
    }

    /// The row a call is read as, for checking the rendering against a reference without the graphs.
    public struct PromptRow: Sendable, Equatable {
        /// Decoder ids: the image block as V + k (k = 0 ..< N, V = 248,320), the text as the tokenizer writes it.
        public let ids: [Int]
        /// The answer slots, ascending, one per question.
        public let slots: [Int]
        /// The grid of the image block; nil for a text-only row.
        public let grid: Grid?

        /// The same row in the author's processor form: every image token as `<|image_pad|>` (248056).
        public var processorIDs: [Int] { DeciderVisionPromptRenderer.processorIDs(ids) }
    }

    /// Everything one call read, for a gate: the row, the letter logits and probabilities at every slot, the
    /// full-vocabulary top-1, the decoder's calls and each stage's time.
    public struct Readout: Sendable {
        public let row: PromptRow
        /// Per question, the logits of its letters (A… over its options), widened from the decoder's fp16.
        public let letterLogits: [[Float]]
        /// Per question, the option probabilities (the letters' softmax at the bundle's temperature).
        public let probabilities: [[Double]]
        /// Per question, the id with the largest logit over the whole vocabulary at its slot (the lowest id on a
        /// tie): the answer letter when the model answers in the prompt's terms.
        public let fullVocabularyTop1: [Int]
        /// Per question, whether every logit at its slot was finite.
        public let finite: [Bool]
        /// Per question, the function whose call produced its slot's logits: "prefill" or "main".
        public let readFrom: [String]
        /// The answers, in question order; every one carries the call's timing.
        public let answers: [Decision.Answer]
        /// Decoder calls: `prefill` (16 tokens each) and `main` (one token each).
        public let prefillCalls: Int
        public let mainCalls: Int
        /// Seconds per stage: "decode" (image → RGB), "resize", "patches", "tower", "tokenize", "static_inputs",
        /// "decoder", "readout", and "wall" for the whole call.
        public let seconds: [String: Double]
    }

    /// What the decoder bundle's metadata.json says about the read-out.
    struct Metadata: Sendable {
        let name: String
        /// The graph's file name (`assets.main`).
        let asset: String
        let vocab: Int
        let maxContext: Int
        let prefillChunk: Int?
        let letterIDs: [Int]
        let temperature: Double

        init(bundle: URL) throws {
            let url = bundle.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: url),
                let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                let language = j["language"] as? [String: Any], let assets = j["assets"] as? [String: Any],
                let asset = assets["main"] as? String, let vocab = language["vocab_size"] as? Int,
                let context = language["max_context_length"] as? Int
            else { throw DeciderVisionError.bundle("\(url.path): no language / assets block") }
            let decision = j["decision"] as? [String: Any] ?? [:]
            name = j["name"] as? String ?? bundle.lastPathComponent
            self.asset = asset
            self.vocab = vocab
            maxContext = context
            prefillChunk = language["prefill_chunk"] as? Int
            letterIDs = decision["letter_ids"] as? [Int] ?? Array(32...41)
            temperature = decision["temperature"] as? Double ?? 1.0
        }
    }

    /// Where a tower that was not loaded up front comes from.
    private struct Source: Sendable {
        let model: DeciderVisionModelID
        let store: ModelStore
        let progress: (@Sendable (DownloadProgress) -> Void)?
    }

    /// Options a choice may list (the letters A–J); a score takes as many levels.
    public static let maxOptions = DeciderVisionPromptRenderer.maxOptions

    /// The catalog id, or the decoder bundle's directory name for a local bundle.
    public nonisolated let id: String
    /// The decoder bundle's name, from its metadata.
    public nonisolated let modelName: String
    /// The temperature the letter logits are read at: the bundle's (1).
    public nonisolated let temperature: Double
    /// The bundle's tokenizer, the one every row is rendered with.
    public nonisolated var tokenizer: any Tokenizer { renderer.tokenizer }

    nonisolated let metadata: Metadata
    nonisolated let renderer: DeciderVisionPromptRenderer
    private let decoder: DeciderVisionDecoder
    private var towers: [Grid: DeciderVisionTower]
    private let source: Source?
    /// Held for a whole call: each graph `await` lets another call into the actor, and two calls would share one
    /// set of decoder states.
    private let lock = AsyncMutex()
    /// Timing of the last call.
    public private(set) var lastTiming: Decision.Timing?

    /// The grids whose towers are loaded.
    public var loadedGrids: [Grid] { towers.keys.sorted { $0.rawValue < $1.rawValue } }

    /// Loads decider-2b-vision by its catalog id (`kind: visionDecision`), downloading on first use: the decoder and
    /// the towers for `grids`. The other grid's tower downloads the first time a decision asks for it.
    public init(
        catalog id: String,
        grids: Set<Grid> = [.g256],
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .visionDecision)
        guard let model = DeciderVisionModelID.byCatalogID[entry.id] else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(
            model: model.pinned(entry.revision), grids: grids, store: store, downloadProgress: downloadProgress,
            id: id)
    }

    /// Downloads the decoder and the towers for `grids` from the Hub (if needed) and loads them.
    public init(
        model: DeciderVisionModelID = .decider2bVision,
        grids: Set<Grid> = [.g256],
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        try await self.init(
            model: model, grids: grids, store: store, downloadProgress: downloadProgress,
            id: model.decoder.path.map { ($0 as NSString).lastPathComponent } ?? model.decoder.repo)
    }

    private init(
        model: DeciderVisionModelID, grids: Set<Grid>, store: ModelStore,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)?, id: String
    ) async throws {
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        var towerURLs: [Grid: URL] = [:]
        for grid in grids.sorted(by: { $0.rawValue < $1.rawValue }) {
            guard let tower = model.towers[grid] else { throw DeciderVisionError.towerNotLoaded(grid: grid.description) }
            towerURLs[grid] = try await store.download(tower, progress: downloadProgress)
        }
        try await self.init(
            decoderAt: decoderURL, towersAt: towerURLs, id: id,
            source: Source(model: model, store: store, progress: downloadProgress))
    }

    /// Loads a local decoder bundle directory (metadata.json, the `.aimodel` or an `.aimodelc` compiled for this
    /// device, tokenizer/) and a tower per grid (the graph itself, or the directory holding it). A decision at a
    /// grid without a tower throws `DeciderVisionError.towerNotLoaded`.
    public init(decoderAt decoderURL: URL, towersAt towerURLs: [Grid: URL]) async throws {
        try await self.init(decoderAt: decoderURL, towersAt: towerURLs, id: decoderURL.lastPathComponent, source: nil)
    }

    private init(decoderAt decoderURL: URL, towersAt towerURLs: [Grid: URL], id: String, source: Source?) async throws {
        let metadata = try Metadata(bundle: decoderURL)
        let tokenizer = try await AutoTokenizer.from(modelFolder: decoderURL.appendingPathComponent("tokenizer"))
        let letters = DeciderVisionPromptRenderer.letters
        guard metadata.letterIDs.count == letters.count else {
            throw DeciderVisionError.bundle("\(metadata.letterIDs.count) letter ids for \(letters.count) letters")
        }
        for (letter, id) in zip(letters, metadata.letterIDs) {
            let got = tokenizer.encode(text: letter, addSpecialTokens: false)
            guard got == [id] else {
                throw DeciderVisionError.bundle("letter \(letter) encodes to \(got), the metadata says \(id)")
            }
        }
        var towers: [Grid: DeciderVisionTower] = [:]
        for (grid, url) in towerURLs.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            towers[grid] = try await Self.loadTower(grid, at: url)
        }
        let stem = (metadata.asset as NSString).deletingPathExtension
        guard let graph = try GraphBundle.graph(named: stem, in: decoderURL) else {
            throw DeciderVisionError.bundle("\(decoderURL.lastPathComponent) holds no \(metadata.asset)")
        }
        self.decoder = try await DeciderVisionDecoder(
            contentsOf: graph, vocab: metadata.vocab, maxContext: metadata.maxContext,
            prefillChunk: metadata.prefillChunk, options: Self.decoderOptions(for: graph))
        self.id = id
        self.modelName = metadata.name
        self.temperature = metadata.temperature
        self.metadata = metadata
        self.renderer = DeciderVisionPromptRenderer(tokenizer: tokenizer)
        self.towers = towers
        self.source = source
    }

    /// The runtime specializes a JIT `.aimodel` here with the settings the zoo's gates ran it with: the GPU, and
    /// frequent reshapes for the decoder (its `position_ids` grows by one or sixteen per call). A compiled graph
    /// loads as compiled.
    static func decoderOptions(for graph: URL) -> SpecializationOptions {
        guard graph.pathExtension == "aimodel" else { return .default }
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        return options
    }

    static func towerOptions(for graph: URL) -> SpecializationOptions {
        graph.pathExtension == "aimodel" ? SpecializationOptions(preferredComputeUnitKind: .gpu) : .default
    }

    private static func loadTower(_ grid: Grid, at url: URL) async throws -> DeciderVisionTower {
        let graph = try GraphBundle.resolve(in: url)
        return try await DeciderVisionTower(contentsOf: graph, grid: grid.side, options: towerOptions(for: graph))
    }

    /// Loads the towers for `grids` now (downloading them if needed), so the first decision at each does not pay
    /// for it.
    public func prepare(_ grids: Grid...) async throws {
        try await lock.withLock {
            for grid in grids { _ = try await tower(for: grid) }
        }
    }

    /// The loaded tower for `grid`, downloading and loading it first when this decider has a source for it.
    private func tower(for grid: Grid) async throws -> DeciderVisionTower {
        if let tower = towers[grid] { return tower }
        guard let source, let model = source.model.towers[grid] else {
            throw DeciderVisionError.towerNotLoaded(grid: grid.description)
        }
        let url = try await source.store.download(model, progress: source.progress)
        let tower = try await Self.loadTower(grid, at: url)
        towers[grid] = tower
        return tower
    }

    // MARK: - Decide

    /// Every question about the image and the state, read in one pass; the answers in question order. `image`
    /// nil reads a text-only row through the same decoder. The image's pixels are read as stored: a camera photo's
    /// EXIF orientation is not applied, as the author's code applies none.
    public func decide(
        image: CGImage?, state: String, questions: [Decision.Question], grid: Grid = .g256
    ) async throws -> [Decision.Answer] {
        guard !questions.isEmpty else { return [] }
        return try await readout(image: image, state: state, questions: questions, grid: grid).answers
    }

    /// Questions keyed by an id of the caller's choosing. They are laid out in the row in key order (sorted), so
    /// the same keys always read the same prompt.
    public func decide(
        image: CGImage?, state: String, questions: [String: Decision.Question], grid: Grid = .g256
    ) async throws -> [String: Decision.Answer] {
        let keys = questions.keys.sorted()
        let answers = try await decide(image: image, state: state, questions: keys.map { questions[$0]! }, grid: grid)
        return Dictionary(uniqueKeysWithValues: zip(keys, answers))
    }

    /// One question about the image and the state.
    public func decide(
        image: CGImage?, state: String, question: Decision.Question, grid: Grid = .g256
    ) async throws -> Decision.Answer {
        try await readout(image: image, state: state, questions: [question], grid: grid).answers[0]
    }

    /// The whole call with what it read (the row, every slot's letter logits and probabilities, the
    /// full-vocabulary top-1, the decoder's calls and each stage's time); `decide` returns its `answers`.
    public func readout(
        image: CGImage?, state: String, questions: [Decision.Question], grid: Grid = .g256
    ) async throws -> Readout {
        try await lock.withLock {
            try await read(image: image, state: state, questions: questions, grid: grid)
        }
    }

    /// The row a call would read, rendered with the bundle's tokenizer only: `grid` nil for a text-only row.
    public nonisolated func promptRow(state: String, questions: [Decision.Question], grid: Grid?) throws -> PromptRow {
        let row = try renderer.row(state: state, questions: questions, grid: grid?.side)
        return PromptRow(ids: row.ids, slots: row.slots, grid: grid)
    }

    private func read(
        image: CGImage?, state: String, questions: [Decision.Question], grid: Grid
    ) async throws -> Readout {
        // A malformed question fails here, before any graph runs, and a missing tower loads before the clock starts.
        let tRow = ContinuousClock.now
        let row = try renderer.row(state: state, questions: questions, grid: image == nil ? nil : grid.side)
        let tokenizeSeconds = deciderVisionSeconds(since: tRow)
        guard row.ids.count <= metadata.maxContext else {
            throw DecisionError.promptTooLong(tokens: row.ids.count, max: metadata.maxContext)
        }
        var loadedTower: DeciderVisionTower? = nil
        if image != nil { loadedTower = try await tower(for: grid) }

        let t0 = ContinuousClock.now
        var seconds: [String: Double] = ["tokenize": tokenizeSeconds]
        var embeds: [Float]? = nil
        var imageSeconds: Double? = nil
        if let image, let tower = loadedTower {
            let prepared = try DeciderVisionPreprocessor.prepare(image, grid: grid.side)
            seconds["decode"] = prepared.decodeSeconds
            seconds["resize"] = prepared.resizeSeconds
            seconds["patches"] = prepared.patchSeconds
            let t = ContinuousClock.now
            embeds = try await tower.encode(patches: prepared.patches)
            seconds["tower"] = deciderVisionSeconds(since: t)
            imageSeconds = deciderVisionSeconds(since: t0)
        }
        let tStatic = ContinuousClock.now
        let inputs = try decoder.staticInputs(
            towerEmbeds: embeds, grid: row.grid, start: row.ropeShiftStart, amount: row.ropeShiftAmount)
        seconds["static_inputs"] = deciderVisionSeconds(since: tStatic)
        let pass = try await decoder.run(ids: row.ids, slots: row.slots, inputs: inputs)
        seconds["decoder"] = pass.seconds

        let tReadout = ContinuousClock.now
        var letterLogits: [[Float]] = [], probabilities: [[Double]] = []
        var top1: [Int] = [], finite: [Bool] = [], readFrom: [String] = []
        for (k, slot) in pass.slots.enumerated() {
            let letters = metadata.letterIDs.prefix(row.optionCounts[k]).map { slot.logits[$0] }
            letterLogits.append(letters)
            probabilities.append(
                DeciderVisionNumerics.softmax(letters.map(Double.init), temperature: metadata.temperature))
            top1.append(DeciderVisionNumerics.argmax(slot.logits))
            finite.append(slot.logits.allSatisfy(\.isFinite))
            readFrom.append(slot.readFrom)
        }
        seconds["readout"] = deciderVisionSeconds(since: tReadout)
        let wall = deciderVisionSeconds(since: t0) + tokenizeSeconds
        seconds["wall"] = wall

        let timing = Decision.Timing(
            promptTokens: row.ids.count, reusedTokens: 0, seconds: wall, imageSeconds: imageSeconds,
            decoderSeconds: pass.seconds)
        lastTiming = timing
        let answers = zip(questions, probabilities).map { question, p in
            DecisionPrompt.answer(for: question, probabilities: p, timing: timing)
        }
        let prefillCalls = pass.callIsPrefill.filter { $0 }.count
        return Readout(
            row: PromptRow(ids: row.ids, slots: row.slots, grid: image == nil ? nil : grid),
            letterLogits: letterLogits, probabilities: probabilities, fullVocabularyTop1: top1, finite: finite,
            readFrom: readFrom, answers: answers, prefillCalls: prefillCalls,
            mainCalls: pass.callIsPrefill.count - prefillCalls, seconds: seconds)
    }
}
