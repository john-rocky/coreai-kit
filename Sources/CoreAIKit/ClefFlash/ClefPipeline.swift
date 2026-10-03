// From the model zoo's apps/ClefFlash/Sources/ClefFlash/ClefDecider.swift (5ef2247, sha256 e52a1d3b3cd4, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// ClefPipeline — a SystemOne request (+ one image at a fixed grid) -> the SystemOne response, the order decide.py runs:
//
//   let decider = try await ClefPipeline(assets: .init(decoderBundle: bundleDir, decoder: aimodelcURL, head: headDir,
//                                                     headAsset: headAimodelcURL, table: tableURL,
//                                                     towers: [.g448: towerURL]))
//   let response = try await decider.decide(request: try ClefRequest(data: json), image: cgImage, grid: .g448)
//
//   request ──ClefPromptBuilder──> ids, question / option spans, the decoder's static inputs
//   image ──ClefImagePreprocess (Pillow's bicubic)──> patches [4 G^2, 1536] ──tower──> image_embeds [G^2, 4096]
//   decoder: fresh zero states, ceil(T / 64) calls of `main` -> hidden [T, 4096] fp16
//   head arrays (span means, last token, lexical table rows, membership, types, masks) padded to the bucket
//   head `t<bucket>` -> one logit per option -> per question a float32 softmax -> the response (4 decimals)
//
// Contract checks at load: the bundle metadata (prefill chunk, context, vocabulary, image rows, prompt lengths), the
// tokenizer's special ids and the prompt's fixed pieces, the decoder's / head buckets' / towers' input, output and
// state names, shapes and types, the table's byte count. An asset that differs fails there, not in a probability.

import CoreAI
import CoreGraphics
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitClefDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 27, iOS 27, *)
final class ClefPipeline: @unchecked Sendable {
    enum Grid: Int, Sendable, CaseIterable, CustomStringConvertible {
        case g256 = 8
        case g448 = 14

        var side: Int { rawValue }
        var tile: Int { ClefImagePreprocess.tileSide(grid: rawValue) }
        var merged: ClefMergedGrid { ClefMergedGrid(h: rawValue, w: rawValue) }
        var description: String { "g\(tile)" }

        init?(tile: Int) {
            guard let g = Grid.allCases.first(where: { $0.tile == tile }) else { return nil }
            self = g
        }
    }

    /// Where the assets are. `decoder` / `headAsset` = the asset to load (`.aimodelc` AOT or `.aimodel`); nil = the
    /// bundle's `.aimodel`.
    struct Assets: Sendable {
        var decoderBundle: URL
        var decoder: URL?
        var head: URL
        var headAsset: URL?
        var table: URL
        var towers: [Grid: URL]

        init(decoderBundle: URL, decoder: URL? = nil, head: URL, headAsset: URL? = nil, table: URL,
                    towers: [Grid: URL] = [:]) {
            self.decoderBundle = decoderBundle
            self.decoder = decoder
            self.head = head
            self.headAsset = headAsset
            self.table = table
            self.towers = towers
        }
    }

    /// What the bundles' metadata.json files say.
    struct Metadata: Sendable {
        let name: String
        let asset: String
        let vocab: Int
        let maxContext: Int
        let chunk: Int
        let nImageMax: Int
        let prefixTokens: Int
        let suffixTokens: Int
        let headName: String
        let headAsset: String
        let buckets: [ClefHead.Bucket]

        init(bundle: URL, head: URL) throws {
            let url = bundle.appendingPathComponent("metadata.json")
            let j = try ClefJSONParser.parse(Data(contentsOf: url))
            guard let lang = j["language"], let asset = j["assets"]?["main"]?.string,
                  let vocab = lang["vocab_size"]?.intValue, let ctx = lang["max_context_length"]?.intValue,
                  let chunk = lang["prefill_chunk"]?.intValue, let nmax = j["vision"]?["n_image_max"]?.intValue,
                  let prompt = j["decision"]?["prompt"], let pre = prompt["prefix_tokens"]?.intValue,
                  let suf = prompt["suffix_tokens"]?.intValue
            else { throw ClefFlashError.bundle("\(url.path): no language / assets / vision / decision.prompt block") }
            name = j["name"]?.string ?? bundle.lastPathComponent
            self.asset = asset
            self.vocab = vocab
            maxContext = ctx
            self.chunk = chunk
            nImageMax = nmax
            prefixTokens = pre
            suffixTokens = suf
            let hurl = head.appendingPathComponent("metadata.json")
            let h = try ClefJSONParser.parse(Data(contentsOf: hurl))
            guard let hm = h["head"], hm["shape"]?.string == "bucket", let b = hm["buckets"]?.array,
                  let hasset = h["assets"]?["main"]?.string
            else { throw ClefFlashError.bundle("\(hurl.path): not a bucket head (head.shape / head.buckets / assets.main)") }
            headName = h["name"]?.string ?? head.lastPathComponent
            headAsset = hasset
            buckets = try b.map { e in
                guard let a = e.array, a.count == 2, let f = a[0].string, let t = a[1].intValue else {
                    throw ClefFlashError.bundle("\(hurl.path): bucket \(ClefPythonJSON.dumps(e))")
                }
                return ClefHead.Bucket(function: f, tokens: t)
            }
        }
    }

    /// Everything one decision did, for gates and timing.
    struct Trace: Sendable {
        let row: ClefPromptRow
        let prepared: ClefImagePreprocess.Prepared?
        /// The tower output, or the image rows given to `trace(embeds:)`.
        let imageRows: [Float]?
        let pass: ClefDecoder.Pass
        let bucket: ClefHead.Bucket
        /// One logit per real option, in the head's option order.
        let logits: [Float]
        let probabilities: [[Float]]
        let response: ClefJSON
        let seconds: [String: Double]
    }

    let metadata: Metadata
    let prompt: ClefPromptBuilder
    let decoder: ClefDecoder
    let head: ClefHead
    let table: ClefLexicalTable
    private(set) var towers: [Grid: ClefVisionTower]
    let tokenizerLoadSeconds: Double

    init(assets: Assets, decoderOptions: SpecializationOptions = .default,
                headOptions: SpecializationOptions = .default, towerOptions: SpecializationOptions = .default) async throws
    {
        let meta = try Metadata(bundle: assets.decoderBundle, head: assets.head)
        guard meta.vocab == ClefPromptBuilder.vocab, meta.nImageMax == ClefDecoder.imageRows else {
            throw ClefFlashError.contract("bundle vocab \(meta.vocab) / n_image_max \(meta.nImageMax) (want \(ClefPromptBuilder.vocab) / \(ClefDecoder.imageRows))")
        }
        let t0 = ContinuousClock.now
        prompt = try await ClefPromptBuilder.load(tokenizerFolder: assets.decoderBundle.appendingPathComponent("tokenizer"),
                                              nImageMax: meta.nImageMax, prefixTokens: meta.prefixTokens,
                                              suffixTokens: meta.suffixTokens)
        tokenizerLoadSeconds = clefSecondsSince(t0)
        table = try ClefLexicalTable(contentsOf: assets.table)
        var loaded: [Grid: ClefVisionTower] = [:]
        for (g, url) in assets.towers.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            loaded[g] = try await ClefVisionTower(contentsOf: url, grid: g.side, options: towerOptions)
        }
        towers = loaded
        head = try await ClefHead(contentsOf: assets.headAsset ?? assets.head.appendingPathComponent(meta.headAsset),
                                  buckets: meta.buckets, options: headOptions)
        decoder = try await ClefDecoder(contentsOf: assets.decoder ?? assets.decoderBundle.appendingPathComponent(meta.asset),
                                        chunk: meta.chunk, maxContext: meta.maxContext, options: decoderOptions)
        metadata = meta
    }

    /// The response for one request (`image == nil` = a text-only row).
    func decide(request: ClefRequest, image: CGImage? = nil, grid: Grid = .g448) async throws -> ClefJSON {
        try await trace(request: request, image: image, grid: grid).response
    }

    /// The whole decision with its intermediate values and times. `embeds` + `embedsGrid` replace the image and
    /// the tower with given image rows (float32 [N * 4096], N = embedsGrid.count): the gate's way to feed the
    /// oracle's rows, e.g. at a native grid no tower graph has.
    func trace(request: ClefRequest, image: CGImage? = nil, grid: Grid = .g448, embeds: [Float]? = nil,
                      embedsGrid: ClefMergedGrid? = nil) async throws -> Trace
    {
        let t0 = ContinuousClock.now
        var seconds: [String: Double] = [:]
        var prepared: ClefImagePreprocess.Prepared? = nil
        var rows: [Float]? = nil
        var merged: ClefMergedGrid? = nil
        if let embeds {
            guard let g = embedsGrid else { throw ClefFlashError.contract("embeds without a grid") }
            rows = embeds
            merged = g
        } else if let image {
            guard let tower = towers[grid] else { throw ClefFlashError.bundle("no \(grid) tower loaded") }
            let p = try ClefImagePreprocess.prepare(image, grid: grid.side)
            seconds["decode_rgb"] = p.decodeSeconds
            seconds["resize"] = p.resizeSeconds
            seconds["patches"] = p.patchSeconds
            let t = ContinuousClock.now
            rows = try await tower.encode(patches: p.patches)
            seconds["tower"] = clefSecondsSince(t)
            prepared = p
            merged = grid.merged
        }
        let t1 = ContinuousClock.now
        let row = try prompt.build(request, grid: merged)
        seconds["tokenize"] = clefSecondsSince(t1)
        let t2 = ContinuousClock.now
        let inputs = try decoder.staticInputs(embeds: rows, row: row)
        seconds["static_inputs"] = clefSecondsSince(t2)
        let pass = try await decoder.run(ids: row.decoderIDs, inputs: inputs)
        seconds["decoder"] = pass.seconds
        seconds["state_reset"] = pass.resetSeconds
        let t3 = ContinuousClock.now
        let arrays = try head.arrays(hidden16: pass.hidden, row: row, table: table)
        seconds["head_arrays"] = clefSecondsSince(t3)
        let t4 = ContinuousClock.now
        let logits = try await head.run(arrays)
        seconds["head"] = clefSecondsSince(t4)
        let t5 = ContinuousClock.now
        let probs = ClefHead.questionProbs(logits, layout: arrays.layout)
        seconds["softmax"] = clefSecondsSince(t5)
        let t6 = ContinuousClock.now
        let perQuestion = zip(row.questions, probs).map { q, p in Array(zip(q.optionIDs, p)).map { (id: $0.0, p: $0.1) } }
        let response = try ClefResponse.make(request, probabilities: perQuestion, tokens: row.ids.count)
        seconds["response"] = clefSecondsSince(t6)
        seconds["wall"] = clefSecondsSince(t0)
        return Trace(row: row, prepared: prepared, imageRows: rows, pass: pass, bucket: arrays.bucket,
                     logits: Array(logits[0..<arrays.options]), probabilities: probs, response: response, seconds: seconds)
    }

    /// A tower loaded after init (kit addition): `KitClefDecider` downloads a grid's tower the first time a decision
    /// asks for it.
    func loadTower(_ grid: Grid, contentsOf url: URL, options: SpecializationOptions) async throws {
        towers[grid] = try await ClefVisionTower(contentsOf: url, grid: grid.side, options: options)
    }
}

#endif
