// From the model zoo's apps/D1Omni/Sources/D1Omni/D1Omni.swift (f9e0e09, sha256 e24b2589aec3), identifiers prefixed D1 for the kit.
// D1OmniPipeline — a SystemOne request -> the SystemOne response on d1-omni-600M's decision graphs, in host.py's order:
//
//   let d1 = try await D1OmniPipeline(folders: D1OmniPipeline.folders(macos: dir))   // the .aimodel of every bucket (JIT)
//   let response = try await d1.systemOne(state: state, questions: questions)
//
//   request ──D1Prompt.rows (the publisher's encode, per mode)──> one row per question
//           ──D1GraphInputs at bucket_for(P + n)──> the bucket's graph (one call per row) ──> scores[P + markers]
//           ──D1Readout.probabilities (÷ T on text rows, float32 softmax, noul flipped)──> D1Readout.response
//
// Every bucket folder holds metadata.json (the decision block: the graph's contract, the nine token ids, the
// temperatures, max_length / image_text_length / audio_text_length / min_text_positions) and the bucket's `.aimodel`;
// the tokenizer folder holds tokenizer.json + tokenizer_config.json. What differs from the contract fails at load. A
// bucket's graph loads on its first row (or every one at init with `preload`). `assets` replaces a bucket's `.aimodel`
// with another file to load (the Mac's AOT `.aimodelc`, compiled from the same `.aimodel`).
//
// Text is round 8. Images and audio are round 9: `systemOne(state:questions:images:)` decodes each image, cuts and
// resizes its crops, runs the vision graph once per crop and concatenates the prefixes (D1ImagePreprocess, D1VisionGraph);
// `systemOne(state:questions:audio:)` reads the clip, makes the mel and the masks at its bucket and runs that bucket's
// audio graph (D1AudioPreprocess, D1AudioGraph); the media folders are `vision-<precision>/` (with position_table.f32) and
// `audio-<precision>-<sec>s/` (with mel_filters_128x257_f32.bin) beside the decision buckets (`MediaFolders`). The
// rows, the graph inputs and the readout are the same for every mode. One decision at a time per instance.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
final class D1OmniPipeline: @unchecked Sendable {
    struct Bucket: Sendable {
        let length: Int
        let folder: URL
        /// what loads: `assets[L]`, or the folder's `.aimodel` (metadata.json assets.main)
        let asset: URL
    }

    /// One row's decision: the marker logits (float32, the graph's), the reported probabilities, the bucket, seconds.
    struct RowResult: Sendable {
        let row: D1Row
        let bucket: Int
        let logits: [Float]
        let probabilities: [Float]
        /// graph inputs, the call, the readout
        let seconds: (inputs: Double, graph: Double, readout: Double)
    }

    let buckets: [Int: Bucket]
    let lengths: [Int]
    let config: D1Config
    let tokenizer: D1Tokenizer
    let options: SpecializationOptions
    let media: MediaFolders?
    private var graphs: [Int: D1DecisionGraph] = [:]
    private var visionGraph: D1VisionGraph? = nil
    private var audioGraphs: [Int: D1AudioGraph] = [:]
    private var positionTable: [Float]? = nil
    private var melFilterbank: [Float]? = nil

    /// The media bundles beside the decision buckets: the vision folder and one audio folder per clip bucket, each
    /// with metadata.json naming its `.aimodel` (`assets.main`); `assets` replaces a graph's file ("vision",
    /// "audio-<sec>") with another one to load (the Mac's AOT `.aimodelc`).
    struct MediaFolders: Sendable {
        let vision: URL?
        let audio: [Int: URL]
        let assets: [String: URL]

        init(vision: URL?, audio: [Int: URL], assets: [String: URL] = [:]) {
            self.vision = vision
            self.audio = audio
            self.assets = assets
        }

        /// `vision-<precision>/` and `audio-<precision>-<sec>s/` under a `macos/` directory (the Hugging Face layout and
        /// the conversion's work folder name them alike).
        static func find(macos dir: URL, precision: String = "fp16", assets: [String: URL] = [:]) -> MediaFolders {
            let fm = FileManager.default
            let v = dir.appendingPathComponent("vision-\(precision)")
            var a: [Int: URL] = [:]
            for sec in D1AudioPreprocess.bucketSeconds {
                let f = dir.appendingPathComponent("audio-\(precision)-\(sec)s")
                if fm.fileExists(atPath: f.appendingPathComponent("metadata.json").path) { a[sec] = f }
            }
            return MediaFolders(vision: fm.fileExists(atPath: v.appendingPathComponent("metadata.json").path) ? v : nil,
                                audio: a, assets: assets)
        }

        /// The file a media graph loads: `assets[key]`, or the folder's metadata.json `assets.main`.
        func asset(_ key: String, folder: URL) throws -> URL {
            if let u = assets[key] { return u }
            let j = try D1JSONParser.parse(Data(contentsOf: folder.appendingPathComponent("metadata.json")))
            guard let main = j["assets"]?["main"]?.string else { throw D1OmniError.bundle("\(folder.path): no assets.main") }
            return folder.appendingPathComponent(main)
        }
    }

    /// The bucket folders under a `macos/` directory: `decide-<precision>-L<L>` (the Hugging Face layout) or
    /// `<precision>-L<L>` (the conversion's work folder).
    static func folders(macos dir: URL, precision: String = "fp16") throws -> [Int: URL] {
        var out: [Int: URL] = [:]
        for name in try FileManager.default.contentsOfDirectory(atPath: dir.path) {
            for prefix in ["decide-\(precision)-L", "\(precision)-L"] where name.hasPrefix(prefix) {
                if let L = Int(name.dropFirst(prefix.count)), out[L] == nil {
                    out[L] = dir.appendingPathComponent(name)
                }
            }
        }
        if out.isEmpty { throw D1OmniError.bundle("\(dir.path): no decide-\(precision)-L<L> or \(precision)-L<L> folder") }
        return out
    }

    /// `folders`: L -> the bucket's folder; `assets`: L -> the file to load instead of the folder's `.aimodel`;
    /// `tokenizerFolder`: nil = the smallest bucket's `tokenizer/`.
    init(folders: [Int: URL], assets: [Int: URL] = [:], tokenizerFolder: URL? = nil,
                options: SpecializationOptions = D1DecisionGraph.gpuOptions, preload: Bool = false,
                media: MediaFolders? = nil) async throws {
        self.media = media
        var buckets: [Int: Bucket] = [:]
        var config: D1Config? = nil
        var tokenIDs: [String: Int]? = nil
        for (L, folder) in folders.sorted(by: { $0.key < $1.key }) {
            let url = folder.appendingPathComponent("metadata.json")
            let j = try D1JSONParser.parse(Data(contentsOf: url))
            guard let d = j["decision"], let main = j["assets"]?["main"]?.string, d["seq_len"]?.intValue == L,
                  let ids = d["token_ids"]?.members, let temps = d["temperatures"]?.members,
                  let maxLength = d["max_length"]?.intValue, let image = d["image_text_length"]?.intValue,
                  let audio = d["audio_text_length"]?.intValue, let minText = d["min_text_positions"]?.intValue,
                  d["prefix_hidden"]?.intValue == D1DecisionGraph.hidden
            else { throw D1OmniError.bundle("\(url.path): no decision block for L = \(L) (seq_len, token_ids, temperatures, lengths)") }
            let c = D1Config(maxLength: maxLength, imageTextLength: image, audioTextLength: audio, minTextPositions: minText,
                             temperatures: Dictionary(uniqueKeysWithValues: try temps.map { m in
                                 guard let t = m.value.double else { throw D1OmniError.bundle("\(url.path): temperature \(m.key)") }
                                 return (m.key, t)
                             }))
            let t = Dictionary(uniqueKeysWithValues: try ids.map { m in
                guard let i = m.value.intValue else { throw D1OmniError.bundle("\(url.path): token id \(m.key)") }
                return (m.key, i)
            })
            if let config, config.temperatures != c.temperatures || config.maxLength != c.maxLength
                || config.imageTextLength != c.imageTextLength || config.audioTextLength != c.audioTextLength
                || config.minTextPositions != c.minTextPositions {
                throw D1OmniError.bundle("\(url.path): the decision block differs from the other buckets'")
            }
            if let tokenIDs, tokenIDs != t { throw D1OmniError.bundle("\(url.path): token ids differ from the other buckets'") }
            config = c
            tokenIDs = t
            buckets[L] = Bucket(length: L, folder: folder, asset: assets[L] ?? folder.appendingPathComponent(main))
        }
        guard let config, let tokenIDs, let first = buckets.keys.min() else { throw D1OmniError.bundle("no bucket folders") }
        self.buckets = buckets
        self.lengths = buckets.keys.sorted()
        self.config = config
        self.options = options
        tokenizer = try await D1Tokenizer.load(folder: tokenizerFolder ?? buckets[first]!.folder.appendingPathComponent("tokenizer"),
                                               expected: tokenIDs)
        if preload { for L in lengths { _ = try await graph(L) } }
    }

    /// The bucket's graph, loaded on first use.
    func graph(_ L: Int) async throws -> D1DecisionGraph {
        if let g = graphs[L] { return g }
        guard let b = buckets[L] else { throw D1OmniError.graphLimit("no bucket of length \(L) (buckets \(lengths))") }
        let g = try await D1DecisionGraph(contentsOf: b.asset, length: L, options: options)
        graphs[L] = g
        return g
    }

    /// The loaded graphs, by length.
    var loaded: [Int: D1DecisionGraph] { graphs }

    /// host.py `request_rows` with this bundle's config.
    func rows(state: D1JSONValue?, questions: D1JSONValue, mode: D1Mode = .text, prefixLength: Int = 0) throws -> [D1Row] {
        try D1Prompt.rows(tokenizer, state: state, questions: questions, mode: mode, prefixLength: prefixLength, config: config)
    }

    /// host.py `bucket_for` over this bundle's buckets.
    func bucket(for row: D1Row) throws -> Int {
        guard let L = D1GraphInputs.bucket(positions: row.positions, buckets: lengths) else {
            throw D1OmniError.graphLimit("question '\(row.qid)': \(row.positions) positions, over the largest bucket \(lengths.last ?? 0)")
        }
        return L
    }

    /// One row through its bucket's graph (`prefix`: the media prefix [P * 1024] when P > 0) and the readout.
    func decide(_ row: D1Row, prefix: [Float]? = nil, bucket: Int? = nil) async throws -> RowResult {
        let L = try bucket ?? self.bucket(for: row)
        let g = try await graph(L)
        let t0 = ContinuousClock.now
        let x = try D1GraphInputs(row: row, length: L)
        let t1 = ContinuousClock.now
        let s = try await g.scores(x, prefix: prefix)
        let t2 = ContinuousClock.now
        let logits = x.markers.map { s[$0] }
        let p = D1Readout.probabilities(logits: logits, question: row.question, calibrate: row.calibrate, config: config)
        let t3 = ContinuousClock.now
        return RowResult(row: row, bucket: L, logits: logits, probabilities: p,
                        seconds: (Self.seconds(t0, t1), Self.seconds(t1, t2), Self.seconds(t2, t3)))
    }

    static func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
        let d = b - a
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    /// A text request's response (`system_one(state, questions)`): `questions` a {name: question} object.
    func systemOne(state: D1JSONValue?, questions: D1JSONValue) async throws -> D1JSONValue {
        let rows = try self.rows(state: state, questions: questions)
        var probs: [[Float]] = []
        for r in rows { probs.append(try await decide(r).probabilities) }
        return D1Readout.response(rows: rows, probabilities: probs)
    }

    /// An image or audio request's response, the media prefix [P * 1024] given (the vision / audio graphs' rows, in
    /// request order; round 9 builds them in Swift).
    func systemOne(state: D1JSONValue?, questions: D1JSONValue, mode: D1Mode, prefix: [Float]) async throws -> D1JSONValue {
        guard mode != .text, prefix.count % D1DecisionGraph.hidden == 0, !prefix.isEmpty else {
            throw D1OmniError.request("a media request needs its prefix (P x \(D1DecisionGraph.hidden) values)")
        }
        let rows = try self.rows(state: state, questions: questions, mode: mode, prefixLength: prefix.count / D1DecisionGraph.hidden)
        var probs: [[Float]] = []
        for r in rows { probs.append(try await decide(r, prefix: prefix).probabilities) }
        return D1Readout.response(rows: rows, probabilities: probs)
    }

    // MARK: - media (round 9)

    /// The vision graph, loaded on first use (with the bundle's position table).
    func vision() async throws -> D1VisionGraph {
        if let g = visionGraph { return g }
        guard let folder = media?.vision else { throw D1OmniError.bundle("no vision folder (MediaFolders.vision)") }
        let g = try await D1VisionGraph(contentsOf: try media!.asset("vision", folder: folder), options: options)
        visionGraph = g
        return g
    }

    /// The vision bundle's position table [16][16][768].
    func visionPositionTable() throws -> [Float] {
        if let t = positionTable { return t }
        guard let folder = media?.vision else { throw D1OmniError.bundle("no vision folder (MediaFolders.vision)") }
        let t = try D1ImagePreprocess.positionTable(contentsOf: folder.appendingPathComponent("position_table.f32"))
        positionTable = t
        return t
    }

    /// The audio graph of one clip bucket, loaded on first use.
    func audio(_ bucket: D1AudioBucket) async throws -> D1AudioGraph {
        if let g = audioGraphs[bucket.seconds] { return g }
        guard let folder = media?.audio[bucket.seconds] else { throw D1OmniError.bundle("no audio folder for the \(bucket.seconds) s bucket") }
        let g = try await D1AudioGraph(contentsOf: try media!.asset("audio-\(bucket.seconds)", folder: folder), bucket: bucket,
                                     options: options)
        audioGraphs[bucket.seconds] = g
        return g
    }

    /// The audio bundles' filterbank [128][257] (every bucket's file is the same; the first bucket's is read).
    func audioFilterbank() throws -> [Float] {
        if let f = melFilterbank { return f }
        guard let folder = media?.audio.sorted(by: { $0.key < $1.key }).first?.value else { throw D1OmniError.bundle("no audio folder") }
        let f = try D1AudioPreprocess.filterbank(contentsOf: folder.appendingPathComponent(D1AudioPreprocess.filterbankFile))
        melFilterbank = f
        return f
    }

    /// Every crop's inputs of every image, in request order (host.image_crops_inputs with the NumPy forms).
    func imageInputs(_ images: [D1RGBImage]) throws -> [D1CropInputs] {
        let table = try visionPositionTable()
        return try images.flatMap { try D1ImagePreprocess.crops($0).map { try D1ImagePreprocess.inputs($0, table: table) } }
    }

    /// The images' prefix [P * 1024]: each crop's first (ph / 2)(pw / 2) rows, crops in order, images in order.
    func imagePrefix(_ crops: [D1CropInputs]) async throws -> [Float] {
        let g = try await vision()
        var prefix: [Float] = []
        for c in crops { prefix += try await g.prefix(c) }
        return prefix
    }

    /// The clip's inputs at its bucket (host.audio_inputs with mel_numpy).
    func audioInputs(samples: [Int16]) throws -> D1AudioInputs {
        try D1AudioPreprocess.inputs(samples: samples, filterbank: try audioFilterbank())
    }

    /// The clip's prefix [P * 1024] from its bucket's graph.
    func audioPrefix(_ x: D1AudioInputs) async throws -> [Float] {
        try await audio(x.bucket).prefix(x)
    }

    /// An image request's response: the images (files, in request order) -> their prefix -> the image rows.
    func systemOne(state: D1JSONValue?, questions: D1JSONValue, images: [URL]) async throws -> D1JSONValue {
        let decoded = try images.map { try D1ImagePreprocess.decode(contentsOf: $0) }
        let prefix = try await imagePrefix(try imageInputs(decoded))
        let p = try D1MediaLength.imagePrefixLength(decoded.map { ($0.width, $0.height) })
        guard prefix.count == p * D1DecisionGraph.hidden else { throw D1OmniError.contract("an image prefix of \(prefix.count / D1DecisionGraph.hidden) rows, P = \(p)") }
        return try await systemOne(state: state, questions: questions, mode: .image, prefix: prefix)
    }

    /// An audio request's response: the clip (a 16 kHz mono 16-bit WAV) -> its prefix -> the audio rows.
    func systemOne(state: D1JSONValue?, questions: D1JSONValue, audio: URL) async throws -> D1JSONValue {
        try await systemOne(state: state, questions: questions, samples: try D1AudioPreprocess.samples(contentsOf: audio))
    }

    /// An audio request's response from the clip's int16 samples.
    func systemOne(state: D1JSONValue?, questions: D1JSONValue, samples: [Int16]) async throws -> D1JSONValue {
        let x = try audioInputs(samples: samples)
        let prefix = try await audioPrefix(x)
        return try await systemOne(state: state, questions: questions, mode: .audio, prefix: prefix)
    }
}
