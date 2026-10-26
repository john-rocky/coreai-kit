// Pipeline — one photo + the words for what to cut out → the boxes the model writes and a matte per box, with each
// step's time. The steps are the package's host steps for `predict(task="segmentation")` on a fixed 448 grid, in Swift:
//
//   1  the upright photo → Pillow BICUBIC 448² → patches → vision tower with taps
//   2  the prompt (the package's template) → decoder: prompt, then greedy; at each fed <OBJ>, its last_hidden
//   3  the answer → the boxes (`parse_seg_boxes`); as many instances as boxes and hidden states both have
//   4  the photo → Pillow BICUBIC 1024² → the canvas
//   5  box_pe on the host (BoxFourier)
//   6  graph A per 16 instances, 7 graph B per instance
//   8  sigmoid → bilinear to the photo's size (`Alpha`, done by the caller at the size it needs)
//
// The model folder holds four parts (`ModelFolder`): the decoder bundle, the tower, graph A and graph B.

import CoreAI
import Foundation

/// Where the four parts of the model are. A folder laid out as the Hugging Face repository's platform subtree:
///
///   <folder>/wevisone_vl_2b_decode_int8hu_pf64/   the decoder bundle: metadata.json, tokenizer/, the asset
///   <folder>/wevisone_vl_2b_vision_taps_g14.aimodel
///   <folder>/wv_mask_a_g28_n16_fp16_hf.aimodel
///   <folder>/wv_mask_b_fp16.aimodel
///
/// A part may be an ahead-of-time asset beside it (`<name>.h16c.aimodelc` on a Mac); a `cutout.json` in the folder can
/// name other parts (`{"decoder": "…", "tower": "…", "graph_a": "…", "graph_b": "…"}`), so the decoder's form can be swapped.
/// For a measurement, the environment variables `CUTOUT_TOWER`, `CUTOUT_GRAPH_A` and `CUTOUT_GRAPH_B` name the tower and
/// the mask graphs for one launch (the forms measured and not shipped: the fp16w32 tower, graph A for the Neural
/// Engine, graph B with K instances per call; a name or a path), and `CUTOUT_GRAPH_A_UNIT=ane` asks for the Neural
/// Engine for graph A's `.aimodel`; without them nothing changes.
public struct ModelFolder: Sendable {
    public static let defaultNames = ["decoder": "wevisone_vl_2b_decode_int8hu_pf64", "tower": "wevisone_vl_2b_vision_taps_g14",
                                      "graph_a": "wv_mask_a_g28_n16_fp16_hf", "graph_b": "wv_mask_b_fp16"]
    public let root: URL
    public let decoder: URL
    public let tokenizer: URL
    public let tower: URL
    public let graphA: URL
    public let graphB: URL
    /// true when every part is an ahead-of-time `.aimodelc`.
    public let aot: Bool
    public let preferAOT: Bool

    public init(root: URL, preferAOT: Bool) throws {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue else {
            throw CutoutError.files("The model files are not on this device: no folder \(root.lastPathComponent). Put the model "
                + "folder there (README: Run it).")
        }
        var names = Self.defaultNames
        if let data = try? Data(contentsOf: root.appendingPathComponent("cutout.json")),
           let j = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            for (k, v) in j where names[k] != nil { names[k] = v }
        }
        let env = ProcessInfo.processInfo.environment
        if let t = env["CUTOUT_TOWER"], !t.isEmpty { names["tower"] = t }
        if let a = env["CUTOUT_GRAPH_A"], !a.isEmpty { names["graph_a"] = a }
        if let b = env["CUTOUT_GRAPH_B"], !b.isEmpty { names["graph_b"] = b }
        func asset(_ base: URL, _ name: String) throws -> URL {
            if name.hasPrefix("/") { return URL(fileURLWithPath: name) }   // an override given as a path
            let aotURL = base.appendingPathComponent("\(name).h16c.aimodelc")
            let jitURL = base.appendingPathComponent("\(name).aimodel")
            if preferAOT, fm.fileExists(atPath: aotURL.path) { return aotURL }
            if fm.fileExists(atPath: jitURL.path) { return jitURL }
            throw CutoutError.files("The model files are not complete: \(name).aimodel is missing from \(root.lastPathComponent).")
        }
        let bundle = root.appendingPathComponent(names["decoder"]!)
        guard fm.fileExists(atPath: bundle.appendingPathComponent("metadata.json").path),
              let data = try? Data(contentsOf: bundle.appendingPathComponent("metadata.json")),
              let meta = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let assets = meta["assets"] as? [String: String], let main = assets["main"]
        else { throw CutoutError.files("The model files are not complete: \(names["decoder"]!)/metadata.json is missing.") }
        let tok = bundle.appendingPathComponent("tokenizer")
        guard fm.fileExists(atPath: tok.appendingPathComponent("tokenizer.json").path) else {
            throw CutoutError.files("The model files are not complete: \(names["decoder"]!)/tokenizer/tokenizer.json is missing.")
        }
        // metadata.json names the bundle's own asset: `<name>.aimodel`, or `<name>.h16c.aimodelc` in an ahead-of-time copy
        var base = main
        for suffix in [".aimodelc", ".aimodel"] where base.hasSuffix(suffix) { base = String(base.dropLast(suffix.count)) }
        if let r = base.range(of: #"\.h[0-9]+[a-z]*$"#, options: .regularExpression) { base = String(base[..<r.lowerBound]) }
        self.root = root
        decoder = try asset(bundle, base)
        tokenizer = tok
        tower = try asset(root, names["tower"]!)
        graphA = try asset(root, names["graph_a"]!)
        graphB = try asset(root, names["graph_b"]!)
        aot = [decoder, tower, graphA, graphB].allSatisfy { $0.pathExtension == "aimodelc" }
        self.preferAOT = preferAOT
    }

    /// Another asset in the folder by name (`<name>.h16c.aimodelc` when ahead-of-time assets are preferred and there is
    /// one, else `<name>.aimodel`), or a path.
    public func asset(named name: String) throws -> URL {
        if name.hasPrefix("/") { return URL(fileURLWithPath: name) }
        let fm = FileManager.default
        let aotURL = root.appendingPathComponent("\(name).h16c.aimodelc")
        let jitURL = root.appendingPathComponent("\(name).aimodel")
        if preferAOT, fm.fileExists(atPath: aotURL.path) { return aotURL }
        if fm.fileExists(atPath: jitURL.path) { return jitURL }
        throw CutoutError.files("The model files are not complete: \(name).aimodel is missing from \(root.lastPathComponent).")
    }
}

@available(macOS 27, iOS 27, *)
public final class CutoutPipeline: @unchecked Sendable {
    public struct Load: Sendable {
        public var seconds: [String: Double] = [:]
        /// phys_footprint after each part, in bytes
        public var footprint: [String: UInt64] = [:]
        public var options: [String: String] = [:]
    }

    public struct Instance: Sendable {
        /// x1, y1, x2, y2 in 0 … 1000 (per mille of the picture's width and height)
        public let box: [Int]
        public let label: String
        /// mask_logits [1024 · 1024] fp32 (nil for the boxes task)
        public let logits: [Float]?
    }

    public struct Result: Sendable {
        public let task: CutoutTask
        public let words: String
        public let row: PromptRow
        public let generation: WeVisOneDecoder.Generation
        public let response: String
        public let instances: [Instance]
        /// box_pe rows per graph A call (kept when asked)
        public let boxPE: [[Float]]
        public let width: Int
        public let height: Int
        /// seconds per step: resize448, patches, tower, prompt, decoder_prompt, decode, resize1024, canvas, box_pe,
        /// graph_a, graph_b (sums), mask_inputs, total
        public let seconds: [String: Double]
        public let graphA: [Double]
        public let graphB: [Double]
        /// the inputs as the model got them (for checks)
        public let patches: [Float16]?
        public let canvas8: RGB8Image?
    }

    public let folder: ModelFolder
    public let prompt: PromptBuilder
    public let tower: VisionTaps
    public let decoder: WeVisOneDecoder
    public let masks: MaskGraphs
    public let load: Load

    /// Loads the four parts: the decoder first (the largest specialization), then the tower and the two mask graphs.
    /// `decoderOptions` / `graphOptions` default to the forms the measurements used: an ahead-of-time asset with
    /// `SpecializationOptions.default`; a `.aimodel` on the GPU, the decoder with frequent reshapes (its position axis
    /// grows every call), the static graphs without (frequent reshapes on a fixed-shape graph crash on iOS).
    public init(folder: ModelFolder, kvLength: Int = 1024, progress: (@Sendable (String) -> Void)? = nil) async throws {
        var load = Load()
        func options(for url: URL, dynamic: Bool) -> SpecializationOptions {
            if url.pathExtension == "aimodelc" { return .default }
            var o = SpecializationOptions(preferredComputeUnitKind: .gpu)
            o.expectFrequentReshapes = dynamic
            return o
        }
        var t = ContinuousClock.now
        progress?("tokenizer")
        prompt = try await PromptBuilder.load(tokenizerFolder: folder.tokenizer)
        load.seconds["tokenizer"] = secondsSince(t)
        progress?("decoder")
        t = ContinuousClock.now
        let dOpt = options(for: folder.decoder, dynamic: true)
        decoder = try await WeVisOneDecoder(contentsOf: folder.decoder, kvLength: kvLength, options: dOpt)
        load.seconds["decoder"] = secondsSince(t)
        load.footprint["decoder"] = memoryFootprint().now
        load.options["decoder"] = describe(dOpt)
        progress?("vision")
        t = ContinuousClock.now
        let tOpt = options(for: folder.tower, dynamic: false)
        tower = try await VisionTaps(contentsOf: folder.tower, options: tOpt)
        load.seconds["tower"] = secondsSince(t)
        load.footprint["tower"] = memoryFootprint().now
        load.options["tower"] = describe(tOpt)
        progress?("mask decoder")
        t = ContinuousClock.now
        let gOpt = options(for: folder.graphA, dynamic: false)
        var aOpt = gOpt
        if ProcessInfo.processInfo.environment["CUTOUT_GRAPH_A_UNIT"] == "ane", folder.graphA.pathExtension != "aimodelc" {
            aOpt = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        }
        let m = try await MaskGraphs(graphA: folder.graphA, graphB: folder.graphB, options: gOpt, graphAOptions: aOpt)
        masks = m
        load.seconds["mask_graphs"] = secondsSince(t)
        load.footprint["mask_graphs"] = memoryFootprint().now
        load.options["mask_graphs"] = describe(gOpt)
        load.options["graph_a"] = "\(folder.graphA.lastPathComponent): \(describe(aOpt))"
        load.options["tower_asset"] = folder.tower.lastPathComponent
        load.options["graph_b_asset"] = "\(folder.graphB.lastPathComponent) (\(m.graphB.rows) per call)"
        self.folder = folder
        self.load = load
    }

    /// One request. `image` is the upright photo. `onToken` gets each generated id as it comes (for a live answer).
    /// `instruction` replaces the prompt built from the words (a fixture's prompt as written); `masks` replaces the
    /// loaded mask graphs (another precision of the same graphs).
    public func run(image: RGB8Image, words: String, task: CutoutTask, schedule: WeVisOneDecoder.Schedule = .pad,
                    maxNew: Int = 768, recordGaps: Bool = false, keepInputs: Bool = false, instruction: String? = nil,
                    masks other: MaskGraphs? = nil, inputs given: (patches: [Float16], canvas: RGB8Image)? = nil,
                    onToken: (@Sendable (Int) -> Void)? = nil) async throws -> Result
    {
        let masks = other ?? self.masks
        let tAll = ContinuousClock.now
        var s: [String: Double] = [:]
        var t = ContinuousClock.now
        let patches: [Float16]
        if let given {
            patches = given.patches   // another host's tensors (a check: the oracle's own)
        } else {
            let tile = ImageHost.pillowBicubic(image, width: ImageHost.towerSide, height: ImageHost.towerSide)
            s["resize448"] = secondsSince(t)
            t = ContinuousClock.now
            patches = try ImageHost.patches(tile)
            s["patches"] = secondsSince(t)
        }
        t = ContinuousClock.now
        let v = try await tower.encode(patches: patches)
        s["tower"] = secondsSince(t)
        t = ContinuousClock.now
        let row = try prompt.row(prompt: instruction ?? PromptBuilder.instruction(task: task, words: words))
        s["prompt"] = secondsSince(t)
        let gen = try await decoder.generate(prompt: row.ids, inputs: decoder.staticInputs(v, row: row), schedule: schedule,
                                             maxNew: maxNew, recordGaps: recordGaps, onToken: onToken)
        s["decoder_prompt"] = gen.promptSeconds
        s["decode"] = gen.decodeSeconds
        let response = prompt.response(gen.ids)
        let labeled = AnswerParse.labeled(response)
        var instances: [Instance] = []
        var boxPE: [[Float]] = []
        var canvas8: RGB8Image? = nil
        var graphA: [Double] = [], graphB: [Double] = []
        switch task {
        case .boxes:
            instances = labeled.map { Instance(box: $0.box, label: $0.label, logits: nil) }
        case .cutout:
            let boxes = AnswerParse.segBoxes(response)
            let objLabels = labeled.filter(\.obj).map(\.label)
            let n = min(boxes.count, gen.objHidden.count)
            if n > 0 {
                t = ContinuousClock.now
                let resized = given?.canvas ?? ImageHost.pillowBicubic(image, width: ImageHost.canvasSide, height: ImageHost.canvasSide)
                s["resize1024"] = secondsSince(t)
                t = ContinuousClock.now
                let canvas = ImageHost.canvas(resized)
                s["canvas"] = secondsSince(t)
                if keepInputs { canvas8 = resized }
                let (logits, timing) = try await masks.logits(
                    canvas: canvas, taps: v.taps, hidden: Array(gen.objHidden.prefix(n)),
                    boxes: boxes.prefix(n).map(AnswerParse.unit), boxPE: { _, pe in boxPE.append(pe) })
                s["box_pe"] = timing.boxPE.reduce(0, +)
                s["graph_a"] = timing.graphA.reduce(0, +)
                s["graph_b"] = timing.graphB.reduce(0, +)
                s["mask_inputs"] = timing.inputs
                graphA = timing.graphA
                graphB = timing.graphB
                for i in 0..<n {
                    let label = objLabels.count == boxes.count ? objLabels[i] : words
                    instances.append(Instance(box: boxes[i], label: label, logits: logits[i]))
                }
            }
        }
        s["total"] = secondsSince(tAll)
        return Result(task: task, words: words, row: row, generation: gen, response: response, instances: instances,
                      boxPE: boxPE, width: image.width, height: image.height, seconds: s, graphA: graphA, graphB: graphB,
                      patches: keepInputs ? patches : nil, canvas8: canvas8)
    }
}

// MARK: - measurements: another graph B, graph A or tower against the loaded one, in one process

@available(macOS 27, iOS 27, *)
extension CutoutPipeline {
    /// The request's mask-decoder inputs for one photo and its words: tower, decoder (pad schedule, greedy), boxes, canvas.
    public func maskInputs(image: RGB8Image, words: String) async throws
        -> (canvas: [Float], taps: NDArray, hidden: [[Float16]], boxes: [[Float]], response: String)
    {
        let tile = ImageHost.pillowBicubic(image, width: ImageHost.towerSide, height: ImageHost.towerSide)
        let v = try await tower.encode(patches: try ImageHost.patches(tile))
        let row = try prompt.row(prompt: PromptBuilder.instruction(task: .cutout, words: words))
        let gen = try await decoder.generate(prompt: row.ids, inputs: decoder.staticInputs(v, row: row), schedule: .pad, maxNew: 768)
        let response = prompt.response(gen.ids)
        let boxes = AnswerParse.segBoxes(response)
        let n = min(boxes.count, gen.objHidden.count)
        guard n > 0 else { throw CutoutError.prompt("no instance in the answer \(response)") }
        let canvas = ImageHost.canvas(ImageHost.pillowBicubic(image, width: ImageHost.canvasSide, height: ImageHost.canvasSide))
        return (canvas, v.taps, Array(gen.objHidden.prefix(n)), boxes.prefix(n).map(AnswerParse.unit), response)
    }

    /// The loaded graph B against `other` (another K) on this request's own graph A outputs. One round = as many
    /// instances as the larger K (the request's instances repeated), each graph in its own chunks of K; `warm` rounds
    /// of each, then `reps` rounds alternating (ABBA). Also the two graphs' mask_logits on the same instances (bit-equal
    /// count, max |Δ|) and the footprint after each graph's first round.
    public func benchGraphB(other: MaskGraphB, image: RGB8Image, words: String, reps: Int, warm: Int = 2) async throws
        -> [String: Any]
    {
        let inputs = try await maskInputs(image: image, words: words)
        var ta0 = MaskGraphs.Timing()
        let stage = try await masks.stagesA(canvas: inputs.canvas, taps: inputs.taps, hidden: inputs.hidden,
                                            boxes: inputs.boxes, timing: &ta0)[0]
        let base = masks.graphB
        let k = max(base.rows, other.rows)
        let plane = 512 * 512
        var q = [Float](), c = [Float]()
        for slot in 0..<k {
            let i = slot % stage.n
            q += stage.queries[(i * 192)..<((i + 1) * 192)]
            c += stage.coarse[(i * plane)..<((i + 1) * plane)]
        }
        let round = MaskGraphs.StageA(n: k, image: stage.image, semantic: stage.semantic, queries: q, coarse: c)
        let fp0 = memoryFootprint().now
        var t = MaskGraphs.Timing()
        let la = try await base.run(round, timing: &t)
        let fpBase = memoryFootprint().now
        let lb = try await other.run(round, timing: &t)
        let fpOther = memoryFootprint().now
        var bitEqual = 0
        var maxd: Float = 0
        for i in 0..<k {
            if la[i] == lb[i] { bitEqual += 1 }
            for j in 0..<la[i].count { maxd = max(maxd, abs(la[i][j] - lb[i][j])) }
        }
        func once(_ g: MaskGraphB) async throws -> Double {
            var tt = MaskGraphs.Timing()
            let t0 = ContinuousClock.now
            _ = try await g.run(round, timing: &tt)
            return secondsSince(t0) * 1000
        }
        for _ in 0..<warm {
            _ = try await once(base)
            _ = try await once(other)
        }
        var msBase: [Double] = [], msOther: [Double] = []
        for r in 0..<reps {
            if r % 2 == 0 {
                msBase.append(try await once(base))
                msOther.append(try await once(other))
            } else {
                msOther.append(try await once(other))
                msBase.append(try await once(base))
            }
        }
        func stat(_ v: [Double]) -> [String: Any] {
            let s = v.sorted()
            var median: Double = 0
            if !s.isEmpty { median = s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }
            let lo: Double = s.first ?? 0, hi: Double = s.last ?? 0
            return ["median": median, "min": lo, "max": hi, "n": s.count, "all": v]
        }
        let mb = stat(msBase), mo = stat(msOther)
        let medBase = mb["median"] as? Double ?? 0, medOther = mo["median"] as? Double ?? 0
        var baseRec: [String: Any] = ["asset": base.url.lastPathComponent, "load_s": base.loadSeconds]
        baseRec["rows"] = base.rows
        var otherRec: [String: Any] = ["asset": other.url.lastPathComponent, "load_s": other.loadSeconds]
        otherRec["rows"] = other.rows
        let footprint: [String: UInt64] = ["before": fp0, "after_base_first": fpBase, "after_other_first": fpOther,
                                           "end": memoryFootprint().now, "peak": memoryFootprint().peak]
        var rec: [String: Any] = ["schema": "cutout-bbench/1", "words": words, "response": inputs.response]
        rec["instances_real"] = stage.n
        rec["round_instances"] = k
        rec["base"] = baseRec
        rec["other"] = otherRec
        rec["base_round_ms"] = mb
        rec["other_round_ms"] = mo
        rec["per_instance_ms"] = ["base": medBase / Double(k), "other": medOther / Double(k)]
        rec["base_over_other"] = medOther > 0 ? medBase / medOther : 0
        rec["logits_bit_equal_instances"] = bitEqual
        rec["logits_max_abs_d"] = Double(maxd)
        rec["footprint"] = footprint
        rec["graph_a_ms"] = ta0.graphA.map { $0 * 1000 }
        return rec
    }

    /// The loaded tower against `other` on one photo: the three outputs' cosine between the two (fp32 reads), their
    /// times alternating (ABBA) after `warm` calls each, and the footprint after each tower's first call.
    public func benchTower(other: VisionTaps, image: RGB8Image, reps: Int, warm: Int = 2) async throws -> [String: Any] {
        let patches = try ImageHost.patches(ImageHost.pillowBicubic(image, width: ImageHost.towerSide, height: ImageHost.towerSide))
        let fp0 = memoryFootprint().now
        let a = try await tower.encode(patches: patches)
        let fpBase = memoryFootprint().now
        let b = try await other.encode(patches: patches)
        let fpOther = memoryFootprint().now
        func cosine(_ x: NDArray, _ y: NDArray) throws -> [String: Double] {
            let u = try ND.readFloats(x), v = try ND.readFloats(y)
            var dot = 0.0, nu = 0.0, nv = 0.0, maxd = 0.0
            for i in 0..<u.count {
                dot += Double(u[i]) * Double(v[i])
                nu += Double(u[i]) * Double(u[i])
                nv += Double(v[i]) * Double(v[i])
                maxd = max(maxd, Double(abs(u[i] - v[i])))
            }
            return ["cos": dot / (nu.squareRoot() * nv.squareRoot() + 1e-300), "max_abs_d": maxd]
        }
        let agree: [String: Any] = ["image_embeds": try cosine(a.imageEmbeds, b.imageEmbeds),
                                    "deepstack_embeds": try cosine(a.deepstackEmbeds, b.deepstackEmbeds),
                                    "taps": try cosine(a.taps, b.taps)]
        func once(_ t: VisionTaps) async throws -> Double {
            let t0 = ContinuousClock.now
            _ = try await t.encode(patches: patches)
            return secondsSince(t0) * 1000
        }
        for _ in 0..<warm {
            _ = try await once(tower)
            _ = try await once(other)
        }
        var msBase: [Double] = [], msOther: [Double] = []
        for r in 0..<reps {
            if r % 2 == 0 {
                msBase.append(try await once(tower))
                msOther.append(try await once(other))
            } else {
                msOther.append(try await once(other))
                msBase.append(try await once(tower))
            }
        }
        func stat(_ v: [Double]) -> [String: Any] {
            let s = v.sorted()
            var median: Double = 0
            if !s.isEmpty { median = s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }
            let lo: Double = s.first ?? 0, hi: Double = s.last ?? 0
            return ["median": median, "min": lo, "max": hi, "n": s.count, "all": v]
        }
        let baseRec: [String: Any] = ["asset": tower.url.lastPathComponent, "load_s": tower.loadSeconds]
        let otherRec: [String: Any] = ["asset": other.url.lastPathComponent, "load_s": other.loadSeconds]
        let footprint: [String: UInt64] = ["before": fp0, "after_base_first": fpBase, "after_other_first": fpOther,
                                           "end": memoryFootprint().now, "peak": memoryFootprint().peak]
        var rec: [String: Any] = ["schema": "cutout-tbench/1", "agreement": agree]
        rec["base"] = baseRec
        rec["other"] = otherRec
        rec["base_ms"] = stat(msBase)
        rec["other_ms"] = stat(msOther)
        rec["footprint"] = footprint
        return rec
    }
}

@available(macOS 27, iOS 27, *)
extension CutoutPipeline {
    /// The loaded graph A against another (`other`: a graph A for the Neural Engine, e.g.) on this request's own inputs:
    /// the outputs graph B reads (queries and coarse logits of the real rows, semantic_s2) compared between the two
    /// (cosine, max |Δ|, bit-equal), graph A's call alone timed after `warm` calls each, `reps` alternating (ABBA), and the
    /// footprint after each first call.
    public func benchGraphA(other: MaskGraphs, image: RGB8Image, words: String, reps: Int, warm: Int = 2) async throws
        -> [String: Any]
    {
        let inputs = try await maskInputs(image: image, words: words)
        func once(_ m: MaskGraphs) async throws -> (ms: Double, stage: MaskGraphs.StageA) {
            var t = MaskGraphs.Timing()
            let s = try await m.stagesA(canvas: inputs.canvas, taps: inputs.taps, hidden: inputs.hidden, boxes: inputs.boxes,
                                        timing: &t)
            return (t.graphA.reduce(0, +) * 1000, s[0])
        }
        let fp0 = memoryFootprint().now
        let a = try await once(masks)
        let fpBase = memoryFootprint().now
        let b = try await once(other)
        let fpOther = memoryFootprint().now
        func agree(_ u: [Float], _ v: [Float]) -> [String: Any] {
            var dot = 0.0, nu = 0.0, nv = 0.0, maxd = 0.0
            for i in 0..<min(u.count, v.count) {
                dot += Double(u[i]) * Double(v[i])
                nu += Double(u[i]) * Double(u[i])
                nv += Double(v[i]) * Double(v[i])
                maxd = max(maxd, Double(abs(u[i] - v[i])))
            }
            var r: [String: Any] = ["cos": dot / (nu.squareRoot() * nv.squareRoot() + 1e-300), "max_abs_d": maxd]
            r["bit_equal"] = u == v
            return r
        }
        var agreement: [String: Any] = [:]
        agreement["queries"] = agree(a.stage.queries, b.stage.queries)
        agreement["coarse_logits"] = agree(a.stage.coarse, b.stage.coarse)
        agreement["semantic_s2"] = agree(try ND.readFloats(a.stage.semantic), try ND.readFloats(b.stage.semantic))
        for _ in 0..<warm {
            _ = try await once(masks)
            _ = try await once(other)
        }
        var msBase: [Double] = [], msOther: [Double] = []
        for r in 0..<reps {
            if r % 2 == 0 {
                msBase.append(try await once(masks).ms)
                msOther.append(try await once(other).ms)
            } else {
                msOther.append(try await once(other).ms)
                msBase.append(try await once(masks).ms)
            }
        }
        func stat(_ v: [Double]) -> [String: Any] {
            let s = v.sorted()
            var median: Double = 0
            if !s.isEmpty { median = s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }
            let lo: Double = s.first ?? 0, hi: Double = s.last ?? 0
            return ["median": median, "min": lo, "max": hi, "n": s.count, "all": v]
        }
        let footprint: [String: UInt64] = ["before": fp0, "after_base_first": fpBase, "after_other_first": fpOther,
                                           "end": memoryFootprint().now, "peak": memoryFootprint().peak]
        var rec: [String: Any] = ["schema": "cutout-abench/1", "words": words, "response": inputs.response]
        rec["instances_real"] = a.stage.n
        rec["base"] = ["asset": masks.urls.a.lastPathComponent, "load_s": masks.loadSeconds.a] as [String: Any]
        rec["other"] = ["asset": other.urls.a.lastPathComponent, "load_s": other.loadSeconds.a] as [String: Any]
        rec["base_ms"] = stat(msBase)
        rec["other_ms"] = stat(msOther)
        rec["agreement"] = agreement
        rec["footprint"] = footprint
        return rec
    }
}
