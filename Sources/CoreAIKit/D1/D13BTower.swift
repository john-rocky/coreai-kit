// From the model zoo's apps/D1/Sources/D1/Tower.swift (e36ad15, sha256 b2860ebb5eeb), identifiers prefixed D13B for the kit.
// Tower — the d1 vision tower on the low-level runtime: the tower bundle of conversion/d1/export_vision.py (SigLIP2 +
// the projector in its exact form, the crop's grid given as inputs), one stateless function `main` called once per
// crop. The type is apps/ClefFlash's VisionTower (zoo d1-3b 8e82d36); the contract is the tower bundle's metadata.json
// `graph`, checked against the loaded function at load.
//
//   inputs   patches [1024, 768], pos_table [1024, d_v], key_bias [1024] (float32, or float16 for an fp16 tower: the
//            host's float32 values are cast here, as gate_tower.py casts them), unshuffle_idx [256, 4] i32
//   output   image_embeds [256, d] (float32, or float16): rows 0 ..< h w / 4 are the crop's image tokens in its merged
//            grid's row-major order, the rest padding
//
// A request's image rows (conversion/d1/host.py's companion vision_host.py §6, metadata `vision.rows`): every crop's
// first h w / 4 rows, crops in order (per picture its tiles row-major, then the thumbnail), pictures in text order,
// cast to float16 (round to nearest even, NumPy's astype) -> the decoder's image_embeds rows 0 ..< n.
//
// A crop's four inputs come from the picture file (`D13BTowerInputs.pictures(files:table:)`: ImagePixels.swift's decode,
// cap_pixels, crops, the torch uint8 bicubic resize, patches, the position table of the tower bundle's
// `host/position_embedding.safetensors` resized per crop, the unshuffle index = vision_host.tower_inputs, the order
// decide.py and readout_gate_vision.py run) or, to split an error between the pixels and the graphs, from raw files
// (`D13BTowerInputs.read`: what vision_host.tower_inputs wrote, little-endian float32 / int32).

import CoreAI
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitD1Decider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 27, iOS 27, *)
final class D13BTower: @unchecked Sendable {
    /// One crop's four inputs, row-major, in float32 / int32 (the host's types).
    struct CropInputs: Sendable {
        let patches: [Float]
        let posTable: [Float]
        let keyBias: [Float]
        let unshuffle: [Int32]
        /// (h, w) patches and the crop's image tokens h w / 4
        let grid: (h: Int, w: Int)
        let tokens: Int

        init(patches: [Float], posTable: [Float], keyBias: [Float], unshuffle: [Int32], grid: (h: Int, w: Int),
                    tokens: Int) {
            self.patches = patches
            self.posTable = posTable
            self.keyBias = keyBias
            self.unshuffle = unshuffle
            self.grid = grid
            self.tokens = tokens
        }
    }

    static let inputNames = ["patches", "pos_table", "key_bias", "unshuffle_idx"]

    let bundle: URL
    let url: URL
    let name: String
    let contract: D13BGraphContract
    /// d (image_embeds [256, d]: the decoder's hidden width)
    let width: Int
    let maxTokens: Int
    /// the bundle's `host/position_embedding.safetensors` [256, d_v] (metadata.json `host_files`), resized per crop
    let positionTable: D13BPixels.PositionTable
    let loadSeconds: Double
    let descriptor: D13BJSONValue
    let options: SpecializationOptions
    /// crops encoded since the load (a gate reads it around a request the host refuses: no call)
    private(set) var callCount = 0
    private let function: InferenceFunction
    private let inputDescriptors: [String: NDArrayDescriptor]

    static let positionTableFile = "host/position_embedding.safetensors"

    /// `bundle` = a tower bundle directory; `asset` = its `.aimodelc` (AOT) or `.aimodel` (JIT), nil = the AOT asset
    /// beside it (`<bundles>_aotc/<name>.h16c.aimodelc`). The contract's input shapes are checked against the host's
    /// (patches [1024, 768], pos_table [1024, d_v] with d_v the position table's width, key_bias [1024]).
    init(bundle: URL, asset: URL? = nil, options: SpecializationOptions? = nil) async throws {
        let meta = try D13BJSONParser.parse(Data(contentsOf: bundle.appendingPathComponent("metadata.json")))
        guard meta["kind"]?.string == "vision-tower", let name = meta["name"]?.string, let g = meta["graph"],
              meta["assets"]?["main"]?.string != nil
        else { throw D13BError.bundle("\(bundle.path)/metadata.json: not a vision-tower bundle with name / graph / assets.main") }
        let contract = try D13BGraphContract(g, what: "\(name) metadata.json graph")
        guard Set(contract.inputs.keys) == Set(Self.inputNames), let out = contract.outputs["image_embeds"], out.shape.count == 2,
              contract.inputs["unshuffle_idx"] == D13BTensorSpec(shape: [out.shape[0], 4], dtype: "int32")
        else { throw D13BError.contract("\(name): the graph is not patches / pos_table / key_bias / unshuffle_idx -> image_embeds") }
        let table = try D13BPixels.PositionTable.load(url: bundle.appendingPathComponent(Self.positionTableFile))
        let p = D13BPixels.maxPatches
        guard out.shape[0] == D13BPixels.maxTokens, contract.inputs["patches"]?.shape == [p, D13BPixels.patchDim],
              contract.inputs["pos_table"]?.shape == [p, table.dim], contract.inputs["key_bias"]?.shape == [p]
        else {
            throw D13BError.contract("\(name): inputs \(contract.inputs.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }) "
                + "-> image_embeds \(out) are not the host's (patches [\(p), \(D13BPixels.patchDim)], pos_table [\(p), "
                + "\(table.dim)] = \(Self.positionTableFile), key_bias [\(p)], \(D13BPixels.maxTokens) rows)")
        }
        let url = asset ?? D13BPaths.aot(bundle: bundle, name: name)
        let opts = options ?? (url.pathExtension == "aimodelc" ? .default : D13BPaths.towerJITOptions)
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: opts)
        guard let fd = model.functionDescriptor(for: contract.function), let fn = try model.loadFunction(named: contract.function)
        else { throw D13BError.contract("tower \(url.lastPathComponent): no \"\(contract.function)\" (functions \(model.functionNames))") }
        loadSeconds = d13bSeconds(since: t0)
        try D13BContract.check(fd, what: "tower \(url.lastPathComponent)", inputs: contract.inputs, outputs: contract.outputs,
                             states: contract.states)
        self.bundle = bundle
        self.url = url
        self.name = name
        self.contract = contract
        width = out.shape[1]
        maxTokens = out.shape[0]
        positionTable = table
        descriptor = D13BContract.describe(fd)
        self.options = opts
        function = fn
        inputDescriptors = Dictionary(uniqueKeysWithValues: Self.inputNames.map { ($0, D13BND.descriptor(fd.inputDescriptor(of: $0))!) })
    }

    /// A tower bundle's `.aimodel` (metadata.json `assets.main`), the asset the runtime specializes here (JIT).
    static func modelAsset(bundle: URL) throws -> URL {
        let meta = try D13BJSONParser.parse(Data(contentsOf: bundle.appendingPathComponent("metadata.json")))
        guard let main = meta["assets"]?["main"]?.string else { throw D13BError.bundle("\(bundle.path)/metadata.json: no assets.main") }
        return bundle.appendingPathComponent(main)
    }

    /// One crop -> image_embeds [256 * d] row-major as Float (an fp16 tower's values widened exactly).
    func encode(_ c: CropInputs) async throws -> [Float] {
        var inputs: [String: NDArray] = [:]
        for (n, v) in [("patches", c.patches), ("pos_table", c.posTable), ("key_bias", c.keyBias)] {
            let d = inputDescriptors[n]!
            guard v.count == d.shape.reduce(1, *) else {
                throw D13BError.contract("tower input \(n): \(v.count) values for \(d.shape)")
            }
            inputs[n] = contract.inputs[n]!.dtype == "float16" ? D13BND.make(v.map { Float16($0) }, d) : D13BND.make(v, d)
        }
        let ud = inputDescriptors["unshuffle_idx"]!
        guard c.unshuffle.count == ud.shape.reduce(1, *) else {
            throw D13BError.contract("tower input unshuffle_idx: \(c.unshuffle.count) values for \(ud.shape)")
        }
        inputs["unshuffle_idx"] = D13BND.make(c.unshuffle, ud)
        callCount += 1
        var outputs = try await function.run(inputs: inputs)
        guard let array = outputs.remove("image_embeds")?.ndArray else {
            throw D13BError.contract("tower: no image_embeds in the outputs")
        }
        let emb: [Float] = contract.outputs["image_embeds"]!.dtype == "float16"
            ? D13BND.read(array, as: Float16.self).map { Float($0) } : D13BND.read(array, as: Float.self)
        guard emb.count == maxTokens * width else {
            throw D13BError.contract("tower: \(emb.count) values for \(maxTokens) x \(width)")
        }
        return emb
    }

    /// Every crop in order -> the decoder's image rows (each crop's first `tokens` rows, float16) and each crop's whole
    /// output (for a gate).
    func imageRows(_ crops: [CropInputs]) async throws -> (rows: [Float16], outputs: [[Float]], seconds: [Double]) {
        var rows: [Float16] = []
        var outs: [[Float]] = []
        var secs: [Double] = []
        for c in crops {
            guard c.tokens <= maxTokens else { throw D13BError.contract("a crop of \(c.tokens) tokens over the tower's \(maxTokens)") }
            let t = ContinuousClock.now
            let e = try await encode(c)
            secs.append(d13bSeconds(since: t))
            rows += e[0..<(c.tokens * width)].map { Float16($0) }
            outs.append(e)
        }
        return (rows, outs, secs)
    }
}

/// The four tower inputs of every crop of a request's pictures, in text order: made from the picture files here
/// (`pictures(files:table:)`, ImagePixels.swift) or read from raw files (`read`, `--tower-inputs <dir>`):
/// `<dir>/manifest.json` = {"pictures": [{"id", "size": [w, h] (as the picture arrives), "crops": [{"crop", "kind",
/// "grid": [h, w], "n_tokens", "patches", "pos_table", "key_bias", "unshuffle_idx"}]}]} with each input a raw
/// little-endian file (float32; unshuffle_idx int32) beside it. The plan Swift makes from the picture's size
/// (`D13BVision.plan`) must give the same crops.
@available(macOS 27, iOS 27, *)
struct D13BTowerInputs: Sendable {
    struct Picture: Sendable {
        let id: String
        /// the picture as it arrives (after its EXIF orientation, before cap_pixels)
        let width: Int
        let height: Int
        let plan: D13BVision.Plan
        let crops: [D13BTower.CropInputs]
    }

    let pictures: [Picture]

    /// Picture files, in text order -> every crop's four inputs (vision_host.py §1–8 through `D13BPixels`: decode with
    /// the EXIF orientation, cap_pixels, the plan, each crop resized and cut, patches / key_bias / unshuffle_idx, the
    /// position table resized to the crop's grid once per distinct grid). A picture's id is its file name without the
    /// extension.
    static func pictures(files: [URL], table: D13BPixels.PositionTable) throws -> D13BTowerInputs {
        var out: [Picture] = []
        for url in files {
            let pic = try D13BPixels.picture(url: url)
            let w = pic.decoded.rgb.width, h = pic.decoded.rgb.height
            let check = D13BVision.plan(pictureWidth: w, pictureHeight: h)
            guard check.crops == pic.plan.crops else {
                throw D13BError.contract("\(url.lastPathComponent): the plan of the capped pixels differs from the plan of \(w) x \(h)")
            }
            let inputs = try D13BPixels.towerInputs(pic, table: table)
            let crops = zip(pic.plan.crops, inputs).map { c, t in
                D13BTower.CropInputs(patches: t.patches, posTable: t.posTable, keyBias: t.keyBias, unshuffle: t.unshuffleIndex,
                                   grid: (t.gridHeight, t.gridWidth), tokens: c.tokens)
            }
            out.append(Picture(id: url.deletingPathExtension().lastPathComponent, width: w, height: h, plan: pic.plan,
                               crops: crops))
        }
        return D13BTowerInputs(pictures: out)
    }

    static func read(_ dir: URL, ids: [String]? = nil) throws -> D13BTowerInputs {
        let m = try D13BJSONParser.parse(Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        var out: [Picture] = []
        var byID: [String: D13BJSONValue] = [:]
        for p in m["pictures"]?.array ?? [] {
            guard let id = p["id"]?.string else { throw D13BError.bundle("\(dir.path)/manifest.json: a picture without an id") }
            byID[id] = p
        }
        let order = try (ids ?? (m["pictures"]?.array ?? []).compactMap { $0["id"]?.string }).map { id -> D13BJSONValue in
            guard let p = byID[id] else { throw D13BError.bundle("\(dir.path)/manifest.json: no picture \(id)") }
            return p
        }
        for p in order {
            guard let id = p["id"]?.string, let size = p["size"]?.array?.compactMap(\.intValue), size.count == 2,
                  let crops = p["crops"]?.array
            else { throw D13BError.bundle("\(dir.path)/manifest.json: a picture without id / size / crops") }
            let plan = D13BVision.plan(pictureWidth: size[0], pictureHeight: size[1])
            guard plan.crops.count == crops.count else {
                throw D13BError.contract("\(id): \(crops.count) crops in the manifest, the plan of \(size[0]) x \(size[1]) has "
                    + "\(plan.crops.count)")
            }
            var cs: [D13BTower.CropInputs] = []
            for (c, pc) in zip(crops, plan.crops) {
                guard let grid = c["grid"]?.array?.compactMap(\.intValue), grid.count == 2, let n = c["n_tokens"]?.intValue,
                      grid[0] == pc.grid.h, grid[1] == pc.grid.w, n == pc.tokens, c["kind"]?.string == pc.kind.rawValue
                else { throw D13BError.contract("\(id) \(c["crop"]?.string ?? "?"): the manifest's crop differs from the plan's") }
                func file(_ k: String) throws -> Data {
                    guard let f = c[k]?.string else { throw D13BError.bundle("\(id): no \(k) file") }
                    return try Data(contentsOf: dir.appendingPathComponent(f))
                }
                cs.append(D13BTower.CropInputs(patches: Self.floats(try file("patches")), posTable: Self.floats(try file("pos_table")),
                                             keyBias: Self.floats(try file("key_bias")), unshuffle: Self.int32s(try file("unshuffle_idx")),
                                             grid: (grid[0], grid[1]), tokens: n))
            }
            out.append(Picture(id: id, width: size[0], height: size[1], plan: plan, crops: cs))
        }
        return D13BTowerInputs(pictures: out)
    }

    static func floats(_ d: Data) -> [Float] {
        d.withUnsafeBytes { b in (0..<(d.count / 4)).map { Float(bitPattern: UInt32(littleEndian: b.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
    }

    static func int32s(_ d: Data) -> [Int32] {
        d.withUnsafeBytes { b in (0..<(d.count / 4)).map { Int32(littleEndian: b.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) } }
    }
}

/// Where a bundle's AOT asset lives and how a `.aimodel` is specialized here.
@available(macOS 27, iOS 27, *)
enum D13BPaths {
    /// `<bundles>/<name>` -> `<bundles>_aotc/<name>.h16c.aimodelc` (the lane's layout, readout_gate.py's rule).
    static func aot(bundle: URL, name: String) -> URL {
        let parent = bundle.deletingLastPathComponent()
        return parent.deletingLastPathComponent().appendingPathComponent("\(parent.lastPathComponent)_aotc")
            .appendingPathComponent("\(name).h16c.aimodelc")
    }

    /// The decoder `.aimodel`'s specialization: GPU preferred with frequent reshapes, the exporter's AOT flags
    /// (`coreai-build compile --preferred-compute gpu --expect-frequent-reshapes`; apps/Kev, apps/ClefFlash).
    static var decoderJITOptions: SpecializationOptions {
        var o = SpecializationOptions(preferredComputeUnitKind: .gpu)
        o.expectFrequentReshapes = true
        return o
    }

    /// The tower `.aimodel`'s: GPU preferred (export_vision.py compiles it without --expect-frequent-reshapes).
    static var towerJITOptions: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .gpu) }
}

#endif
