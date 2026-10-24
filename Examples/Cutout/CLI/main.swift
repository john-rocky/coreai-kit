// cutout-cli — the headless shell of Cutout on macOS: the same Sources/Engine code and Sources/QuickStart.swift the app
// ships, no Xcode project and no device.
//
//   cutout-cli run --models <folder> --image <file> --words "the dog" [--task cutout|boxes] [--out <dir>]
//       one request: the answer, the boxes, a PNG per instance and the combined PNG in <out>
//   cutout-cli fixture --models <folder> --rows <rows.json> --out <dir> [--forms pad-fp16,s1-fp32]
//       [--graph-a32 <asset> --graph-b32 <asset>] [--kv 1024] [--max-new 768] [--alpha]
//       every row in every form (Engine/Fixture.swift)
//   cutout-cli tail --models <folder> --image <file> --prompt "…" [--tails 13,45,63] [--reps 8] --out <json>
//       the prompt's tail fed both ways (64-pad prefill vs S=1 main), alternated
//   cutout-cli bench --models <folder> --image <file> --words "…" [--reps 3] --out <json>
//       the same request `reps` times after one warm-up, each step's seconds
//   cutout-cli boxpe --out <dir>
//       every box_pe value a 0–1000 box can give, for the check against the package's torch arithmetic
//   cutout-cli bbench --models <folder> --image <file> --words "…" --form <graph B name or path> [--reps 8] --out <json>
//   cutout-cli tbench --models <folder> --image <file> --form <tower name or path> [--reps 8] --out <json>
//   cutout-cli abench --models <folder> --image <file> --words "…" --form <graph A name or path> [--unit ane] --out <json>
//       another graph B (K instances per call) or tower against the loaded one, alternating (Engine/Pipeline.swift)
// CUTOUT_TOWER / CUTOUT_GRAPH_B in the environment load another tower / graph B (ModelFolder).

import CoreAI
import Foundation

@available(macOS 27, *)
struct CLI {
    var args: [String]

    func value(_ name: String) -> String? {
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    func flag(_ name: String) -> Bool { args.contains(name) }

    func need(_ name: String) throws -> String {
        guard let v = value(name) else { throw CutoutError.prompt("missing \(name)") }
        return v
    }

    func url(_ name: String) throws -> URL { URL(fileURLWithPath: try need(name)) }

    func pipeline() async throws -> CutoutPipeline {
        let folder = try ModelFolder(root: try url("--models"), preferAOT: !flag("--jit"))
        let kv = Int(value("--kv") ?? "1024") ?? 1024
        let p = try await CutoutPipeline(folder: folder, kvLength: kv) { FileHandle.standardError.write("loading \($0)\n".data(using: .utf8)!) }
        log("LOAD \(json(["seconds": p.load.seconds, "footprint": p.load.footprint.mapValues { Double($0) / 1e9 }, "options": p.load.options, "assets": [folder.decoder.path, folder.tower.path, folder.graphA.path, folder.graphB.path], "kv": kv]))")
        return p
    }

    func log(_ s: String) { print(s); fflush(stdout) }

    func json(_ o: Any) -> String {
        (try? JSONSerialization.data(withJSONObject: o, options: [.sortedKeys])).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    func writeJSON(_ o: Any, _ url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: o, options: [.prettyPrinted, .sortedKeys]).write(to: url)
    }

    func image(_ path: String) throws -> RGB8Image {
        let l = try ImageHost.load(url: URL(fileURLWithPath: path))
        return try ImageHost.uprightRGB8(l.image, orientation: l.orientation)
    }

    func main() async throws {
        switch args.first {
        case "run": try await run()
        case "fixture": try await fixture()
        case "tail": try await tail()
        case "bench": try await bench()
        case "boxpe": try boxpe()
        case "pe-effect": try await peEffect()
        case "bbench", "tbench", "abench": try await partBench(args.first!)
        default:
            print("usage: cutout-cli run | fixture | tail | bench | boxpe (see the comment at the top of CLI/main.swift)")
        }
    }

    func run() async throws {
        let p = try await pipeline()
        let img = try image(try need("--image"))
        let task = CutoutTask(rawValue: value("--task") ?? "cutout") ?? .cutout
        let r = try await p.run(image: img, words: try need("--words"), task: task)
        log("ANSWER \(r.response)")
        log("BOXES \(r.instances.map(\.box))")
        log("SECONDS \(json(r.seconds))")
        if let out = value("--out") {
            let dir = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            var probs: [[Float]] = []
            for (i, inst) in r.instances.enumerated() {
                guard let l = inst.logits else { continue }
                let pr = Alpha.probabilities(l, width: r.width, height: r.height)
                probs.append(pr)
                try CutoutImage.png(img, alpha: CutoutImage.combinedAlpha([pr])).write(to: dir.appendingPathComponent("instance_\(i).png"))
            }
            if !probs.isEmpty {
                try CutoutImage.png(img, alpha: CutoutImage.combinedAlpha(probs)).write(to: dir.appendingPathComponent("cutout.png"))
            }
            try writeJSON(["answer": r.response, "boxes": r.instances.map(\.box), "labels": r.instances.map(\.label),
                           "seconds": r.seconds, "ids": r.generation.ids], dir.appendingPathComponent("run.json"))
        }
    }

    func fixture() async throws {
        let p = try await pipeline()
        let rowsURL = try url("--rows")
        let rows = try Fixture.rows(at: rowsURL)
        var forms: [(WeVisOneDecoder.Schedule, MaskGraphs, String)] = []
        var fp32: MaskGraphs? = nil
        if let a = value("--graph-a32"), let b = value("--graph-b32") {
            fp32 = try await MaskGraphs(graphA: URL(fileURLWithPath: a), graphB: URL(fileURLWithPath: b), options: .default)
        }
        for name in (value("--forms") ?? "pad-fp16").split(separator: ",").map(String.init) {
            let parts = name.split(separator: "-").map(String.init)
            guard parts.count == 2, let s = WeVisOneDecoder.Schedule(rawValue: parts[0]) else { throw CutoutError.prompt("form \(name)") }
            let m: MaskGraphs
            if parts[1] == p.masks.precision.a { m = p.masks } else if let f = fp32, parts[1] == f.precision.a { m = f } else {
                throw CutoutError.prompt("form \(name): no \(parts[1]) mask graphs loaded")
            }
            forms.append((s, m, name))
        }
        let base = URL(fileURLWithPath: value("--images") ?? rowsURL.deletingLastPathComponent().path)
        let t0 = ContinuousClock.now
        let summary = try await Fixture.run(pipeline: p, rows: rows, imagesBase: base, out: try url("--out"), forms: forms,
                                            maxNew: Int(value("--max-new") ?? "768") ?? 768, writeAlpha: flag("--alpha"),
                                            inputs: value("--inputs").map { URL(fileURLWithPath: $0) }, log: log)
        log("DONE fixture \(rows.count) rows x \(forms.count) forms in \(String(format: "%.1f", secondsSince(t0))) s, footprint peak \(memoryFootprint().peak)")
        try writeJSON(["rows": summary, "load": ["seconds": p.load.seconds, "options": p.load.options]],
                      try url("--out").appendingPathComponent("summary.json"))
    }

    func tail() async throws {
        let p = try await pipeline()
        let img = try image(try need("--image"))
        let tile = ImageHost.pillowBicubic(img, width: ImageHost.towerSide, height: ImageHost.towerSide)
        let v = try await p.tower.encode(patches: try ImageHost.patches(tile))
        let row = try p.prompt.row(prompt: try need("--prompt"))
        let tails = (value("--tails") ?? "13,45,63").split(separator: ",").compactMap { Int($0) }
        let runs = try await p.decoder.tailBench(prompt: row.ids, inputs: p.decoder.staticInputs(v, row: row), tails: tails,
                                                 reps: Int(value("--reps") ?? "8") ?? 8)
        try writeJSON(TailReport.make(runs, promptLength: row.ids.count), try url("--out"))
        log("TAIL \(json(TailReport.make(runs, promptLength: row.ids.count)["summary"] ?? [:]))")
    }

    func bench() async throws {
        let p = try await pipeline()
        let img = try image(try need("--image"))
        let words = try need("--words")
        let schedule = WeVisOneDecoder.Schedule(rawValue: value("--schedule") ?? "pad") ?? .pad
        var runs: [[String: Any]] = []
        let reps = Int(value("--reps") ?? "3") ?? 3
        for i in 0...reps {
            let r = try await p.run(image: img, words: words, task: .cutout, schedule: schedule)
            var s = r.seconds
            s["alpha"] = 0
            let ta = ContinuousClock.now
            let probs = r.instances.compactMap { $0.logits.map { Alpha.probabilities($0, width: r.width, height: r.height) } }
            s["alpha"] = secondsSince(ta)
            let tp = ContinuousClock.now
            if !probs.isEmpty { _ = try CutoutImage.png(img, alpha: CutoutImage.combinedAlpha(probs)) }
            s["png"] = secondsSince(tp)
            runs.append(["warm": i > 0, "seconds": s, "tokens": r.generation.ids.count, "instances": r.instances.count,
                         "response": r.response, "graph_a": r.graphA, "graph_b": r.graphB,
                         "prompt_calls": r.generation.promptCalls.map { ["kind": $0.kind, "tokens": $0.tokens, "seconds": $0.seconds] },
                         "footprint": ["now": memoryFootprint().now, "peak": memoryFootprint().peak]])
            log("BENCH \(i) \(json(s)) tokens=\(r.generation.ids.count) instances=\(r.instances.count)")
        }
        try writeJSON(["runs": runs, "load": ["seconds": p.load.seconds, "options": p.load.options]], try url("--out"))
    }

    func partBench(_ mode: String) async throws {
        let p = try await pipeline()
        let img = try image(try need("--image"))
        let asset = try p.folder.asset(named: try need("--form"))
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = false
        if asset.pathExtension == "aimodelc" { options = .default }
        let reps = Int(value("--reps") ?? "8") ?? 8
        var rec: [String: Any]
        if mode == "abench" {
            var aOptions = options
            if value("--unit") == "ane", asset.pathExtension != "aimodelc" {
                aOptions = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            }
            let other = try await MaskGraphs(graphA: asset, graphB: p.folder.graphB, options: options, graphAOptions: aOptions)
            rec = try await p.benchGraphA(other: other, image: img, words: try need("--words"), reps: reps)
        } else if mode == "bbench" {
            let other = try await MaskGraphB(contentsOf: asset, options: options)
            rec = try await p.benchGraphB(other: other, image: img, words: try need("--words"), reps: reps)
        } else {
            let other = try await VisionTaps(contentsOf: asset, options: options)
            rec = try await p.benchTower(other: other, image: img, reps: reps)
        }
        rec["load_options"] = p.load.options
        try writeJSON(rec, try url("--out"))
        let keys: Set<String> = ["per_instance_ms", "base_over_other", "logits_bit_equal_instances", "logits_max_abs_d", "agreement",
                                 "footprint", "base_ms", "other_ms"]
        log("\(mode.uppercased()) \(json(rec.filter { keys.contains($0.key) }))")
    }

    /// What the torch table in BoxFourier changes in a matte: for each of its widths, a box near the row's own first box
    /// with that width and height, through graph A / B twice — box_pe with the table (torch's value) and with libm's logf
    /// alone — and the two alphas compared at the photo's size (MAE, IoU of α > 0.5, logits max |Δ|).
    func peEffect() async throws {
        let p = try await pipeline()
        let img = try image(try need("--image"))
        let tile = ImageHost.pillowBicubic(img, width: ImageHost.towerSide, height: ImageHost.towerSide)
        let v = try await p.tower.encode(patches: try ImageHost.patches(tile))
        let row = try p.prompt.row(prompt: try need("--prompt"))
        let gen = try await p.decoder.generate(prompt: row.ids, inputs: p.decoder.staticInputs(v, row: row), schedule: .pad, maxNew: 768)
        let response = p.prompt.response(gen.ids)
        let boxes = AnswerParse.segBoxes(response)
        guard let own = boxes.first, let hidden = gen.objHidden.first else { throw CutoutError.prompt("no box in \(response)") }
        let canvas = ImageHost.canvas(ImageHost.pillowBicubic(img, width: ImageHost.canvasSide, height: ImageHost.canvasSide))
        let unit = (0...1000).map { Float(Double($0) / 1000.0) }
        func pair(_ target: UInt32, near lo: Int, _ hi: Int) -> (Int, Int)? {
            var best: (Int, Int)? = nil, cost = Int.max
            for a in 0..<1000 {
                for b in (a + 1)...1000 where max(unit[b] - unit[a], Float(1e-6)).bitPattern == target {
                    let c = abs(a - lo) + abs(b - hi)
                    if c < cost { cost = c; best = (a, b) }
                }
            }
            return best
        }
        var rows: [[String: Any]] = []
        for target in BoxFourier.torchEncoded.keys.sorted() {
            guard let (x1, x2) = pair(target, near: own[0], own[2]), let (y1, y2) = pair(target, near: own[1], own[3]) else { continue }
            let box = [x1, y1, x2, y2]
            let (withTable, _) = try await p.masks.logits(canvas: canvas, taps: v.taps, hidden: [hidden], boxes: [AnswerParse.unit(box)])
            let (withLibm, _) = try await p.masks.logits(canvas: canvas, taps: v.taps, hidden: [hidden], boxes: [AnswerParse.unit(box)],
                                                          libmBoxPE: true)
            let a1 = Alpha.probabilities(withTable[0], width: img.width, height: img.height)
            let a2 = Alpha.probabilities(withLibm[0], width: img.width, height: img.height)
            var mae = 0.0, inter = 0, union = 0
            for i in 0..<a1.count {
                mae += Double(abs(a1[i] - a2[i]))
                let u = a1[i] > 0.5, w = a2[i] > 0.5
                if u && w { inter += 1 }
                if u || w { union += 1 }
            }
            var maxd: Float = 0
            for i in 0..<withTable[0].count { maxd = max(maxd, abs(withTable[0][i] - withLibm[0][i])) }
            rows.append(["width_bits": String(format: "0x%08X", target), "width": Float(bitPattern: target), "box": box,
                         "alpha_mae": mae / Double(a1.count), "iou_05": union == 0 ? 1.0 : Double(inter) / Double(union),
                         "logits_max_abs_d": maxd])
            log("PE-EFFECT width 0x\(String(target, radix: 16)) box \(box) MAE \(mae / Double(a1.count)) IoU \(union == 0 ? 1.0 : Double(inter) / Double(union))")
        }
        try writeJSON(["image": try need("--image"), "prompt": try need("--prompt"), "response": response, "own_box": own,
                       "rows": rows], try url("--out"))
    }

    /// Every box_pe value a box with 0–1000 coordinates can give: the fp32 widths w = max(fp32(b/1000) − fp32(a/1000), 1e-6)
    /// for 0 ≤ a < b ≤ 1000 (and the clamp), their encoded value ew, and the 64 features of every input value (the 1,001 x1 /
    /// y1 values and the distinct ew values).
    func boxpe() throws {
        let out = try url("--out")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let unit = (0...1000).map { Float(Double($0) / 1000.0) }
        var widths = Set<UInt32>()
        widths.insert(Float(1e-6).bitPattern)
        for a in 0..<1000 {
            for b in (a + 1)...1000 { widths.insert(max(unit[b] - unit[a], Float(1e-6)).bitPattern) }
        }
        let w = widths.sorted().map { Float(bitPattern: $0) }
        let t0 = ContinuousClock.now
        let ew = w.map(BoxFourier.encodedSide)
        let encodeSeconds = secondsSince(t0)
        var values = unit
        values += Array(Set(ew.map(\.bitPattern))).sorted().map { Float(bitPattern: $0) }
        var feats = [Float](repeating: 0, count: values.count * 64)
        let t1 = ContinuousClock.now
        for (i, v) in values.enumerated() { BoxFourier.features(v, into: &feats, at: i * 64) }
        let featureSeconds = secondsSince(t1)
        try Fixture.write(w, out.appendingPathComponent("w.f32.bin"))
        try Fixture.write(ew, out.appendingPathComponent("ew.f32.bin"))
        try Fixture.write(values, out.appendingPathComponent("values.f32.bin"))
        try Fixture.write(feats, out.appendingPathComponent("features.f32.bin"))
        try writeJSON(["widths": w.count, "values": values.count, "encode_seconds": encodeSeconds,
                       "feature_seconds": featureSeconds, "per_box_features_seconds": featureSeconds / Double(values.count) * 4],
                      out.appendingPathComponent("boxpe.json"))
        log("BOXPE widths=\(w.count) values=\(values.count) encode=\(encodeSeconds) s features=\(featureSeconds) s")
    }
}

if #available(macOS 27, *) {
    do {
        try await CLI(args: Array(CommandLine.arguments.dropFirst())).main()
    } catch {
        FileHandle.standardError.write("ERROR \(error)\n".data(using: .utf8)!)
        exit(1)
    }
} else {
    print("cutout-cli needs macOS 27")
    exit(1)
}
