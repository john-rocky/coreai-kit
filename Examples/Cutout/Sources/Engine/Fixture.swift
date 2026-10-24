// Fixture — the same request code over a list of rows, writing what a check needs: the ids, the answer, the boxes, the
// inputs the model got and the mask logits, per row. The Mac CLI (`cutout-cli fixture`) and the app (`-fixture`, on the
// phone) both run it, so the phone's numbers come from the code the screen runs.
//
//   rows file   {"rows": [{"id": …, "image": <path relative to the rows file or `images`>, "prompt": <as written>,
//               "task": "segmentation" | "matting" | "detection" | …}, …]}
//   per row     <out>/<form>/<id>/result.json, patches.f16.bin [784 · 1536], canvas.u8.bin [1024 · 1024 · 3],
//               hidden.f16.bin [k · 2048], box_pe.f32.bin [16 · 256 per graph A call], logits_<i>.f32.bin [1024²],
//               and with `writeAlpha` alpha_<i>.f32.bin (the probabilities at the photo's size)
//   form        "<schedule>-<graphs>", e.g. "s1-fp32" or "pad-fp16"

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public enum Fixture {
    public struct Row: Codable, Sendable {
        public let id: String
        public let image: String
        public let prompt: String
        public let task: String?
    }

    struct RowsFile: Codable {
        let rows: [Row]
    }

    public static func rows(at url: URL) throws -> [Row] {
        try JSONDecoder().decode(RowsFile.self, from: Data(contentsOf: url)).rows
    }

    static func write<T>(_ values: [T], _ url: URL) throws {
        try values.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
    }

    /// Runs every row in every form and writes the outputs; returns one summary line per row and form.
    @discardableResult
    public static func run(pipeline: CutoutPipeline, rows: [Row], imagesBase: URL, out: URL,
                           forms: [(schedule: WeVisOneDecoder.Schedule, masks: MaskGraphs, name: String)],
                           maxNew: Int = 768, writeAlpha: Bool, inputs: URL? = nil,
                           log: (String) -> Void) async throws -> [[String: Any]] {
        var summary: [[String: Any]] = []
        for row in rows {
            let file = imagesBase.appendingPathComponent(row.image)
            let loaded = try ImageHost.load(url: file)
            let tDecode = ContinuousClock.now
            let image = try ImageHost.uprightRGB8(loaded.image, orientation: loaded.orientation)
            let decodeSeconds = secondsSince(tDecode)
            let task: CutoutTask = row.task == "detection" ? .boxes : .cutout
            // another host's tensors for this row (<inputs>/<id>/patches.f16.bin [784 · 1536], canvas.u8.bin [1024² · 3])
            var given: (patches: [Float16], canvas: RGB8Image)? = nil
            if let inputs {
                let p = inputs.appendingPathComponent(row.id).appendingPathComponent("patches.f16.bin")
                let c = inputs.appendingPathComponent(row.id).appendingPathComponent("canvas.u8.bin")
                if let pd = try? Data(contentsOf: p), let cd = try? Data(contentsOf: c) {
                    let patches = pd.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
                    given = (patches, RGB8Image(width: 1024, height: 1024, pixels: [UInt8](cd), decodePath: "given"))
                }
            }
            for form in forms {
                let dir = out.appendingPathComponent(form.name).appendingPathComponent(row.id)
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let r = try await pipeline.run(image: image, words: "", task: task, schedule: form.schedule, maxNew: maxNew,
                                               recordGaps: true, keepInputs: true, instruction: row.prompt, masks: form.masks,
                                               inputs: given)
                var files: [String: String] = [:]
                if let p = r.patches { try write(p, dir.appendingPathComponent("patches.f16.bin")); files["patches"] = "patches.f16.bin" }
                if let c = r.canvas8 { try Data(c.pixels).write(to: dir.appendingPathComponent("canvas.u8.bin")); files["canvas"] = "canvas.u8.bin" }
                try write(r.generation.objHidden.flatMap { $0 }, dir.appendingPathComponent("hidden.f16.bin"))
                try write(r.boxPE.flatMap { $0 }, dir.appendingPathComponent("box_pe.f32.bin"))
                var alphaSeconds: [Double] = []
                for (i, inst) in r.instances.enumerated() {
                    guard let logits = inst.logits else { continue }
                    try write(logits, dir.appendingPathComponent("logits_\(i).f32.bin"))
                    if writeAlpha {
                        let ta = ContinuousClock.now
                        let p = Alpha.probabilities(logits, width: r.width, height: r.height)
                        alphaSeconds.append(secondsSince(ta))
                        try write(p, dir.appendingPathComponent("alpha_\(i).f32.bin"))
                    }
                }
                let rec: [String: Any] = [
                    "id": row.id, "form": form.name, "schedule": form.schedule.rawValue,
                    "graphs": "\(form.masks.precision.a)/\(form.masks.precision.b)",
                    "prompt": row.prompt, "prompt_ids": r.row.ids, "processor_ids": r.row.processorIDs,
                    "rope_shift": [r.row.ropeShiftStart, r.row.ropeShiftAmount],
                    "generated_ids": r.generation.ids, "response": r.response,
                    "boxes": r.instances.map(\.box), "labels": r.instances.map(\.label),
                    "n_hidden": r.generation.objHidden.count, "obj_positions": r.generation.objPositions,
                    "gaps": r.generation.gaps, "stop": r.generation.stop,
                    "prompt_calls": r.generation.promptCalls.map { ["kind": $0.kind, "tokens": $0.tokens, "seconds": $0.seconds] },
                    "decode_calls": r.generation.decodeCalls, "seconds": r.seconds, "decode_image_seconds": decodeSeconds,
                    "graph_a_seconds": r.graphA, "graph_b_seconds": r.graphB, "alpha_seconds": alphaSeconds,
                    "image": ["file": row.image, "width": r.width, "height": r.height, "decode_path": image.decodePath],
                    "footprint": ["now": memoryFootprint().now, "peak": memoryFootprint().peak],
                    "files": files,
                ]
                let json = try JSONSerialization.data(withJSONObject: rec, options: [.prettyPrinted, .sortedKeys])
                try json.write(to: dir.appendingPathComponent("result.json"))
                summary.append(["id": row.id, "form": form.name, "boxes": r.instances.map(\.box), "stop": r.generation.stop,
                                "total_s": r.seconds["total"] ?? 0])
                log(String(format: "FIXTURE %@ %@ boxes=%d hidden=%d tokens=%d stop=%@ total=%.3f s response=%@",
                           row.id, form.name, r.instances.count, r.generation.objHidden.count, r.generation.ids.count,
                           r.generation.stop, r.seconds["total"] ?? 0, r.response))
            }
        }
        return summary
    }
}
