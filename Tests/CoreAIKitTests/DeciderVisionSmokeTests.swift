import CoreAIKitVision
import CoreGraphics
import Foundation
import XCTest

@testable import CoreAIKit

/// decider-2b-vision end to end against the author's fp32 read-out: the zoo fixture's `g256`, `g448` and `text` runs
/// (76 runs, 108 answer slots) through `KitVisionDecider` — PNG decode, Pillow-order resize, the tower, the author's
/// prompt, the decoder's chunk order, the letter softmax. Opt-in (loads 3.3 GB of graphs; a minute or two on an M4
/// Max after the first specialization):
///
///     KIT_DECIDERVISION_DECODER=<decoder bundle dir> \
///     KIT_DECIDERVISION_TOWER_G256=<g256 tower dir or .aimodel> KIT_DECIDERVISION_TOWER_G448=<g448 tower> \
///     KIT_DECIDERVISION_FIXTURE=<zoo>/models/decider-2b-vision/fixtures-decider-2b-vision.json \
///     KIT_DECIDERVISION_IMAGES=<the fixture's PNGs> [KIT_DECIDERVISION_REFERENCE=<zoo Swift run .json>] \
///     swift test --filter DeciderVisionSmoke
///
/// The bar is the zoo's, fixed before its first result: every run's ids and slots equal the author's, the letter
/// argmax and the full-vocabulary top-1 equal the author's on every slot, every logit finite, max |Δp| ≤ 0.02 and the
/// mean over runs of each run's mean |Δp| ≤ 0.002; and the first run read again repeats its slot logits bit for bit.
/// With a reference (the zoo's Swift run of the same files, `decider-vision fixture --out`), the argmax must agree on
/// every slot; |Δp| and the letter logits bit-equal to it are reported.
@available(macOS 27, iOS 27, *)
final class DeciderVisionSmokeTests: XCTestCase {
    static let maxDelta = 0.02
    static let meanDelta = 0.002

    struct Fixture: Decodable {
        struct Request: Decodable {
            struct Question: Decodable {
                let text: String
                let options: [String]
            }
            let id: String
            let image: String?
            let context: String
            let questions: [Question]
        }
        struct Row: Decodable {
            let id: String
            let arm: String
            let request_id: String
            let ids: [Int]
            let slots: [Int]
            let p_oracle: [[Double]]
            let argmax: [Int]
            let full_vocab_top1_id: [Int]
        }
        let schema: String
        let requests: [Request]
        let rows: [Row]
    }

    struct Reference: Decodable {
        struct Run: Decodable {
            struct Answer: Decodable {
                let letter_logits: [Double]
                let probs: [Double]
                let argmax: Int
            }
            let id: String
            let arm: String
            let answers: [Answer]
        }
        let runs: [Run]
    }

    static func env(_ key: String) -> String? { ProcessInfo.processInfo.environment[key] }

    func testTheFixtureMeetsTheZoosBar() async throws {
        guard let decoder = Self.env("KIT_DECIDERVISION_DECODER"), let g256 = Self.env("KIT_DECIDERVISION_TOWER_G256"),
            let g448 = Self.env("KIT_DECIDERVISION_TOWER_G448"), let fixturePath = Self.env("KIT_DECIDERVISION_FIXTURE"),
            let images = Self.env("KIT_DECIDERVISION_IMAGES")
        else {
            throw XCTSkip(
                "Set KIT_DECIDERVISION_DECODER, KIT_DECIDERVISION_TOWER_G256, KIT_DECIDERVISION_TOWER_G448, "
                    + "KIT_DECIDERVISION_FIXTURE and KIT_DECIDERVISION_IMAGES to run.")
        }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
        XCTAssertEqual(fixture.schema, "coreai-decider-vision-fixtures/1")
        var reference: [String: Reference.Run] = [:]
        if let path = Self.env("KIT_DECIDERVISION_REFERENCE") {
            let runs = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: path))).runs
            for run in runs { reference["\(run.id)/\(run.arm)"] = run }
        }

        let decider = try await KitVisionDecider(
            decoderAt: URL(fileURLWithPath: decoder),
            towersAt: [.g256: URL(fileURLWithPath: g256), .g448: URL(fileURLWithPath: g448)])
        let requests = Dictionary(uniqueKeysWithValues: fixture.requests.map { ($0.id, $0) })
        let imageRoot = URL(fileURLWithPath: images)

        func read(_ row: Fixture.Row) async throws -> KitVisionDecider.Readout {
            let request = try XCTUnwrap(requests[row.request_id])
            var image: CGImage? = nil
            if row.arm != "text", let name = request.image {
                image = try ImageFile.load(imageRoot.appendingPathComponent("\(name).png")).cgImage
            }
            return try await decider.readout(
                image: image, state: request.context,
                questions: request.questions.map { .choice($0.text, $0.options) },
                grid: row.arm == "g448" ? .g448 : .g256)
        }

        var runs = 0, slots = 0, argmax = 0, top1 = 0
        var worst = 0.0, worstAt = ""
        var runMeans: [Double] = []
        var refSlots = 0, refArgmax = 0, refBitEqual = 0
        var refWorst = 0.0
        var first: (row: Fixture.Row, logits: [[Float]])? = nil
        for row in fixture.rows where ["g256", "g448", "text"].contains(row.arm) {
            let r = try await read(row)
            if first == nil { first = (row, r.letterLogits) }
            runs += 1
            XCTAssertEqual(r.row.processorIDs, row.ids, "\(row.id): ids")
            XCTAssertEqual(r.row.slots, row.slots, "\(row.id): slots")
            XCTAssertEqual(r.probabilities.count, row.p_oracle.count, "\(row.id): answers")
            var deltas: [Double] = []
            for (s, (p, po)) in zip(r.probabilities, row.p_oracle).enumerated() {
                slots += 1
                let d = zip(p, po).map { abs($0 - $1) }
                deltas += d
                if let m = d.max(), m > worst { (worst, worstAt) = (m, "\(row.id) slot \(s)") }
                let best = DeciderVisionNumerics.argmax(p)
                if best == row.argmax[s] { argmax += 1 }
                if r.fullVocabularyTop1[s] == row.full_vocab_top1_id[s] { top1 += 1 }
                XCTAssertEqual(best, row.argmax[s], "\(row.id) slot \(s): argmax")
                XCTAssertTrue(r.finite[s], "\(row.id) slot \(s): a logit is not finite")
                if let ref = reference["\(row.request_id)/\(row.arm)"], s < ref.answers.count {
                    let a = ref.answers[s]
                    refSlots += 1
                    if best == a.argmax { refArgmax += 1 }
                    if r.letterLogits[s] == a.letter_logits.map(Float.init) { refBitEqual += 1 }
                    refWorst = max(refWorst, zip(p, a.probs).map { abs($0 - $1) }.max() ?? 0)
                }
            }
            runMeans.append(deltas.reduce(0, +) / Double(max(1, deltas.count)))
        }
        let mean = runMeans.reduce(0, +) / Double(max(1, runMeans.count))
        print(
            "PROBE decider-2b-vision: runs \(runs) slots \(slots) argmax \(argmax)/\(slots) "
                + "full-vocab top-1 \(top1)/\(slots) max |dp| \(worst) (\(worstAt)) mean of run means \(mean)")
        XCTAssertEqual(runs, 76)
        XCTAssertEqual(slots, 108)
        XCTAssertEqual(argmax, slots)
        XCTAssertEqual(top1, slots)
        XCTAssertLessThanOrEqual(worst, Self.maxDelta, worstAt)
        XCTAssertLessThanOrEqual(mean, Self.meanDelta)
        if !reference.isEmpty {
            print("PROBE vs reference: argmax \(refArgmax)/\(refSlots) letter logits bit-equal \(refBitEqual)/\(refSlots) max |dp| \(refWorst)")
            XCTAssertEqual(refSlots, 108)
            XCTAssertEqual(refArgmax, refSlots)
        }

        // The states are zeroed per row: the first run, read again after every other, repeats bit for bit.
        let f = try XCTUnwrap(first)
        let again = try await read(f.row)
        XCTAssertEqual(again.letterLogits, f.logits, "\(f.row.id): the re-read differs")
    }
}
