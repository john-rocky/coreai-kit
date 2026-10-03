// ClefFlashTests.swift — clef-flash in the kit: the catalog entry and what loads it, the `images` field of the
// `/v1/systemone` wire, a request in the author's form, and — opt-in, with the weights — twelve runs of the model
// zoo's Swift gate, bit for bit (Fixtures/clef_flash_swift12.json: the author's ids and spans, the zoo host's logits,
// probabilities, hidden-state digest and response for the same assets):
//
//     KIT_CLEFFLASH_GATE=1 KIT_CLEFFLASH_IMAGES=<the fixture's images folder> [KIT_CLEFFLASH_OUT=<transcript.json>] \
//         swift test --filter ClefFlash
//
// The gate downloads clef-flash from the catalog pin on its first run (18.2 GB, and the 912 MB g448 tower), and the
// first load specializes the decoder (56 s and 31.8 GB of the runtime's cache in the zoo's run).

import CoreGraphics
import CryptoKit
import Foundation
import Testing

@testable import CoreAIKit

struct ClefFlashTests {
    static let revision = "220ac149ed1136465f7d9dac93aaf0dfc55d6b73"
    static let repo = "mlboydaisuke/clef-flash-CoreAI"

    // MARK: - The catalog entry

    @available(macOS 27, iOS 27, *)
    @Test func theCatalogEntryNamesEveryPartAtItsPin() throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "clef-flash"))
        #expect(entry.kind == .decision)
        #expect(entry.format == Decision.Format.jointHead.rawValue)
        #expect(entry.revision == Self.revision)
        #expect(entry.variants["ios"] == nil)  // the Mac only: the decoder is 15.9 GB
        let model = try ClefFlashModelID(entry: entry)
        #expect(model.decoder == ModelID(Self.repo, path: "gpu-pipelined/clef_flash_decode_fp16_pf64", revision: Self.revision))
        #expect(model.head == ModelID(Self.repo, path: "gpu-pipelined/clef_flash_head_bucket_fp16w32", revision: Self.revision))
        #expect(model.tableFolder == ModelID(Self.repo, path: "host", revision: Self.revision))
        #expect(model.tableFile == "lm_head_fp16.bin")
        #expect(model.towers == [
            .g256: ModelID(Self.repo, path: "gpu-pipelined/clef_flash_g256_vision_fp16w32", revision: Self.revision),
            .g448: ModelID(Self.repo, path: "gpu-pipelined/clef_flash_g448_vision_fp16w32", revision: Self.revision),
        ])
    }

    /// `TypedDecisions` would read the decoder's hidden states as logits: a joint-head entry is `KitClefDecider`'s,
    /// and every other decision entry stays where it was.
    @available(macOS 27, iOS 27, *)
    @Test func onlyKitClefDeciderLoadsAJointHeadEntry() throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "clef-flash"))
        #expect(!TypedDecisions.supports(entry))
        #expect(KitClefDecider.supports(entry))
        #expect(SystemOne.supports(entry))
        for other in ModelCatalog.builtin.available(.decision) where other.id != entry.id {
            #expect(TypedDecisions.supports(other), "\(other.id)")
            #expect(!KitClefDecider.supports(other), "\(other.id)")
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func theGridsAreTheTowersTiles() {
        #expect(KitClefDecider.Grid(tile: 256) == .g256 && KitClefDecider.Grid(tile: 448) == .g448)
        #expect(KitClefDecider.Grid(tile: 512) == nil)
        #expect(KitClefDecider.Grid.g448.imageTokens == 196 && KitClefDecider.Grid.g256.imageTokens == 64)
    }

    // MARK: - The wire's image

    static func request(images: String, extra: String = "") throws -> SystemOne.Request {
        try SystemOne.request(from: Data(
            #"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}, "images": \#(images)\#(extra)}"#.utf8))
    }

    @Test func theWireCarriesOneImage() throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
        let b64 = png.base64EncodedString()
        #expect(try Self.request(images: #"["data:image/png;base64,\#(b64)"]"#).images == [.data(png)])
        #expect(try Self.request(images: #"["\#(b64)"]"#).images == [.data(png)])
        // Plain base64 of a JPEG starts with "/9j/": base64, not a path.
        #expect(try Self.request(images: #"["\#(jpeg.base64EncodedString())"]"#).images == [.data(jpeg)])
        #expect(try Self.request(images: #"["/tmp/x.png"]"#).images == [.file(URL(fileURLWithPath: "/tmp/x.png"))])
        #expect(try Self.request(images: #"["file:///tmp/x.png"]"#).images == [.file(URL(string: "file:///tmp/x.png")!)])
        #expect(try Self.request(images: "null").images.isEmpty)
        #expect(try Self.request(images: #"["\#(b64)"]"#, extra: #", "grid": 256"#).grid == 256)
        #expect(throws: SystemOne.WireError.self) { try Self.request(images: #"["\#(b64)", "\#(b64)"]"#) }
        #expect(throws: SystemOne.WireError.self) { try Self.request(images: "[1]") }
        #expect(throws: SystemOne.WireError.self) { try Self.request(images: #""\#(b64)""#) }
        #expect(throws: SystemOne.WireError.self) { try Self.request(images: #"["\#(b64)"]"#, extra: #", "grid": "448""#) }
        // A request without the field reads as it always did.
        let plain = try SystemOne.request(from: Data(#"{"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}"#.utf8))
        #expect(plain.images.isEmpty && plain.grid == nil)
    }

    /// A backend that does not read images.
    struct TextOnly: DecisionBackend {
        let id = "text-only"
        let maxOptions = 10
        var modelName: String { get async { "text-only" } }
        func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer { throw DecisionError.noLogits }
        func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response { throw DecisionError.noLogits }
    }

    @Test func aBackendThatReadsNoImagesRefusesOneByName() throws {
        let refusal = try #require(TextOnly().imageRefusal(Self.request(images: #"["/tmp/x.png"]"#)))
        #expect(refusal.message.contains("'text-only' reads no images"))
        #expect(TextOnly().imageRefusal(try Self.request(images: "[]")) == nil)
        #expect(SystemOneServer.isLoopback("127.0.0.1") && SystemOneServer.isLoopback("::1"))
        #expect(!SystemOneServer.isLoopback("0.0.0.0") && !SystemOneServer.isLoopback("192.168.1.2"))
    }

    // MARK: - The request in the author's form

    /// The wire's own values: a structured state as canonical JSON with its numbers as written, a choice listed as
    /// names read as options alone, `model` filled in.
    @available(macOS 27, iOS 27, *)
    @Test func aWireRequestIsReadFromItsOwnValues() throws {
        let body = #"""
            {"state": {"total": 1250.0, "count": 3, "note": "é/x"},
             "questions": {"pick": {"type": "choice", "instructions": "Which?", "criteria": ["b", "a"]},
                           "ok": {"type": "noul", "instructions": {"rule": 1e-5}}}}
            """#
        let clef = try KitClefDecider.clefRequest(try SystemOne.request(from: Data(body.utf8)), id: "clef-flash")
        #expect(clef.model == "clef-flash")
        #expect(ClefPythonJSON.render(clef.state) == #"{"count":3,"note":"é/x","total":1250.0}"#)
        let pick = try clef.questions[0].options()
        #expect(pick.map(\.id) == ["a", "b"])  // by code point
        #expect(pick.allSatisfy { $0.description == nil })
        #expect(clef.questions[1].instructions.map(ClefPythonJSON.render) == #"{"rule":1e-05}"#)
        #expect(try clef.questions[1].options().map(\.id) == ["true", "false"])
    }

    /// A request built in Swift: an option whose description is its id reads as an option alone.
    @available(macOS 27, iOS 27, *)
    @Test func aSwiftRequestIsReadFromItsTypedQuestions() throws {
        let request = SystemOne.Request(state: "s", questions: [
            (id: "c", question: .choice("Which?", options: [.init(id: "x", description: "the x"), .init("y")])),
            (id: "s", question: .score("How?", levels: ["low", "high"])),
            (id: "n", question: .noul("Yes?", yes: "it holds")),
        ])
        let clef = try KitClefDecider.clefRequest(request, id: "clef-flash")
        let c = try clef.questions[0].options()
        #expect(c.map(\.id) == ["x", "y"] && c[0].description == .string("the x") && c[1].description == nil)
        #expect(try clef.questions[1].options().map(\.id) == ["0", "1"])
        let n = try clef.questions[2].options()
        #expect(n[0].description == .string("it holds"))
        #expect(n[1].description == .string("The proposition is false or the answer is no."))
    }

    @available(macOS 27, iOS 27, *)
    @Test func probabilitiesComeBackInTheKitsOptionOrder() throws {
        #expect(try KitClefDecider.kitOrder(.noul("?"), ["true": 0.75, "false": 0.25], id: "q") == [0.25, 0.75])
        #expect(try KitClefDecider.kitOrder(.choice("?", ["b", "a"]), ["a": 0.25, "b": 0.75], id: "q") == [0.75, 0.25])
        #expect(try KitClefDecider.kitOrder(.score("?", levels: ["x", "y"]), ["0": 0.5, "1": 0.5], id: "q") == [0.5, 0.5])
        #expect(throws: SystemOne.WireError.self) { try KitClefDecider.kitOrder(.choice("?", ["a", "c"]), ["a": 1], id: "q") }
    }

    @Test func theRendererWritesWhatPythonWrites() {
        #expect(ClefPythonJSON.render(.string("as is")) == "as is")
        #expect(ClefPythonJSON.render(.number("1250.0")) == "1250.0")
        #expect(ClefPythonJSON.render(.number("1e-5")) == "1e-05")
        #expect(ClefPythonJSON.render(.number("-0")) == "0")
        #expect(ClefPythonJSON.render(.object([ClefJSONMember("é", .number("1")), ClefJSONMember("b", .string("\u{1}/"))]))
            == #"{"b":"\u0001/","é":1}"#)
        #expect(ClefPythonJSON.pyRound(0.97825, 4) == 0.9782)
    }

    // MARK: - Twelve runs of the zoo's Swift gate, bit for bit (opt-in: the weights)

    static var environment: [String: String] { ProcessInfo.processInfo.environment }

    static func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/\(name)")
    }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: environment["KIT_CLEFFLASH_GATE"] == "1"))
    func twelveRunsMatchTheZooSwiftHostBitForBit() async throws {
        let reference = try JSONValue.parse(Data(contentsOf: Self.fixtureURL("clef_flash_swift12.json")))
        let runs = try #require(reference["runs"]?.elements)
        let imageDirs = (Self.environment["KIT_CLEFFLASH_IMAGES"] ?? "").split(separator: ",").map { URL(fileURLWithPath: String($0)) }
        let start = ContinuousClock.now
        let decider = try await KitClefDecider(catalog: "clef-flash")
        let load = ContinuousClock.now - start
        var rows: [JSONValue] = []
        var bitEqual = 0
        for run in runs {
            let key = "\(run["id"]?.stringValue ?? "")/\(run["arm"]?.stringValue ?? "")"
            var image: CGImage? = nil
            if let name = run["image_file"]?.stringValue {
                let url = try #require(
                    imageDirs.map { $0.appendingPathComponent(name) }.first { FileManager.default.fileExists(atPath: $0.path) },
                    "\(key): KIT_CLEFFLASH_IMAGES holds no \(name)")
                image = try KitClefDecider.cgImage(.file(url))
            }
            let request = Data(try #require(run["request"]).dumps().utf8)
            let r = try await decider.readout(requestJSON: request, image: image, grid: .g448)

            let idsDigest = SHA256.hash(data: Data(r.ids.map(String.init).joined(separator: ",").utf8))
                .map { String(format: "%02x", $0) }.joined()
            let idsEqual = idsDigest == run["ids_sha256"]?.stringValue && r.ids.count == Int(run["tokens"]?.doubleValue ?? -1)
            let ints: (JSONValue?) -> [Int] = { ($0?.elements ?? []).compactMap { $0.doubleValue.map(Int.init) } }
            let spansEqual = r.questions.map { [$0.questionSpan] + $0.optionSpans } == (run["questions"]?.elements ?? []).map {
                [ints($0["question_span"])] + ($0["option_spans"]?.elements ?? []).map(ints)
            } && r.questions.map(\.optionIDs) == (run["questions"]?.elements ?? []).map { ($0["option_ids"]?.elements ?? []).compactMap(\.stringValue) }
            let floats: (JSONValue?) -> [Float] = { ($0?.elements ?? []).compactMap { $0.doubleValue.map(Float.init) } }
            let logitsEqual = r.logits.map(\.bitPattern) == floats(run["logits"]).map(\.bitPattern)
            let probabilitiesEqual = r.probabilities.map { $0.map(\.bitPattern) }
                == (run["probabilities"]?.elements ?? []).map { floats($0).map(\.bitPattern) }
            let hiddenEqual = r.hiddenSHA256 == run["hidden_sha256"]?.stringValue
            let responseEqual = (try? JSONValue.parse(r.response)) == run["response"]
            let shapeEqual = r.bucket == run["bucket"]?.stringValue && r.calls == Int(run["calls"]?.doubleValue ?? -1)
            #expect(idsEqual, "\(key): ids differ from the author's")
            #expect(spansEqual, "\(key): spans or option order differ from the author's")
            #expect(logitsEqual, "\(key): logits differ from the zoo host's")
            #expect(probabilitiesEqual, "\(key): probabilities differ from the zoo host's")
            #expect(hiddenEqual, "\(key): the decoder's hidden rows differ from the zoo host's")
            #expect(responseEqual, "\(key): the response differs from the zoo host's")
            #expect(shapeEqual, "\(key): bucket or decoder calls differ")
            let all = idsEqual && spansEqual && logitsEqual && probabilitiesEqual && hiddenEqual && responseEqual && shapeEqual
            if all { bitEqual += 1 }
            rows.append(.object([
                .init("run", .string(key)), .init("tokens", .int(r.ids.count)), .init("bucket", .string(r.bucket)),
                .init("calls", .int(r.calls)), .init("ids_equal_author", .bool(idsEqual)),
                .init("spans_equal_author", .bool(spansEqual)), .init("logits_bit_equal_zoo_swift", .bool(logitsEqual)),
                .init("probabilities_bit_equal_zoo_swift", .bool(probabilitiesEqual)),
                .init("hidden_sha256_equal_zoo_swift", .bool(hiddenEqual)), .init("response_equal_zoo_swift", .bool(responseEqual)),
                .init("hidden_sha256", .string(r.hiddenSHA256)),
            ]))
        }
        #expect(bitEqual == runs.count)
        if let out = Self.environment["KIT_CLEFFLASH_OUT"] {
            let transcript = JSONValue.object([
                .init("schema", .string("coreai-kit-clef-flash-test/1")),
                .init("test", .string("ClefFlashTests.twelveRunsMatchTheZooSwiftHostBitForBit")),
                .init("model", .string(decider.id)), .init("bundle", .string(decider.modelName)),
                .init("load_seconds", .number("\(Double(load.components.seconds) + Double(load.components.attoseconds) * 1e-18)")),
                .init("runs", .int(runs.count)), .init("bit_equal_runs", .int(bitEqual)), .init("rows", .array(rows)),
            ])
            try transcript.dumps().appending("\n").write(toFile: out, atomically: true, encoding: .utf8)
        }
    }
}
