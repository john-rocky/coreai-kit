// D13BTests.swift — d1-3B in the kit: the catalog entry and what loads it, the request in the provider's form, the
// order the probabilities come back in, the texts and the readout against the Python reference, the refusals, and —
// with the files — the fixture's rows token for token and runs of the shipped Mac decoder bit for bit against the
// Python reference's run of the same bundle (Fixtures/d1_3b_swift.json: 24 text questions of 16 records and 4 questions
// about 2 drawn pictures, with each row's ids, hidden-row digest, z and p, and each record's response in the provider's
// form; the zoo's Swift host gave every one of these rows bit for bit on the same assets):
//
//     swift test --filter D13B                                                  # without files
//     KIT_D1_3B_TOKENIZER=<a decoder bundle's tokenizer/> swift test --filter D13B   # + every fixture row's ids
//     KIT_D1_3B_GATE=1 [KIT_D1_3B_IMAGES=<the drawn pictures>] [KIT_D1_3B_OUT=<transcript.json>] swift test --filter D13B
//     KIT_D1_3B_RECORDS=<records.json> KIT_D1_3B_RENDER=<render_ids.json> swift test --filter D13B   # the zoo's 393 rows
//
// The gate downloads the Mac's decoder bundle (5.6 GB) from the catalog pin when the store does not hold it and the tower
// (854 MB) for the pictures; `KIT_D1_3B_STORE=<dir>` points the tests at a ModelStore directory other than the default, and a store
// that already holds the decoder runs the gate without the variable. The pictures are not in the repository: the
// zoo's conversion/d1/make_fixture_images.py draws them, and the fixture holds their decoded pixels' sha256.

import CryptoKit
import Foundation
import Testing

@testable import CoreAIKit

struct D13BTests {
    static let pin = "3cf8b7ca20ee5e2acf20ff522f4ffa7201aa8fb8"
    static let environment = ProcessInfo.processInfo.environment

    // MARK: - The catalog entry

    @available(macOS 27, iOS 27, *)
    @Test func theCatalogEntryIsD13BAtItsPin() throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "d1-3b"))
        #expect(entry.kind == .tokenDecision && entry.format == KitD1Decider.format)
        #expect(entry.repo == "mlboydaisuke/d1-3B-CoreAI" && entry.revision == Self.pin)
        #expect(entry.variants["macos"]?.path == "gpu-pipelined/d1_3b_decode_fp16_pf64_s")
        #expect(entry.variants["ios"]?.path == "gpu-pipelined/d1_3b_decode_int8mlp_pf64_s")
        #expect(entry.variants["macos"]?.sizeMB == 5598 && entry.variants["ios"]?.sizeMB == 3740)
        #expect(entry.assets == CatalogEntry.Assets(tower: "gpu-pipelined/d1_3b_vision_fp16w32_s"))
        #expect(entry.engine == nil && entry.calibration == nil && entry.license == nil)
        #expect(entry.modelID(path: "gpu-pipelined/d1_3b_vision_fp16w32_s")
            == ModelID("mlboydaisuke/d1-3B-CoreAI", path: "gpu-pipelined/d1_3b_vision_fp16w32_s", revision: Self.pin))
    }

    /// The catalog's `assets.tower` decodes beside clef-flash's fields, and a kit that does not know the field reads the
    /// entry without it.
    @Test func theTowerAssetDecodes() throws {
        let json = #"{"head": null, "tower": "gpu-pipelined/t"}"#
        let assets = try JSONDecoder().decode(CatalogEntry.Assets.self, from: Data(json.utf8))
        #expect(assets.tower == "gpu-pipelined/t" && assets.head == nil && assets.towers == nil)
        let clef = try JSONDecoder().decode(CatalogEntry.Assets.self, from: Data(#"{"head": "h", "table": "t"}"#.utf8))
        #expect(clef.tower == nil && clef.head == "h")
    }

    /// The bundles carry `kind: decision-backbone` and no `decision.head`; only `KitD1Decider` reads them. An entry of
    /// this kind is `KitD1Decider`'s alone, `TypedDecisions(catalog:)` (and so `CoreAI.decide` and `CoreAI.systemOne`,
    /// which load through it) refuses it by its kind before it downloads anything, and `KitD1Decider(catalog:)` refuses
    /// another kind the same way.
    @available(macOS 27, iOS 27, *)
    @Test func onlyKitD1DeciderLoadsATokenDecisionEntry() async throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "d1-3b"))
        #expect(!TypedDecisions.supports(entry) && !KitClefDecider.supports(entry) && !KitKevDecider.supports(entry))
        #expect(!KitD1OmniDecider.supports(entry))
        #expect(KitD1Decider.supports(entry) && SystemOne.supports(entry))
        for other in ModelCatalog.builtin.available(.decision) + ModelCatalog.builtin.available(.jointDecision)
            + ModelCatalog.builtin.available(.rowDecision) + ModelCatalog.builtin.available(.omniDecision)
        {
            #expect(!KitD1Decider.supports(other), "\(other.id)")
        }
        let otherFormat = CatalogEntry(
            id: entry.id, name: entry.name, repo: entry.repo, revision: entry.revision, kind: .tokenDecision,
            variants: entry.variants, format: "markerScores")
        #expect(!KitD1Decider.supports(otherFormat))
        #expect(ModelCatalog.builtin.available(.tokenDecision).map(\.id) == ["d1-3b"])

        // The store's Hub is a closed local port: a download would fail on the transport, not on the kind.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(directory: root, hubBaseURL: URL(string: "http://127.0.0.1:9")!)
        do {
            _ = try await TypedDecisions(catalog: "d1-3b", store: store)
            Issue.record("TypedDecisions loaded d1-3b")
        } catch let error as CoreAIKitError {
            guard case .catalogKindMismatch(let id, let expected, let found) = error else {
                Issue.record("not the kind refusal: \(error)")
                return
            }
            #expect(id == "d1-3b" && expected == "chat or decision" && found == "tokenDecision")
        }
        do {
            _ = try await KitD1Decider(catalog: "kev-0.8b", store: store)
            Issue.record("KitD1Decider loaded kev-0.8b")
        } catch let error as CoreAIKitError {
            guard case .catalogKindMismatch(let id, let expected, let found) = error else {
                Issue.record("not the kind refusal: \(error)")
                return
            }
            #expect(id == "kev-0.8b" && expected == "tokenDecision" && found == "rowDecision")
        }
        #expect(((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []).isEmpty)
    }

    // MARK: - The request in the provider's form

    /// The wire's own values: a structured state with its numbers as written and a repeated key in the place it had with
    /// its last value (`json.loads`), a choice listed as names read as labels alone.
    @available(macOS 27, iOS 27, *)
    @Test func aWireRequestIsReadFromItsOwnValues() throws {
        let body = #"""
            {"state": {"total": 1250.0, "n": 3, "total": 1e-5, "ok": true, "who": "é"},
             "questions": {"pick": {"type": "choice", "instructions": "Which?", "criteria": ["b", "a"]},
                           "ok": {"type": "noul", "instructions": "Holds?", "criteria": {"true": "it holds"}},
                           "how": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}}
            """#
        let request = try SystemOne.request(from: Data(body.utf8))
        let root = KitD1Decider.providerRequest(request)
        let d1 = try D13BRequest(json: root)
        #expect(D13BText.prefix(d1.state)
            == "<|startoftext|><|im_start|>user\n{\n  \"total\": 1e-05,\n  \"n\": 3,\n  \"ok\": true,\n  \"who\": \"é\"\n}\n\n\nQUESTION:\n")
        #expect(d1.questions.map(\.name) == ["pick", "ok", "how"])
        #expect(d1.questions[0].labels == ["b", "a"] && d1.questions[0].options.allSatisfy { $0.desc == nil })
        #expect(D13BText.questionBlock(d1.questions[1], codes: nil)
            == "Holds?\nYes: it holds\nNo: None\n\nReply with yes or no only.")
        #expect(D13BText.questionBlock(d1.questions[2], codes: nil)
            == "How?\n\n0 low\n1 high\n\nReply with a single digit 0-1 only.")
    }

    /// Text the wire carries as a JSON value — structured instructions, a score's levels as objects — reads as its JSON
    /// text, the kit's wire rule for every model; the provider's own form keeps its refusal of anything but text.
    @available(macOS 27, iOS 27, *)
    @Test func aWireRequestReadsJSONValuedTextAsItsJSONText() throws {
        let body = #"""
            {"state": "s", "questions": {
              "same": {"type": "noul", "instructions": {"who": {"name": "A"}, "question": "Same person?"}},
              "priority": {"type": "score", "instructions": "How?", "criteria": [{"level": "low"}, {"level": "high"}]}}}
            """#
        let d1 = try D13BRequest(json: KitD1Decider.providerRequest(try SystemOne.request(from: Data(body.utf8))))
        #expect(d1.questions[0].instructions == #"{"who": {"name": "A"}, "question": "Same person?"}"#)
        #expect(d1.questions[1].levels == [#"{"level": "low"}"#, #"{"level": "high"}"#])
        #expect(throws: D13BError.self) { try D13BRequest(data: Data(body.utf8)) }
    }

    /// A request built in Swift: an option whose description is its id reads as a label alone.
    @available(macOS 27, iOS 27, *)
    @Test func aSwiftRequestIsReadFromItsTypedQuestions() throws {
        let request = SystemOne.Request(state: "s", questions: [
            (id: "c", question: .choice("Which?", options: [.init(id: "x", description: "the x"), .init("y")])),
            (id: "s", question: .score("How?", levels: ["low", "high"])),
            (id: "n", question: .noul("Yes?", no: "it fails")),
        ])
        let d1 = try D13BRequest(json: KitD1Decider.providerRequest(request))
        #expect(d1.state == .string("s"))
        #expect(d1.questions.map(\.name) == ["c", "s", "n"])
        #expect(d1.questions[0].options.map(\.label) == ["x", "y"] && d1.questions[0].options.map(\.desc) == ["the x", nil])
        #expect(d1.questions[1].levels == ["low", "high"])
        #expect(D13BText.questionBlock(d1.questions[2], codes: nil) == "Yes?\nYes: None\nNo: it fails\n\nReply with yes or no only.")
    }

    @available(macOS 27, iOS 27, *)
    @Test func probabilitiesComeBackInTheKitsOrder() throws {
        // The provider reports a noul as [yes, no]; the kit's order is [no, yes].
        #expect(try KitD1Decider.kitOrder(.noul("?"), keys: ["yes", "no"], [0.75, 0.25], id: "q") == [0.25, 0.75])
        #expect(try KitD1Decider.kitOrder(.choice("?", ["b", "a"]), keys: ["b", "a"], [0.75, 0.25], id: "q") == [0.75, 0.25])
        #expect(try KitD1Decider.kitOrder(.score("?", levels: ["x", "y", "z"]), keys: ["0", "1", "2"], [0.1, 0.2, 0.7], id: "q")
            == [0.1, 0.2, 0.7])
        #expect(throws: SystemOne.WireError.self) {
            try KitD1Decider.kitOrder(.choice("?", ["a", "b"]), keys: ["b", "a"], [0.5, 0.5], id: "q")
        }
    }

    // MARK: - The fixture

    struct Fixture {
        struct Question {
            let name: String
            let type: String
            let keys: [String]
            let codes: [String]?
            let rowLength: Int
            let slot: Int
            let rowIDsSHA256: String
            let textSHA256: String?
            let groups: [[Int]]
            let hiddenSHA256: String
            let logitIDs: [Int]
            let zBits: [UInt64]
            let pBits: [UInt64]
            let oracle: [Double]
            let nearTie: Bool
        }

        struct Record {
            let id: String
            /// The request's JSON text, as `json.dumps(request)` writes it (members in order).
            let requestJSON: String
            let request: JSONValue
            let inputTokens: Int
            let response: String
            let questions: [Question]
            /// Pictures: the file name, the size as it arrives, the decoded pixels' sha256; the crops and their tower
            /// outputs' sha256.
            let picture: (file: String, width: Int, height: Int, rgbSHA256: String)?
            let towerOutputs: [String]
        }

        let records: [Record]
        let imageRecords: [Record]

        static func load() throws -> Fixture {
            let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/d1_3b_swift.json")
            let root = try JSONValue.parse(Data(contentsOf: url))
            #expect(root["schema"]?.stringValue == "coreai-kit-d1-3b-swift-reference/1")
            #expect(root["model"]?["revision"]?.stringValue == D13BTests.pin)
            func ints(_ v: JSONValue?) -> [Int] { (v?.elements ?? []).compactMap { $0.doubleValue.map(Int.init) } }
            func hex(_ v: JSONValue?) -> [UInt64] { (v?.elements ?? []).compactMap { $0.stringValue.flatMap { UInt64($0, radix: 16) } } }
            func record(_ r: JSONValue) -> Record {
                let qs = (r["questions"]?.elements ?? []).map { q in
                    Question(
                        name: q["name"]?.stringValue ?? "", type: q["type"]?.stringValue ?? "",
                        keys: (q["keys"]?.elements ?? []).compactMap(\.stringValue),
                        codes: q["codes"]?.elements.map { $0.compactMap(\.stringValue) },
                        rowLength: q["row_len"]?.doubleValue.map(Int.init) ?? -1, slot: q["slot"]?.doubleValue.map(Int.init) ?? -1,
                        rowIDsSHA256: q["row_ids_sha256"]?.stringValue ?? "", textSHA256: q["text_sha256"]?.stringValue,
                        groups: (q["groups"]?.elements ?? []).map(ints), hiddenSHA256: q["hidden_sha256"]?.stringValue ?? "",
                        logitIDs: ints(q["logit_ids"]), zBits: hex(q["z_bits"]), pBits: hex(q["p_bits"]),
                        oracle: (q["oracle_p"]?.elements ?? []).compactMap(\.doubleValue),
                        nearTie: q["near_tie"] == .bool(true))
                }
                var picture: (String, Int, Int, String)?
                if let p = r["picture"] {
                    let size = ints(p["size"])
                    picture = (p["file"]?.stringValue ?? "", size.first ?? 0, size.last ?? 0, p["rgb_sha256"]?.stringValue ?? "")
                }
                let request = r["request"] ?? .null
                return Record(
                    id: r["id"]?.stringValue ?? "", requestJSON: request.dumps(), request: request,
                    inputTokens: r["input_tokens"]?.doubleValue.map(Int.init) ?? -1, response: r["response"]?.stringValue ?? "",
                    questions: qs, picture: picture,
                    towerOutputs: (r["tower_outputs_sha256"]?.elements ?? []).compactMap(\.stringValue))
            }
            return Fixture(
                records: (root["records"]?.elements ?? []).map(record),
                imageRecords: (root["image_records"]?.elements ?? []).map(record))
        }
    }

    static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func sha256(_ text: String) -> String { sha256(Data(text.utf8)) }
    /// sha256 of `json.dumps(ids)`, the fixture's digest of a row.
    static func idsSHA256(_ ids: [Int]) -> String { sha256("[" + ids.map(String.init).joined(separator: ", ") + "]") }

    /// The provider's question objects as the kit's typed questions: a choice's options with their descriptions (a
    /// null description is the label alone), a noul's criteria, a score's levels.
    static func typedQuestions(_ questions: JSONValue) -> [(id: String, question: Decision.Question)] {
        (questions.members ?? []).map { m in
            let q = m.value
            let instructions = q["instructions"]?.stringValue ?? ""
            switch q["type"]?.stringValue {
            case "noul":
                return (m.key, .noul(
                    instructions, yes: q["criteria"]?["true"]?.stringValue, no: q["criteria"]?["false"]?.stringValue))
            case "score":
                return (m.key, .score(instructions, levels: (q["criteria"]?.elements ?? []).compactMap(\.stringValue)))
            default:
                return (m.key, .choice(instructions, options: (q["criteria"]?.members ?? []).map {
                    Decision.Option(id: $0.key, description: $0.value.stringValue ?? $0.key)
                }))
            }
        }
    }

    /// The provider's request (the fixture's), parsed as `json.loads` parses it.
    @available(macOS 27, iOS 27, *)
    static func provider(_ record: Fixture.Record) throws -> D13BRequest {
        try D13BRequest(data: Data(record.requestJSON.utf8))
    }

    /// Every fixture text row's prompt is the provider's (its sha256; the option codes as the fixture's aliases gave
    /// them), with no tokenizer.
    @available(macOS 27, iOS 27, *)
    @Test func theTextsAreTheProviders() throws {
        let fixture = try Fixture.load()
        #expect(fixture.records.count == 16 && fixture.imageRecords.count == 2)
        #expect(fixture.records.map(\.questions.count).reduce(0, +) == 24)
        var rows = 0
        for r in fixture.records {
            let request = try Self.provider(r)
            for (q, f) in zip(request.questions, r.questions) {
                let text = D13BText.prefix(request.state) + D13BText.suffix(q, codes: f.codes)
                #expect(Self.sha256(text) == f.textSHA256, "\(r.id)/\(f.name): the row's text differs from the provider's")
                #expect(q.keys == f.keys, "\(r.id)/\(f.name)")
                rows += 1
            }
        }
        #expect(rows == 24)
    }

    /// The readout from the Python reference's z (float64, h_slot · E[id] of the shipped decoder's rows): the group max
    /// and the softmax give its p bit for bit, and each record's response is the provider's text.
    @available(macOS 27, iOS 27, *)
    @Test func theReadoutWritesTheProvidersResponse() throws {
        let fixture = try Fixture.load()
        var rows = 0
        for r in fixture.records {
            let request = try Self.provider(r)
            var probs: [[Double]] = []
            for f in r.questions {
                let z = Dictionary(uniqueKeysWithValues: zip(f.logitIDs, f.zBits.map(Double.init(bitPattern:))))
                #expect(D13BReadout.groupIDs(f.groups) == f.logitIDs, "\(r.id)/\(f.name)")
                let p = D13BReadout.probabilities(logits: z, groups: f.groups)
                #expect(p.map(\.bitPattern) == f.pBits, "\(r.id)/\(f.name): p differ from the Python reference's")
                probs.append(p)
                rows += 1
            }
            let body = D13BReadout.response(request.questions, probabilities: probs, inputTokens: r.inputTokens)
            #expect(D13BPythonFormat.dumps(body, indent: 2, asciiOnly: false) == r.response, "\(r.id): the response differs")
        }
        #expect(rows == 24)
    }

    /// The provider's refusals of a request's shape, in its words, before any tokenizer; the wire's own refusals of
    /// the same shapes, in the kit's words; the length and picture limits' texts.
    @available(macOS 27, iOS 27, *)
    @Test func theRefusalsAreTheProvidersWords() throws {
        func refusal(_ json: String) -> String? {
            do {
                _ = try D13BRequest(data: Data(json.utf8))
                return nil
            } catch let error as D13BError {
                return error.message
            } catch {
                return "\(error)"
            }
        }
        #expect(refusal(#"{"state": "s", "questions": {"q": {"type": "noul"}}}"#) == "questions.q.instructions: field required")
        let eleven = (0..<11).map { "\"l\($0)\"" }.joined(separator: ", ")
        #expect(refusal(#"{"state": "s", "questions": {"q": {"type": "score", "instructions": "i", "criteria": [\#(eleven)]}}}"#)
            == "questions.q.criteria: a score question takes a list of 1..10 levels")
        #expect(refusal(#"{"questions": {"q": {"type": "noul", "instructions": "i"}}}"#) == "state: field required (null for no state)")
        #expect(throws: SystemOne.WireError("question 'q' has no 'instructions'")) {
            try SystemOne.request(from: Data(#"{"state": "s", "questions": {"q": {"type": "noul"}}}"#.utf8))
        }
        #expect(throws: SystemOne.WireError("question 'q': a score needs 2–10 levels, got 11")) {
            try SystemOne.request(from: Data(#"{"state": "s", "questions": {"q": {"type": "score", "instructions": "i", "criteria": [\#(eleven)]}}}"#.utf8))
        }
        #expect(throws: D13BError.self) { try D13BTokenizer.graphContextCheck(length: 4033, chunk: 64, maxContext: 4096) }
        do {
            try D13BTokenizer.graphContextCheck(length: 4033, chunk: 64, maxContext: 4096)
        } catch let error as D13BError {
            #expect(error.message
                == "a row of 4033 tokens runs 4096 padded positions, over the graph's 4095 (rows of at most 4032 tokens at S = 64)")
        }
        try D13BTokenizer.graphContextCheck(length: 4032, chunk: 64, maxContext: 4096)
        #expect(D13BTokenizer.imageRowsRefusal(tokens: 3540, imageRows: 2816) == "images: 3540 image tokens over the graph's 2816 image rows")
        // One picture needs at most 2,810 image tokens: the wire's one picture never meets the limit.
        #expect(D13BVision.plan(pictureWidth: 2048, pictureHeight: 1536).tokens <= 2816)
    }

    // MARK: - The rows, token for token (a decoder bundle's tokenizer/)

    /// KIT_D1_3B_STORE, else the default store.
    static var store: ModelStore {
        if let path = environment["KIT_D1_3B_STORE"] { return ModelStore(directory: URL(fileURLWithPath: path)) }
        return .default
    }

    @available(macOS 27, iOS 27, *)
    static var decoderFolder: URL? {
        guard let entry = ModelCatalog.builtin.entry(id: "d1-3b"), let model = entry.modelID else { return nil }
        return store.localURL(for: model)
    }

    /// KIT_D1_3B_TOKENIZER, else the stored decoder bundle's tokenizer/.
    static var tokenizerFolder: URL? {
        if let path = environment["KIT_D1_3B_TOKENIZER"] { return URL(fileURLWithPath: path) }
        guard #available(macOS 27, iOS 27, *) else { return nil }
        return decoderFolder?.appendingPathComponent("tokenizer")
    }

    static var tokenizerEnabled: Bool { tokenizerFolder != nil }

    /// Every fixture row rebuilt by the kit from its request with the kit's swift-transformers: the provider's ids, the
    /// text and the readout groups; a picture's row from its size alone (the processor's crops, `<image>` k as
    /// 128,000 + k).
    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: tokenizerEnabled, "set KIT_D1_3B_TOKENIZER (or keep the decoder bundle in the store)"))
    func everyFixtureRowIsTheProviders() async throws {
        let fixture = try Fixture.load()
        let tokenizer = try await D13BTokenizer.load(folder: try #require(Self.tokenizerFolder))
        var equal = 0
        for r in fixture.records {
            let request = try Self.provider(r)
            let rows = try tokenizer.rows(request, table: nil)
            #expect(rows.inputTokens == r.inputTokens, "\(r.id): input_tokens")
            for (row, f) in zip(rows.rows, r.questions) {
                let same = Self.idsSHA256(row.ids) == f.rowIDsSHA256 && row.ids.count == f.rowLength && row.slot == f.slot
                    && row.groups == f.groups && Self.sha256(row.text) == f.textSHA256
                #expect(same, "\(r.id)/\(f.name): the kit's row differs from the provider's")
                if same { equal += 1 }
            }
        }
        for r in fixture.imageRecords {
            let request = try Self.provider(r)
            let picture = try #require(r.picture)
            let plan = D13BVision.plan(pictureWidth: picture.width, pictureHeight: picture.height)
            let prefix = D13BText.prefix(request.state, images: D13BVision.imageToken)
            for (q, f) in zip(request.questions, r.questions) {
                let codes = q.kind == .choice ? try tokenizer.aliases(q.labels).map(\.code) : nil
                let ids = D13BVision.extensionIDs(try D13BVision.promptIDs(tokenizer, text: prefix + D13BText.suffix(q, codes: codes), plans: [plan]))
                let same = Self.idsSHA256(ids) == f.rowIDsSHA256 && ids.count == f.rowLength
                #expect(same, "\(r.id)/\(f.name): the kit's picture row differs from the provider's")
                if same { equal += 1 }
            }
        }
        #expect(equal == 28)
    }

    /// The zoo's whole fixture (the lane's files, not in this repository): every record's rows from its raw request
    /// against host.py's (`render_ids.json`: the text, the ids, the groups and keys), and the refused record refused.
    @available(macOS 27, iOS 27, *)
    @Test(.enabled(
        if: tokenizerEnabled && environment["KIT_D1_3B_RECORDS"] != nil && environment["KIT_D1_3B_RENDER"] != nil,
        "set KIT_D1_3B_RECORDS and KIT_D1_3B_RENDER (the lane's files) and a tokenizer"))
    func everyZooRowIsHostPys() async throws {
        let tokenizer = try await D13BTokenizer.load(folder: try #require(Self.tokenizerFolder))
        let records = try D13BJSONParser.parse(Data(contentsOf: URL(fileURLWithPath: Self.environment["KIT_D1_3B_RECORDS"]!)))
        let render = try D13BJSONParser.parse(Data(contentsOf: URL(fileURLWithPath: Self.environment["KIT_D1_3B_RENDER"]!)))
        var byID: [String: D13BJSONValue] = [:]
        for r in render["records"]?.array ?? [] { if let id = r["id"]?.string { byID[id] = r } }
        var rowsEqual = 0, rowsTotal = 0, refusedEqual = 0, recordsTotal = 0
        for rec in records["records"]?.array ?? [] {
            guard let id = rec["id"]?.string, let want = byID[id], let request = rec["request"] else { continue }
            recordsTotal += 1
            func check(_ built: D13BRows) {
                let wanted = Dictionary(uniqueKeysWithValues: (want["questions"]?.array ?? []).compactMap { q in
                    q["name"]?.string.map { ($0, q) }
                })
                for row in built.rows {
                    guard let q = wanted[row.name] else { continue }
                    rowsTotal += 1
                    let ids = (q["row_ids"]?.array ?? []).compactMap(\.intValue)
                    let groups = (q["groups"]?.array ?? []).map { ($0.array ?? []).compactMap(\.intValue) }
                    let same = row.ids == ids && row.text == q["text"]?.string && row.groups == groups
                        && row.keys == (q["keys"]?.array ?? []).compactMap(\.string)
                    #expect(same, "\(id)/\(row.name): differs from host.py")
                    if same { rowsEqual += 1 }
                }
            }
            do {
                let built = try tokenizer.rows(try D13BRequest(json: request), table: nil)
                #expect(built.inputTokens == want["input_tokens"]?.intValue, "\(id): input_tokens")
                check(built)
            } catch let error as D13BError {
                let refused = want["refused"]?.members?.first?.value.string
                #expect(refused != nil, "\(id): refused (\(error.message)), host.py did not")
                if refused == error.message { refusedEqual += 1 }
                // host.py's rows of the record's answerable questions, each alone (test_host.py's order)
                for q in want["questions"]?.array ?? [] {
                    guard let name = q["name"]?.string, let one = request["questions"]?[name] else { continue }
                    check(try tokenizer.rows(
                        try D13BRequest(json: .object([D13BJSONMember("state", request["state"] ?? .null),
                                                       D13BJSONMember("questions", .object([D13BJSONMember(name, one)]))])),
                        table: nil))
                }
            }
        }
        #expect(recordsTotal == 361 && rowsTotal == 393 && rowsEqual == 393 && refusedEqual == 1)
        if let out = Self.environment["KIT_D1_3B_ROWS_OUT"] {
            let doc = JSONValue.object([
                .init("schema", .string("coreai-kit-d1-3b-rows/1")), .init("records", .int(recordsTotal)),
                .init("rows", .int(rowsTotal)), .init("rows_equal_host_py", .int(rowsEqual)),
                .init("refused_equal_host_py", .int(refusedEqual)),
            ])
            try doc.dumps().appending("\n").write(toFile: out, atomically: true, encoding: .utf8)
        }
    }

    // MARK: - Runs of the shipped Mac decoder, bit for bit (the bundles)

    @available(macOS 27, iOS 27, *)
    static var bundlesPresent: Bool { decoderFolder != nil }

    static var gateEnabled: Bool {
        guard #available(macOS 27, iOS 27, *) else { return false }
        #if os(macOS)
        return environment["KIT_D1_3B_GATE"] == "1" || bundlesPresent
        #else
        return false
        #endif
    }

    /// KIT_D1_3B_IMAGES: the drawn pictures (the zoo's make_fixture_images.py), checked by their decoded pixels.
    static var imagesFolder: URL? { environment["KIT_D1_3B_IMAGES"].map { URL(fileURLWithPath: $0) } }

    /// The fixture through `KitD1Decider(catalog:)` on the Mac's decoder: per text row the ids, the hidden rows, z and p
    /// against the Python reference's run of the same bundle (bit for bit), each response against the provider's text;
    /// with the shared prefix and a prepared state the same bits; the pictures (with `KIT_D1_3B_IMAGES`) with the tower's
    /// outputs, from a file and from bytes; the wire; the refusals in the provider's words, before any graph call; and
    /// FACTS §7's bar against the provider's fp32 oracle.
    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: gateEnabled, "set KIT_D1_3B_GATE=1, or keep the decoder bundle in the store"))
    func runsMatchThePythonReferenceBitForBit() async throws {
        let fixture = try Fixture.load()
        let store = Self.store
        let t0 = ContinuousClock.now
        let decider = try await KitD1Decider(catalog: "d1-3b", store: store)
        let load = ContinuousClock.now - t0
        #expect(decider.modelName == "d1_3b_decode_fp16_pf64_s")
        var transcript: [JSONValue] = []
        var rowsEqual = 0, rowsTotal = 0, responsesEqual = 0
        var maxAbsDP = 0.0, meanSum = 0.0, argmaxEqual = 0, argmaxCounted = 0
        typealias Readout = KitD1Decider.Readout
        var direct: [String: Readout] = [:]

        func compare(_ r: Fixture.Record, _ out: Readout, mode: String) -> Bool {
            var all = true
            for (row, f) in zip(out.rows, r.questions) {
                let ids = Self.idsSHA256(row.ids) == f.rowIDsSHA256 && row.ids.count == f.rowLength && row.slot == f.slot
                    && row.groups == f.groups && row.keys == f.keys
                let hidden = row.hiddenSHA256 == f.hiddenSHA256
                let z = f.zBits.isEmpty || (row.logitIDs == f.logitIDs && row.logits.map(\.bitPattern) == f.zBits)
                let p = row.probabilities.map(\.bitPattern) == f.pBits
                #expect(ids, "\(r.id)/\(f.name) \(mode): the row's ids differ from the provider's")
                #expect(hidden && z && p, "\(r.id)/\(f.name) \(mode): hidden \(hidden), z \(z), p \(p)")
                all = all && ids && hidden && z && p
                transcript.append(.object([
                    .init("row", .string("\(r.id)/\(f.name)")), .init("mode", .string(mode)),
                    .init("tokens", .int(row.ids.count)), .init("ids_equal_provider", .bool(ids)),
                    .init("hidden_sha256_equal_python", .bool(hidden)), .init("z_bits_equal_python", .bool(z)),
                    .init("p_bits_equal_python", .bool(p)),
                ]))
            }
            return all && out.rows.count == r.questions.count
        }
        func bar(_ r: Fixture.Record, _ out: Readout) {
            for (row, f) in zip(out.rows, r.questions) {
                let d = zip(row.probabilities, f.oracle).map { abs($0 - $1) }
                maxAbsDP = max(maxAbsDP, d.max() ?? 0)
                meanSum += d.reduce(0, +) / Double(max(d.count, 1))
                if !f.nearTie {
                    argmaxCounted += 1
                    let a = row.probabilities.indices.max { row.probabilities[$0] < row.probabilities[$1] }
                    let b = f.oracle.indices.max { f.oracle[$0] < f.oracle[$1] }
                    if a == b { argmaxEqual += 1 }
                }
            }
        }

        // Every text record, each row from zero states.
        for r in fixture.records {
            let out = try await decider.readout(requestJSON: Data(r.requestJSON.utf8))
            #expect(out.mode == "direct" && out.inputTokens == r.inputTokens, "\(r.id)")
            if compare(r, out, mode: "direct") { rowsEqual += out.rows.count }
            rowsTotal += r.questions.count
            #expect(out.response == r.response, "\(r.id): the response differs from the provider's")
            if out.response == r.response { responsesEqual += 1 }
            bar(r, out)
            direct[r.id] = out
        }

        // The shared prefix and a prepared state: the same rows as the direct run, bit for bit.
        var sharedEqual = 0, preparedEqual = 0, multi = 0
        for r in fixture.records where r.questions.count > 1 {
            multi += 1
            let shared = try await decider.readout(requestJSON: Data(r.requestJSON.utf8), shared: true)
            let d = try #require(direct[r.id])
            let same = shared.rows == d.rows && shared.response == d.response
            #expect(same && shared.mode == "shared", "\(r.id): shared differs from direct")
            if same { sharedEqual += 1 }
            let state = try JSONValue.parse(Data(try #require(r.request["state"]).dumps().utf8))
            let prepared = try await decider.prepare(state: state)
            let questions = try #require(r.request["questions"]).dumps()
            let later = try await decider.readout(prepared: prepared, questionsJSON: Data(questions.utf8))
            let samePrepared = later.rows == shared.rows && later.mode == "prepared" && later.sharedTokens == shared.sharedTokens
            #expect(samePrepared, "\(r.id): prepared differs from shared")
            if samePrepared { preparedEqual += 1 }
            transcript.append(.object([
                .init("record", .string(r.id)), .init("shared_equal_direct", .bool(same)),
                .init("prepared_equal_shared", .bool(samePrepared)), .init("shared_tokens", .int(shared.sharedTokens)),
                .init("state_tokens", .int(shared.stateTokens)), .init("calls_direct", .int(d.calls)),
                .init("calls_shared", .int(shared.calls)),
            ]))
        }
        #expect(multi == 3 && sharedEqual == 3 && preparedEqual == 3)

        // The wire: the card's request through the server's route, the typed answers in the kit's order.
        let card = try #require(fixture.records.first { $0.id == "card_refund" })
        let server = SystemOneServer(host: "127.0.0.1", modelID: "d1-3b", backend: decider) { _ in }
        let wire = await server.route(HTTPRequest(
            method: "POST", path: SystemOne.path, headers: [:], body: Data(card.requestJSON.utf8)))
        #expect(wire.status == 200)
        let response = try await decider.systemOne(try SystemOne.request(from: Data(card.requestJSON.utf8)))
        let refs = try #require(direct["card_refund"]).rows
        var wireEqual = true
        for (answer, ref) in zip(response.answers, refs) {
            let p = answer.answer.noul.map { [1 - $0, $0] } ?? answer.answer.probabilities
            let want = try KitD1Decider.kitOrder(answer.question, keys: ref.keys, ref.probabilities, id: answer.id)
            let ok = answer.answer.noul.map { $0 == want[1] } ?? (p == want)
            #expect(ok, "card_refund/\(answer.id): the wire's p differ from the readout's")
            wireEqual = wireEqual && ok
        }

        // The pictures: a file and the same bytes, the tower's outputs too.
        var imageRowsEqual = 0, towersEqual = 0, imageResponsesEqual = 0, bytesEqual = 0
        if let images = Self.imagesFolder {
            for r in fixture.imageRecords {
                let picture = try #require(r.picture)
                let file = images.appendingPathComponent(picture.file)
                let decoded = try D13BPixels.decode(url: file)
                #expect(Self.sha256(Data(decoded.rgb.bytes)) == picture.rgbSHA256, "\(picture.file): not the drawn picture")
                let out = try await decider.readout(requestJSON: Data(r.requestJSON.utf8), images: [file])
                if compare(r, out, mode: "picture") { imageRowsEqual += out.rows.count }
                #expect(out.towerOutputSHA256 == r.towerOutputs, "\(r.id): the tower's outputs differ")
                if out.towerOutputSHA256 == r.towerOutputs { towersEqual += 1 }
                #expect(out.response == r.response, "\(r.id): the response differs from the provider's")
                if out.response == r.response { imageResponsesEqual += 1 }
                bar(r, out)
                // The same picture as bytes, through the Swift API (the provider's None as nil).
                let asked = Self.typedQuestions(try #require(r.request["questions"]))
                let state = r.request["state"].flatMap { $0 == .null ? nil : $0 }
                let typed = try await decider.systemOne(
                    state: state, questions: asked, images: [.data(try Data(contentsOf: file))])
                // The Swift API runs the shared prefix (the decider's default): the file again with it, to tell the
                // two apart.
                let sharedFile = try await decider.readout(requestJSON: Data(r.requestJSON.utf8), images: [file], shared: true)
                var same = true
                var maxDelta = 0.0
                var bytesP: [JSONValue] = []
                for (answer, row) in zip(typed.answers, sharedFile.rows) {
                    let want = try KitD1Decider.kitOrder(answer.question, keys: row.keys, row.probabilities, id: answer.id)
                    let got = answer.answer.probabilities
                    same = same && (answer.answer.noul.map { $0 == want[1] } ?? (got == want))
                    maxDelta = max(maxDelta, zip(got, want).map { abs($0 - $1) }.max() ?? 0)
                    bytesP.append(.array(got.map { .double($0) }))
                }
                let sharedEqual = sharedFile.rows == out.rows
                #expect(same, "\(r.id): the picture as bytes gives other p than the file with the shared prefix")
                if same { bytesEqual += 1 }
                transcript.append(.object([
                    .init("picture_record", .string(r.id)), .init("shared_equal_direct", .bool(sharedEqual)),
                    .init("shared_tokens", .int(sharedFile.sharedTokens)), .init("state_tokens", .int(sharedFile.stateTokens)),
                    .init("bytes_equal_file_shared", .bool(same)), .init("bytes_max_abs_dp", .double(maxDelta)),
                    .init("bytes_p_kit_order", .array(bytesP)),
                    .init("shared_hidden", .array(sharedFile.rows.map { .string($0.hiddenSHA256) })),
                    .init("direct_hidden", .array(out.rows.map { .string($0.hiddenSHA256) })),
                ]))
            }
            #expect(imageRowsEqual == 4 && towersEqual == 2 && imageResponsesEqual == 2 && bytesEqual == 2)
        }

        // The refusals: the provider's words through the readout and the wire, before any graph call.
        let tokenizer = try await D13BTokenizer.load(folder: try #require(Self.tokenizerFolder))
        var refusals: [JSONValue] = []
        func refused(_ name: String, _ body: () async throws -> Void) async -> String? {
            do {
                try await body()
                Issue.record("\(name): not refused")
                return nil
            } catch let error as SystemOne.WireError {
                return error.message
            } catch {
                return "\(error)"
            }
        }
        let calls0 = try await decider.graphCalls()
        // 1. A row past 4,032 tokens.
        let long = String(repeating: "The pallet count was checked again at the dock. ", count: 500)
        let longJSON = #"{"state": "\#(long)", "questions": {"q": {"type": "noul", "instructions": "Was it checked?"}}}"#
        let longRow = try tokenizer.row(state: .string(long), try D13BRequest.validateQuestion(
            name: "q", try D13BJSONParser.parse(#"{"type": "noul", "instructions": "Was it checked?"}"#)))
        let padded = (longRow.ids.count + 63) / 64 * 64
        let longText = "a row of \(longRow.ids.count) tokens runs \(padded) padded positions, over the graph's 4095 "
            + "(rows of at most 4032 tokens at S = 64)"
        #expect(longRow.ids.count > 4032)
        let r1 = await refused("long") { _ = try await decider.readout(requestJSON: Data(longJSON.utf8)) }
        let w1 = await refused("long, wire") { _ = try await decider.systemOne(try SystemOne.request(from: Data(longJSON.utf8))) }
        let s1 = await server.route(HTTPRequest(method: "POST", path: SystemOne.path, headers: [:], body: Data(longJSON.utf8)))
        #expect(r1 == longText && w1 == longText && s1.status == 422 && String(decoding: s1.body, as: UTF8.self).contains(longText))
        // 2. Pictures over the graph's 2,816 image rows (two of 1,770 tokens): the Swift API, as the wire carries one.
        if let images = Self.imagesFolder, let tiled = fixture.imageRecords.first(where: { $0.id.hasPrefix("img06") })?.picture {
            let file = images.appendingPathComponent(tiled.file)
            let json = #"{"state": null, "questions": {"q": {"type": "noul", "instructions": "Is there a grid?"}}}"#
            let r2 = await refused("pictures") { _ = try await decider.readout(requestJSON: Data(json.utf8), images: [file, file]) }
            #expect(r2 == "images: 3540 image tokens over the graph's 2816 image rows")
            refusals.append(.object([.init("kind", .string("image_tokens")), .init("readout", .string(r2 ?? ""))]))
        }
        // 3. A label whose id is not in the option table.
        let table = try D13BOptionTable.ids(json: try #require(Self.decoderFolder).appendingPathComponent("head/option_rows.json"))
        let accentJSON = #"{"state": "s", "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"é": null, "α": null}}}}"#
        let accentID = try #require(tokenizer.singleToken("é"))
        #expect(!table.idSet.contains(accentID))
        let accentText = "questions.q: the token id \(accentID) of label 'é' is not in the option table"
        let r3 = await refused("label") { _ = try await decider.readout(requestJSON: Data(accentJSON.utf8)) }
        let w3 = await refused("label, wire") { _ = try await decider.systemOne(try SystemOne.request(from: Data(accentJSON.utf8))) }
        #expect(r3 == accentText && w3 == accentText)
        // 4. A score of 11 levels: the provider's words in its form, the wire's own on the wire.
        let eleven = (0..<11).map { "\"l\($0)\"" }.joined(separator: ", ")
        let scoreJSON = #"{"state": "s", "questions": {"q": {"type": "score", "instructions": "How?", "criteria": [\#(eleven)]}}}"#
        let r4 = await refused("score") { _ = try await decider.readout(requestJSON: Data(scoreJSON.utf8)) }
        let s4 = await server.route(HTTPRequest(method: "POST", path: SystemOne.path, headers: [:], body: Data(scoreJSON.utf8)))
        #expect(r4 == "questions.q.criteria: a score question takes a list of 1..10 levels")
        #expect(s4.status == 422 && String(decoding: s4.body, as: UTF8.self).contains("a score needs 2–10 levels, got 11"))
        // 5. A question without instructions.
        let bareJSON = #"{"state": "s", "questions": {"q": {"type": "noul"}}}"#
        let r5 = await refused("instructions") { _ = try await decider.readout(requestJSON: Data(bareJSON.utf8)) }
        let s5 = await server.route(HTTPRequest(method: "POST", path: SystemOne.path, headers: [:], body: Data(bareJSON.utf8)))
        #expect(r5 == "questions.q.instructions: field required")
        #expect(s5.status == 422 && String(decoding: s5.body, as: UTF8.self).contains("question 'q' has no 'instructions'"))
        let calls1 = try await decider.graphCalls()
        #expect(calls1.decoder == calls0.decoder && calls1.tower == calls0.tower, "a refused request made a graph call")
        refusals += [
            .object([.init("kind", .string("row_over_4032")), .init("readout", .string(r1 ?? "")), .init("wire", .string(w1 ?? "")),
                     .init("server_status", .int(s1.status))]),
            .object([.init("kind", .string("label_outside_table")), .init("readout", .string(r3 ?? "")), .init("wire", .string(w3 ?? ""))]),
            .object([.init("kind", .string("score_11_levels")), .init("readout", .string(r4 ?? "")),
                     .init("server", .string(String(decoding: s4.body, as: UTF8.self)))]),
            .object([.init("kind", .string("no_instructions")), .init("readout", .string(r5 ?? "")),
                     .init("server", .string(String(decoding: s5.body, as: UTF8.self)))]),
        ]

        #expect(rowsEqual == 24 && rowsTotal == 24 && responsesEqual == 16 && wireEqual)
        #expect(maxAbsDP <= 0.02 && argmaxEqual == argmaxCounted)

        if let out = Self.environment["KIT_D1_3B_OUT"] {
            let seconds = Double(load.components.seconds) + Double(load.components.attoseconds) * 1e-18
            let questions = rowsTotal + (Self.imagesFolder == nil ? 0 : 4)
            let doc = JSONValue.object([
                .init("schema", .string("coreai-kit-d1-3b-test-run/1")), .init("model", .string(decider.id)),
                .init("bundle", .string(decider.modelName)), .init("init_seconds", .double(seconds)),
                .init("text_rows", .int(rowsTotal)), .init("text_rows_bit_equal", .int(rowsEqual)),
                .init("responses_equal", .int(responsesEqual)), .init("multi_question_records", .int(multi)),
                .init("shared_equal_direct", .int(sharedEqual)), .init("prepared_equal_shared", .int(preparedEqual)),
                .init("picture_rows_bit_equal", .int(imageRowsEqual)), .init("tower_outputs_equal", .int(towersEqual)),
                .init("picture_responses_equal", .int(imageResponsesEqual)), .init("picture_bytes_equal_file", .int(bytesEqual)),
                .init("wire_equal_readout", .bool(wireEqual)), .init("bar_questions", .int(questions)),
                .init("bar_max_abs_dp_vs_oracle", .double(maxAbsDP)),
                .init("bar_mean_of_row_mean_abs_dp", .double(meanSum / Double(max(questions, 1)))),
                .init("bar_argmax_equal_non_near_tie", .int(argmaxEqual)), .init("bar_non_near_tie", .int(argmaxCounted)),
                .init("graph_calls_during_refusals", .int(calls1.decoder - calls0.decoder)),
                .init("refusals", .array(refusals)), .init("per_row", .array(transcript)),
            ])
            try doc.dumps().appending("\n").write(toFile: out, atomically: true, encoding: .utf8)
        }
    }
}
