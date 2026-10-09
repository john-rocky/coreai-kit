// D1OmniTests.swift — d1-omni-600M in the kit: the catalog entry and what loads it, the wire's `audio`, the request in
// the publisher's form, the order the probabilities come back in, the readout and the response against the publisher's
// code, and — with the files — the zoo's public rows token for token and runs of the zoo's Swift host, bit for bit
// (Fixtures/d1_omni_swift.json: every public row's ids as a digest, its markers and its length; for 14 requests — 20 text
// rows at every length a fixture row reaches, 3 images, 3 clips — the zoo Swift host's marker logits and probabilities
// and the response the publisher's code writes for them):
//
//     KIT_D1_OMNI_TOKENIZER=<tokenizer/> KIT_D1_OMNI_REFERENCE=<reference/> swift test --filter D1Omni   # the 360 rows
//     KIT_D1_OMNI_GATE=1 [KIT_D1_OMNI_OUT=<transcript.json>] swift test --filter D1Omni                 # + the graphs
//
// The rows need the repo's `tokenizer/` and `reference/` (4.7 MB and 9.3 MB at the catalog pin); the gate downloads
// them and the `macos/` bundles (6.5 GB) on its first run. `KIT_D1_OMNI_STORE=<dir>` points the tests at a ModelStore
// directory other than the default; a store that already holds the bundles and `reference/` runs the gate without the
// variable.

import CryptoKit
import Foundation
import Testing

@testable import CoreAIKit

struct D1OmniTests {
    static let pin = "914184e50fb1e4ef2ece2454d656cbd9031728ff"
    static let environment = ProcessInfo.processInfo.environment

    // MARK: - The catalog entry

    @available(macOS 27, iOS 27, *)
    @Test func theCatalogEntryIsD1OmniAtItsPin() throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "d1-omni-600m"))
        #expect(entry.kind == .omniDecision && entry.format == KitD1OmniDecider.format)
        #expect(entry.repo == "mlboydaisuke/d1-omni-600M-CoreAI" && entry.revision == Self.pin)
        #expect(entry.variants["macos"]?.path == "macos" && entry.variants["ios"]?.path == "ios")
        #expect(entry.variants["macos"]?.sizeMB == 6458 && entry.variants["ios"]?.sizeMB == 6458)
        #expect(entry.assets == nil && entry.engine == nil && entry.calibration == nil && entry.license == nil)
        #expect(KitD1OmniDecider.folders.count == 12)
        #expect(entry.modelID(path: "macos/decide-fp16-L64")
            == ModelID("mlboydaisuke/d1-omni-600M-CoreAI", path: "macos/decide-fp16-L64", revision: Self.pin))
    }

    /// The bundle says `decision.head = "encoder"`, but laya's encoder path cannot read it: an `omniDecision` entry is
    /// `KitD1OmniDecider`'s alone, `TypedDecisions(catalog:)` refuses it by its kind before it downloads anything, and
    /// `KitD1OmniDecider(catalog:)` refuses another kind the same way.
    @available(macOS 27, iOS 27, *)
    @Test func onlyKitD1OmniDeciderLoadsAnOmniDecisionEntry() async throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "d1-omni-600m"))
        #expect(!TypedDecisions.supports(entry) && !KitClefDecider.supports(entry) && !KitKevDecider.supports(entry))
        #expect(KitD1OmniDecider.supports(entry) && SystemOne.supports(entry))
        for other in ModelCatalog.builtin.available(.decision) + ModelCatalog.builtin.available(.jointDecision)
            + ModelCatalog.builtin.available(.rowDecision)
        {
            #expect(!KitD1OmniDecider.supports(other), "\(other.id)")
        }
        let otherFormat = CatalogEntry(
            id: entry.id, name: entry.name, repo: entry.repo, revision: entry.revision, kind: .omniDecision,
            variants: entry.variants, format: "encoder")
        #expect(!KitD1OmniDecider.supports(otherFormat))
        #expect(ModelCatalog.builtin.available(.omniDecision).map(\.id) == ["d1-omni-600m"])

        // The store's Hub is a closed local port: a download would fail on the transport, not on the kind.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(directory: root, hubBaseURL: URL(string: "http://127.0.0.1:9")!)
        do {
            _ = try await TypedDecisions(catalog: "d1-omni-600m", store: store)
            Issue.record("TypedDecisions loaded d1-omni-600m")
        } catch let error as CoreAIKitError {
            guard case .catalogKindMismatch(let id, let expected, let found) = error else {
                Issue.record("not the kind refusal: \(error)")
                return
            }
            #expect(id == "d1-omni-600m" && expected == "chat or decision" && found == "omniDecision")
        }
        do {
            _ = try await KitD1OmniDecider(catalog: "kev-0.8b", store: store)
            Issue.record("KitD1OmniDecider loaded kev-0.8b")
        } catch let error as CoreAIKitError {
            guard case .catalogKindMismatch(let id, let expected, let found) = error else {
                Issue.record("not the kind refusal: \(error)")
                return
            }
            #expect(id == "kev-0.8b" && expected == "omniDecision" && found == "rowDecision")
        }
        #expect(((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []).isEmpty)
    }

    // MARK: - The wire's audio

    static func request(_ fields: String) throws -> SystemOne.Request {
        try SystemOne.request(from: Data(#"{\#(fields)"questions": {"q": {"type": "noul", "instructions": "i"}}}"#.utf8))
    }

    @Test func theWireCarriesOneClip() throws {
        let wav = Data("RIFF....WAVE".utf8)
        let b64 = wav.base64EncodedString()
        #expect(try Self.request(#""state": "s", "audio": "data:audio/wav;base64,\#(b64)", "#).audio == .data(wav))
        #expect(try Self.request(#""state": "s", "audio": "\#(b64)", "#).audio == .data(wav))
        #expect(try Self.request(#""state": "s", "audio": "/tmp/a clip.wav", "#).audio
            == .file(URL(fileURLWithPath: "/tmp/a clip.wav")))
        #expect(try Self.request(#""state": "s", "audio": "file:///tmp/x.wav", "#).audio
            == .file(URL(string: "file:///tmp/x.wav")!))
        #expect(throws: SystemOne.WireError.self) { try Self.request(#""state": "s", "audio": ["\#(b64)"], "#) }
        #expect(throws: SystemOne.WireError.self) { try Self.request(#""state": "s", "audio": 1, "#) }
        #expect(throws: SystemOne.WireError.self) { try Self.request(#""state": "s", "audio": "data:audio/wav,raw", "#) }
        // A clip may come without a state, the publisher's None; the raw request keeps the absence.
        let stateless = try Self.request(#""audio": "\#(b64)", "#)
        #expect(stateless.state == "" && !stateless.structuredState && stateless.json?["state"] == nil)
        let null = try Self.request(#""state": null, "audio": "\#(b64)", "#)
        #expect(null.state == "" && null.json?["state"] == .null)
        // Without a clip the state stays required, and null stays refused.
        #expect(throws: SystemOne.WireError("'state' is required")) { try Self.request("") }
        #expect(throws: SystemOne.WireError("'state' must be a string, an object or an array")) {
            try Self.request(#""state": null, "#)
        }
        // A request without the field reads as it always did.
        let plain = try Self.request(#""state": "s", "#)
        #expect(plain.audio == nil && plain.images.isEmpty)
    }

    /// A backend that reads neither images nor audio.
    struct TextOnly: DecisionBackend {
        let id = "text-only"
        let maxOptions = 10
        var modelName: String { get async { "text-only" } }
        func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer { throw DecisionError.noLogits }
        func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response { throw DecisionError.noLogits }
    }

    @Test func aBackendThatReadsNoAudioRefusesAClipByName() throws {
        let clip = try Self.request(#""state": "s", "audio": "/tmp/x.wav", "#)
        let refusal = try #require(TextOnly().mediaRefusal(clip))
        #expect(refusal.message.contains("'text-only' reads no audio") && refusal.message.contains("d1-omni-600m"))
        let text = try Self.request(#""state": "s", "#)
        #expect(TextOnly().imageRefusal(clip) == nil && TextOnly().mediaRefusal(text) == nil)
        let image = try Self.request(#""state": "s", "images": ["/tmp/x.png"], "#)
        #expect(TextOnly().mediaRefusal(image)?.message.contains("(clef-flash, d1-omni-600m, d1-3b)") == true)
    }

    @Test func aServerOffLoopbackReadsNoClipPath() async throws {
        let server = SystemOneServer(host: "0.0.0.0", modelID: "text-only", backend: TextOnly()) { _ in }
        let clip = #"{"state": "s", "audio": "/tmp/x.wav", "questions": {"q": {"type": "noul", "instructions": "i"}}}"#
        let response = await server.route(HTTPRequest(method: "POST", path: SystemOne.path, headers: [:], body: Data(clip.utf8)))
        #expect(response.status == 422)
        // The backend's refusal comes first: it reads no audio at all.
        #expect(String(decoding: response.body, as: UTF8.self).contains("reads no audio"))
    }

    // MARK: - The request in the publisher's form

    /// The wire's own values: a structured state with its numbers as written and a repeated key in its first place with
    /// its last value (`json.loads`), a choice listed as names read as options alone, a missing state as None.
    @available(macOS 27, iOS 27, *)
    @Test func aWireRequestIsReadFromItsOwnValues() throws {
        let body = #"""
            {"state": {"total": 1250.0, "n": 3, "total": 1e-5, "ok": true},
             "questions": {"pick": {"type": "choice", "instructions": "Which?", "criteria": ["b", "a"]},
                           "ok": {"type": "noul", "instructions": {"rule": 2}, "criteria": {"yes": "it holds"}},
                           "how": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}}
            """#
        let request = try SystemOne.request(from: Data(body.utf8))
        let (state, questions) = KitD1OmniDecider.publisherRequest(request)
        #expect(D1Prompt.serialize(try #require(state)) == #"{"total": 1e-05, "n": 3, "ok": true}"#)
        let pick = try D1Question(json: try #require(questions["pick"]))
        #expect(pick.names == ["b", "a"] && D1Prompt.renderOptions(pick, noulDefault: nil, audio: false) == ["b", "a"])
        let ok = try D1Question(json: try #require(questions["ok"]))
        #expect(ok.instructions == "{'rule': 2}")
        #expect(D1Prompt.renderOptions(ok, noulDefault: nil, audio: false)
            == ["false: no, the statement does not hold", "true: it holds"])
        #expect(D1Prompt.renderOptions(try D1Question(json: try #require(questions["how"])), noulDefault: nil, audio: false)
            == ["level 0: low", "level 1: high"])
        let clip = try SystemOne.request(from: Data(
            #"{"audio": "UklGRg==", "questions": {"q": {"type": "noul", "instructions": "i"}}}"#.utf8))
        #expect(KitD1OmniDecider.publisherRequest(clip).state == nil)
    }

    /// A request built in Swift: an option whose description is its id reads as a name alone.
    @available(macOS 27, iOS 27, *)
    @Test func aSwiftRequestIsReadFromItsTypedQuestions() throws {
        let request = SystemOne.Request(state: "s", questions: [
            (id: "c", question: .choice("Which?", options: [.init(id: "x", description: "the x"), .init("y")])),
            (id: "s", question: .score("How?", levels: ["low", "high"])),
            (id: "n", question: .noul("Yes?", no: "it fails")),
        ])
        let (state, questions) = KitD1OmniDecider.publisherRequest(request)
        #expect(state == .string("s"))
        let render = { (key: String) in
            D1Prompt.renderOptions(try D1Question(json: questions[key]!), noulDefault: nil, audio: false)
        }
        #expect(try render("c") == ["x: the x", "y"])
        #expect(try render("s") == ["level 0: low", "level 1: high"])
        #expect(try render("n") == ["false: it fails", "true: yes, the statement holds"])
        #expect(questions.members?.map(\.key) == ["c", "s", "n"])
    }

    @available(macOS 27, iOS 27, *)
    @Test func probabilitiesComeBackInTheKitsOptionOrder() throws {
        func row(_ q: String) throws -> D1Row {
            D1Row(qid: "q", question: try D1Question(json: try D1JSONParser.parse(q)), ids: [], markers: [], calibrate: true,
                  prefixLength: 0, mode: .text, maxLen: 0)
        }
        let noul = try row(#"{"type": "noul", "instructions": "?"}"#)
        // The publisher reports a noul as [yes, no]; the kit's order is [no, yes].
        #expect(try KitD1OmniDecider.kitOrder(.noul("?"), row: noul, [0.75, 0.25], id: "q") == [0.25, 0.75])
        let choice = try row(#"{"type": "choice", "instructions": "?", "criteria": {"b": null, "a": null}}"#)
        #expect(try KitD1OmniDecider.kitOrder(.choice("?", ["b", "a"]), row: choice, [0.75, 0.25], id: "q") == [0.75, 0.25])
        #expect(throws: SystemOne.WireError.self) {
            try KitD1OmniDecider.kitOrder(.choice("?", ["a", "b"]), row: choice, [0.5, 0.5], id: "q")
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func theTextsAreWhatPythonWrites() throws {
        #expect(D1PythonFormat.floatRepr(1e-5) == "1e-05" && D1PythonFormat.floatRepr(1250) == "1250.0")
        #expect(D1PythonFormat.floatRepr(1e16) == "1e+16" && D1PythonFormat.floatRepr(0.1) == "0.1")
        #expect(D1PythonFormat.pyStr(try D1JSONParser.parse(#"{"a": [1, 2.50, null, true]}"#)) == "{'a': [1, 2.5, None, True]}")
        #expect(D1Prompt.criterion(try D1JSONParser.parse(#"{"k": [1, "é"]}"#)) == #"{"k": [1, "é"]}"#)
        #expect(D1Prompt.serialize(try D1JSONParser.parse(#"["é", 1e3]"#)) == #"["é", 1000.0]"#)
        #expect(D1Tokenizer.escape("a <|pad|> b <|x y|>") == "a <\u{A6}pad\u{A6}> b <|x y|>")
        #expect(D1PythonFormat.pySum([0.1, 0.2, 0.3]) == 0.6)
        #expect(D1GraphInputs.bucket(positions: 65, buckets: KitD1OmniDecider.decisionLengths) == 128)
        #expect(D1GraphInputs.bucket(positions: 4097, buckets: KitD1OmniDecider.decisionLengths) == nil)
    }

    // MARK: - The fixture

    struct Fixture {
        struct Row {
            let id: String
            let record: String
            let mode: String
            let qid: String
            let prefixRows: Int
            let idsCount: Int
            let idsSHA256: String
            let markers: [Int]
            let bucket: Int
        }

        struct SmokeRow {
            let qid: String
            let bucket: Int
            let logitsBits: [UInt32]
            let probsBits: [UInt32]
        }

        struct Smoke {
            let record: String
            let mode: String
            let stateIsNone: Bool
            let questions: JSONValue
            let media: [String]
            let response: String
            let rows: [SmokeRow]
        }

        let rows: [Row]
        let smoke: [Smoke]

        static func load() throws -> Fixture {
            let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/d1_omni_swift.json")
            let root = try JSONValue.parse(Data(contentsOf: url))
            #expect(root["schema"]?.stringValue == "coreai-kit-d1-omni-test/1")
            #expect(root["model"]?["revision"]?.stringValue == D1OmniTests.pin)
            let ints: (JSONValue?) -> [Int] = { ($0?.elements ?? []).compactMap { $0.doubleValue.map(Int.init) } }
            let rows = (root["rows"]?.elements ?? []).map { r -> Row in
                let f = r.elements ?? []
                let id = f[0].stringValue ?? ""
                let parts = id.split(separator: "/", maxSplits: 2).map(String.init)
                return Row(
                    id: id, record: parts[0], mode: parts[1], qid: parts[2], prefixRows: ints(.array([f[1]]))[0],
                    idsCount: ints(.array([f[2]]))[0], idsSHA256: f[3].stringValue ?? "", markers: ints(f[4]),
                    bucket: ints(.array([f[5]]))[0])
            }
            let smoke = (root["smoke"]?.elements ?? []).map { s in
                Smoke(
                    record: s["record"]?.stringValue ?? "", mode: s["mode"]?.stringValue ?? "",
                    stateIsNone: s["state_is_none"] == .bool(true), questions: s["questions"] ?? .null,
                    media: (s["media"]?.elements ?? []).compactMap(\.stringValue),
                    response: s["response"]?.stringValue ?? "",
                    rows: (s["rows"]?.elements ?? []).map { r in
                        SmokeRow(
                            qid: r["qid"]?.stringValue ?? "", bucket: ints(r["bucket"].map { .array([$0]) })[0],
                            logitsBits: ints(r["logits_bits"]).map { UInt32($0) },
                            probsBits: ints(r["probs_bits"]).map { UInt32($0) })
                    })
            }
            return Fixture(rows: rows, smoke: smoke)
        }
    }

    /// sha256 of `json.dumps(ids)`, the fixture's digest of a row.
    static func idsSHA256(_ ids: [Int]) -> String {
        let text = "[" + ids.map(String.init).joined(separator: ", ") + "]"
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    /// The fixture's 38 smoke rows through the kit's readout, from the zoo Swift host's marker logits: the probabilities
    /// bit for bit, and each request's response as the publisher's code writes it for them.
    @available(macOS 27, iOS 27, *)
    @Test func theReadoutWritesThePublishersResponse() throws {
        let fixture = try Fixture.load()
        #expect(fixture.rows.count == 360 && fixture.smoke.count == 14)
        #expect(fixture.smoke.map(\.rows.count).reduce(0, +) == 38)
        let rows = Dictionary(uniqueKeysWithValues: fixture.rows.map { ($0.id, $0) })
        for s in fixture.smoke {
            let questions = KitD1OmniDecider.d1JSON(s.questions)
            var d1Rows: [D1Row] = []
            var probs: [[Float]] = []
            for r in s.rows {
                let q = try D1Question(json: try #require(questions[r.qid]))
                let row = try #require(rows["\(s.record)/\(s.mode)/\(r.qid)"])
                let p = D1Readout.probabilities(
                    logits: r.logitsBits.map(Float.init(bitPattern:)), question: q, calibrate: s.mode == "text")
                #expect(p.map(\.bitPattern) == r.probsBits, "\(s.record)/\(r.qid): p differ from the zoo host's")
                d1Rows.append(D1Row(
                    qid: r.qid, question: q, ids: [Int](repeating: 0, count: row.idsCount), markers: row.markers,
                    calibrate: s.mode == "text", prefixLength: row.prefixRows, mode: D1Mode(rawValue: s.mode)!, maxLen: 0))
                probs.append(p)
            }
            let response = D1PythonFormat.dumps(D1Readout.response(rows: d1Rows, probabilities: probs), asciiOnly: false)
            #expect(response == s.response, "\(s.record): the response differs from the publisher's")
        }
    }

    // MARK: - The public rows, token for token (the tokenizer and reference/)

    /// KIT_D1_OMNI_STORE, else the default store.
    static var store: ModelStore {
        if let path = environment["KIT_D1_OMNI_STORE"] { return ModelStore(directory: URL(fileURLWithPath: path)) }
        return .default
    }

    @available(macOS 27, iOS 27, *)
    static func local(_ path: String) -> URL? {
        guard let entry = ModelCatalog.builtin.entry(id: "d1-omni-600m") else { return nil }
        return store.localURL(for: entry.modelID(path: path))
    }

    /// KIT_D1_OMNI_TOKENIZER, else the smallest bucket's tokenizer, else the repo's `tokenizer/`, in the store.
    static var tokenizerFolder: URL? {
        if let path = environment["KIT_D1_OMNI_TOKENIZER"] { return URL(fileURLWithPath: path) }
        guard #available(macOS 27, iOS 27, *) else { return nil }
        return local("macos/decide-fp16-L64").map { $0.appendingPathComponent("tokenizer") } ?? local("tokenizer")
    }

    /// KIT_D1_OMNI_REFERENCE, else the repo's `reference/` in the store.
    static var referenceFolder: URL? {
        if let path = environment["KIT_D1_OMNI_REFERENCE"] { return URL(fileURLWithPath: path) }
        guard #available(macOS 27, iOS 27, *) else { return nil }
        return local("reference")
    }

    static var rowsEnabled: Bool { tokenizerFolder != nil && referenceFolder != nil }

    /// The records of `reference/records.json` by id, read as `json.loads` reads them.
    @available(macOS 27, iOS 27, *)
    static func records(_ reference: URL) throws -> [String: D1JSONValue] {
        let root = try D1JSONParser.parse(Data(contentsOf: reference.appendingPathComponent("records.json")))
        var out: [String: D1JSONValue] = [:]
        for r in root["records"]?.array ?? [] { if let id = r["id"]?.string { out[id] = r } }
        return out
    }

    /// Every public row of the zoo's fixture rebuilt by the kit from the record (the media prefix rows given): the ids,
    /// the markers and the length the row runs at, with the kit's swift-transformers.
    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: rowsEnabled, "set KIT_D1_OMNI_TOKENIZER and KIT_D1_OMNI_REFERENCE (or keep the bundles in the store)"))
    func everyPublicRowIsThePublishers() async throws {
        let fixture = try Fixture.load()
        let records = try Self.records(try #require(Self.referenceFolder))
        let tokenizer = try await D1Tokenizer.load(
            folder: try #require(Self.tokenizerFolder),
            expected: ["<|pad|>": 0, "<|startoftext|>": 1, "<|im_end|>": 7, "<|mask|>": 16, "<|reserved_7|>": 17,
                       "<|reserved_8|>": 18, "<|reserved_9|>": 19, "<|reserved_10|>": 20, "<|reserved_11|>": 21])
        var cache: [String: [D1Row]] = [:]
        var equal = 0
        for row in fixture.rows {
            let key = "\(row.record)/\(row.mode)/\(row.prefixRows)"
            if cache[key] == nil {
                let request = try #require(records[row.record]?["request"], "\(row.record): not in records.json")
                cache[key] = try D1Prompt.rows(
                    tokenizer, state: request["state"], questions: try #require(request["questions"]),
                    mode: try #require(D1Mode(rawValue: row.mode)), prefixLength: row.prefixRows)
            }
            let built = try #require(cache[key]?.first { $0.qid == row.qid }, "\(row.id): no row")
            let same = built.ids.count == row.idsCount && Self.idsSHA256(built.ids) == row.idsSHA256
                && built.markers == row.markers
                && D1GraphInputs.bucket(positions: built.positions, buckets: KitD1OmniDecider.decisionLengths) == row.bucket
            #expect(same, "\(row.id): the kit's row differs from the publisher's")
            if same { equal += 1 }
        }
        #expect(equal == 360)
    }

    // MARK: - Runs of the zoo's Swift host, bit for bit (the bundles)

    @available(macOS 27, iOS 27, *)
    static var bundlesPresent: Bool {
        guard let variant = ModelCatalog.builtin.entry(id: "d1-omni-600m")?.variant?.path else { return false }
        return KitD1OmniDecider.folders.allSatisfy { local("\(variant)/\($0)") != nil } && referenceFolder != nil
    }

    static var gateEnabled: Bool {
        guard #available(macOS 27, iOS 27, *) else { return false }
        return environment["KIT_D1_OMNI_GATE"] == "1" || bundlesPresent
    }

    /// The fixture's 14 smoke requests through `KitD1OmniDecider(catalog:)`: per row the ids, the markers and the length
    /// against the publisher's, the marker logits and the probabilities against the zoo Swift host's (bit for bit), and
    /// each response against the publisher's text; an image and a clip again over the wire (as base64, and the clip with
    /// no state), whose answers carry the same probabilities in the kit's order; and only the lengths the rows needed
    /// loaded.
    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: gateEnabled, "set KIT_D1_OMNI_GATE=1, or keep the bundles and reference/ in the store"))
    func runsMatchTheZooSwiftHostBitForBit() async throws {
        let fixture = try Fixture.load()
        let rows = Dictionary(uniqueKeysWithValues: fixture.rows.map { ($0.id, $0) })
        let store = Self.store
        let start = ContinuousClock.now
        let decider = try await KitD1OmniDecider(catalog: "d1-omni-600m", store: store)
        let load = ContinuousClock.now - start
        let entry = try #require(ModelCatalog.builtin.entry(id: "d1-omni-600m"))
        var reference: URL
        if let local = Self.referenceFolder {
            reference = local
        } else {
            reference = try await store.download(entry.modelID(path: "reference"))
        }
        let records = try Self.records(reference)
        #expect(try await decider.loadedLengths().isEmpty)
        var transcript: [JSONValue] = []
        var requestsEqual = 0, rowsEqual = 0
        for s in fixture.smoke {
            let request = try #require(records[s.record]?["request"])
            let media = s.media.map { reference.appendingPathComponent($0) }
            let r = try await decider.readout(
                requestJSON: Data(D1PythonFormat.dumps(request, asciiOnly: false).utf8),
                images: s.mode == "image" ? media : [], audio: s.mode == "audio" ? media.first : nil)
            #expect(r.mode == s.mode && r.rows.count == s.rows.count, "\(s.record)")
            for (row, ref) in zip(r.rows, s.rows) {
                let fix = try #require(rows["\(s.record)/\(s.mode)/\(ref.qid)"])
                let idsOK = row.questionID == ref.qid && row.ids.count == fix.idsCount
                    && Self.idsSHA256(row.ids) == fix.idsSHA256 && row.markers == fix.markers
                    && row.prefixRows == fix.prefixRows && row.bucket == ref.bucket
                let logitsOK = row.logits.map(\.bitPattern) == ref.logitsBits
                let probsOK = row.probabilities.map(\.bitPattern) == ref.probsBits
                #expect(idsOK, "\(s.record)/\(ref.qid): the row differs from the publisher's")
                #expect(logitsOK && probsOK, "\(s.record)/\(ref.qid): the marker logits or p differ from the zoo host's")
                if idsOK && logitsOK && probsOK { rowsEqual += 1 }
                transcript.append(.object([
                    .init("row", .string("\(s.record)/\(s.mode)/\(ref.qid)")), .init("bucket", .int(row.bucket)),
                    .init("prefix_rows", .int(row.prefixRows)), .init("positions", .int(row.prefixRows + row.ids.count)),
                    .init("ids_equal_publisher", .bool(idsOK)), .init("logits_bits_equal_zoo_swift", .bool(logitsOK)),
                    .init("p_bits_equal_zoo_swift", .bool(probsOK)),
                ]))
            }
            let responseOK = r.response == s.response
            #expect(responseOK, "\(s.record): the response differs from the publisher's")
            if responseOK { requestsEqual += 1 }
        }
        #expect(rowsEqual == 38 && requestsEqual == 14)

        // Over the wire: img_01 (no state: the publisher reads None as "" for an image) and aud_09 (no state at all).
        var wire: [JSONValue] = []
        for (record, field) in [("img_01", "images"), ("aud_09", "audio")] {
            let s = try #require(fixture.smoke.first { $0.record == record })
            let request = try #require(records[record]?["request"])
            let bytes = try Data(contentsOf: reference.appendingPathComponent(s.media[0]))
            let media = "data:application/octet-stream;base64,\(bytes.base64EncodedString())"
            let value = field == "images" ? "[\"\(media)\"]" : "\"\(media)\""
            let state = field == "images" ? #""state": "", "# : ""
            let body = "{\(state)\"\(field)\": \(value), \"questions\": "
                + D1PythonFormat.dumps(try #require(request["questions"]), asciiOnly: false) + "}"
            let response = try await decider.systemOne(try SystemOne.request(from: Data(body.utf8)))
            #expect(response.model == "d1-omni-600m" && response.metadata?["mode"]?.stringValue == s.mode)
            var equal = true
            for (answer, ref) in zip(response.answers, s.rows) {
                let p = ref.probsBits.map { Double(Float(bitPattern: $0)) }
                // A noul keeps P(yes) alone; the others every option's p, in the kit's order.
                let ok = answer.answer.noul.map { [$0] == [p[0]] } ?? (answer.answer.probabilities == p)
                #expect(ok, "\(record) \(answer.id): systemOne's p differ from the zoo host's")
                equal = equal && ok
            }
            wire.append(.object([.init("request", .string(record)), .init("p_equal_zoo_swift", .bool(equal))]))
        }
        let loaded = try await decider.loadedLengths()
        #expect(loaded == [64, 128, 256, 512, 2048, 4096], "only the lengths the rows needed: \(loaded)")

        if let out = Self.environment["KIT_D1_OMNI_OUT"] {
            let seconds = Double(load.components.seconds) + Double(load.components.attoseconds) * 1e-18
            let doc = JSONValue.object([
                .init("schema", .string("coreai-kit-d1-omni-test-run/1")), .init("model", .string(decider.id)),
                .init("bundle", .string(decider.modelName)), .init("init_seconds", .double(seconds)),
                .init("requests", .int(fixture.smoke.count)), .init("requests_equal", .int(requestsEqual)),
                .init("rows", .int(transcript.count)), .init("rows_bit_equal", .int(rowsEqual)),
                .init("loaded_lengths", .array(loaded.map { .int($0) })), .init("wire", .array(wire)),
                .init("per_row", .array(transcript)),
            ])
            try doc.dumps().appending("\n").write(toFile: out, atomically: true, encoding: .utf8)
        }
    }
}
