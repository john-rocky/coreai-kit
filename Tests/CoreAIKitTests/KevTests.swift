// KevTests.swift — Kev in the kit: the catalog entries and what loads them, the request in the author's form, the order
// the probabilities come back in, the refusal of a graph that does not ship, and — opt-in, with the weights — runs of
// the model zoo's Swift host, bit for bit (Fixtures/kev_swift13.json: the author's row ids, and the zoo host's hidden-row
// digests, probabilities and answers for the same bundles, direct and with the shared prefix):
//
//     KIT_KEV_GATE=1 [KIT_KEV_OUT=<transcript.json>] swift test --filter Kev          # Kev-0.8B, 10 records
//     KIT_KEV_GATE_4B=1 [KIT_KEV_OUT_4B=<transcript.json>] swift test --filter Kev    # Kev-4B, 3 records
//
// Each gate downloads its model from the catalog pin on its first run (1.5 GB and 8.4 GB), and the first load
// specializes the graph (3.3 s for 0.8B, 16.6 s and 15.55 GB of the runtime's cache for 4B in the zoo's runs).

import CryptoKit
import Foundation
import Testing

@testable import CoreAIKit

struct KevTests {
    static let pins = [
        "kev-0.8b": "3368b4e970fa17dd145c9341c387d628b0277cb3", "kev-4b": "dad3eb2b2e338b6be6d259a9317db37bbeb2e0d7",
    ]

    // MARK: - The catalog entries

    @available(macOS 27, iOS 27, *)
    @Test func theCatalogEntriesAreKevAtTheirPins() throws {
        let small = try #require(ModelCatalog.builtin.entry(id: "kev-0.8b"))
        let large = try #require(ModelCatalog.builtin.entry(id: "kev-4b"))
        for entry in [small, large] {
            #expect(entry.kind == .rowDecision)
            #expect(entry.format == KitKevDecider.format)
            #expect(entry.revision == Self.pins[entry.id])
            #expect(entry.assets == nil && entry.engine == nil && entry.calibration == nil)
        }
        let path = "gpu-pipelined/kev_0_8b_decode_fp16_metal_pf128"
        #expect(small.variants["macos"]?.path == path && small.variants["ios"]?.path == path)
        #expect(small.modelID == ModelID("mlboydaisuke/Kev-0.8B-CoreAI", path: path, revision: Self.pins["kev-0.8b"]!))
        #expect(large.variants["macos"]?.path == "gpu-pipelined/kev_4b_decode_fp16_metal_pf128")
        #expect(large.variants["ios"] == nil)  // the Mac only
    }

    /// `TypedDecisions` would read the decoder's hidden states as logits: a `rowDecision` entry is `KitKevDecider`'s,
    /// every other decision entry stays where it was, and `TypedDecisions(catalog:)` refuses Kev by its kind before it
    /// downloads anything.
    @available(macOS 27, iOS 27, *)
    @Test func onlyKitKevDeciderLoadsARowDecisionEntry() async throws {
        let entry = try #require(ModelCatalog.builtin.entry(id: "kev-0.8b"))
        #expect(!TypedDecisions.supports(entry) && !KitClefDecider.supports(entry))
        #expect(KitKevDecider.supports(entry) && SystemOne.supports(entry))
        for other in ModelCatalog.builtin.available(.decision) + ModelCatalog.builtin.available(.jointDecision) {
            #expect(!KitKevDecider.supports(other), "\(other.id)")
        }
        let otherFormat = CatalogEntry(
            id: entry.id, name: entry.name, repo: entry.repo, revision: entry.revision, kind: .rowDecision,
            variants: entry.variants, format: "jointHead")
        #expect(!KitKevDecider.supports(otherFormat))

        // The store's Hub is a closed local port: a download would fail on the transport, not on the kind.
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(directory: root, hubBaseURL: URL(string: "http://127.0.0.1:9")!)
        do {
            _ = try await TypedDecisions(catalog: "kev-0.8b", store: store)
            Issue.record("TypedDecisions loaded kev-0.8b")
        } catch let error as CoreAIKitError {
            guard case .catalogKindMismatch(let id, let expected, let found) = error else {
                Issue.record("not the kind refusal: \(error)")
                return
            }
            #expect(id == "kev-0.8b" && expected == "chat or decision" && found == "rowDecision")
        }
        #expect(((try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []).isEmpty)
    }

    /// The graph that takes any call length up to its cap was measured and does not ship (its memory grows): a bundle
    /// that declares one is refused by name before the graph loads.
    @available(macOS 27, iOS 27, *)
    @Test func aDynamicQueryLengthBundleIsRefused() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let delimiters = [("state", 248060), ("q", 248061), ("opt", 248049), ("opt_end", 248050), ("decide", 248062)]
            .map { #""\#($0.0)": {"id": \#($0.1)}"# }.joined(separator: ", ")
        let metadata = #"""
            {"name": "kev_dyn", "assets": {"main": "kev_dyn.aimodel"},
             "language": {"vocab_size": 248320, "max_context_length": 4096, "query_len_range": [2, 512],
                          "query_len_call_max": 512, "query_len_multiple": 16},
             "decision": {"row": {"delimiters": {\#(delimiters)}, "pad": {"id": 248044}},
                          "head": {"files": ["head/head.safetensors", "head/kev_head.json"], "scale": 0.0625}}}
            """#
        try Data(metadata.utf8).write(to: dir.appendingPathComponent("metadata.json"))
        do {
            _ = try await KitKevDecider(bundleAt: dir)
            Issue.record("a dynamic query length bundle loaded")
        } catch let error as DecisionError {
            #expect(error.localizedDescription.contains("dynamic query length"))
        }
    }

    // MARK: - The request in the author's form

    /// The wire's own values: a structured state with its numbers as written, a choice listed as names read as options
    /// alone, `model` filled in; the author's text for them.
    @available(macOS 27, iOS 27, *)
    @Test func aWireRequestIsReadFromItsOwnValues() throws {
        let body = #"""
            {"state": {"total": 1250.0, "count": 3, "ok": true, "items": ["a", {"n": 1e-5}]},
             "questions": {"pick": {"type": "choice", "instructions": "Which?", "criteria": ["b", "a"]},
                           "ok": {"type": "noul", "instructions": {"rule": 2}, "criteria": {"true": "it holds"}},
                           "how": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}}
            """#
        let request = try SystemOne.request(from: Data(body.utf8))
        let root = KitKevDecider.kevRoot(request, id: "kev-0.8b")
        #expect(root["model"] == .string("kev-0.8b"))
        let kev = try KitKevDecider.kevRequest(root)
        let (record, meta) = KevText.record(kev)
        #expect(record.state == "total: 1250.0\ncount: 3\nok: True\nitems:\n  - a\n  - n: 1e-05")
        #expect(meta.map(\.keys) == [["b", "a"], ["false", "true"], ["0", "1"]])
        #expect(record.questions[0].options == ["b", "a"])
        #expect(record.questions[1].instr == "rule: 2" && record.questions[1].options == ["no", "yes: it holds"])
        #expect(record.questions[2].options == ["low", "high"])
    }

    /// A request built in Swift: an option whose description is its id reads as a name alone.
    @available(macOS 27, iOS 27, *)
    @Test func aSwiftRequestIsReadFromItsTypedQuestions() throws {
        let request = SystemOne.Request(state: "s", questions: [
            (id: "c", question: .choice("Which?", options: [.init(id: "x", description: "the x"), .init("y")])),
            (id: "s", question: .score("How?", levels: ["low", "high"])),
            (id: "n", question: .noul("Yes?", no: "it fails")),
        ])
        let (record, meta) = KevText.record(try KitKevDecider.kevRequest(KitKevDecider.kevRoot(request, id: "kev-4b")))
        #expect(record.state == "s")
        #expect(record.questions[0].options == ["x: the x", "y"])
        #expect(record.questions[1].options == ["low", "high"])
        #expect(record.questions[2].options == ["no: it fails", "yes"])
        #expect(meta.map(\.id) == ["c", "s", "n"])
    }

    @available(macOS 27, iOS 27, *)
    @Test func probabilitiesComeBackInTheKitsOptionOrder() throws {
        #expect(try KitKevDecider.kitOrder(.noul("?"), keys: ["false", "true"], [0.25, 0.75], id: "q") == [0.25, 0.75])
        #expect(try KitKevDecider.kitOrder(.choice("?", ["b", "a"]), keys: ["b", "a"], [0.75, 0.25], id: "q") == [0.75, 0.25])
        #expect(throws: SystemOne.WireError.self) {
            try KitKevDecider.kitOrder(.choice("?", ["a", "c"]), keys: ["a", "b"], [0.5, 0.5], id: "q")
        }
    }

    @Test func theRendererWritesWhatPythonWrites() throws {
        #expect(KevText.render(.number("1250.0")) == "1250.0")
        #expect(KevText.render(.number("1e-5")) == "1e-05")
        #expect(KevText.render(.number("-0")) == "0")
        #expect(KevText.render(.bool(false)) == "False")
        #expect(KevText.render(.null) == "")
        #expect(KevTokenizer.rewriteDelimiterText("a <|fim_prefix|> b <|x y|>") == "a <\u{A6}fim_prefix\u{A6}> b <|x y|>")
        #expect(KevPythonFormat.pyRound(0.97825, 4) == 0.9782)
        #expect(KevAnswers.pySum([0.1, 0.2, 0.3]) == 0.6)
    }

    // MARK: - Runs of the zoo's Swift host, bit for bit (opt-in: the weights)

    static var environment: [String: String] { ProcessInfo.processInfo.environment }

    static func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/\(name)")
    }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: environment["KIT_KEV_GATE"] == "1"))
    func kev08bRunsMatchTheZooSwiftHostBitForBit() async throws {
        try await Self.gate(model: "kev-0.8b", records: 10, out: Self.environment["KIT_KEV_OUT"])
    }

    @available(macOS 27, iOS 27, *)
    @Test(.enabled(if: environment["KIT_KEV_GATE_4B"] == "1"))
    func kev4bRunsMatchTheZooSwiftHostBitForBit() async throws {
        try await Self.gate(model: "kev-4b", records: 3, out: Self.environment["KIT_KEV_OUT_4B"])
    }

    /// Every run of `model` in the reference file through `KitKevDecider(catalog:)`: direct and shared, the row ids
    /// against the author's (their sha256 in the zoo's fixture), every row's hidden digest and p bits and the answers
    /// against the zoo host's; then the request through `systemOne` (the shared prefix on), whose answers carry the same
    /// p in the kit's option order.
    @available(macOS 27, iOS 27, *)
    static func gate(model: String, records: Int, out: String?) async throws {
        let reference = try JSONValue.parse(Data(contentsOf: fixtureURL("kev_swift13.json")))
        let runs = (reference["runs"]?.elements ?? []).filter { $0["model"]?.stringValue == model }
        #expect(runs.count == records)
        let start = ContinuousClock.now
        let decider = try await KitKevDecider(catalog: model)
        let load = ContinuousClock.now - start
        let ints: (JSONValue?) -> [Int] = { ($0?.elements ?? []).compactMap { $0.doubleValue.map(Int.init) } }
        var rows: [JSONValue] = []
        var bitEqual = 0
        for run in runs {
            let id = run["id"]?.stringValue ?? ""
            let request = try #require(run["request"])
            let requestJSON = Data(request.dumps().utf8)
            var equal = true
            for mode in ["direct", "shared"] {
                let r = try await decider.readout(requestJSON: requestJSON, shared: mode == "shared")
                let refRows = run["rows"]?.elements ?? []
                #expect(r.rows.count == refRows.count, "\(id) \(mode): rows")
                for (k, (row, ref)) in zip(r.rows, refRows).enumerated() {
                    let littleEndian = row.ids.flatMap { id in withUnsafeBytes(of: Int32(id).littleEndian) { Array($0) } }
                    let idsDigest = SHA256.hash(data: Data(littleEndian)).map { String(format: "%02x", $0) }.joined()
                    let idsOK = idsDigest == ref["row_ids_sha256"]?.stringValue
                        && row.ids.count == Int(ref["row_len"]?.doubleValue ?? -1)
                        && row.decide == Int(ref["decide"]?.doubleValue ?? -1) && row.options == ints(ref["opts"])
                    let hiddenKey = mode == "shared" ? "shared_hidden_sha256" : "hidden_sha256"
                    let hiddenOK = row.hiddenSHA256 == ref[hiddenKey]?.stringValue
                    let pKey = mode == "shared" ? "shared_p_bits" : "p_bits"
                    let pOK = row.probabilities.map { Int($0.bitPattern) } == ints(ref[pKey])
                    #expect(idsOK, "\(id) q\(k) \(mode): row ids differ from the author's")
                    #expect(hiddenOK, "\(id) q\(k) \(mode): hidden rows differ from the zoo host's")
                    #expect(pOK, "\(id) q\(k) \(mode): p differs from the zoo host's")
                    equal = equal && idsOK && hiddenOK && pOK
                    rows.append(.object([
                        .init("run", .string("\(id)/q\(k)/\(mode)")), .init("tokens", .int(row.ids.count)),
                        .init("ids_equal_author", .bool(idsOK)), .init("hidden_sha256_equal_zoo_swift", .bool(hiddenOK)),
                        .init("p_bits_equal_zoo_swift", .bool(pOK)), .init("hidden_sha256", .string(row.hiddenSHA256)),
                    ]))
                }
                let answersOK = r.answersJSON == run[mode == "shared" ? "shared_answers_json" : "answers_json"]?.stringValue
                let usageOK = r.inputTokens == Int(run["input_tokens"]?.doubleValue ?? -1)
                    && r.outputTokens == Int(run["output_tokens"]?.doubleValue ?? -1)
                #expect(answersOK, "\(id) \(mode): answers differ from the zoo host's")
                #expect(usageOK, "\(id) \(mode): usage differs from the zoo host's")
                #expect(r.sharedPrefixTokens == (mode == "shared" ? Int(run["shared_tokens"]?.doubleValue ?? -1) : nil))
                equal = equal && answersOK && usageOK
            }
            // The same request through the wire: the kit's answers carry the zoo host's p.
            let response = try await decider.systemOne(try SystemOne.request(from: requestJSON))
            for (k, answer) in response.answers.enumerated() {
                let ref = (run["rows"]?.elements ?? [])[k]
                let p = ints(ref["shared_p_bits"]).map { Double(Float(bitPattern: UInt32($0))) }
                // A noul answer keeps P(yes) alone; the others every option's p.
                let pOK = answer.answer.noul.map { [$0] == p.suffix(1) } ?? (answer.answer.probabilities == p)
                #expect(pOK, "\(id) \(answer.id): systemOne's p differ from the zoo host's")
                equal = equal && pOK
            }
            #expect(response.model == model)
            if equal { bitEqual += 1 }
        }
        #expect(bitEqual == runs.count)
        if let out {
            let transcript = JSONValue.object([
                .init("schema", .string("coreai-kit-kev-test/1")), .init("model", .string(decider.id)),
                .init("bundle", .string(decider.modelName)),
                .init("load_seconds", .number("\(Double(load.components.seconds) + Double(load.components.attoseconds) * 1e-18)")),
                .init("records", .int(runs.count)), .init("bit_equal_records", .int(bitEqual)), .init("rows", .array(rows)),
            ])
            try transcript.dumps().appending("\n").write(toFile: out, atomically: true, encoding: .utf8)
        }
    }
}
