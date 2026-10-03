// From the model zoo's apps/ClefFlash/Sources/ClefFlash/Head.swift (5ef2247, sha256 671fc5adbe34, the code its Swift gate ran), identifiers prefixed Clef for the kit.
// Head — the joint schema head graph (conversion/clef_flash/clef_head.py `ClefHeadGraph`, bucket form) and the host
// arrays that feed it (`clef_head.head_inputs`), plus the lm_head gather table it reads through the host.
//
//   functions t512 / t1024 / t2048 / t4096 (T = the smallest bucket >= the row's tokens), Q = 16, O = 128:
//   hidden    [T, 4096] f32   the decoder's fp16 hidden rows as float32, rows past the real tokens zero
//   key_valid [T]       f32   1 = a real token
//   q_avg     [Q, T]    f32   row q = float32(1) / float32(len) over question q's span
//   o_avg     [O, T]    f32   row o = float32(1) / float32(len) over option o's span
//   g_avg     [1, T]    f32   1 at the last real token
//   lexical   [O, 4096] f32   the mean of the table's fp16 rows of option o's span ids (float64 sum / len -> float32)
//   member    [Q, O]    f32   1 where option o belongs to question q
//   type_ids  [Q]       i32   noul 0, choice 1, score 2
//   q_valid   [Q]       f32   1 = a real question
//   o_valid   [O]       f32   1 = a real option
//   -> logits [O]       f32   in the author's option order; padding options 0
//
// The table (lm_head_fp16.bin) is the untied lm_head.weight as raw little-endian fp16 [248320, 4096], row t at byte
// t * 8192, no header; it is memory-mapped, and only the rows of the option spans are read.

import CoreAI
import Darwin
import Foundation

// Float16 is unavailable on an Intel Mac, which a universal Release build still compiles the kit for: this file is
// Apple silicon only, and KitClefDecider refuses to load there.
#if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))

@available(macOS 27, iOS 27, *)
final class ClefLexicalTable: @unchecked Sendable {
    static let rows = 248_320
    static let width = 4096
    static let bytes = rows * width * 2          // 2,034,237,440

    let url: URL
    private let base: UnsafeRawPointer
    private let length: Int

    init(contentsOf url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw ClefFlashError.bundle("cannot open \(url.path)") }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw ClefFlashError.bundle("cannot stat \(url.path)") }
        guard Int(st.st_size) == Self.bytes else {
            throw ClefFlashError.contract("lm_head table \(url.lastPathComponent): \(st.st_size) bytes, the contract says \(Self.bytes) ([248320, 4096] fp16)")
        }
        guard let p = mmap(nil, Self.bytes, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            throw ClefFlashError.bundle("cannot map \(url.path)")
        }
        self.url = url
        self.base = UnsafeRawPointer(p)
        self.length = Self.bytes
    }

    deinit { munmap(UnsafeMutableRawPointer(mutating: base), length) }

    /// `clef_head.lexical_rows` for one span: the mean of rows ids[a ..< b], summed in float64 (exact for fp16
    /// inputs), divided by the count in float64, rounded to float32. Writes 4096 values at `out`.
    func spanMean(ids: [Int], _ a: Int, _ b: Int, into out: UnsafeMutablePointer<Float>) {
        var acc = [Double](repeating: 0, count: Self.width)
        let rows = base.assumingMemoryBound(to: Float16.self)
        acc.withUnsafeMutableBufferPointer { s in
            for i in a..<b {
                let row = rows + ids[i] * Self.width
                for j in 0..<Self.width { s[j] += Double(row[j]) }
            }
        }
        let n = Double(b - a)
        for j in 0..<Self.width { out[j] = Float(acc[j] / n) }
    }
}

@available(macOS 27, iOS 27, *)
final class ClefHead: @unchecked Sendable {
    static let qPad = 16
    static let oPad = 128
    static let hidden = 4096
    static let inputNames = ["hidden", "key_valid", "q_avg", "o_avg", "g_avg", "lexical", "member", "type_ids",
                                    "q_valid", "o_valid"]

    struct Bucket: Sendable {
        let function: String
        let tokens: Int
    }

    /// The ten arrays of one row (row-major, graph dtypes) and each question's [start, end) option slice.
    struct Arrays: Sendable {
        let bucket: Bucket
        var hidden: [Float]
        var keyValid: [Float]
        var qAvg: [Float]
        var oAvg: [Float]
        var gAvg: [Float]
        var lexical: [Float]
        var member: [Float]
        var typeIDs: [Int32]
        var qValid: [Float]
        var oValid: [Float]
        let layout: [(Int, Int)]
        let options: Int
    }

    let url: URL
    let buckets: [Bucket]
    let loadSeconds: Double
    let descriptors: ClefJSON
    private let functions: [String: InferenceFunction]
    private let inputDescriptors: [String: [String: NDArrayDescriptor]]

    init(contentsOf url: URL, buckets: [Bucket], options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        var fns: [String: InferenceFunction] = [:]
        var ins: [String: [String: NDArrayDescriptor]] = [:]
        var desc: [ClefJSONMember] = []
        for b in buckets {
            guard let fd = model.functionDescriptor(for: b.function), let fn = try model.loadFunction(named: b.function) else {
                throw ClefFlashError.contract("head \(url.lastPathComponent): no function \(b.function) (has \(model.functionNames))")
            }
            let t = b.tokens, q = Self.qPad, o = Self.oPad, h = Self.hidden
            try clefCheckFunction(fd, what: "head \(b.function)", inputs: [
                "hidden": ClefTensorSpec(shape: [t, h], type: .float32),
                "key_valid": ClefTensorSpec(shape: [t], type: .float32),
                "q_avg": ClefTensorSpec(shape: [q, t], type: .float32),
                "o_avg": ClefTensorSpec(shape: [o, t], type: .float32),
                "g_avg": ClefTensorSpec(shape: [1, t], type: .float32),
                "lexical": ClefTensorSpec(shape: [o, h], type: .float32),
                "member": ClefTensorSpec(shape: [q, o], type: .float32),
                "type_ids": ClefTensorSpec(shape: [q], type: .int32),
                "q_valid": ClefTensorSpec(shape: [q], type: .float32),
                "o_valid": ClefTensorSpec(shape: [o], type: .float32),
            ], outputs: ["logits": ClefTensorSpec(shape: [o], type: .float32)], states: [:])
            fns[b.function] = fn
            ins[b.function] = Dictionary(uniqueKeysWithValues: Self.inputNames.map { ($0, ClefND.descriptor(fd.inputDescriptor(of: $0))!) })
            desc.append(ClefJSONMember(b.function, clefDescribe(fd)))
        }
        self.url = url
        self.buckets = buckets.sorted { $0.tokens < $1.tokens }
        self.loadSeconds = clefSecondsSince(t0)
        self.descriptors = .object(desc)
        self.functions = fns
        self.inputDescriptors = ins
    }

    /// The smallest bucket that holds `tokens`.
    func bucket(for tokens: Int) -> Bucket? { buckets.first { $0.tokens >= tokens } }

    /// `clef_head.head_inputs(hidden, ids, questions, table, t_pad=bucket, q_pad=16, o_pad=128)`.
    func arrays(hidden16: [Float16], row: ClefPromptRow, table: ClefLexicalTable) throws -> Arrays {
        let T = row.ids.count
        guard let bucket = bucket(for: T) else {
            throw ClefFlashError.prompt("\(T) tokens exceed the head's largest bucket (\(buckets.last?.tokens ?? 0))")
        }
        let Tp = bucket.tokens, Q = row.questions.count, O = row.questions.reduce(0) { $0 + $1.optionSpans.count }
        guard Q <= Self.qPad, O <= Self.oPad else {
            throw ClefFlashError.prompt("\(Q) questions / \(O) options exceed the head (Q <= \(Self.qPad), O <= \(Self.oPad))")
        }
        guard hidden16.count == T * Self.hidden else {
            throw ClefFlashError.contract("hidden has \(hidden16.count / Self.hidden) rows for \(T) tokens")
        }
        let H = Self.hidden
        var h = [Float](repeating: 0, count: Tp * H)
        h.withUnsafeMutableBufferPointer { d in
            hidden16.withUnsafeBufferPointer { s in
                for i in 0..<(T * H) { d[i] = Float(s[i]) }
            }
        }
        var keyValid = [Float](repeating: 0, count: Tp)
        for i in 0..<T { keyValid[i] = 1 }
        var qAvg = [Float](repeating: 0, count: Self.qPad * Tp)
        var oAvg = [Float](repeating: 0, count: Self.oPad * Tp)
        var gAvg = [Float](repeating: 0, count: Tp)
        gAvg[T - 1] = 1
        var member = [Float](repeating: 0, count: Self.qPad * Self.oPad)
        var typeIDs = [Int32](repeating: 0, count: Self.qPad)
        var qValid = [Float](repeating: 0, count: Self.qPad)
        var oValid = [Float](repeating: 0, count: Self.oPad)
        var lexical = [Float](repeating: 0, count: Self.oPad * H)
        var layout: [(Int, Int)] = []
        var o = 0
        for (qi, q) in row.questions.enumerated() {
            let s = q.questionSpan[0], e = q.questionSpan[1]
            guard 0 <= s, s < e, e <= T else { throw ClefFlashError.prompt("question span \(q.questionSpan) outside [0, \(T))") }
            let qv = Float(1.0) / Float(e - s)
            for t in s..<e { qAvg[qi * Tp + t] = qv }
            typeIDs[qi] = Int32(q.typeID)
            qValid[qi] = 1
            let start = o
            for span in q.optionSpans {
                let a = span[0], b = span[1]
                guard 0 <= a, a < b, b <= T else { throw ClefFlashError.prompt("option span \(span) outside [0, \(T))") }
                let ov = Float(1.0) / Float(b - a)
                for t in a..<b { oAvg[o * Tp + t] = ov }
                member[qi * Self.oPad + o] = 1
                oValid[o] = 1
                lexical.withUnsafeMutableBufferPointer { table.spanMean(ids: row.ids, a, b, into: $0.baseAddress! + o * H) }
                o += 1
            }
            layout.append((start, o))
        }
        return Arrays(bucket: bucket, hidden: h, keyValid: keyValid, qAvg: qAvg, oAvg: oAvg, gAvg: gAvg, lexical: lexical,
                      member: member, typeIDs: typeIDs, qValid: qValid, oValid: oValid, layout: layout, options: O)
    }

    /// One call of the bucket's function -> logits [128].
    func run(_ a: Arrays) async throws -> [Float] {
        guard let fn = functions[a.bucket.function], let d = inputDescriptors[a.bucket.function] else {
            throw ClefFlashError.contract("head: no function \(a.bucket.function)")
        }
        let inputs: [String: NDArray] = [
            "hidden": ClefND.make(a.hidden, d["hidden"]!), "key_valid": ClefND.make(a.keyValid, d["key_valid"]!),
            "q_avg": ClefND.make(a.qAvg, d["q_avg"]!), "o_avg": ClefND.make(a.oAvg, d["o_avg"]!),
            "g_avg": ClefND.make(a.gAvg, d["g_avg"]!), "lexical": ClefND.make(a.lexical, d["lexical"]!),
            "member": ClefND.make(a.member, d["member"]!), "type_ids": ClefND.make(a.typeIDs, d["type_ids"]!),
            "q_valid": ClefND.make(a.qValid, d["q_valid"]!), "o_valid": ClefND.make(a.oValid, d["o_valid"]!),
        ]
        var outputs = try await fn.run(inputs: inputs)
        guard let array = outputs.remove("logits")?.ndArray else { throw ClefFlashError.contract("head: no logits") }
        let logits = ClefND.read(array, as: Float.self)
        guard logits.count == Self.oPad else { throw ClefFlashError.contract("head: \(logits.count) logits") }
        return logits
    }

    /// `clef_head.question_probs`: per question the float32 softmax over its options.
    static func questionProbs(_ logits: [Float], layout: [(Int, Int)]) -> [[Float]] {
        layout.map { a, b in clefSoftmaxFloat32(Array(logits[a..<b])) }
    }
}

#endif
