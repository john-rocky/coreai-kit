import Foundation

/// Minimal safetensors reader — exactly what the host needs, nothing more.
///
/// Format: 8-byte little-endian header length, a JSON header mapping tensor name to
/// `{dtype, shape, data_offsets}` (offsets relative to the end of the header), then the
/// raw buffer. Supported dtypes: F32, BF16 (widened to F32 — the checkpoint stores its
/// constants in bfloat16 and upstream loads them into fp32 modules, so widening here is
/// bit-faithful to what PyTorch computes with), and I64 (the voice-state offsets).
struct PocketTTSSafeTensors {
    struct Entry {
        let dtype: String
        let shape: [Int]
        let byteRange: Range<Int>
    }

    private let data: Data
    private let dataStart: Int
    let entries: [String: Entry]

    init(url: URL) throws {
        let raw = try Data(contentsOf: url, options: .mappedIfSafe)
        guard raw.count >= 8 else { throw PocketTTSError.message("\(url.lastPathComponent): truncated") }
        var headerLen: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &headerLen) { raw.copyBytes(to: $0, from: 0..<8) }
        let hEnd = 8 + Int(headerLen)
        guard hEnd <= raw.count else { throw PocketTTSError.message("\(url.lastPathComponent): bad header length") }
        let header = try JSONSerialization.jsonObject(with: raw.subdata(in: 8..<hEnd))
        guard let dict = header as? [String: Any] else {
            throw PocketTTSError.message("\(url.lastPathComponent): header is not a JSON object")
        }
        var out: [String: Entry] = [:]
        for (name, v) in dict where name != "__metadata__" {
            guard let t = v as? [String: Any],
                  let dtype = t["dtype"] as? String,
                  let shape = t["shape"] as? [Int],
                  let offs = t["data_offsets"] as? [Int], offs.count == 2 else {
                throw PocketTTSError.message("\(url.lastPathComponent): malformed entry '\(name)'")
            }
            out[name] = Entry(dtype: dtype, shape: shape, byteRange: offs[0]..<offs[1])
        }
        self.entries = out
        self.dataStart = hEnd
        self.data = raw
    }

    func shape(_ name: String) throws -> [Int] {
        guard let e = entries[name] else { throw PocketTTSError.message("missing tensor '\(name)'") }
        return e.shape
    }

    /// Tensor as float32, widening BF16 (a bf16 value is exactly the top 16 bits of the
    /// equal-valued float32, so the widening is a bit shift, not a rounding).
    func floats(_ name: String) throws -> [Float] {
        guard let e = entries[name] else { throw PocketTTSError.message("missing tensor '\(name)'") }
        let lo = dataStart + e.byteRange.lowerBound
        let hi = dataStart + e.byteRange.upperBound
        let raw = data.subdata(in: lo..<hi)
        switch e.dtype {
        case "F32":
            return raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        case "BF16":
            return raw.withUnsafeBytes { buf in
                let u = buf.bindMemory(to: UInt16.self)
                var out = [Float](repeating: 0, count: u.count)
                for i in 0..<u.count { out[i] = Float(bitPattern: UInt32(u[i]) << 16) }
                return out
            }
        default:
            throw PocketTTSError.message("tensor '\(name)' has unsupported dtype \(e.dtype)")
        }
    }

    func int64s(_ name: String) throws -> [Int64] {
        guard let e = entries[name] else { throw PocketTTSError.message("missing tensor '\(name)'") }
        guard e.dtype == "I64" else { throw PocketTTSError.message("tensor '\(name)' is \(e.dtype), expected I64") }
        let lo = dataStart + e.byteRange.lowerBound
        let hi = dataStart + e.byteRange.upperBound
        return data.subdata(in: lo..<hi).withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
    }
}
import Foundation

/// PocketTTSModel geometry, shared by every stage. Values are the measured facts from NOTES.md §3
/// and the derived KV budget from §16.1 — S_MAX is a property of the exported assets
/// (`flowlm_*_s512.aimodel`), so it is a constant here, not a knob.
enum PocketTTSModel {
    static let sMax = 512
    static let tPre = 16
    static let dModel = 1024
    static let ldim = 32
    static let nLayers = 6
    static let nHeads = 16
    static let headDim = 64
    static let vocabBins = 4000        // sentencepiece vocab; LUT has 4001 rows (pad)
    static let sampleRate = 24_000
    static let frameSamples = 1920     // one 12.5 Hz latent -> 1920 PCM samples
    static let mimiDim = 512
    static let mimiRing = 272
    static let temp: Float = 0.7
    static let eosThreshold: Float = -4.0
    static let maxTokensPerChunk = 50  // upstream MAX_TOKEN_PER_CHUNK
    /// `_estimate_max_gen_len`: ceil((tokens / 3 + 2) * 12.5)
    static func maxGenLen(tokens: Int) -> Int {
        Int(((Double(tokens) / 3.0 + 2.0) * 12.5).rounded(.up))
    }
}

/// Locations of the model constants inside a HuggingFace snapshot tree
/// (`hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots/<rev>/languages/english/`).
/// Discovery globs the snapshots directory rather than pinning revision hashes, because
/// the checkpoint revision and the voice-embedding revision are two different snapshots.
struct PocketTTSWeightsLayout {
    let modelURL: URL          // model.safetensors (flow-LM + Mimi constants)
    let tokenizerURL: URL      // tokenizer.model (sentencepiece)
    let embeddingsDir: URL     // embeddings/<voice>.safetensors (pre-baked KV states)

    /// Direct-path init for hosts that do not have an HF snapshot tree — the iOS bench
    /// app pushes the three pieces into its data container and points at them here.
    init(modelURL: URL, tokenizerURL: URL, embeddingsDir: URL) {
        self.modelURL = modelURL
        self.tokenizerURL = tokenizerURL
        self.embeddingsDir = embeddingsDir
    }

    static func discover(root: URL) throws -> PocketTTSWeightsLayout {
        let snapshots = root
            .appendingPathComponent("weights/hf/hub/models--kyutai--pocket-tts-without-voice-cloning/snapshots")
        let fm = FileManager.default
        guard let revs = try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil) else {
            throw PocketTTSError.message("no snapshots under \(snapshots.path); run from the repo root or pass --root")
        }
        var model: URL?, tok: URL?, emb: URL?
        for rev in revs {
            let english = rev.appendingPathComponent("languages/english")
            let m = english.appendingPathComponent("model.safetensors")
            let t = english.appendingPathComponent("tokenizer.model")
            let e = english.appendingPathComponent("embeddings")
            if model == nil, fm.fileExists(atPath: m.path) { model = m }
            if tok == nil, fm.fileExists(atPath: t.path) { tok = t }
            if emb == nil, fm.fileExists(atPath: e.path) { emb = e }
        }
        guard let model, let tok, let emb else {
            throw PocketTTSError.message("incomplete weights tree under \(snapshots.path) "
                + "(need languages/english/{model.safetensors, tokenizer.model, embeddings/})")
        }
        return PocketTTSWeightsLayout(modelURL: model, tokenizerURL: tok, embeddingsDir: emb)
    }
}

/// The host-side model constants: everything the pipeline computes *outside* the graphs.
///
/// Since M2 this is only the text-embedding LUT — the `latent*emb_std+emb_mean` rescale
/// and the k=1 quantizer conv were folded into the Mimi graph (the `_q` asset), so the
/// host passes the raw [1,32] flow-decoder latent straight through. The LUT is stored
/// BF16 in the checkpoint and widened to fp32 here, which is exactly what upstream's
/// fp32 module loading computes with (widening is exact).
struct PocketTTSHostWeights {
    /// 4001 × 1024 text-embedding lookup table (graph (a) — a table lookup, not worth a graph).
    let lut: [Float]

    init(modelURL: URL) throws {
        let st = try PocketTTSSafeTensors(url: modelURL)
        self.lut = try st.floats("flow_lm.conditioner.embed.weight")
        guard lut.count == 4001 * 1024 else {
            throw PocketTTSError.message("model.safetensors constants have unexpected shapes")
        }
    }

    /// Token ids → [1, T, 1024] embeddings, flattened row-major.
    func embed(_ tokens: [Int]) -> [Float] {
        var out = [Float](repeating: 0, count: tokens.count * PocketTTSModel.dModel)
        for (t, id) in tokens.enumerated() {
            precondition(id >= 0 && id <= PocketTTSModel.vocabBins, "token id \(id) outside the LUT")
            let src = id * PocketTTSModel.dModel
            let dst = t * PocketTTSModel.dModel
            for j in 0..<PocketTTSModel.dModel { out[dst + j] = lut[src + j] }
        }
        return out
    }

}

/// One shipped voice: the pre-baked flow-LM KV state (prefill already done for the voice
/// conditioning), loaded from `embeddings/<voice>.safetensors` and permuted from upstream's
/// `[2, 1, S_v, H, D]` capture layout into the packed `[L, 1, H, S_MAX, D]` graph layout.
///
/// The conditioning length is **not** 126 for every voice (NOTES.md §16.1 — the shipped
/// eight run 126–162, the wider catalogue 76–162), so `positions` comes from the file's
/// own `offset` tensor, never from a constant.
struct PocketTTSVoiceState {
    let name: String
    let positions: Int
    let kSeed: [Float]      // [6 * 16 * S_MAX * 64], packed [L,1,H,S,D]
    let vSeed: [Float]

    init(name: String, embeddingsDir: URL) throws {
        self.name = name
        let url = embeddingsDir.appendingPathComponent("\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            let have = (try? FileManager.default.contentsOfDirectory(atPath: embeddingsDir.path))?
                .filter { $0.hasSuffix(".safetensors") }
                .map { String($0.dropLast(".safetensors".count)) }
                .sorted() ?? []
            throw PocketTTSError.message("unknown voice '\(name)'; available: \(have.joined(separator: ", "))")
        }
        let st = try PocketTTSSafeTensors(url: url)

        let planes = PocketTTSModel.nHeads * PocketTTSModel.sMax * PocketTTSModel.headDim
        var k = [Float](repeating: 0, count: PocketTTSModel.nLayers * planes)
        var v = [Float](repeating: 0, count: PocketTTSModel.nLayers * planes)
        var offsets = Set<Int>()
        for layer in 0..<PocketTTSModel.nLayers {
            let cache = try st.floats("transformer.layers.\(layer).self_attn/cache")
            let shape = try st.shape("transformer.layers.\(layer).self_attn/cache")
            let off = try st.int64s("transformer.layers.\(layer).self_attn/offset")[0]
            offsets.insert(Int(off))
            guard shape.count == 5, shape[0] == 2, shape[1] == 1,
                  shape[3] == PocketTTSModel.nHeads, shape[4] == PocketTTSModel.headDim else {
                throw PocketTTSError.message("voice '\(name)' layer \(layer): unexpected cache shape \(shape)")
            }
            let sVoice = shape[2]
            guard sVoice <= PocketTTSModel.sMax, Int(off) <= sVoice else {
                throw PocketTTSError.message("voice '\(name)': \(sVoice) positions exceed S_MAX=\(PocketTTSModel.sMax)")
            }
            // cache[kv, 0, s, h, d] -> packed[layer, 0, h, s, d]
            let vBase = sVoice * PocketTTSModel.nHeads * PocketTTSModel.headDim   // start of cache[1]
            for s in 0..<sVoice {
                for h in 0..<PocketTTSModel.nHeads {
                    let src = (s * PocketTTSModel.nHeads + h) * PocketTTSModel.headDim
                    let dst = layer * planes + (h * PocketTTSModel.sMax + s) * PocketTTSModel.headDim
                    for d in 0..<PocketTTSModel.headDim {
                        k[dst + d] = cache[src + d]
                        v[dst + d] = cache[vBase + src + d]
                    }
                }
            }
        }
        guard offsets.count == 1 else {
            throw PocketTTSError.message("voice '\(name)': per-layer offsets disagree: \(offsets.sorted())")
        }
        self.positions = offsets.first!
        self.kSeed = k
        self.vSeed = v
    }
}

/// Per-voice output-gain normalisation, from the validation sweep's median-RMS table
/// (NOTES.md §16.2). The shipped voices differ by up to 12.3 dB at identical settings —
/// inherited from their conditioning clips, not a bug — so without this the voice picker
/// doubles as a volume control. Voices outside the measured eight play at unity gain.
///
/// The target (0.10) sits at the loud end of the measured spread (alba 0.124 … cosette
/// 0.030), so quiet voices come up a lot and loud voices barely move. Gate (a) compares
/// ungained audio (`--no-gain`): gain is a playback normalisation, not part of parity.
enum PocketTTSVoiceGain {
    static let targetRMS = 0.10
    static let sweepMedianRMS: [String: Double] = [
        "alba": 0.124, "azelma": 0.101, "cosette": 0.030, "eponine": 0.094,
        "fantine": 0.095, "javert": 0.045, "jean": 0.114, "marius": 0.042,
    ]

    /// Gain factor for a voice, before the peak-safety clamp.
    static func gain(for voice: String) -> Double {
        guard let r = sweepMedianRMS[voice] else { return 1.0 }
        return targetRMS / r
    }

    /// Apply the voice gain in place, then clamp so the post-gain peak stays <= 0.99
    /// (a +10 dB boost on a quiet voice must not clip a loud transient). Returns the
    /// effective gain actually applied.
    static func apply(_ samples: inout [Float], voice: String) -> Double {
        var g = gain(for: voice)
        if g == 1.0 { return 1.0 }
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        if Double(peak) * g > 0.99 { g = 0.99 / Double(peak) }
        let gf = Float(g)
        for i in 0..<samples.count { samples[i] *= gf }
        return g
    }
}
