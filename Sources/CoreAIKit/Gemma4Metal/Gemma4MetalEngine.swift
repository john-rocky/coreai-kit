// Gemma4MetalEngine — user-facing chat engine around the raw-Metal Gemma-4-E2B
// mixed-bit decode loop.
//
// The kernel set and the per-token dispatch sequence are copied VERBATIM from the
// gated bench runner (_metal_loop/g4bench/Sources/G4Runner.swift). Do not reorder
// dispatches, change buffer/index bindings, or relax mathMode: the loop is proven
// LOSSLESS (device S1 token gate + MTP gate) only for this exact sequence with
// mathMode .safe. Any change here requires re-running the S1 token gate
// (`tokenGate(refsURL:)`) on device.
//
// Weights: Google's Gemma-4-E2B QAT mixed-bit weights (int2/int4/int8 + PLE tables),
// repacked from the litert-community release into gemma4_pack.(bin|json).
// Use is subject to the Gemma Terms of Use.

import Foundation
import Metal

/// NOT internally synchronized — drive from ONE thread/task at a time (the
/// backends run generate on a single detached task). @unchecked Sendable only
/// so the instance can cross actor boundaries under Swift 6.
public final class Gemma4MetalEngine: @unchecked Sendable {

    // ---------- errors ----------
    public enum EngineError: Error, CustomStringConvertible {
        case noMetalDevice
        case packMissing(String)
        case badPack(String)
        case kernelCompile(String)
        public var description: String {
            switch self {
            case .noMetalDevice: return "no Metal device"
            case .packMissing(let p): return "pack missing at \(p)"
            case .badPack(let m): return "bad pack: \(m)"
            case .kernelCompile(let m): return "kernel compile failed: \(m)"
            }
        }
    }

    // ---------- manifest (identical to G4Runner) ----------
    struct TensorInfo: Decodable { let offset: Int; let nbytes: Int; let dtype: String; let shape: [Int] }
    struct LayerMeta: Decodable {
        let full: Bool, write: Bool, int2: Bool
        let cache: Int, hd: Int, ffn_n: Int, ffn_k: Int
        let layer_scalar: Double
    }
    struct DrafterLayerMeta: Decodable {
        let hd: Int, cache: Int
        let skip: Double, aq_q: Double, aq_attn_vec: Double, aq_gating: Double, aq_down: Double
    }
    struct DrafterMeta: Decodable {
        let layers: [DrafterLayerMeta]
        let aq_pre_proj: Double, aq_post_proj: Double
    }
    struct Meta: Decodable {
        let vocab: Int, hidden: Int, window: Int, max_ctx: Int, n_heads: Int
        let pli_scale: Double
        let layers: [LayerMeta]
        let drafter: DrafterMeta?
        let interleave4: Bool?
    }
    struct Manifest: Decodable { let tensors: [String: TensorInfo]; let meta: Meta }

    typealias Buf = (b: MTLBuffer, o: Int)

    // ---------- public config ----------
    public struct Stats {
        public var prefillTokens = 0, decodeTokens = 0
        public var prefillSeconds = 0.0, decodeSeconds = 0.0
        public var prefillTPS: Double { prefillSeconds > 0 ? Double(prefillTokens) / prefillSeconds : 0 }
        public var decodeTPS: Double { decodeSeconds > 0 ? Double(decodeTokens) / decodeSeconds : 0 }
    }

    /// Gemma-4 chat stop ids: <eos>=1, <turn|>=106.
    public static let defaultStopIds: Set<Int> = [1, 106]

    public private(set) var maxContext: Int = 0
    public private(set) var vocabSize: Int = 0
    public private(set) var lastStats = Stats()

    /// Cross-thread stop request (host cancelled the stream). Checked between CBs.
    nonisolated(unsafe) private var stopRequested = false
    public func requestStop() { stopRequested = true }

    // ---------- device state ----------
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private var blobBuf: MTLBuffer!
    private var manifest: Manifest!
    private var meta: Meta { manifest.meta }
    private var pso: [String: MTLComputePipelineState] = [:]
    private var il4 = false
    // tuning knobs (ship defaults; il4 pack forces the fused R=4 lanes)
    private var r1AttnO = false, r1Down = false
    private var occGS = 8, occGF = 8
    private var fuse = true, lut2 = true, fsr = true
    private let tokensPerCB = 8
    private let pipelineDepth = 3

    private var HID = 0, V = 0, NHEADS = 0, WINDOW = 0, MAXCTX = 0, NPARTS = 0

    // scratch buffers (S=1 lane only — the MTP/verify lane stays in the bench app)
    private var xBuf: MTLBuffer!, xnBuf: MTLBuffer!, xABuf: MTLBuffer!, xFBuf: MTLBuffer!
    private var kv2Buf: [Int: MTLBuffer] = [:]
    private var pliBuf: MTLBuffer!, pleTokBuf: MTLBuffer!, p8Buf: MTLBuffer!
    private var qBuf: [Int: MTLBuffer] = [:], qnBuf: [Int: MTLBuffer] = [:]
    private var kvBuf: [Int: MTLBuffer] = [:], kvnBuf: [Int: MTLBuffer] = [:]
    private var ctxBuf: [Int: MTLBuffer] = [:]
    private var attnBuf: MTLBuffer!, hffnBuf: MTLBuffer!, ffnBuf: MTLBuffer!
    private var g256Buf: MTLBuffer!, p1536Buf: MTLBuffer!, hiddenBuf: MTLBuffer!
    private var logitsBuf: MTLBuffer!, partvBuf: MTLBuffer!, partiBuf: MTLBuffer!
    private var toksBuf: MTLBuffer!
    private var kCache: [MTLBuffer] = [], vCache: [MTLBuffer] = []
    // batched-prefill lane (wide buffers sized for the max chunk width; the active
    // width is `prefillM` — 16/8/4 chunks + smaller remainders, all bit-exact)
    private let SVMAX = 16
    /// Max prefill chunk width (1 = S=1 only, 4/8/16 = wide kernels). Ship default
    /// m8 interchanged on BOTH platforms (M4 Max: +24% over m4; A19: parity with m4
    /// within-day, halves the byte floor for headroom — session-5 A/B). Override with
    /// env G4_PREFILL_M (bench/gate A/B) or directly.
    public var prefillM = 8
    /// Wide-kernel body variant: false = interchanged (register-frugal), true =
    /// staged (m4-literal burst loads; _m8s/_m16s). Same bit-exact math, different
    /// load scheduling — per-device winner. Env G4_PREFILL_V=s selects staged.
    public var prefillStaged = false
    /// byte-LUT int2 decode in the wide prefill lane (_m4l/_m8l, A19 ALU relief —
    /// the S=1 lane's ⭐ lever ported to the chunked kernels). Bit-exact (code-order
    /// preserved). Env G4_PREFILL_L=1/0 overrides. Applies to m ≤ 8, interchanged.
    public var prefillLut = false
    /// Fused wide prefill lane (session-6 prototype pair): m=8 interchanged chunks
    /// dispatch gateup_*_m8_nxa (post_attn fold + pre_ffw norm in the gateup
    /// prologue) and matvec_int4aff_m8_nx_qkv (pre_attn norm + q/k/v, write layers).
    /// Bit-exact (S=1 reduction lane order kept; gated like every lane). Default OFF
    /// until the interleaved device A/B says otherwise. Env G4_PREFILL_F=1 enables.
    /// LUT knob does not apply to the fused gateup (prefill-neutral, session-5).
    public var prefillFused = false
    private var x4Buf: MTLBuffer!, xn4Buf: MTLBuffer!, xa4Buf: MTLBuffer!
    private var kv24Buf: [Int: MTLBuffer] = [:]
    private var pli4Buf: MTLBuffer!, ple4Buf: MTLBuffer!, p84Buf: MTLBuffer!
    private var q4Buf: [Int: MTLBuffer] = [:], qn4Buf: [Int: MTLBuffer] = [:]
    private var kv4Buf: [Int: MTLBuffer] = [:], ctx4Buf: [Int: MTLBuffer] = [:]
    private var attn4Buf: MTLBuffer!, hffn4Buf: MTLBuffer!, ffn4Buf: MTLBuffer!
    private var g4Buf: MTLBuffer!, p4Buf: MTLBuffer!

    private struct LayerW {
        let idx: Int
        let m: LayerMeta
        let wqQP: Buf, wqSC: Buf, wqBI: Buf
        let woQP: Buf, woSC: Buf, woBI: Buf
        var wkQP: Buf?, wkSC: Buf?, wkBI: Buf?
        var wvQP: Buf?, wvSC: Buf?, wvBI: Buf?
        var keyNorm: Buf?
        let gQP: Buf, gSC: Buf, gBI: Buf?
        let uQP: Buf, uSC: Buf, uBI: Buf?
        let dQP: Buf, dSC: Buf, dBI: Buf?
        let pgW8: Buf, pgSC: Buf, ppW8: Buf, ppSC: Buf
        let preAttn: Buf, postAttn: Buf, preFfw: Buf, postFfw: Buf, postPle: Buf, queryNorm: Buf
        let ls: Float
    }
    private var layers: [LayerW] = []
    private var embPacked: Buf!, embScale: Buf!, plePacked: Buf!, pleScale: Buf!
    private var mpW8: Buf!, mpSC: Buf!, projNorm: Buf!, finalNorm: Buf!
    private var headQP: Buf!, headSC: Buf!
    private var invfSliding: Buf!, invfFull: Buf!
    private var PLI_SCALE: Float = 0

    // ---------- int8 static activation scales (act_scales.json sidecar) ----------
    // The mobile-QAT learned clamp: per-tensor scalar scales at every linear boundary
    // + int8 KV-cache storage. Scale 0 = unquantized; with the sidecar missing or
    // G4_ACTQ=0 every scale is 0 and all kernels run bit-identical to the fp16 path.
    // Same structures/bindings as the gated runner (g4loop main.swift).
    struct AqLin { var i: Float = 0; var o: Float = 0 }
    struct LayerAq {
        var q = AqLin(), k = AqLin(), v = AqLin(), o = AqLin()
        var gate = AqLin(), up = AqLin(), down = AqLin()
        var pleGate = AqLin(), pleProj = AqLin()
        var kCache: Float = 0, vCache: Float = 0
    }
    struct ActLayerJ: Decodable {
        let q: [Double], o: [Double], gate: [Double], up: [Double], down: [Double]
        let ple_gate: [Double], ple_proj: [Double]
        let k: [Double]?, v: [Double]?
        let k_cache: Double, v_cache: Double
    }
    struct ActScalesJ: Decodable {
        struct MP: Decodable { let s_in: Double; let s_out: Double }
        let mp: MP
        let layers: [ActLayerJ]
    }
    private var layerAq: [LayerAq] = []
    private var mpAq = AqLin()
    /// true when the pack carries act_scales.json and G4_ACTQ != 0 — the engine then
    /// runs the int8 static activation model (GSM8K 48 -> 83-class on this checkpoint).
    public private(set) var actqOn = false

    private func loadActScales(packDir: URL) {
        layerAq = [LayerAq](repeating: LayerAq(), count: meta.layers.count)
        guard ProcessInfo.processInfo.environment["G4_ACTQ"] != "0",
              let aqData = try? Data(contentsOf: packDir.appendingPathComponent("act_scales.json")),
              let aj = try? JSONDecoder().decode(ActScalesJ.self, from: aqData) else { return }
        func lin(_ a: [Double]?) -> AqLin {
            guard let a = a else { return AqLin() }
            return AqLin(i: Float(a[0]), o: Float(a[1]))
        }
        for (li, al) in aj.layers.enumerated() where li < layerAq.count {
            layerAq[li] = LayerAq(
                q: lin(al.q), k: lin(al.k), v: lin(al.v), o: lin(al.o),
                gate: lin(al.gate), up: lin(al.up), down: lin(al.down),
                pleGate: lin(al.ple_gate), pleProj: lin(al.ple_proj),
                kCache: Float(al.k_cache), vCache: Float(al.v_cache))
        }
        // model_proj weight scales carry the HID^-0.5 fold; fq(y*c, s*c) == c*fq(y, s)
        mpAq = AqLin(i: Float(aj.mp.s_in),
                     o: Float(aj.mp.s_out * pow(Double(meta.hidden), -0.5)))
        actqOn = true
    }

    /// ids whose KV is already committed (multi-turn incremental prefill)
    private var committed: [Int] = []

    // ---------- init ----------
    /// - Parameters:
    ///   - packDir: directory holding gemma4_pack.bin + gemma4_pack.json
    ///   - mslDir: directory holding the gemma4_*.metal.txt kernel sources
    ///     (sdpa/matvec/glue/verify/prefill)
    public init(packDir: URL, mslDir: URL) throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw EngineError.noMetalDevice }
        device = dev
        queue = device.makeCommandQueue()!

        let manURL = packDir.appendingPathComponent("gemma4_pack.json")
        guard let manData = try? Data(contentsOf: manURL) else {
            throw EngineError.packMissing(manURL.path)
        }
        manifest = try JSONDecoder().decode(Manifest.self, from: manData)

        let blobPath = packDir.appendingPathComponent("gemma4_pack.bin").path
        let fd = open(blobPath, O_RDONLY)
        guard fd >= 0 else { throw EngineError.packMissing(blobPath) }
        let blobSize = Int(lseek(fd, 0, SEEK_END))
        let page = Int(getpagesize())
        let mapLen = (blobSize + page - 1) / page * page
        let mapPtr = mmap(nil, mapLen, PROT_READ, MAP_PRIVATE, fd, 0)
        guard mapPtr != MAP_FAILED else { throw EngineError.badPack("mmap failed") }
        if let b = device.makeBuffer(bytesNoCopy: mapPtr!, length: mapLen,
                                     options: .storageModeShared, deallocator: nil) {
            blobBuf = b
        } else {
            // bytesNoCopy can refuse read-only pages — fall back to an owned copy
            blobBuf = device.makeBuffer(bytes: mapPtr!, length: blobSize,
                                        options: .storageModeShared)!
        }

        try compileKernels(mslDir: mslDir)
        try buildTables()
        loadActScales(packDir: packDir)
    }

    private func compileKernels(mslDir: URL) throws {
        // .safe keeps the source accumulation order in every kernel shape — the
        // losslessness proof depends on it (fast math contracts/reassociates).
        let options = MTLCompileOptions()
        options.mathMode = .safe
        il4 = meta.interleave4 ?? false
        // il4 pack has no R=1 / non-fused kernels — force the A19 ship config there
        r1AttnO = false; r1Down = false
        fuse = true; lut2 = true; fsr = true
        #if os(iOS)
        occGS = 8; occGF = 8            // A19-tuned SDPA split
        #else
        occGS = 16; occGF = 8           // M4-tuned
        #endif
        if let s = ProcessInfo.processInfo.environment["G4_PREFILL_M"], let v = Int(s),
           [1, 4, 8, 16].contains(v) { prefillM = v }
        if let s = ProcessInfo.processInfo.environment["G4_PREFILL_V"] { prefillStaged = s == "s" }
        if let s = ProcessInfo.processInfo.environment["G4_PREFILL_L"] { prefillLut = s == "1" }
        if let s = ProcessInfo.processInfo.environment["G4_PREFILL_F"] { prefillFused = s == "1" }
        let ilNames: Set<String> = [
            "matvec_int2sym", "matvec_int4aff", "matvec_int2sym_nx", "matvec_int4aff_nx",
            "matvec_int4aff_nxaa", "matvec_int4aff_nx_qkv", "matvec_int4aff_nxaa_qkv",
            "gateup_int2sym_nxa", "gateup_int4aff_nxa",
            "gateup_int2sym_nxa_lut", "matvec_int2sym_lut", "matvec_int2sym_nx_lut",
            "matvec_int2sym_m4", "matvec_int4aff_m4", "gateup_int2sym_m4", "gateup_int4aff_m4",
            "matvec_int2sym_m8", "matvec_int4aff_m8", "gateup_int2sym_m8", "gateup_int4aff_m8",
            "matvec_int2sym_m16", "matvec_int4aff_m16", "gateup_int2sym_m16", "gateup_int4aff_m16",
            "matvec_int2sym_m8s", "matvec_int4aff_m8s", "gateup_int2sym_m8s", "gateup_int4aff_m8s",
            "matvec_int2sym_m16s", "matvec_int4aff_m16s", "gateup_int2sym_m16s", "gateup_int4aff_m16s",
            "matvec_int2sym_m4l", "matvec_int2sym_m8l", "gateup_int2sym_m4l", "gateup_int2sym_m8l",
            "gateup_int2sym_m8_nxa", "gateup_int4aff_m8_nxa", "matvec_int4aff_m8_nx_qkv"]
        // gemma4_verify supplies the M=4 kernels the batched prefill lane reuses
        // (the MTP verify lane — S=1-bitwise-identical by the .safe accumulation-order
        // contract, proven by the MTP lossless gate); gemma4_prefill supplies the
        // M=8/16 widenings (same per-output accumulation order — see its header).
        for file in ["gemma4_sdpa.metal.txt", "gemma4_matvec.metal.txt", "gemma4_glue.metal.txt",
                     "gemma4_verify.metal.txt", "gemma4_prefill.metal.txt",
                     "gemma4_prefill_fused.metal.txt"] {
            let url = mslDir.appendingPathComponent(file)
            guard let src = try? String(contentsOf: url, encoding: .utf8) else {
                throw EngineError.kernelCompile("missing \(file)")
            }
            let lib: MTLLibrary
            do { lib = try device.makeLibrary(source: src, options: options) }
            catch { throw EngineError.kernelCompile("\(file): \(error.localizedDescription)") }
            for name in lib.functionNames {
                pso[name] = try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
            }
            if il4 && (file.hasPrefix("gemma4_matvec") || file.hasPrefix("gemma4_verify")
                       || file.hasPrefix("gemma4_prefill")) {
                let ilOptions = MTLCompileOptions()
                ilOptions.mathMode = options.mathMode
                ilOptions.preprocessorMacros = ["IL4": 1 as NSNumber]
                let ilLib: MTLLibrary
                do { ilLib = try device.makeLibrary(source: src, options: ilOptions) }
                catch { throw EngineError.kernelCompile("\(file)@il: \(error.localizedDescription)") }
                for name in ilLib.functionNames where ilNames.contains(name) {
                    pso[name + "@il"] = try! device.makeComputePipelineState(
                        function: ilLib.makeFunction(name: name)!)
                }
            }
        }
    }

    private func T(_ name: String) throws -> Buf {
        guard let t = manifest.tensors[name] else { throw EngineError.badPack("missing tensor \(name)") }
        return (blobBuf, t.offset)
    }

    private func buildTables() throws {
        HID = meta.hidden; V = meta.vocab; NHEADS = meta.n_heads; WINDOW = meta.window
        MAXCTX = meta.max_ctx; NPARTS = V / 512
        maxContext = MAXCTX; vocabSize = V

        func mkBuf(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: bytes, options: .storageModeShared)! }
        xBuf = mkBuf(HID * 2); xnBuf = mkBuf(HID * 2)
        xABuf = mkBuf(HID * 2); xFBuf = mkBuf(HID * 2)
        kv2Buf = [256: mkBuf(256 * 2), 512: mkBuf(512 * 2)]
        pliBuf = mkBuf(35 * 256 * 2); pleTokBuf = mkBuf(35 * 256 * 2); p8Buf = mkBuf(35 * 256 * 2)
        qBuf = [256: mkBuf(8 * 256 * 2), 512: mkBuf(8 * 512 * 2)]
        qnBuf = [256: mkBuf(8 * 256 * 2), 512: mkBuf(8 * 512 * 2)]
        kvBuf = [256: mkBuf(256 * 2), 512: mkBuf(512 * 2)]
        kvnBuf = [256: mkBuf(256 * 2), 512: mkBuf(512 * 2)]
        ctxBuf = [256: mkBuf(8 * 256 * 2), 512: mkBuf(8 * 512 * 2)]
        attnBuf = mkBuf(HID * 2); hffnBuf = mkBuf(12288 * 2); ffnBuf = mkBuf(HID * 2)
        g256Buf = mkBuf(256 * 2); p1536Buf = mkBuf(HID * 2); hiddenBuf = mkBuf(HID * 2)
        logitsBuf = mkBuf(V * 2)
        partvBuf = mkBuf(NPARTS * 4); partiBuf = mkBuf(NPARTS * 4)
        toksBuf = mkBuf((MAXCTX + 8) * 4)
        kCache = []; vCache = []
        for li in 0..<15 {
            let hd = meta.layers[li].hd
            kCache.append(mkBuf(MAXCTX * hd * 2))
            vCache.append(mkBuf(MAXCTX * hd * 2))
        }
        x4Buf = mkBuf(SVMAX * HID * 2); xn4Buf = mkBuf(SVMAX * HID * 2)
        xa4Buf = mkBuf(SVMAX * HID * 2)
        kv24Buf = [256: mkBuf(SVMAX * 256 * 2), 512: mkBuf(SVMAX * 512 * 2)]
        pli4Buf = mkBuf(SVMAX * 35 * 256 * 2); ple4Buf = mkBuf(SVMAX * 35 * 256 * 2)
        p84Buf = mkBuf(SVMAX * 35 * 256 * 2)
        q4Buf = [256: mkBuf(SVMAX * 8 * 256 * 2), 512: mkBuf(SVMAX * 8 * 512 * 2)]
        qn4Buf = [256: mkBuf(SVMAX * 8 * 256 * 2), 512: mkBuf(SVMAX * 8 * 512 * 2)]
        kv4Buf = [256: mkBuf(SVMAX * 256 * 2), 512: mkBuf(SVMAX * 512 * 2)]
        ctx4Buf = [256: mkBuf(SVMAX * 8 * 256 * 2), 512: mkBuf(SVMAX * 8 * 512 * 2)]
        attn4Buf = mkBuf(SVMAX * HID * 2); hffn4Buf = mkBuf(SVMAX * 12288 * 2); ffn4Buf = mkBuf(SVMAX * HID * 2)
        g4Buf = mkBuf(SVMAX * 256 * 2); p4Buf = mkBuf(SVMAX * HID * 2)

        layers = []
        for (li, m) in meta.layers.enumerated() {
            let p = String(format: "L%02d.", li)
            var lw = LayerW(
                idx: li,
                m: m,
                wqQP: try T(p + "wq.qp"), wqSC: try T(p + "wq.sc"), wqBI: try T(p + "wq.bi"),
                woQP: try T(p + "wo.qp"), woSC: try T(p + "wo.sc"), woBI: try T(p + "wo.bi"),
                gQP: try T(p + "gate.qp"), gSC: try T(p + "gate.sc"),
                gBI: m.int2 ? nil : (try T(p + "gate.bi")),
                uQP: try T(p + "up.qp"), uSC: try T(p + "up.sc"),
                uBI: m.int2 ? nil : (try T(p + "up.bi")),
                dQP: try T(p + "down.qp"), dSC: try T(p + "down.sc"),
                dBI: m.int2 ? nil : (try T(p + "down.bi")),
                pgW8: try T(p + "ple_gate.w8"), pgSC: try T(p + "ple_gate.sc"),
                ppW8: try T(p + "ple_proj.w8"), ppSC: try T(p + "ple_proj.sc"),
                preAttn: try T(p + "pre_attn"), postAttn: try T(p + "post_attn"),
                preFfw: try T(p + "pre_ffw"), postFfw: try T(p + "post_ffw"),
                postPle: try T(p + "post_ple"), queryNorm: try T(p + "query_norm"),
                ls: Float(m.layer_scalar))
            if m.write {
                lw.wkQP = try T(p + "wk.qp"); lw.wkSC = try T(p + "wk.sc"); lw.wkBI = try T(p + "wk.bi")
                lw.wvQP = try T(p + "wv.qp"); lw.wvSC = try T(p + "wv.sc"); lw.wvBI = try T(p + "wv.bi")
                lw.keyNorm = try T(p + "key_norm")
            }
            layers.append(lw)
        }
        embPacked = try T("embed.packed"); embScale = try T("embed.scale")
        plePacked = try T("ple.packed"); pleScale = try T("ple.scale")
        mpW8 = try T("model_proj.w8"); mpSC = try T("model_proj.sc")
        projNorm = try T("proj_norm"); finalNorm = try T("final_norm")
        headQP = try T("lm_head.qp"); headSC = try T("lm_head.sc")
        invfSliding = try T("invf.sliding"); invfFull = try T("invf.full")
        PLI_SCALE = Float(meta.pli_scale)
    }

    // ---------- encode (verbatim dispatch sequence from G4Runner) ----------
    @inline(__always) private func P(_ n: String) -> MTLComputePipelineState { pso[n]! }
    @inline(__always) private func PI(_ n: String) -> MTLComputePipelineState { pso[il4 ? n + "@il" : n]! }
    @inline(__always) private func sz(_ w: Int, _ h: Int = 1, _ d: Int = 1) -> MTLSize {
        MTLSize(width: w, height: h, depth: d)
    }

    private func encodeStep(_ e: MTLComputeCommandEncoder, pos: Int, wantHead: Bool) {
        let posU = UInt32(pos)
        e.setComputePipelineState(P("embed_gather"))
        e.bufs([embPacked, embScale, (xBuf, 0), (toksBuf, 0)]); e.u32(posU, 4)
        e.dispatchThreads(sz(384), threadsPerThreadgroup: sz(128))
        e.setComputePipelineState(P("matvec_int8"))
        e.bufs([(xBuf, 0), mpW8, mpSC, (p8Buf, 0), (p8Buf, 0)])
        e.u32(UInt32(HID), 5); e.u32(0, 6); e.u32(0, 7)
        e.f32(mpAq.i, 8); e.f32(mpAq.o, 9)
        e.dispatchThreads(sz(32, 8960 / 4), threadsPerThreadgroup: sz(32, 8))
        e.setComputePipelineState(P("ple_gather"))
        e.bufs([plePacked, pleScale, (pleTokBuf, 0), (toksBuf, 0)]); e.u32(posU, 4)
        e.dispatchThreads(sz(4480), threadsPerThreadgroup: sz(256))
        e.setComputePipelineState(P("rmsnorm_glue"))
        e.bufs([(p8Buf, 0), projNorm, (pleTokBuf, 0), (pliBuf, 0)])
        e.u32(256, 4); e.u32(1, 5); e.f32(PLI_SCALE, 6); e.u32(0, 7); e.f32(0, 8)
        e.dispatchThreads(sz(32, 35), threadsPerThreadgroup: sz(32, 8))

        for (li, lw) in layers.enumerated() {
            let aq = layerAq[li]
            let hd = lw.m.hd
            let invf = lw.m.full ? invfFull! : invfSliding!
            let kc = kCache[lw.m.cache], vc = vCache[lw.m.cache]
            let q = qBuf[hd]!, qn = qnBuf[hd]!, kv = kvBuf[hd]!, ctx = ctxBuf[hd]!
            let prev = li > 0 ? layers[li - 1] : nil
            if fuse, lw.m.write {
                let kv2 = kv2Buf[hd]!
                if li == 0 {
                    e.setComputePipelineState(PI("matvec_int4aff_nx_qkv"))
                    e.bufs([(xBuf, 0), lw.preAttn,
                            lw.wqQP, lw.wqSC, lw.wqBI,
                            lw.wkQP!, lw.wkSC!, lw.wkBI!,
                            lw.wvQP!, lw.wvSC!, lw.wvBI!,
                            (q, 0), (kv, 0), (kv2, 0)])
                    e.u32(UInt32(HID), 14); e.u32(UInt32(hd), 15)
                    e.f32(aq.q.i, 16); e.f32(aq.q.o, 17); e.f32(aq.k.o, 18); e.f32(aq.v.o, 19)
                } else {
                    e.setComputePipelineState(PI("matvec_int4aff_nxaa_qkv"))
                    e.bufs([(p1536Buf, 0), prev!.postPle, (xFBuf, 0), (xBuf, 0), lw.preAttn,
                            lw.wqQP, lw.wqSC, lw.wqBI,
                            lw.wkQP!, lw.wkSC!, lw.wkBI!,
                            lw.wvQP!, lw.wvSC!, lw.wvBI!,
                            (q, 0), (kv, 0), (kv2, 0)])
                    e.u32(UInt32(HID), 17); e.u32(UInt32(hd), 18); e.f32(prev!.ls, 19)
                    e.f32(aq.q.i, 20); e.f32(aq.q.o, 21); e.f32(aq.k.o, 22); e.f32(aq.v.o, 23)
                }
                e.dispatchThreads(sz(32, 10 * hd / 4), threadsPerThreadgroup: sz(32, 8))
                if !fsr {
                    e.setComputePipelineState(P("qkv_norm_rope_v"))
                    e.bufs([(q, 0), (kv, 0), (kv2, 0), lw.queryNorm, lw.keyNorm!, invf,
                            (qn, 0), (kc, 0), (vc, 0)])
                    e.u32(UInt32(hd), 9); e.u32(posU, 10)
                    e.f32(aq.kCache, 11); e.f32(aq.vCache, 12)
                    e.dispatchThreads(sz(32, 10), threadsPerThreadgroup: sz(32, 10))
                }
            } else if fuse {
                e.setComputePipelineState(PI("matvec_int4aff_nxaa"))
                e.bufs([(p1536Buf, 0), prev!.postPle, (xFBuf, 0), (xBuf, 0), lw.preAttn,
                        lw.wqQP, lw.wqSC, lw.wqBI, (q, 0)])
                e.u32(UInt32(HID), 9); e.f32(prev!.ls, 10)
                e.f32(aq.q.i, 11); e.f32(aq.q.o, 12)
                e.dispatchThreads(sz(32, 8 * hd / 4), threadsPerThreadgroup: sz(32, 8))
                if !fsr {
                    e.setComputePipelineState(P("qknorm_rope"))
                    e.bufs([(q, 0), lw.queryNorm, invf, (qn, 0)])
                    e.u32(UInt32(hd), 4); e.u32(posU, 5); e.u32(0, 6); e.f32(0, 7)
                    e.dispatchThreads(sz(32, 8), threadsPerThreadgroup: sz(32, 8))
                }
            } else {
                e.setComputePipelineState(P("matvec_int4aff_nx"))
                e.bufs([(xBuf, 0), lw.preAttn, lw.wqQP, lw.wqSC, lw.wqBI, (q, 0)])
                e.u32(UInt32(HID), 6); e.f32(aq.q.i, 7); e.f32(aq.q.o, 8)
                e.dispatchThreads(sz(32, 8 * hd / 4), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(P("qknorm_rope"))
                e.bufs([(q, 0), lw.queryNorm, invf, (qn, 0)])
                e.u32(UInt32(hd), 4); e.u32(posU, 5); e.u32(0, 6); e.f32(0, 7)
                e.dispatchThreads(sz(32, 8), threadsPerThreadgroup: sz(32, 8))
                if lw.m.write {
                    e.setComputePipelineState(P("matvec_int4aff_nx"))
                    e.bufs([(xBuf, 0), lw.preAttn, lw.wkQP!, lw.wkSC!, lw.wkBI!, (kv, 0)])
                    e.u32(UInt32(HID), 6); e.f32(aq.k.i, 7); e.f32(aq.k.o, 8)
                    e.dispatchThreads(sz(32, hd / 4), threadsPerThreadgroup: sz(32, 8))
                    e.setComputePipelineState(P("qknorm_rope"))
                    e.bufs([(kv, 0), lw.keyNorm!, invf, (kc, 0)])
                    e.u32(UInt32(hd), 4); e.u32(posU, 5); e.u32(posU, 6); e.f32(aq.kCache, 7)
                    e.dispatchThreads(sz(32, 1), threadsPerThreadgroup: sz(32, 1))
                    e.setComputePipelineState(P("matvec_int4aff_nx"))
                    e.bufs([(xBuf, 0), lw.preAttn, lw.wvQP!, lw.wvSC!, lw.wvBI!, (kv, 0)])
                    e.u32(UInt32(HID), 6); e.f32(aq.v.i, 7); e.f32(aq.v.o, 8)
                    e.dispatchThreads(sz(32, hd / 4), threadsPerThreadgroup: sz(32, 8))
                    e.setComputePipelineState(P("rmsnorm_glue"))
                    e.bufs([(kv, 0), lw.keyNorm!, (kv, 0), (vc, 0)])
                    e.u32(UInt32(hd), 4); e.u32(2, 5); e.f32(1.0, 6); e.u32(posU, 7)
                    e.f32(aq.vCache, 8)
                    e.dispatchThreads(sz(32, 1), threadsPerThreadgroup: sz(32, 1))
                }
            }
            let j0 = lw.m.full ? 0 : max(0, pos + 1 - WINDOW)
            let occG = lw.m.full ? occGF : occGS
            if fuse && fsr {
                let kv2 = kv2Buf[hd]!
                e.setComputePipelineState(P("flash_sdpa_rope_occ"))
                if lw.m.write {
                    e.bufs([(q, 0), (kv, 0), (kv2, 0), lw.queryNorm, lw.keyNorm!, invf,
                            (kc, 0), (vc, 0), (ctx, 0)])
                } else {
                    e.bufs([(q, 0), (q, 0), (q, 0), lw.queryNorm, lw.queryNorm, invf,
                            (kc, 0), (vc, 0), (ctx, 0)])
                }
                e.u32(UInt32(hd), 9); e.u32(UInt32(j0), 10); e.u32(UInt32(pos + 1 - j0), 11)
                e.u32(UInt32(occG), 12); e.u32(posU, 13); e.u32(lw.m.write ? 1 : 0, 14)
                e.f32(aq.kCache, 15); e.f32(aq.vCache, 16)
                e.dispatchThreads(sz(32, occG, NHEADS), threadsPerThreadgroup: sz(32, occG, 1))
            } else {
                e.setComputePipelineState(P("flash_sdpa_decode_occ"))
                e.bufs([(qn, 0), (kc, 0), (vc, 0), (ctx, 0)])
                e.u32(UInt32(hd), 4); e.u32(UInt32(j0), 5); e.u32(UInt32(pos + 1 - j0), 6)
                e.u32(UInt32(occG), 7)
                e.dispatchThreads(sz(32, occG, NHEADS), threadsPerThreadgroup: sz(32, occG, 1))
            }
            e.setComputePipelineState(PI(r1AttnO ? "matvec_int4aff_r1" : "matvec_int4aff"))
            e.bufs([(ctx, 0), lw.woQP, lw.woSC, lw.woBI, (attnBuf, 0)]); e.u32(UInt32(8 * hd), 5)
            e.f32(aq.o.i, 6); e.f32(aq.o.o, 7)
            e.dispatchThreads(sz(32, r1AttnO ? HID : HID / 4), threadsPerThreadgroup: sz(32, 8))
            if fuse {
                if lw.m.int2 {
                    e.setComputePipelineState(PI(lut2 ? "gateup_int2sym_nxa_lut" : "gateup_int2sym_nxa"))
                    e.bufs([(attnBuf, 0), lw.postAttn, (xBuf, 0), (xABuf, 0), lw.preFfw,
                            lw.gQP, lw.gSC, lw.uQP, lw.uSC, (hffnBuf, 0)])
                    e.u32(UInt32(HID), 10)
                    e.f32(aq.gate.i, 11); e.f32(aq.gate.o, 12); e.f32(aq.up.o, 13)
                } else {
                    e.setComputePipelineState(PI("gateup_int4aff_nxa"))
                    e.bufs([(attnBuf, 0), lw.postAttn, (xBuf, 0), (xABuf, 0), lw.preFfw,
                            lw.gQP, lw.gSC, lw.gBI!, lw.uQP, lw.uSC, lw.uBI!, (hffnBuf, 0)])
                    e.u32(UInt32(HID), 12)
                    e.f32(aq.gate.i, 13); e.f32(aq.gate.o, 14); e.f32(aq.up.o, 15)
                }
                e.dispatchThreads(sz(32, lw.m.ffn_n / 4), threadsPerThreadgroup: sz(32, 8))
                if lw.m.int2 {
                    e.setComputePipelineState(PI(r1Down ? "matvec_int2sym_r1" : (lut2 ? "matvec_int2sym_lut" : "matvec_int2sym")))
                    e.bufs([(hffnBuf, 0), lw.dQP, lw.dSC, (ffnBuf, 0)]); e.u32(UInt32(lw.m.ffn_k), 4)
                    e.f32(aq.down.i, 5); e.f32(aq.down.o, 6)
                    e.dispatchThreads(sz(32, r1Down ? HID : HID / 4), threadsPerThreadgroup: sz(32, 8))
                } else {
                    e.setComputePipelineState(PI(r1Down ? "matvec_int4aff_r1" : "matvec_int4aff"))
                    e.bufs([(hffnBuf, 0), lw.dQP, lw.dSC, lw.dBI!, (ffnBuf, 0)])
                    e.u32(UInt32(lw.m.ffn_k), 5)
                    e.f32(aq.down.i, 6); e.f32(aq.down.o, 7)
                    e.dispatchThreads(sz(32, r1Down ? HID : HID / 4), threadsPerThreadgroup: sz(32, 8))
                }
                e.setComputePipelineState(P("matvec_int8_gate_nxa"))
                e.bufs([(ffnBuf, 0), lw.postFfw, (xABuf, 0), (xFBuf, 0),
                        lw.pgW8, lw.pgSC, (pliBuf, 0), (g256Buf, 0)])
                e.u32(UInt32(HID), 8); e.u32(UInt32(lw.idx * 256), 9)
                e.f32(aq.pleGate.i, 10); e.f32(aq.pleGate.o, 11)
                e.dispatchThreads(sz(32, 256 / 4), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(P("matvec_int8"))
                e.bufs([(g256Buf, 0), lw.ppW8, lw.ppSC, (p1536Buf, 0), (p1536Buf, 0)])
                e.u32(256, 5); e.u32(0, 6); e.u32(0, 7)
                e.f32(aq.pleProj.i, 8); e.f32(aq.pleProj.o, 9)
                e.dispatchThreads(sz(32, HID / 4), threadsPerThreadgroup: sz(32, 8))
                continue
            }
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(attnBuf, 0), lw.postAttn, (xBuf, 0), (xBuf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(1.0, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, 1), threadsPerThreadgroup: sz(32, 1))
            if lw.m.int2 {
                e.setComputePipelineState(P("gateup_int2sym_nx"))
                e.bufs([(xBuf, 0), lw.preFfw, lw.gQP, lw.gSC, lw.uQP, lw.uSC, (hffnBuf, 0)])
                e.u32(UInt32(HID), 7)
                e.f32(aq.gate.i, 8); e.f32(aq.gate.o, 9); e.f32(aq.up.o, 10)
                e.dispatchThreads(sz(32, lw.m.ffn_n / 4), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(PI(r1Down ? "matvec_int2sym_r1" : (lut2 ? "matvec_int2sym_lut" : "matvec_int2sym")))
                e.bufs([(hffnBuf, 0), lw.dQP, lw.dSC, (ffnBuf, 0)]); e.u32(UInt32(lw.m.ffn_k), 4)
                e.f32(aq.down.i, 5); e.f32(aq.down.o, 6)
                e.dispatchThreads(sz(32, r1Down ? HID : HID / 4), threadsPerThreadgroup: sz(32, 8))
            } else {
                e.setComputePipelineState(P("gateup_int4aff_nx"))
                e.bufs([(xBuf, 0), lw.preFfw, lw.gQP, lw.gSC, lw.gBI!, lw.uQP, lw.uSC, lw.uBI!,
                        (hffnBuf, 0)])
                e.u32(UInt32(HID), 9)
                e.f32(aq.gate.i, 10); e.f32(aq.gate.o, 11); e.f32(aq.up.o, 12)
                e.dispatchThreads(sz(32, lw.m.ffn_n / 4), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(PI(r1Down ? "matvec_int4aff_r1" : "matvec_int4aff"))
                e.bufs([(hffnBuf, 0), lw.dQP, lw.dSC, lw.dBI!, (ffnBuf, 0)])
                e.u32(UInt32(lw.m.ffn_k), 5)
                e.f32(aq.down.i, 6); e.f32(aq.down.o, 7)
                e.dispatchThreads(sz(32, r1Down ? HID : HID / 4), threadsPerThreadgroup: sz(32, 8))
            }
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(ffnBuf, 0), lw.postFfw, (xBuf, 0), (xBuf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(1.0, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, 1), threadsPerThreadgroup: sz(32, 1))
            e.setComputePipelineState(P("matvec_int8"))
            e.bufs([(xBuf, 0), lw.pgW8, lw.pgSC, (pliBuf, 0), (g256Buf, 0)])
            e.u32(UInt32(HID), 5); e.u32(1, 6); e.u32(UInt32(lw.idx * 256), 7)
            e.f32(aq.pleGate.i, 8); e.f32(aq.pleGate.o, 9)
            e.dispatchThreads(sz(32, 256 / 4), threadsPerThreadgroup: sz(32, 8))
            e.setComputePipelineState(P("matvec_int8"))
            e.bufs([(g256Buf, 0), lw.ppW8, lw.ppSC, (p1536Buf, 0), (p1536Buf, 0)])
            e.u32(256, 5); e.u32(0, 6); e.u32(0, 7)
            e.f32(aq.pleProj.i, 8); e.f32(aq.pleProj.o, 9)
            e.dispatchThreads(sz(32, HID / 4), threadsPerThreadgroup: sz(32, 8))
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(p1536Buf, 0), lw.postPle, (xBuf, 0), (xBuf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(lw.ls, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, 1), threadsPerThreadgroup: sz(32, 1))
        }

        if fuse {
            let last = layers.last!
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(p1536Buf, 0), last.postPle, (xFBuf, 0), (xBuf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(last.ls, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, 1), threadsPerThreadgroup: sz(32, 1))
        }

        if wantHead {
            e.setComputePipelineState(PI("matvec_int2sym_nx"))
            e.bufs([(xBuf, 0), finalNorm, headQP, headSC, (logitsBuf, 0)])
            e.u32(UInt32(HID), 5); e.f32(0, 6); e.f32(0, 7)   // lm_head unquantized
            e.dispatchThreads(sz(32, V / 4), threadsPerThreadgroup: sz(32, 8))
            e.setComputePipelineState(P("argmax_stage1"))
            e.bufs([(logitsBuf, 0), (partvBuf, 0), (partiBuf, 0)]); e.u32(UInt32(V), 3)
            e.dispatchThreads(sz(NPARTS * 256), threadsPerThreadgroup: sz(256))
            e.setComputePipelineState(P("argmax_stage2"))
            e.bufs([(partvBuf, 0), (partiBuf, 0), (toksBuf, 0)])
            e.u32(UInt32(NPARTS), 3); e.u32(posU, 4)
            e.dispatchThreads(sz(256), threadsPerThreadgroup: sz(256))
        }
    }

    /// One m-position prefill chunk at [pos0, pos0+m), m in {4, 8, 16}: the MTP
    /// verify-lane dispatch sequence (G4Runner.encodeVerify) MINUS the final-norm/
    /// head/argmax tail — it writes KV for all m positions reading the weights once.
    /// m=4 uses the gated verify kernels verbatim; m=8/16 use the gemma4_prefill
    /// widenings (same per-output accumulation order). The helper kernels
    /// (embed/ple gather, qknorm_rope4, rmsnorm_glue, flash_sdpa_verify) are
    /// grid-parameterized — dispatched wider, byte-identical per row.
    private func encodePrefillM(_ e: MTLComputeCommandEncoder, pos0: Int, m: Int) {
        let p0 = UInt32(pos0)
        // Under actq only the INTERCHANGED wide family carries the int8 activation
        // path — the staged/LUT/fused losers keep fp16-only signatures and would run
        // a DIFFERENT model, so they are forced off here (bench A/Bs use G4_ACTQ=0).
        let stagedEff = prefillStaged && !actqOn
        let lutEff = prefillLut && !actqOn
        var sfx = m == 16 ? "_m16" : (m == 8 ? "_m8" : "_m4")
        if m > 4 && stagedEff { sfx += "s" }
        // int2 kernels can take the byte-LUT decode lane (A19 ALU relief)
        let i2sfx = (lutEff && !stagedEff && m <= 8) ? sfx + "l" : sfx
        let gh = min(m, 8)                       // group height for row-parallel glue
        // session-6 fused pair (gateup nxa + qkv nx): m8 interchanged lane only
        let fusedLane = prefillFused && !actqOn && m == 8 && !stagedEff
        e.setComputePipelineState(P("embed_gather4"))
        e.bufs([embPacked, embScale, (x4Buf, 0), (toksBuf, 0)]); e.u32(p0, 4)
        e.dispatchThreads(sz(384, m), threadsPerThreadgroup: sz(128, 1))
        e.setComputePipelineState(P("matvec_int8" + sfx))
        e.bufs([(x4Buf, 0), mpW8, mpSC, (p84Buf, 0), (p84Buf, 0)])
        e.u32(UInt32(HID), 5); e.u32(8960, 6); e.u32(0, 7); e.u32(0, 8); e.u32(0, 9)
        e.f32(mpAq.i, 10); e.f32(mpAq.o, 11)
        e.dispatchThreads(sz(32, 8960 / 2), threadsPerThreadgroup: sz(32, 8))
        e.setComputePipelineState(P("ple_gather4"))
        e.bufs([plePacked, pleScale, (ple4Buf, 0), (toksBuf, 0)]); e.u32(p0, 4)
        e.dispatchThreads(sz(4480, m), threadsPerThreadgroup: sz(256, 1))
        e.setComputePipelineState(P("rmsnorm_glue"))
        e.bufs([(p84Buf, 0), projNorm, (ple4Buf, 0), (pli4Buf, 0)])
        e.u32(256, 4); e.u32(1, 5); e.f32(PLI_SCALE, 6); e.u32(0, 7); e.f32(0, 8)
        e.dispatchThreads(sz(32, m * 35), threadsPerThreadgroup: sz(32, 8))

        for lw in layers {
            let aq = layerAq[lw.idx]
            let hd = lw.m.hd
            let invf = lw.m.full ? invfFull! : invfSliding!
            let kc = kCache[lw.m.cache], vc = vCache[lw.m.cache]
            let win = lw.m.full ? 0 : WINDOW
            let occG = lw.m.full ? occGF : occGS
            let q = q4Buf[hd]!, qn = qn4Buf[hd]!, kv = kv4Buf[hd]!, ctx = ctx4Buf[hd]!
            if fusedLane && lw.m.write {
                // fused pre_attn norm + q/k/v (v -> kv2 scratch keeps k-rope and
                // v-norm hazard-parallel, like the S=1 fused lane)
                let kv2 = kv24Buf[hd]!
                e.setComputePipelineState(PI("matvec_int4aff_m8_nx_qkv"))
                e.bufs([(x4Buf, 0), lw.preAttn,
                        lw.wqQP, lw.wqSC, lw.wqBI,
                        lw.wkQP!, lw.wkSC!, lw.wkBI!,
                        lw.wvQP!, lw.wvSC!, lw.wvBI!,
                        (q, 0), (kv, 0), (kv2, 0)])
                e.u32(UInt32(HID), 14); e.u32(UInt32(hd), 15)
                e.dispatchThreads(sz(32, 10 * hd / 2), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(P("qknorm_rope4"))
                e.bufs([(q, 0), lw.queryNorm, invf, (qn, 0)])
                e.u32(UInt32(hd), 4); e.u32(p0, 5); e.u32(8, 6); e.u32(0, 7); e.f32(0, 8)
                e.dispatchThreads(sz(32, m * 8), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(P("qknorm_rope4"))
                e.bufs([(kv, 0), lw.keyNorm!, invf, (kc, 0)])
                e.u32(UInt32(hd), 4); e.u32(p0, 5); e.u32(1, 6); e.u32(p0, 7); e.f32(aq.kCache, 8)
                e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
                e.setComputePipelineState(P("rmsnorm_glue"))
                e.bufs([(kv2, 0), lw.keyNorm!, (kv2, 0), (vc, 0)])
                e.u32(UInt32(hd), 4); e.u32(2, 5); e.f32(1.0, 6); e.u32(p0, 7); e.f32(aq.vCache, 8)
                e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
            } else {
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(x4Buf, 0), lw.preAttn, (x4Buf, 0), (xn4Buf, 0)])
            e.u32(UInt32(HID), 4); e.u32(0, 5); e.f32(1.0, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
            e.setComputePipelineState(PI("matvec_int4aff" + sfx))
            e.bufs([(xn4Buf, 0), lw.wqQP, lw.wqSC, lw.wqBI, (q, 0)])
            e.u32(UInt32(HID), 5); e.u32(UInt32(8 * hd), 6)
            e.f32(aq.q.i, 7); e.f32(aq.q.o, 8)
            e.dispatchThreads(sz(32, 8 * hd / 2), threadsPerThreadgroup: sz(32, 8))
            e.setComputePipelineState(P("qknorm_rope4"))
            e.bufs([(q, 0), lw.queryNorm, invf, (qn, 0)])
            e.u32(UInt32(hd), 4); e.u32(p0, 5); e.u32(8, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, m * 8), threadsPerThreadgroup: sz(32, 8))
            if lw.m.write {
                e.setComputePipelineState(PI("matvec_int4aff" + sfx))
                e.bufs([(xn4Buf, 0), lw.wkQP!, lw.wkSC!, lw.wkBI!, (kv, 0)])
                e.u32(UInt32(HID), 5); e.u32(UInt32(hd), 6)
                e.f32(aq.k.i, 7); e.f32(aq.k.o, 8)
                e.dispatchThreads(sz(32, hd / 2), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(P("qknorm_rope4"))
                e.bufs([(kv, 0), lw.keyNorm!, invf, (kc, 0)])
                e.u32(UInt32(hd), 4); e.u32(p0, 5); e.u32(1, 6); e.u32(p0, 7); e.f32(aq.kCache, 8)
                e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
                e.setComputePipelineState(PI("matvec_int4aff" + sfx))
                e.bufs([(xn4Buf, 0), lw.wvQP!, lw.wvSC!, lw.wvBI!, (kv, 0)])
                e.u32(UInt32(HID), 5); e.u32(UInt32(hd), 6)
                e.f32(aq.v.i, 7); e.f32(aq.v.o, 8)
                e.dispatchThreads(sz(32, hd / 2), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(P("rmsnorm_glue"))
                e.bufs([(kv, 0), lw.keyNorm!, (kv, 0), (vc, 0)])
                e.u32(UInt32(hd), 4); e.u32(2, 5); e.f32(1.0, 6); e.u32(p0, 7); e.f32(aq.vCache, 8)
                e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
            }
            }
            e.setComputePipelineState(P("flash_sdpa_verify"))
            e.bufs([(qn, 0), (kc, 0), (vc, 0), (ctx, 0)])
            e.u32(UInt32(hd), 4); e.u32(p0, 5); e.u32(UInt32(win), 6)
            e.u32(UInt32(occG), 7); e.u32(8, 8)
            e.dispatchThreads(sz(32, occG, m * 8), threadsPerThreadgroup: sz(32, occG, 1))
            e.setComputePipelineState(PI("matvec_int4aff" + sfx))
            e.bufs([(ctx, 0), lw.woQP, lw.woSC, lw.woBI, (attn4Buf, 0)])
            e.u32(UInt32(8 * hd), 5); e.u32(UInt32(HID), 6)
            e.f32(aq.o.i, 7); e.f32(aq.o.o, 8)
            e.dispatchThreads(sz(32, HID / 2), threadsPerThreadgroup: sz(32, 8))
            if fusedLane {
                // fused post_attn fold + pre_ffw norm + gateup; x' ping-pongs to
                // xa4 (never in place — cross-threadgroup RES reads would race)
                if lw.m.int2 {
                    e.setComputePipelineState(PI("gateup_int2sym_m8_nxa"))
                    e.bufs([(attn4Buf, 0), lw.postAttn, (x4Buf, 0), (xa4Buf, 0), lw.preFfw,
                            lw.gQP, lw.gSC, lw.uQP, lw.uSC, (hffn4Buf, 0)])
                    e.u32(UInt32(HID), 10); e.u32(UInt32(lw.m.ffn_n), 11)
                    e.dispatchThreads(sz(32, lw.m.ffn_n / 2), threadsPerThreadgroup: sz(32, 8))
                    e.setComputePipelineState(PI("matvec_int2sym" + i2sfx))
                    e.bufs([(hffn4Buf, 0), lw.dQP, lw.dSC, (ffn4Buf, 0)])
                    e.u32(UInt32(lw.m.ffn_k), 4); e.u32(UInt32(HID), 5)
                    e.f32(aq.down.i, 6); e.f32(aq.down.o, 7)
                    e.dispatchThreads(sz(32, HID / 2), threadsPerThreadgroup: sz(32, 8))
                } else {
                    e.setComputePipelineState(PI("gateup_int4aff_m8_nxa"))
                    e.bufs([(attn4Buf, 0), lw.postAttn, (x4Buf, 0), (xa4Buf, 0), lw.preFfw,
                            lw.gQP, lw.gSC, lw.gBI!, lw.uQP, lw.uSC, lw.uBI!, (hffn4Buf, 0)])
                    e.u32(UInt32(HID), 12); e.u32(UInt32(lw.m.ffn_n), 13)
                    e.dispatchThreads(sz(32, lw.m.ffn_n / 2), threadsPerThreadgroup: sz(32, 8))
                    e.setComputePipelineState(PI("matvec_int4aff" + sfx))
                    e.bufs([(hffn4Buf, 0), lw.dQP, lw.dSC, lw.dBI!, (ffn4Buf, 0)])
                    e.u32(UInt32(lw.m.ffn_k), 5); e.u32(UInt32(HID), 6)
                    e.f32(aq.down.i, 7); e.f32(aq.down.o, 8)
                    e.dispatchThreads(sz(32, HID / 2), threadsPerThreadgroup: sz(32, 8))
                }
            } else {
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(attn4Buf, 0), lw.postAttn, (x4Buf, 0), (x4Buf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(1.0, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(x4Buf, 0), lw.preFfw, (x4Buf, 0), (xn4Buf, 0)])
            e.u32(UInt32(HID), 4); e.u32(0, 5); e.f32(1.0, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
            if lw.m.int2 {
                e.setComputePipelineState(PI("gateup_int2sym" + i2sfx))
                e.bufs([(xn4Buf, 0), lw.gQP, lw.gSC, lw.uQP, lw.uSC, (hffn4Buf, 0)])
                e.u32(UInt32(HID), 6); e.u32(UInt32(lw.m.ffn_n), 7)
                e.f32(aq.gate.i, 8); e.f32(aq.gate.o, 9); e.f32(aq.up.o, 10)
                e.dispatchThreads(sz(32, lw.m.ffn_n / 2), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(PI("matvec_int2sym" + i2sfx))
                e.bufs([(hffn4Buf, 0), lw.dQP, lw.dSC, (ffn4Buf, 0)])
                e.u32(UInt32(lw.m.ffn_k), 4); e.u32(UInt32(HID), 5)
                e.f32(aq.down.i, 6); e.f32(aq.down.o, 7)
                e.dispatchThreads(sz(32, HID / 2), threadsPerThreadgroup: sz(32, 8))
            } else {
                e.setComputePipelineState(PI("gateup_int4aff" + sfx))
                e.bufs([(xn4Buf, 0), lw.gQP, lw.gSC, lw.gBI!, lw.uQP, lw.uSC, lw.uBI!, (hffn4Buf, 0)])
                e.u32(UInt32(HID), 8); e.u32(UInt32(lw.m.ffn_n), 9)
                e.f32(aq.gate.i, 10); e.f32(aq.gate.o, 11); e.f32(aq.up.o, 12)
                e.dispatchThreads(sz(32, lw.m.ffn_n / 2), threadsPerThreadgroup: sz(32, 8))
                e.setComputePipelineState(PI("matvec_int4aff" + sfx))
                e.bufs([(hffn4Buf, 0), lw.dQP, lw.dSC, lw.dBI!, (ffn4Buf, 0)])
                e.u32(UInt32(lw.m.ffn_k), 5); e.u32(UInt32(HID), 6)
                e.f32(aq.down.i, 7); e.f32(aq.down.o, 8)
                e.dispatchThreads(sz(32, HID / 2), threadsPerThreadgroup: sz(32, 8))
            }
            }
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(ffn4Buf, 0), lw.postFfw, (fusedLane ? xa4Buf! : x4Buf!, 0), (x4Buf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(1.0, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
            e.setComputePipelineState(P("matvec_int8" + sfx))
            e.bufs([(x4Buf, 0), lw.pgW8, lw.pgSC, (pli4Buf, 0), (g4Buf, 0)])
            e.u32(UInt32(HID), 5); e.u32(256, 6); e.u32(1, 7)
            e.u32(UInt32(lw.idx * 256), 8); e.u32(UInt32(35 * 256), 9)
            e.f32(aq.pleGate.i, 10); e.f32(aq.pleGate.o, 11)
            e.dispatchThreads(sz(32, 256 / 2), threadsPerThreadgroup: sz(32, 8))
            e.setComputePipelineState(P("matvec_int8" + sfx))
            e.bufs([(g4Buf, 0), lw.ppW8, lw.ppSC, (p4Buf, 0), (p4Buf, 0)])
            e.u32(256, 5); e.u32(UInt32(HID), 6); e.u32(0, 7); e.u32(0, 8); e.u32(0, 9)
            e.f32(aq.pleProj.i, 10); e.f32(aq.pleProj.o, 11)
            e.dispatchThreads(sz(32, HID / 2), threadsPerThreadgroup: sz(32, 8))
            e.setComputePipelineState(P("rmsnorm_glue"))
            e.bufs([(p4Buf, 0), lw.postPle, (x4Buf, 0), (x4Buf, 0)])
            e.u32(UInt32(HID), 4); e.u32(1, 5); e.f32(lw.ls, 6); e.u32(0, 7); e.f32(0, 8)
            e.dispatchThreads(sz(32, m), threadsPerThreadgroup: sz(32, gh))
        }
        // (verify's final-norm/head/argmax tail intentionally omitted — prefill only
        // needs the KV writes; the first decode step recomputes the last position's head)
    }

    /// Chunked prefill over [range): widest available chunks (16 -> 8 -> 4, capped by
    /// prefillM) + S=1 remainder, pipelined like decode.
    private func runPrefill(_ range: Range<Int>) {
        var pending: [MTLCommandBuffer] = []
        var pos = range.lowerBound
        let cbTokens = max(tokensPerCB, prefillM)
        while pos < range.upperBound {
            let cb = queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            var inCB = 0
            while pos < range.upperBound && inCB < cbTokens {
                let rem = range.upperBound - pos
                if prefillM >= 16 && rem >= 16 {
                    encodePrefillM(e, pos0: pos, m: 16); pos += 16; inCB += 16
                } else if prefillM >= 8 && rem >= 8 {
                    encodePrefillM(e, pos0: pos, m: 8); pos += 8; inCB += 8
                } else if prefillM >= 4 && rem >= 4 {
                    encodePrefillM(e, pos0: pos, m: 4); pos += 4; inCB += 4
                } else {
                    encodeStep(e, pos: pos, wantHead: false); pos += 1; inCB += 1
                }
            }
            e.endEncoding()
            cb.commit()
            pending.append(cb)
            if pending.count >= pipelineDepth { pending.removeFirst().waitUntilCompleted() }
        }
        while !pending.isEmpty { pending.removeFirst().waitUntilCompleted() }
    }

    // ---------- token IO ----------
    private func setToks(_ ids: [Int], at offset: Int) {
        let p = toksBuf.contents().bindMemory(to: UInt32.self, capacity: MAXCTX + 8)
        for (i, t) in ids.enumerated() { p[offset + i] = UInt32(t) }
    }
    private func readTok(_ i: Int) -> Int {
        let p = toksBuf.contents().bindMemory(to: UInt32.self, capacity: MAXCTX + 8)
        return Int(p[i])
    }

    /// Forget the whole conversation (next generate re-prefills from position 0).
    public func reset() { committed = [] }

    /// Tokens currently committed to the KV cache (the `processedTokenCount` face).
    public var committedCount: Int { committed.count }

    /// KV-rewind for cross-turn prefix reuse (the kit ChatSession `reset(to:)` contract):
    /// keep the leading `n` committed tokens, drop the rest. Positions >= n are simply
    /// overwritten by the next prefill (causal attention never reads them first).
    /// Returns the retained length.
    @discardableResult
    public func trimCommitted(to n: Int) -> Int {
        if n < committed.count { committed = Array(committed.prefix(max(0, n))) }
        return committed.count
    }

    // ---------- generation ----------
    /// Greedy generation with streaming.
    /// - Parameters:
    ///   - promptIds: the FULL conversation ids (bos + all turns + generation header).
    ///     KV for the longest common prefix with the previous call is reused.
    ///   - maxNew: cap on new tokens.
    ///   - stopIds: generation stops after any of these (stop id NOT emitted).
    ///   - onTokens: called with each batch of new token ids as they complete.
    /// - Returns: the generated ids (stop id excluded).
    @discardableResult
    public func generate(promptIds: [Int], maxNew: Int,
                         stopIds: Set<Int> = Gemma4MetalEngine.defaultStopIds,
                         onTokens: (([Int]) -> Void)? = nil) -> [Int] {
        precondition(!promptIds.isEmpty)
        stopRequested = false
        let plen = promptIds.count
        let budget = min(maxNew, MAXCTX - plen - 8)
        guard budget > 0 else { return [] }

        // longest common prefix with what's already in KV
        var pfx = 0
        while pfx < committed.count && pfx < plen && committed[pfx] == promptIds[pfx] { pfx += 1 }
        setToks(Array(promptIds[pfx...]), at: pfx)

        var stats = Stats()
        // prefill: steps [start, plen-1) write KV without the head
        let prefillStart = min(pfx, plen - 1)
        stats.prefillTokens = plen - 1 - prefillStart
        let t0 = DispatchTime.now()
        if prefillStart < plen - 1 {
            runPrefill(prefillStart..<(plen - 1))
        }
        stats.prefillSeconds = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9

        // decode: steps [plen-1, plen-1+budget), head on every step
        let t1 = DispatchTime.now()
        let emitted = runBatch((plen - 1)..<(plen - 1 + budget), headFrom: plen - 1,
                               stopIds: stopIds, onTokens: onTokens)
        stats.decodeSeconds = Double(DispatchTime.now().uptimeNanoseconds - t1.uptimeNanoseconds) / 1e9
        stats.decodeTokens = emitted.count
        lastStats = stats

        committed = promptIds + emitted
        return emitted
    }

    /// Encode steps in CB chunks with a bounded pipeline; emit + stop-check tokens as
    /// each CB completes. Returns emitted tokens (only when stopIds != nil).
    @discardableResult
    private func runBatch(_ range: Range<Int>, headFrom: Int,
                          stopIds: Set<Int>?, onTokens: (([Int]) -> Void)?) -> [Int] {
        var emitted: [Int] = []
        var stopped = false
        var pending: [(cb: MTLCommandBuffer, lo: Int, hi: Int)] = []
        var pos = range.lowerBound

        func drainOne() {
            let head = pending.removeFirst()
            head.cb.waitUntilCompleted()
            guard let stopIds else { return }
            // step at position p produced the token at p+1
            var batch: [Int] = []
            for p in head.lo..<head.hi where p >= headFrom && !stopped {
                let t = readTok(p + 1)
                if stopIds.contains(t) { stopped = true; break }
                batch.append(t)
            }
            if !batch.isEmpty {
                emitted.append(contentsOf: batch)
                onTokens?(batch)
            }
        }

        while pos < range.upperBound && !stopped && !stopRequested {
            let end = min(pos + tokensPerCB, range.upperBound)
            let cb = queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            for p in pos..<end { encodeStep(e, pos: p, wantHead: p >= headFrom) }
            e.endEncoding()
            cb.commit()
            pending.append((cb, pos, end))
            pos = end
            if pending.count >= pipelineDepth { drainOne() }
        }
        while !pending.isEmpty { drainOne() }
        return emitted
    }

    // ---------- validation ----------
    public struct GateResult: Sendable { public let pass: Bool; public let detail: [String] }

    /// Gate support: byte snapshot of K/V cache rows [0, n) per cache slot (K then V,
    /// slots 0..<15). The wide-prefill KV gate byte-compares these across prefill
    /// widths (chunked vs S=1) — the core losslessness proof for the prefill lane.
    public func kvSnapshot(positions n: Int) -> [Data] {
        var out: [Data] = []
        for li in 0..<15 {
            let bytes = min(n, MAXCTX) * meta.layers[li].hd * 2
            out.append(Data(bytes: kCache[li].contents(), count: bytes))
            out.append(Data(bytes: vCache[li].contents(), count: bytes))
        }
        return out
    }

    /// S1 token gate: greedy ids for the 3 oracle prompts must match the fp16 (or
    /// bf16 near-tie) reference EXACTLY. Run after any host or kernel change.
    public func tokenGate(refsURL: URL) throws -> GateResult {
        struct PromptRefs: Decodable { let prompt_ids: [Int]; let bf16: [Int]; let fp16: [Int] }
        struct Refs: Decodable { let prompts: [String: PromptRefs] }
        let refs = try JSONDecoder().decode(Refs.self, from: Data(contentsOf: refsURL))
        var pass = true
        var detail: [String] = []
        for (prompt, d) in refs.prompts.sorted(by: { $0.key < $1.key }) {
            reset()
            let out = generate(promptIds: d.prompt_ids, maxNew: d.fp16.count, stopIds: [])
            let ok = out == d.fp16 || out == d.bf16
            pass = pass && ok
            detail.append("\(ok ? "PASS" : "FAIL") \(prompt)")
            if !ok { detail.append("  got \(out)") }
        }
        reset()
        return GateResult(pass: pass, detail: detail)
    }
}

// MARK: - encoder sugar (identical to G4Runner)
private extension MTLComputeCommandEncoder {
    @inline(__always) func bufs(_ list: [Gemma4MetalEngine.Buf]) {
        for (i, b) in list.enumerated() { setBuffer(b.b, offset: b.o, index: i) }
    }
    @inline(__always) func u32(_ v: UInt32, _ index: Int) {
        var x = v; setBytes(&x, length: 4, index: index)
    }
    @inline(__always) func f32(_ v: Float, _ index: Int) {
        var x = v; setBytes(&x, length: 4, index: index)
    }
}
