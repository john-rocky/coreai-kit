// BoxFourier — graph A's `box_pe` input [16, 256]: the mask decoder's box encoding before its MLP (`BoxFourierEncoder`
// in the package's mask_decoder.py), made on the host as `box_fourier_np` in the zoo's
// conversion/wevisone/mask/mask_graph.py makes it, with torch's values for the 7 widths below:
//
//   x1, y1                      the box's fp32 values (x / 1000 rounded to fp32)
//   w = max(x2 − x1, 1e-6)      fp32 subtraction; h the same
//   ew = fp32(0.2) · log(w) + fp32(0.5)    fp32 product and sum rounded separately (no fused multiply-add); eh the same
//   per value v of (x1, y1, ew, eh), k = 0 … 31:
//     phase = fp32(fp32(2π) · v) · 2^k     (the · 2^k is exact)
//     sin, cos of the phase in double, rounded to fp32
//   order: per value [sin k = 0 … 31, cos k = 0 … 31], the four values in turn → 256
//
// Why on the host: the phases reach 1e10 rad, so one ulp of difference in ew or eh moves the k ≳ 20 components to
// unrelated values. Inside the GPU graph, 0.2 · log(w) + 0.5 came out one ulp away from the CPU's on 28 of 92 box terms
// and an alpha left the bar (the zoo's knowledge/wevisone-vl-2b-port.md). The host's fp32 arithmetic has to equal the
// package's torch CPU arithmetic bit for bit; the CLI's `boxpe` command writes every value a box can give for the check.

import Foundation

public enum BoxFourier {
    public static let numFreqs = 32
    public static let width = 4 * 2 * numFreqs   // 256
    static let twoPi = Float(2.0 * Double.pi)     // 6.2831855 = 0x40C90FDB

    /// The widths where Darwin's `logf` and torch's CPU `log` (2.9 and 2.11, any tensor length) round apart: torch's
    /// encoded value for each. Integer boxes (0–1000) give 3,494 distinct fp32 widths; these 7 are the only ones that
    /// differ (`cutout-cli boxpe` + the zoo's conversion/wevisone/host/check_boxpe.py, 2026-10-25), so with the table
    /// every box's value is the package's.
    static let torchEncoded: [UInt32: UInt32] = [
        0x3E36_45A0: 0x3E1E_855E,  // w 0.17799997: logf gives 0x3E1E855C
        0x3E56_041A: 0x3E3F_66DA,  // w 0.20900002: 0x3E3F66DC
        0x3E5E_353F: 0x3E47_183C,  // w 0.21699999: 0x3E47183E
        0x3EB0_A3D6: 0x3E93_065A,  // w 0.34499997: 0x3E930658
        0x3EC4_1892: 0x3E9D_B982,  // w 0.38299996: 0x3E9DB981
        0x3F1B_E76C: 0x3ECD_374F,  // w 0.60899997: 0x3ECD374E
        0x3F39_DB24: 0x3EDF_3604,  // w 0.72600007: 0x3EDF3603
    ]

    /// fp32(0.2) · log(v) + fp32(0.5), the product and the sum rounded separately; torch's value where libm's differs.
    @inline(never) public static func encodedSide(_ v: Float) -> Float {
        if let t = torchEncoded[v.bitPattern] { return Float(bitPattern: t) }
        let p = Float(0.2) * logf(v)
        return roundedSum(p, Float(0.5))
    }

    /// The same without the table (libm's logf alone), for the check that measures what the table changes.
    @inline(never) public static func encodedSideLibm(_ v: Float) -> Float {
        roundedSum(Float(0.2) * logf(v), Float(0.5))
    }

    /// A separate function so the compiler cannot fuse the product into the sum.
    @inline(never) static func roundedSum(_ a: Float, _ b: Float) -> Float { a + b }

    /// The 64 features of one value: [sin(phase_k), k = 0 … 31] then [cos(phase_k), k = 0 … 31].
    public static func features(_ v: Float, into out: inout [Float], at offset: Int) {
        let base = twoPi * v
        var scale: Float = 1
        for k in 0..<numFreqs {
            let phase = Double(base * scale)
            out[offset + k] = Float(Foundation.sin(phase))
            out[offset + numFreqs + k] = Float(Foundation.cos(phase))
            scale *= 2
        }
    }

    /// The four encoded values of one box (x1, y1, x2, y2 in fp32): x1, y1, ew, eh.
    public static func encoded(_ box: [Float], libm: Bool = false) -> [Float] {
        let w = max(box[2] - box[0], Float(1e-6))
        let h = max(box[3] - box[1], Float(1e-6))
        return libm ? [box[0], box[1], encodedSideLibm(w), encodedSideLibm(h)] : [box[0], box[1], encodedSide(w), encodedSide(h)]
    }

    /// `box_pe` rows for fp32 boxes [n][4] → [n * 256]. `libm` = without the torch table (a check only).
    public static func rows(_ boxes: [[Float]], libm: Bool = false) -> [Float] {
        var out = [Float](repeating: 0, count: boxes.count * width)
        for (i, box) in boxes.enumerated() {
            for (j, v) in encoded(box, libm: libm).enumerated() {
                features(v, into: &out, at: i * width + j * 2 * numFreqs)
            }
        }
        return out
    }
}
