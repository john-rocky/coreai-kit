// Gemma4 E2B mixed-bit RAW-METAL decode loop — FUSED WIDE PREFILL kernels (M=8).
//
// Session-6 prototype pair (GEMMA4_PREFILL_FUSE_KICKOFF.md step 1):
//   gateup_int{2sym,4aff}_m8_nxa : post_attn residual-fold + pre_ffw rmsnorm folded
//                                  into the wide gateup prologue (removes 2
//                                  rmsnorm_glue dispatches/layer + the x4/xn4
//                                  device round-trip). x' ping-pongs to a separate
//                                  XOUT buffer (xa4) exactly like the S=1 fused
//                                  lane's xBuf->xABuf rotation — never in place.
//   matvec_int4aff_m8_nx_qkv     : pre_attn rmsnorm + q/k/v matvecs in ONE dispatch
//                                  (write layers; rows [0,8hd)=q, [8hd,9hd)=k,
//                                  [9hd,10hd)=v like the S=1 _nx_qkv). v lands in
//                                  a dedicated scratch (kv2_4) so k's rope and v's
//                                  norm stay hazard-parallel.
//
// BIT-EXACTNESS CONTRACT (same as every lane before):
// - Reductions: token t's rmsnorm sums are computed by ONE SIMD-group with the
//   EXACT S=1 lane->k mapping (k = lane*nept + i, i ascending; fp32 accumulate;
//   simd_sum across 32 lanes) — identical op order => identical bytes. inv1/inv2
//   are published through threadgroup floats (a value copy, not a re-computation).
// - Element-wise values: xp = RES + half(float(D)*inv1)*PW and
//   xn = half(float(xp)*inv2)*NW are recomputed ON DEMAND at each consumption site
//   with the same expression shapes as the S=1 NXA/NX bodies (explicit half
//   temporaries, mathMode .safe). Element-wise ops are order-free, so recompute
//   sites produce byte-identical halfs (the S=1 _nx_qkv already consumes x this
//   way — this is its literal widening, not a new numerics class).
// - Dot accumulation: the k-block j-then-block chains, simd_sum epilogues and
//   gelu expressions are copied UNCHANGED from gemma4_prefill.metal's m8 bodies.
// Proven by the Mac KV byte-compare + tokgate + device S1 gates, not argued.
//
// Dispatch (matvecs): threads = [32, N/2], group_size = [32, 8] — identical to the
// unfused m8 lane. M=8 is internal (per-thread accumulator width MV=2).

#include <metal_stdlib>
using namespace metal;

constant uint RPF   = 2;    // rows per SIMD-group (same as the m8 lane's RP)
constant uint SGYPF = 8;
constant uint GPF   = 64;   // INT4 affine group size

#ifndef IL4
#define IL4 0
#endif
#define QPF_WORD(QP, n, kw, w) \
    (IL4 ? (QP)[((((n) >> 2) * (kw) + (w)) << 2) + ((n) & 3)] : (QP)[(n) * (kw) + (w)])

// ---- shared fold prologue ------------------------------------------------------------------
// SIMD-group t (t < SV tokens) runs token t's two reductions in the exact S=1 NXA
// order and publishes inv1/inv2; tgid.y == 0 also writes the folded x' back to XOUT
// (the next fold's RES input). Consumers recompute xp/xn per element from the
// published inv values — no threadgroup x staging, no register x-stage.
// tg_inv1/tg_inv2 are threadgroup float[SV] declared in the KERNEL body (Metal
// forbids threadgroup declarations inside inline functions) and passed down.
#define NXA_FOLD_WIDE(SV)                                                   \
    {                                                                       \
        const uint nept = K / 32;                                           \
        if (sg < SV) {                                                      \
            const uint t = sg;                                              \
            float ssq1 = 0.0f;                                              \
            for (uint i = 0; i < nept; ++i) {                               \
                float dv = float(D[t * K + lane * nept + i]);               \
                ssq1 += dv * dv;                                            \
            }                                                               \
            const float inv1 =                                              \
                metal::precise::rsqrt(simd_sum(ssq1) / float(K) + 1e-6f);   \
            float ssq2 = 0.0f;                                              \
            for (uint i = 0; i < nept; ++i) {                               \
                const uint k = lane * nept + i;                             \
                half nrm = half(float(D[t * K + k]) * inv1) * PW[k];        \
                half xp = RES[t * K + k] + nrm;                             \
                if (tgid.y == 0) XOUT[t * K + k] = xp;                      \
                float xv = float(xp);                                       \
                ssq2 += xv * xv;                                            \
            }                                                               \
            const float s2 = simd_sum(ssq2);                                \
            if (lane == 0) {                                                \
                tg_inv1[t] = inv1;                                          \
                tg_inv2[t] = metal::precise::rsqrt(s2 / float(K) + 1e-6f);  \
            }                                                               \
        }                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                    \
    }

// On-demand folded+normed x for one token/element (expression shapes = S=1 NXA
// consumption: hx = half(float(xp)*inv2)*NW — with xp recomputed as in the fold).
inline float xn_at(device const half* D, device const half* RES, uint idx,
                   float inv1, float inv2, half pwk, half nwk)
{
    half nrm = half(float(D[idx]) * inv1) * pwk;
    half xp = RES[idx] + nrm;
    half hx = half(float(xp) * inv2) * nwk;
    return float(hx);
}
#define XN_AT(t, k, pwk, nwk) \
    xn_at(D, RES, (t) * K + (k), tg_inv1[(t)], tg_inv2[(t)], (pwk), (nwk))

// ---- FUSED gate+up+gelu+mul + NXA fold, INT2, M = 4*MV --------------------------------------
template <uint MV>
inline void gateup_int2sym_mw_nxa(device const half* D, device const half* PW,
                                  device const half* RES, device half* XOUT,
                                  device const half* NW,
                                  device const uint* QPG, device const float* SCG,
                                  device const uint* QPU, device const float* SCU,
                                  device half* C, uint K, uint N,
                                  threadgroup float* tg_inv1, threadgroup float* tg_inv2,
                                  uint2 tid, uint2 tgid)
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGYPF + sg) * RPF;
    const uint kw = K >> 4;

    NXA_FOLD_WIDE(4 * MV)

    float4 accg[RPF][MV], accu[RPF][MV];
    for (uint r = 0; r < RPF; ++r)
        for (uint v = 0; v < MV; ++v) { accg[r][v] = float4(0.0f); accu[r][v] = float4(0.0f); }

    for (uint kb = 0; kb < K; kb += 512) {
        const uint k0 = kb + lane * 16;
        const uint w0 = (kb >> 4) + lane;
        uint pkg[RPF], pku[RPF];
        for (uint r = 0; r < RPF; ++r) {
            const uint n = base_row + r;
            pkg[r] = QPF_WORD(QPG, n, kw, w0);
            pku[r] = QPF_WORD(QPU, n, kw, w0);
        }
        float4 sg4[RPF][MV], su4[RPF][MV];
        for (uint r = 0; r < RPF; ++r)
            for (uint v = 0; v < MV; ++v) { sg4[r][v] = float4(0.0f); su4[r][v] = float4(0.0f); }
        for (uint j = 0; j < 16; ++j) {
            const uint k = k0 + j;
            const half pwk = PW[k];
            const half nwk = NW[k];
            float4 xv[MV];
            for (uint v = 0; v < MV; ++v)
                xv[v] = float4(XN_AT(4 * v, k, pwk, nwk), XN_AT(4 * v + 1, k, pwk, nwk),
                               XN_AT(4 * v + 2, k, pwk, nwk), XN_AT(4 * v + 3, k, pwk, nwk));
            for (uint r = 0; r < RPF; ++r) {
                float wg = float(int(((pkg[r] >> (2 * j)) & 0x3) ^ 2u) - 2);
                float wu = float(int(((pku[r] >> (2 * j)) & 0x3) ^ 2u) - 2);
                for (uint v = 0; v < MV; ++v) {
                    sg4[r][v] += xv[v] * wg;
                    su4[r][v] += xv[v] * wu;
                }
            }
        }
        for (uint r = 0; r < RPF; ++r)
            for (uint v = 0; v < MV; ++v) { accg[r][v] += sg4[r][v]; accu[r][v] += su4[r][v]; }
    }
    for (uint r = 0; r < RPF; ++r) {
        const uint n = base_row + r;
        const float scg = SCG[n];
        const float scu = SCU[n];
        for (uint v = 0; v < MV; ++v) {
            for (uint m = 0; m < 4; ++m) {
                float tg = simd_sum(m == 0 ? accg[r][v].x : (m == 1 ? accg[r][v].y : (m == 2 ? accg[r][v].z : accg[r][v].w)));
                float tu = simd_sum(m == 0 ? accu[r][v].x : (m == 1 ? accu[r][v].y : (m == 2 ? accu[r][v].z : accu[r][v].w)));
                if (lane == 0) {
                    float xg = tg * scg;
                    float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                        0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
                    C[(4 * v + m) * N + n] = half(gel * (tu * scu));
                }
            }
        }
    }
}

kernel void gateup_int2sym_m8_nxa(
    device const half*  D    [[buffer(0)]],
    device const half*  PW   [[buffer(1)]],
    device const half*  RES  [[buffer(2)]],
    device half*        XOUT [[buffer(3)]],
    device const half*  NW   [[buffer(4)]],
    device const uint*  QPG  [[buffer(5)]],
    device const float* SCG  [[buffer(6)]],
    device const uint*  QPU  [[buffer(7)]],
    device const float* SCU  [[buffer(8)]],
    device half*        C    [[buffer(9)]],
    constant uint&      K    [[buffer(10)]],
    constant uint&      N    [[buffer(11)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tg_inv1[8];
    threadgroup float tg_inv2[8];
    gateup_int2sym_mw_nxa<2>(D, PW, RES, XOUT, NW, QPG, SCG, QPU, SCU, C, K, N,
                             tg_inv1, tg_inv2, tid, tgid);
}

// ---- FUSED gate+up+gelu+mul + NXA fold, INT4, M = 4*MV --------------------------------------
template <uint MV>
inline void gateup_int4aff_mw_nxa(device const half* D, device const half* PW,
                                  device const half* RES, device half* XOUT,
                                  device const half* NW,
                                  device const uint* QPG, device const half* SCG, device const half* BIG,
                                  device const uint* QPU, device const half* SCU, device const half* BIU,
                                  device half* C, uint K, uint N,
                                  threadgroup float* tg_inv1, threadgroup float* tg_inv2,
                                  uint2 tid, uint2 tgid)
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGYPF + sg) * RPF;
    const uint kw = K >> 3;
    const uint ng = K / GPF;

    NXA_FOLD_WIDE(4 * MV)

    float4 accg[RPF][MV], accu[RPF][MV];
    for (uint r = 0; r < RPF; ++r)
        for (uint v = 0; v < MV; ++v) { accg[r][v] = float4(0.0f); accu[r][v] = float4(0.0f); }

    for (uint kb = 0; kb < K; kb += 256) {
        const uint k0 = kb + lane * 8;
        const uint w0 = (kb >> 3) + lane;
        const uint grp = k0 / GPF;
        uint pkg[RPF], pku[RPF];
        float scgr[RPF], bigr[RPF], scur[RPF], biur[RPF];
        for (uint r = 0; r < RPF; ++r) {
            const uint n = base_row + r;
            pkg[r] = QPF_WORD(QPG, n, kw, w0);
            pku[r] = QPF_WORD(QPU, n, kw, w0);
            scgr[r] = float(SCG[n * ng + grp]); bigr[r] = float(BIG[n * ng + grp]);
            scur[r] = float(SCU[n * ng + grp]); biur[r] = float(BIU[n * ng + grp]);
        }
        float4 sg4[RPF][MV], su4[RPF][MV];
        for (uint r = 0; r < RPF; ++r)
            for (uint v = 0; v < MV; ++v) { sg4[r][v] = float4(0.0f); su4[r][v] = float4(0.0f); }
        for (uint j = 0; j < 8; ++j) {
            const uint k = k0 + j;
            const half pwk = PW[k];
            const half nwk = NW[k];
            float4 xv[MV];
            for (uint v = 0; v < MV; ++v)
                xv[v] = float4(XN_AT(4 * v, k, pwk, nwk), XN_AT(4 * v + 1, k, pwk, nwk),
                               XN_AT(4 * v + 2, k, pwk, nwk), XN_AT(4 * v + 3, k, pwk, nwk));
            for (uint r = 0; r < RPF; ++r) {
                float wg = scgr[r] * float((pkg[r] >> (j * 4)) & 0xf) + bigr[r];
                float wu = scur[r] * float((pku[r] >> (j * 4)) & 0xf) + biur[r];
                for (uint v = 0; v < MV; ++v) {
                    sg4[r][v] += xv[v] * wg;
                    su4[r][v] += xv[v] * wu;
                }
            }
        }
        for (uint r = 0; r < RPF; ++r)
            for (uint v = 0; v < MV; ++v) { accg[r][v] += sg4[r][v]; accu[r][v] += su4[r][v]; }
    }
    for (uint r = 0; r < RPF; ++r) {
        const uint n = base_row + r;
        for (uint v = 0; v < MV; ++v) {
            for (uint m = 0; m < 4; ++m) {
                float tg = simd_sum(m == 0 ? accg[r][v].x : (m == 1 ? accg[r][v].y : (m == 2 ? accg[r][v].z : accg[r][v].w)));
                float tu = simd_sum(m == 0 ? accu[r][v].x : (m == 1 ? accu[r][v].y : (m == 2 ? accu[r][v].z : accu[r][v].w)));
                if (lane == 0) {
                    float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                        0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
                    C[(4 * v + m) * N + n] = half(gel * tu);
                }
            }
        }
    }
}

kernel void gateup_int4aff_m8_nxa(
    device const half* D    [[buffer(0)]],
    device const half* PW   [[buffer(1)]],
    device const half* RES  [[buffer(2)]],
    device half*       XOUT [[buffer(3)]],
    device const half* NW   [[buffer(4)]],
    device const uint* QPG  [[buffer(5)]],
    device const half* SCG  [[buffer(6)]],
    device const half* BIG  [[buffer(7)]],
    device const uint* QPU  [[buffer(8)]],
    device const half* SCU  [[buffer(9)]],
    device const half* BIU  [[buffer(10)]],
    device half*       C    [[buffer(11)]],
    constant uint&     K    [[buffer(12)]],
    constant uint&     N    [[buffer(13)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tg_inv1[8];
    threadgroup float tg_inv2[8];
    gateup_int4aff_mw_nxa<2>(D, PW, RES, XOUT, NW, QPG, SCG, BIG, QPU, SCU, BIU, C, K, N,
                             tg_inv1, tg_inv2, tid, tgid);
}

// ---- FUSED pre_attn norm + q/k/v matvec, INT4, M = 4*MV (write layers) ----------------------
// Rows [0, 8hd) = q -> CQ [M, 8hd], [8hd, 9hd) = k -> CK [M, hd], [9hd, 10hd) = v ->
// CV [M, hd]. Segment boundaries are multiples of RPF=2, so weight/output selection
// is uniform per SIMD-group. Norm: every SIMD-group redundantly computes all M invs
// (the S=1 NX_PROLOGUE shape — no barrier), then x is normed on demand at read.
template <uint MV>
inline void matvec_int4aff_mw_nx_qkv(device const half* X, device const half* NW,
                                     device const uint* QPQ, device const half* SCQ, device const half* BIQ,
                                     device const uint* QPK, device const half* SCK, device const half* BIK,
                                     device const uint* QPV, device const half* SCV, device const half* BIV,
                                     device half* CQ, device half* CK, device half* CV,
                                     uint K, uint hd, uint2 tid, uint2 tgid)
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint grow = (tgid.y * SGYPF + sg) * RPF;   // global row (q ++ k ++ v)
    const uint kw = K >> 3;
    const uint ng = K / GPF;

    device const uint* QP; device const half* SC; device const half* BI; device half* C;
    uint base_row, nout;
    if (grow < 8 * hd)      { QP = QPQ; SC = SCQ; BI = BIQ; C = CQ; base_row = grow;          nout = 8 * hd; }
    else if (grow < 9 * hd) { QP = QPK; SC = SCK; BI = BIK; C = CK; base_row = grow - 8 * hd; nout = hd; }
    else                    { QP = QPV; SC = SCV; BI = BIV; C = CV; base_row = grow - 9 * hd; nout = hd; }

    float inv[4 * MV];
    {
        const uint nept = K / 32;
        for (uint t = 0; t < 4 * MV; ++t) {
            float ssq = 0.0f;
            for (uint i = 0; i < nept; ++i) {
                float xv = float(X[t * K + lane * nept + i]);
                ssq += xv * xv;
            }
            const float ms = simd_sum(ssq) / float(K);
            inv[t] = metal::precise::rsqrt(ms + 1e-6f);
        }
    }

    float4 acc[RPF][MV];
    for (uint r = 0; r < RPF; ++r)
        for (uint v = 0; v < MV; ++v) acc[r][v] = float4(0.0f);

    for (uint kb = 0; kb < K; kb += 256) {
        const uint k0 = kb + lane * 8;
        const uint w0 = (kb >> 3) + lane;
        const uint grp = k0 / GPF;
        uint pk[RPF];
        float scr[RPF], bir[RPF];
        for (uint r = 0; r < RPF; ++r) {
            const uint n = base_row + r;
            pk[r] = QPF_WORD(QP, n, kw, w0);
            scr[r] = float(SC[n * ng + grp]);
            bir[r] = float(BI[n * ng + grp]);
        }
        float4 s4[RPF][MV];
        for (uint r = 0; r < RPF; ++r)
            for (uint v = 0; v < MV; ++v) s4[r][v] = float4(0.0f);
        for (uint j = 0; j < 8; ++j) {
            const uint k = k0 + j;
            const half nwk = NW[k];
            float4 xv[MV];
            for (uint v = 0; v < MV; ++v) {
                half hx0 = half(float(X[(4 * v) * K + k]) * inv[4 * v]) * nwk;
                half hx1 = half(float(X[(4 * v + 1) * K + k]) * inv[4 * v + 1]) * nwk;
                half hx2 = half(float(X[(4 * v + 2) * K + k]) * inv[4 * v + 2]) * nwk;
                half hx3 = half(float(X[(4 * v + 3) * K + k]) * inv[4 * v + 3]) * nwk;
                xv[v] = float4(float(hx0), float(hx1), float(hx2), float(hx3));
            }
            for (uint r = 0; r < RPF; ++r) {
                uint q = (pk[r] >> (j * 4)) & 0xf;
                float wv = scr[r] * float(q) + bir[r];
                for (uint v = 0; v < MV; ++v) s4[r][v] += xv[v] * wv;
            }
        }
        for (uint r = 0; r < RPF; ++r)
            for (uint v = 0; v < MV; ++v) acc[r][v] += s4[r][v];
    }
    for (uint r = 0; r < RPF; ++r) {
        const uint n = base_row + r;
        for (uint v = 0; v < MV; ++v) {
            const float t0 = simd_sum(acc[r][v].x);
            const float t1 = simd_sum(acc[r][v].y);
            const float t2 = simd_sum(acc[r][v].z);
            const float t3 = simd_sum(acc[r][v].w);
            if (lane == 0) {
                C[(4 * v) * nout + n] = half(t0);
                C[(4 * v + 1) * nout + n] = half(t1);
                C[(4 * v + 2) * nout + n] = half(t2);
                C[(4 * v + 3) * nout + n] = half(t3);
            }
        }
    }
}

kernel void matvec_int4aff_m8_nx_qkv(
    device const half* X    [[buffer(0)]],
    device const half* NW   [[buffer(1)]],
    device const uint* QPQ  [[buffer(2)]],
    device const half* SCQ  [[buffer(3)]],
    device const half* BIQ  [[buffer(4)]],
    device const uint* QPK  [[buffer(5)]],
    device const half* SCK  [[buffer(6)]],
    device const half* BIK  [[buffer(7)]],
    device const uint* QPV  [[buffer(8)]],
    device const half* SCV  [[buffer(9)]],
    device const half* BIV  [[buffer(10)]],
    device half*       CQ   [[buffer(11)]],
    device half*       CK   [[buffer(12)]],
    device half*       CV   [[buffer(13)]],
    constant uint&     K    [[buffer(14)]],
    constant uint&     hd   [[buffer(15)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{ matvec_int4aff_mw_nx_qkv<2>(X, NW, QPQ, SCQ, BIQ, QPK, SCK, BIK, QPV, SCV, BIV,
                              CQ, CK, CV, K, hd, tid, tgid); }
