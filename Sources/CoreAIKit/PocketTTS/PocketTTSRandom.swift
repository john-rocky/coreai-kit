import Foundation

/// PyTorch's CPU RNG, reproduced exactly: an MT19937 engine plus the `normal_fill`
/// Box–Muller transform torch uses for float tensors with 16+ elements.
///
/// This is what makes gate (a) possible at all — the oracle's `--rng explicit` protocol
/// is "a fresh `torch.Generator` re-seeded `seed + call_index` per flow call, then
/// `tensor.normal_(0, temp**0.5)` on a [1,32] float32 tensor", so the host must produce
/// the same 32 floats from the same seed. Verified bit-for-bit against torch 2.13
/// (`tools/gates/` in NOTES.md §18).
///
/// Details that are load-bearing and easy to get wrong:
///   * torch's `mt19937_engine` seeds with the classic `init_genrand` recurrence and
///     truncates the 64-bit seed to 32 bits.
///   * uniforms come from the LOW 24 bits of each 32-bit draw: `(x & 0xFFFFFF) / 2^24`.
///   * `normal_fill_16` works on 16-lane blocks: lanes 0–7 are radii, lanes 8–15 angles;
///     `u1 = 1 - lane[j]`, radius `sqrt(-2 ln u1)` in *float* math, theta promoted to
///     double for the `2π·u2` product then truncated back to float, `cos` for the low
///     lane and `sin` for the high one. A [1,32] tensor is exactly two blocks.
struct PocketTTSTorchRNG {
    private var state = [UInt32](repeating: 0, count: 624)
    private var index = 624

    init(seed: UInt64) {
        state[0] = UInt32(truncatingIfNeeded: seed)
        for i in 1..<624 {
            state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i)
        }
    }

    private mutating func regenerate() {
        for i in 0..<624 {
            let y = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7FFF_FFFF)
            var n = state[(i + 397) % 624] ^ (y >> 1)
            if y & 1 == 1 { n ^= 2_567_483_615 }
            state[i] = n
        }
        index = 0
    }

    mutating func next() -> UInt32 {
        if index >= 624 { regenerate() }
        var y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 2_636_928_640
        y ^= (y << 15) & 4_022_730_752
        y ^= y >> 18
        return y
    }

    /// One uniform float in [0, 1), torch's `uniform_float_from_uint32`.
    mutating func uniform() -> Float {
        Float(next() & 0xFF_FFFF) * (1.0 / Float(1 << 24))
    }

    /// `torch.empty(n).normal_(0, std, generator=g)` for n a multiple of 16.
    mutating func normalFill(count: Int, std: Float) -> [Float] {
        precondition(count >= 16 && count % 16 == 0,
                     "normalFill mirrors torch's block path; count must be a multiple of 16")
        var data = [Float](repeating: 0, count: count)
        for i in 0..<count { data[i] = uniform() }
        var base = 0
        while base + 16 <= count {
            for j in 0..<8 {
                let u1: Float = 1 - data[base + j]
                let u2: Float = data[base + j + 8]
                let radius = sqrtf(-2 * logf(u1))
                let theta = Float(2.0 * Double.pi * Double(u2))   // 2.0f * pi<double> * u2, then cast
                data[base + j] = radius * cosf(theta) * std
                data[base + j + 8] = radius * sinf(theta) * std
            }
            base += 16
        }
        return data
    }
}

/// The oracle noise protocol (`conversion/e2e_coreai.py`): one global call counter per
/// synthesis run, starting at 0; every AR step first increments it, then draws 32 floats
/// from a generator seeded `seed + counter`. Counter slot 0 is *reserved* for the
/// text-prefill draw upstream computes and throws away, which is why steps start at
/// `seed + 1`. The counter runs across chunks and is never reset mid-run.
struct PocketTTSNoiseSource {
    let seed: UInt64
    let std: Float
    private(set) public var calls = 0

    init(seed: UInt64, temp: Float) {
        self.seed = seed
        self.std = Float(Double(temp).squareRoot())
    }

    mutating func nextNoise(count: Int) -> [Float] {
        calls += 1
        var rng = PocketTTSTorchRNG(seed: seed &+ UInt64(calls))
        return rng.normalFill(count: count, std: std)
    }
}
