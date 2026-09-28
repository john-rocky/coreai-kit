// Audio8Sampler.swift — what the host keeps of the Audio8-TTS sampler. The draws themselves (top-k 50 / top-p 0.9 /
// temperature 0.7, Gumbel-max `argmax(softmax / -log(u))`, the RAS repetition rule with its (0.9, 1.0) second draw)
// run inside the `frame` graph (zoo `conversion/audio8_tts/audio8_frame.py`, spec `sampler.py`); the host supplies the
// uniform draws `u` — the randomness IS that vector, so a recorded `u` replays a choice exactly on identical logits
// (how the smoke test and the gate app compare this host with the Python engine run frame for frame) — and keeps the
// publisher's 10-token RAS window: nil before the first frame (-1s to the graph), zeros(10) after it, then rolling.

import Foundation

public enum Audio8Sampling {
    public static let semanticBegin: Int32 = 151678
    static let semanticEnd: Int32 = 155773
    public static let eos: Int32 = 151645
    public static let pad: Int32 = 151643
    public static let allowedCount = 4097          // semantic 0..4095 then eos at index 4096
    static let eosIndex = 4096
    public static let codebookSize = 4096
    public static let numCodebooks = 10
    static let rasTopP: Float = 0.9
    static let rasTemperature: Float = 1.0
    public static let rasWindow = 10

    /// The generation settings baked into the graph (the checkpoint's generation_config.json; the RAS branch's
    /// 0.9 / 1.0 are the model config's).
    public static let topK = 50
    public static let topP: Float = 0.9
    public static let temperature: Float = 0.7
    public static let maxNewTokens = 512

    /// The publisher's `previous` tensor: nil before step 0, zeros(10) after it, then rolling.
    public struct RASWindow: Sendable {
        public private(set) var values: [Int32]? = nil
        public init() {}
        public mutating func push(_ semantic: Int32) {
            if values == nil { values = [Int32](repeating: 0, count: rasWindow) }
            else { values!.removeFirst(); values!.append(semantic) }
        }
    }
}

/// Where the sampler's uniforms come from. The default is a seeded xoshiro256** stream; the smoke test and the gate
/// app inject the oracle's recorded draws to compare this host with the Python engine run choice for choice.
public protocol Audio8NoiseSource {
    /// (normal-branch draws [4097], RAS-high draws [4097]) for the semantic sample of one frame.
    mutating func slow() -> ([Float], [Float])
    /// Draws [4096] for one codebook sample.
    mutating func fast() -> [Float]
}

public struct Audio8SeededNoise: Audio8NoiseSource, Sendable {
    private var s: (UInt64, UInt64, UInt64, UInt64)

    public init(seed: UInt64) {
        // splitmix64 seeding
        var x = seed
        func next() -> UInt64 {
            x &+= 0x9E37_79B9_7F4A_7C15
            var z = x
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        s = (next(), next(), next(), next())
    }

    private mutating func nextUInt64() -> UInt64 {
        let result = ((s.1 &* 5) << 7 | (s.1 &* 5) >> 57) &* 9
        let t = s.1 << 17
        s.2 ^= s.0; s.3 ^= s.1; s.1 ^= s.2; s.0 ^= s.3
        s.2 ^= t
        s.3 = (s.3 << 45) | (s.3 >> 19)
        return result
    }

    /// Uniform in (0, 1]: the top 24 bits, never exactly 0 (a 0 draw would make -log(u) infinite).
    private mutating func uniform() -> Float {
        (Float(nextUInt64() >> 40) + 1) * (1.0 / Float(1 << 24))
    }

    private mutating func draws(_ n: Int) -> [Float] {
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n { out[i] = uniform() }
        return out
    }

    public mutating func slow() -> ([Float], [Float]) {
        (draws(Audio8Sampling.allowedCount), draws(Audio8Sampling.allowedCount))
    }

    public mutating func fast() -> [Float] { draws(Audio8Sampling.codebookSize) }
}
