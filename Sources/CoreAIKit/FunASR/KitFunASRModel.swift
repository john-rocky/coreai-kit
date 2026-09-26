// KitFunASRModel.swift — a Core AI Fun-ASR-Nano-2512 transcription model (FunAudioLLM): SAN-M audio
// encoder + adaptor and a Qwen3-0.6B decoder, for Chinese (and Chinese dialects), English and
// Japanese.
//
// ```swift
// let asr = try await KitFunASRModel(model: .funASRNano2512)
// let result = try await asr.transcribe(samples: pcm16kMono)                   // -> Transcription
// let hinted = try await asr.transcribe(samples: pcm16kMono, hotwords: ["开放时间"])
// ```
//
// funasr's three options ride the prompt: `hotwords` (a context list the model may prefer),
// `language` (a hint such as "中文", or zh / en / ja / ko; nil = auto) and `itn` (inverse text
// normalization, on by default). Audio longer than 30 s is cut into 30 s windows, transcribed in
// order and joined with a space — no VAD, so a word on a window boundary can be split. The model
// emits no language tag: `Transcription.language` is empty.

import CoreAIKitVision
import Foundation

/// A downloadable Fun-ASR model = a decoder bundle + its paired audio encoder, addressed as paths
/// inside one HF repo, plus the architecture geometry.
@available(macOS 27, iOS 27, *)
public struct FunASRModelID: Sendable, Hashable {
    public let decoder: ModelID
    public let encoder: ModelID
    public let arch: FunASRArchitecture

    public init(decoder: ModelID, encoder: ModelID, arch: FunASRArchitecture) {
        self.decoder = decoder
        self.encoder = encoder
        self.arch = arch
    }

    /// Fun-ASR-Nano-2512: decoder int8 linears with the fp16 tied head/embedding, encoder fp16
    /// weights with fp32 compute. ≤ 30 s per window.
    public static let funASRNano2512 = FunASRModelID(
        decoder: ModelID(
            "mlboydaisuke/Fun-ASR-Nano-2512-CoreAI",
            path: "gpu-pipelined/funasr_nano_2512_decode_int8lin_n63_s1"),
        encoder: ModelID(
            "mlboydaisuke/Fun-ASR-Nano-2512-CoreAI",
            path: "gpu-pipelined/funasr_nano_audio_encoder_fp16w32_l500"),
        arch: .funASRNano2512)

    /// Presets by catalog id. The catalog carries one variant path per entry, while a Fun-ASR
    /// model is a decoder + paired audio encoder + geometry — so every `asr` catalog entry driven
    /// by `KitFunASRModel` pairs with a preset here, keyed by the id its card shows.
    static let byCatalogID: [String: FunASRModelID] = [
        "fun-asr-nano-2512": .funASRNano2512
    ]

    /// A copy with both sub-bundle ids pinned to a Hub revision (nil = unchanged), so a catalog
    /// entry's pin covers the decoder and its paired encoder alike.
    func pinned(_ revision: String?) -> FunASRModelID {
        guard let revision else { return self }
        return FunASRModelID(
            decoder: decoder.pinned(revision), encoder: encoder.pinned(revision), arch: arch)
    }
}

/// A Core AI Fun-ASR-Nano bundle behind one `transcribe(samples:)` call. Serial use (one
/// transcription at a time).
@available(macOS 27, iOS 27, *)
public struct KitFunASRModel: Sendable {
    let runtime: FunASRRuntime

    /// Loads a Fun-ASR model by its catalog id — the id shown on the model's card:
    ///
    /// ```swift
    /// let asr = try await KitFunASRModel(catalog: "fun-asr-nano-2512")
    /// ```
    ///
    /// Resolves the decoder/encoder pair and geometry from the preset table, and the platform
    /// availability + revision pin from the live catalog (built-in snapshot offline), then
    /// downloads if needed.
    public init(
        catalog id: String,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .asr)
        guard let model = FunASRModelID.byCatalogID[entry.id] else {
            throw CoreAIKitError.modelNotInCatalog(id: id)
        }
        try await self.init(
            model: model.pinned(entry.revision), store: store, downloadProgress: downloadProgress)
    }

    /// Downloads the decoder + encoder bundles from the Hub (if needed) and loads them.
    public init(
        model: FunASRModelID = .funASRNano2512,
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let decoderURL = try await store.download(model.decoder, progress: downloadProgress)
        let encoderRoot = try await store.download(model.encoder, progress: downloadProgress)
        try await self.init(
            decoderBundleAt: decoderURL, encoderModelAt: encoderRoot, arch: model.arch)
    }

    /// Loads a local decoder bundle directory + a local encoder graph (the `.aimodel`/`.aimodelc`
    /// itself, or the bundle directory holding it).
    public init(
        decoderBundleAt decoderURL: URL, encoderModelAt encoderURL: URL,
        arch: FunASRArchitecture = .funASRNano2512,
        encoderComputeUnits: GraphModel.ComputeUnits = .gpu
    ) async throws {
        self.runtime = try await FunASRRuntime(
            decoderBundleAt: decoderURL, encoderModelAt: encoderURL, arch: arch,
            encoderComputeUnits: encoderComputeUnits)
    }

    public init(runtime: FunASRRuntime) {
        self.runtime = runtime
    }

    /// Transcribe a raw 16 kHz mono waveform of any length (30 s windows, joined with a space).
    /// Pass `onPartial` to stream the running transcript.
    public func transcribe(
        samples: [Float], hotwords: [String] = [], language: String? = nil, itn: Bool = true,
        onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> Transcription {
        let windows = try await transcribeWindows(
            samples: samples, hotwords: hotwords, language: language, itn: itn, onPartial: onPartial)
        return Transcription(language: "", text: Self.join(windows.map(\.text)))
    }

    /// The window loop behind `transcribe(samples:)`, one result per window that holds at least one
    /// fbank frame (25 ms).
    func transcribeWindows(
        samples: [Float], hotwords: [String] = [], language: String? = nil, itn: Bool = true,
        maxTokens: Int = 512, onPartial: (@Sendable (String) -> Void)? = nil
    ) async throws -> [FunASRRuntime.Decoded] {
        var results: [FunASRRuntime.Decoded] = []
        var start = 0
        repeat {
            let end = min(start + runtime.arch.windowSamples, samples.count)
            let window = Array(samples[start..<end])
            start = end
            guard FunASRFbankPreprocessor.lfrFrames(samples: window.count) > 0 else { continue }
            let timing = try await runtime.attachTimed(samples: window)
            let before = Self.join(results.map(\.text))
            var sink: (@Sendable (String) -> Void)?
            if let onPartial {
                sink = { partial in onPartial(Self.join([before, partial])) }
            }
            var decoded = try await runtime.decode(
                hotwords: hotwords, language: language, itn: itn, maxTokens: maxTokens,
                onPartial: sink)
            decoded.frontendMs = timing.frontendMs
            decoded.encoderMs = timing.encoderMs
            results.append(decoded)
        } while start < samples.count
        return results
    }

    public func detachAudio() { runtime.detach() }

    private static func join(_ texts: [String]) -> String {
        texts.filter { !$0.isEmpty }.joined(separator: " ")
    }
}
