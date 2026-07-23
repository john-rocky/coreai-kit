// KitDialogue.swift — multi-speaker / dialogue text-to-speech (VibeVoice-Realtime-0.5B).
//
// The zoo's other TTS models speak one utterance in one voice. This one performs a *script*: each
// "Speaker N:" turn is generated from its own voice preset and the turns are concatenated, so a
// two-person conversation or a podcast segment renders in a single call — on device, no network.
//
//   let dialogue = try await KitDialogue(catalog: "vibevoice-realtime-0.5b")
//   let audio = try await dialogue.perform("""
//       Speaker 1: Did you know this runs entirely on the phone?
//       Speaker 2: No cloud at all? That is wild.
//       """)
//
// Pair it with `KitDiarizer` for the round trip: generate a conversation, then have the zoo tell
// you who spoke when.

import CoreAIKitVision
import Foundation

/// One parsed line of a script.
public struct DialogueTurn: Sendable {
    /// 1-based speaker number as written in the script (`Speaker 2:` → 2).
    public let speaker: Int
    public let text: String
    /// The voice preset this turn was rendered with.
    public let voice: String

    public init(speaker: Int, text: String, voice: String) {
        self.speaker = speaker
        self.text = text
        self.voice = voice
    }
}

/// Multi-speaker TTS by catalog id.
public struct KitDialogue: Sendable {
    private let engine: VibeVoiceTTS
    /// The catalog id this was loaded from.
    public let catalogID: String
    /// Voice presets available in this bundle (25: EN/ZH and more, `name` + `language`).
    public let voices: [VibeVoiceVoice]

    /// Loads the dialogue model by catalog id. First use downloads the five graph bundles for this
    /// platform plus the host assets (voice prefill caches, glue, tokenizer, embedding table).
    public init(
        catalog id: String = "vibevoice-realtime-0.5b",
        store: ModelStore = .default,
        computeUnits: GraphModel.ComputeUnits = .gpu,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        let entry = try await ModelCatalog.entry(forID: id, expecting: .tts)
        guard let variant = entry.variant else { throw CoreAIKitError.modelNotInCatalog(id: id) }
        let bundles = try await store.download(
            entry.modelID(path: variant.path), progress: downloadProgress)
        let glue = try await store.download(
            entry.modelID(path: "coreai_host/glue"), progress: downloadProgress)
        let voicesDir = try await store.download(
            entry.modelID(path: "coreai_host/voices"), progress: downloadProgress)
        let embedDir = try await store.download(
            entry.modelID(path: "coreai_host/embed"), progress: downloadProgress)
        let embed = embedDir.appendingPathComponent("embed_tokens_fp16.bin")

        let paths = VibeVoicePaths.inBundleDir(
            bundles, glueDir: glue, voicesDir: voicesDir, embedTokens: embed)
        self.engine = try await VibeVoiceTTS(paths: paths, computeUnits: computeUnits)
        self.catalogID = entry.id
        self.voices = await engine.voices
    }

    /// Loads from already-downloaded assets.
    public init(paths: VibeVoicePaths, computeUnits: GraphModel.ComputeUnits = .gpu) async throws {
        self.engine = try await VibeVoiceTTS(paths: paths, computeUnits: computeUnits)
        self.catalogID = "vibevoice-realtime-0.5b"
        self.voices = await engine.voices
    }

    /// The default voice order used when a script does not name voices: the packaged English ones
    /// first (so `Speaker 1` / `Speaker 2` sound distinct), then whatever else is available.
    public var defaultVoices: [String] {
        let en = voices.filter { $0.language == "en" }.map(\.name)
        return en.isEmpty ? voices.map(\.name) : en
    }

    /// Speaks one utterance. Text longer than the decoder's ~8.5 s window is split on sentence
    /// boundaries and concatenated.
    public func speak(
        _ text: String, voice: String? = nil, seed: UInt64 = 0x5EED_1234
    ) async throws -> SpokenAudio {
        let name = voice ?? defaultVoices.first ?? ""
        let tagged = text.lowercased().hasPrefix("speaker") ? text : "Speaker 1: \(text)"
        return try await engine.synthesize(tagged, voice: name, seed: seed)
    }

    /// Speaks with an explicit per-frame DDPM noise table instead of the seeded generator, so a
    /// caller can reproduce a published reference exactly. Verification hook — normal use wants
    /// `speak(_:voice:seed:)`.
    public func speak(
        _ text: String, voice: String, noiseFrames: [[Float]]
    ) async throws -> SpokenAudio {
        try await engine.synthesize(text, voice: voice, noiseFrames: noiseFrames)
    }

    /// Performs a `Speaker N:`-style script: one voice per speaker, turns concatenated in order.
    /// Lines without a `Speaker N:` prefix continue the current speaker (speaker 1 to begin with).
    ///
    /// `voices` overrides the per-speaker voice assignment (index 0 = Speaker 1).
    public func perform(
        _ script: String, voices assigned: [String]? = nil, gapSeconds: Double = 0.25,
        seed: UInt64 = 0x5EED_1234
    ) async throws -> (audio: SpokenAudio, turns: [DialogueTurn]) {
        let pool = assigned ?? defaultVoices
        guard !pool.isEmpty else { throw VibeVoiceError.voiceNotFound("(no voices)") }
        let parsed = Self.parse(script, voices: pool)
        var samples: [Float] = []
        let gap = [Float](repeating: 0, count: Int(gapSeconds * 24_000))
        for (i, turn) in parsed.enumerated() {
            // Feed the model the tagged line (that is the form it was trained on); `turn.text`
            // stays clean for callers that want to display or align the script.
            let piece = try await engine.synthesize(
                "Speaker \(turn.speaker): \(turn.text)", voice: turn.voice, seed: seed &+ UInt64(i))
            if !samples.isEmpty { samples.append(contentsOf: gap) }
            samples.append(contentsOf: piece.samples)
        }
        return (SpokenAudio(samples: samples, sampleRate: 24_000), parsed)
    }

    /// Splits a script into turns. `Speaker 1:` / `speaker 2 :` / `Speaker1:` all parse; the number
    /// selects the voice (wrapping if the script has more speakers than voices).
    static func parse(_ script: String, voices: [String]) -> [DialogueTurn] {
        var turns: [DialogueTurn] = []
        var speaker = 1
        var buffer = ""

        func flush() {
            let text = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            buffer = ""
            guard !text.isEmpty else { return }
            turns.append(DialogueTurn(
                speaker: speaker, text: text, voice: voices[(speaker - 1) % voices.count]))
        }

        for rawLine in script.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            if let (n, rest) = Self.speakerPrefix(line) {
                flush()
                speaker = n
                buffer = rest
            } else {
                buffer += buffer.isEmpty ? line : " " + line
            }
        }
        flush()
        return turns
    }

    /// `"Speaker 2: hello"` → `(2, "hello")`, else nil.
    private static func speakerPrefix(_ line: String) -> (Int, String)? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.lowercased().hasPrefix("speaker") else { return nil }
        var rest = Substring(trimmed.dropFirst("speaker".count))
        let digits = rest.prefix(while: { $0.isNumber || $0.isWhitespace })
        guard let n = Int(digits.trimmingCharacters(in: .whitespaces)) else { return nil }
        rest = rest.dropFirst(digits.count)
        guard rest.trimmingCharacters(in: .whitespaces).hasPrefix(":") else { return nil }
        let body = rest.drop(while: { $0 != ":" }).dropFirst()
        return (n, String(body).trimmingCharacters(in: .whitespaces))
    }
}
