// MeetingTranscriber.swift — "who said what", fully on-device, and the one speech capability
// Apple ships nothing for.
//
// A `KitDiarizer` (Streaming Sortformer) segments the clip into speaker turns, then a
// `SpeechToText` transcribes each turn's slice — no ASR word timestamps needed, the diarizer
// supplies the boundaries.
//
// **The ASR side defaults to Apple's**, which costs zero bytes of app download. That is the
// whole shape of the thing: Apple transcribes, and the 238 MB this package adds is the part
// Apple cannot do. Pointing it at a catalog model instead is a deliberate upgrade, not the
// starting point — it turns a 238 MB feature into a 3.4 GB one.
//
// ```swift
// let meeting = try await MeetingTranscriber()                       // 238 MB total
// let transcript = try await meeting.transcribe(samples: pcm16kMono)
// print(transcript.text)
// // Speaker 1 [0.3–4.1s]: With her white paint and her scarlet smokestack…
// // Speaker 2 [4.6–6.3s]: …
//
// // Opt in to a catalog ASR when Apple's locale support or determinism is not enough:
// let better = try await MeetingTranscriber(asr: KitTranscriber(catalog: "whisper-large-v3-turbo"))
// ```

import Foundation

/// One transcribed speaker turn. `speaker` is a 1-based display label in order of first
/// appearance (the diarizer's raw 0..<4 speaker index is not meaningful to a reader).
public struct MeetingTurn: Sendable, Hashable {
    public let speaker: Int
    public let startSec: Double
    public let endSec: Double
    public let text: String

    public init(speaker: Int, startSec: Double, endSec: Double, text: String) {
        self.speaker = speaker
        self.startSec = startSec
        self.endSec = endSec
        self.text = text
    }

    /// `Speaker 1 [0.3–4.1s]: …` — one transcript line.
    public var line: String {
        String(format: "Speaker %d [%.1f–%.1fs]: %@", speaker, startSec, endSec, text)
    }
}

/// A diarized transcript: the transcribed turns plus the speaker count.
public struct MeetingTranscript: Sendable {
    public let turns: [MeetingTurn]
    public let speakerCount: Int

    public init(turns: [MeetingTurn], speakerCount: Int) {
        self.turns = turns
        self.speakerCount = speakerCount
    }

    /// The whole transcript, one `Speaker N [t0–t1s]: text` line per turn.
    public var text: String { turns.map(\.line).joined(separator: "\n") }
}

/// Speaker-attributed transcription behind one `transcribe(samples:)` call, composed from any
/// diarization + any speech-to-text catalog model. Serial use (one clip at a time).
public struct MeetingTranscriber: Sendable {
    /// Widen each turn slightly so word onsets at the boundary aren't clipped.
    static let turnPadSec = 0.1
    /// Skip turns shorter than this — too short to transcribe meaningfully.
    static let minTurnSec = 0.3

    let diarizer: KitDiarizer
    let transcriber: any SpeechToText

    /// The ASR this was built with, for a transcript header — `system-speech(en-US)` or a
    /// catalog id.
    public var asrID: String { transcriber.id }

    /// Compose from an already-loaded diarizer and any transcriber.
    public init(diarizer: KitDiarizer, transcriber: any SpeechToText) {
        self.diarizer = diarizer
        self.transcriber = transcriber
    }

    /// The default: Apple transcribes, Sortformer says who was talking.
    ///
    /// Downloads the diarizer on first use (238 MB) and nothing else — the locale assets
    /// Apple's transcriber needs are the OS's, shared with every app, and do not count
    /// against this one.
    public init(
        locale: Locale = .current,
        diarizer diarizerID: String = "sortformer-diar-v2",
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        self.diarizer = try await KitDiarizer(
            catalog: diarizerID, store: store, downloadProgress: downloadProgress)
        self.transcriber = try await SystemTranscriber(locale: locale)
    }

    /// Explicit ASR — a catalog model, or anything else conforming to `SpeechToText`.
    public init(
        asr transcriber: any SpeechToText,
        diarizer diarizerID: String = "sortformer-diar-v2",
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        self.diarizer = try await KitDiarizer(
            catalog: diarizerID, store: store, downloadProgress: downloadProgress)
        self.transcriber = transcriber
    }

    /// Both by catalog id, the pre-`SystemTranscriber` shape.
    @available(*, deprecated, message: "Apple's transcriber is the default and costs no download; pass a KitTranscriber to `asr:` only when a catalog model is genuinely needed.")
    public init(
        asr asrID: String,
        diarizer diarizerID: String = "sortformer-diar-v2",
        store: ModelStore = .default,
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws {
        self.diarizer = try await KitDiarizer(
            catalog: diarizerID, store: store, downloadProgress: downloadProgress)
        self.transcriber = try await KitTranscriber(
            catalog: asrID, store: store, downloadProgress: downloadProgress)
    }

    /// Transcribe a 16 kHz mono clip into a speaker-attributed transcript. `language` nil =
    /// auto-detect (passed through to the ASR). `onTurn` streams each transcribed turn as it
    /// lands, for progressive UI.
    public func transcribe(
        samples: [Float], language: String? = nil,
        onTurn: (@Sendable (MeetingTurn) -> Void)? = nil
    ) async throws -> MeetingTranscript {
        let segments = try await diarizer.diarize(samples: samples)

        let sr = Double(KitDiarizer.sampleRate)
        let minTurn = Int(Self.minTurnSec * sr)
        var speakerLabel: [Int: Int] = [:]  // raw speaker -> display order (1-based, first-seen)
        var turns: [MeetingTurn] = []

        for seg in segments {
            let a = max(0, Int((seg.startSec - Self.turnPadSec) * sr))
            let b = min(samples.count, Int((seg.endSec + Self.turnPadSec) * sr))
            guard b - a >= minTurn else { continue }
            let text = try await transcriber.transcribe(
                samples: Array(samples[a..<b]), language: language, onPartial: nil
            ).text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let label = speakerLabel[seg.speaker] ?? (speakerLabel.count + 1)
            speakerLabel[seg.speaker] = label
            let turn = MeetingTurn(
                speaker: label, startSec: seg.startSec, endSec: seg.endSec, text: text)
            turns.append(turn)
            onTurn?(turn)
        }
        return MeetingTranscript(turns: turns, speakerCount: speakerLabel.count)
    }
}
