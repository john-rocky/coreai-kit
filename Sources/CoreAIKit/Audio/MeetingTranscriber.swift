// MeetingTranscriber.swift — "who said what", fully on-device: the first kit API composed from
// two catalog models instead of one. A `KitDiarizer` (Streaming Sortformer) segments the clip
// into speaker turns, then the chosen `KitTranscriber` ASR transcribes each turn's audio slice —
// no ASR word timestamps needed, the diarizer supplies the turn boundaries. This is the
// coreai-audio app's device-verified "Diarize" pipeline promoted to a kit API so card snippets
// stay honest.
//
// ```swift
// let meeting = try await MeetingTranscriber(asr: "whisper-large-v3-turbo")
// let transcript = try await meeting.transcribe(samples: pcm16kMono)
// print(transcript.text)
// // Speaker 1 [0.3–4.1s]: With her white paint and her scarlet smokestack…
// // Speaker 2 [4.6–6.3s]: …
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
    let transcriber: KitTranscriber

    /// Compose from already-loaded models (mix any diarizer with any ASR).
    public init(diarizer: KitDiarizer, transcriber: KitTranscriber) {
        self.diarizer = diarizer
        self.transcriber = transcriber
    }

    /// Loads both models by their catalog ids — the ids shown on the model cards. Downloads on
    /// first use (progress via `downloadProgress`, both models), then loads from the local cache.
    public init(
        asr asrID: String = "whisper-large-v3-turbo",
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
                samples: Array(samples[a..<b]), language: language
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
