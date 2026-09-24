// QuickStart.swift — the snippet the model cards show, compiled so it stays honest. Everything
// here is public kit API: decode any audio file to 16 kHz mono, load the diarizer + the chosen
// ASR by catalog id, and get back "who said what".

import CoreAIKit
import Foundation

/// Transcribe a meeting recording into a speaker-attributed transcript, fully on-device.
/// `diarizer` is `sortformer-diar-v2` (up to 4 speakers) or `nemotron-3-diarization` (up to 8).
/// `onTurn` streams each transcribed turn as it lands.
func transcribeMeeting(
    audio url: URL,
    asr asrID: String = "whisper-large-v3-turbo",
    diarizer diarizerID: String = "sortformer-diar-v2",
    onTurn: (@Sendable (MeetingTurn) -> Void)? = nil
) async throws -> MeetingTranscript {
    let samples = try AudioFile.pcm16kMono(url)
    let meeting = try await MeetingTranscriber(asr: asrID, diarizer: diarizerID)
    return try await meeting.transcribe(samples: samples, onTurn: onTurn)
}
