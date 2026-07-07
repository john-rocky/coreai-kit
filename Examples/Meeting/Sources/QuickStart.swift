// QuickStart.swift — the snippet the model cards show, compiled so it stays honest. Everything
// here is public kit API: decode any audio file to 16 kHz mono, load the diarizer + the chosen
// ASR by catalog id, and get back "who said what".

import CoreAIKit
import Foundation

/// Transcribe a meeting recording into a speaker-attributed transcript, fully on-device.
/// `onTurn` streams each transcribed turn as it lands.
func transcribeMeeting(
    audio url: URL,
    asr asrID: String = "whisper-large-v3-turbo",
    onTurn: (@Sendable (MeetingTurn) -> Void)? = nil
) async throws -> MeetingTranscript {
    let samples = try AudioFile.pcm16kMono(url)
    let meeting = try await MeetingTranscriber(asr: asrID)
    return try await meeting.transcribe(samples: samples, onTurn: onTurn)
}
