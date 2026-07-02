// MicRecorder.swift — capture microphone audio and hand back the 16 kHz mono Float waveform the
// kit speech models consume. Records to a temp file with AVAudioRecorder (avoids AVAudioEngine's
// live-tap / render-queue dispatch assertions, which crash in this use), then decodes it via
// `AudioFile.pcm16kMono`. Promoted from the AudioChat example (principle: capture engines that
// feed a model are kit APIs; the record button and permission prompt chrome stay in the app —
// iOS apps need `NSMicrophoneUsageDescription` in Info.plist).

import AVFoundation
import Foundation

/// One start/stop microphone capture at a time; not thread-safe across concurrent recordings.
///
/// ```swift
/// let recorder = MicRecorder()
/// try await recorder.start()          // first use prompts for mic permission
/// // … speak …
/// let samples = recorder.stop()       // [Float] @ 16 kHz mono
/// ```
public final class MicRecorder: NSObject, @unchecked Sendable {
    public enum MicError: LocalizedError {
        case denied, failed

        public var errorDescription: String? {
            switch self {
            case .denied: return "Microphone access was denied (enable it in Settings)."
            case .failed: return "Could not start recording."
            }
        }
    }

    private var recorder: AVAudioRecorder?
    private var fileURL: URL?

    public override init() { super.init() }

    public var isRecording: Bool { recorder?.isRecording ?? false }

    /// Requests permission (first time prompts), then starts recording to a temp WAV.
    public func start() async throws {
        #if os(iOS)
            guard await AVAudioApplication.requestRecordPermission() else {
                throw MicError.denied
            }
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default)
            try session.setActive(true)
        #endif
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreaikit-miccap.wav")
        try? FileManager.default.removeItem(at: url)
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
        ]
        let rec = try AVAudioRecorder(url: url, settings: settings)
        guard rec.record() else { throw MicError.failed }
        recorder = rec
        fileURL = url
    }

    /// Stops recording and returns the captured 16 kHz mono samples (empty if nothing captured).
    public func stop() -> [Float] {
        stopFile().flatMap { try? AudioFile.pcm16kMono($0) } ?? []
    }

    /// Stops recording and returns the clip as a temp WAV URL (nil if nothing was captured) —
    /// for handing to file-shaped APIs. The recorder writes a WAV natively, so this is the
    /// primitive; `stop()` is its decoded convenience. The next `start()` overwrites the file.
    public func stopFile() -> URL? {
        recorder?.stop()
        recorder = nil
        #if os(iOS)
            try? AVAudioSession.sharedInstance().setActive(false)
        #endif
        return fileURL
    }
}
