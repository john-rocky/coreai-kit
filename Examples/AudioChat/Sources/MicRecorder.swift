// MicRecorder — capture microphone audio to a temp file with AVAudioRecorder, then decode it to
// 16 kHz mono Float via AudioLoader. AVAudioRecorder avoids AVAudioEngine's live-tap / render-queue
// dispatch assertions (which crash here). Permission is requested before any session work.

import AVFoundation
import Foundation

final class MicRecorder: NSObject, @unchecked Sendable {
    enum MicError: LocalizedError {
        case denied, failed
        var errorDescription: String? {
            switch self {
            case .denied: return "Microphone access was denied (enable it in Settings)."
            case .failed: return "Could not start recording."
            }
        }
    }

    private var recorder: AVAudioRecorder?
    private var fileURL: URL?

    var isRecording: Bool { recorder?.isRecording ?? false }

    /// Request permission (first time prompts), then start recording to a temp WAV.
    func start(_ completion: @escaping @Sendable (Error?) -> Void) {
        let begin: @Sendable (Bool) -> Void = { granted in
            DispatchQueue.main.async {
                guard granted else { completion(MicError.denied); return }
                do {
                    #if os(iOS)
                        let session = AVAudioSession.sharedInstance()
                        try session.setCategory(.playAndRecord, mode: .default)
                        try session.setActive(true)
                    #endif
                    let url = FileManager.default.temporaryDirectory
                        .appendingPathComponent("miccap.wav")
                    try? FileManager.default.removeItem(at: url)
                    let settings: [String: Any] = [
                        AVFormatIDKey: kAudioFormatLinearPCM,
                        AVSampleRateKey: 16000.0,
                        AVNumberOfChannelsKey: 1,
                        AVLinearPCMBitDepthKey: 16,
                        AVLinearPCMIsFloatKey: false,
                    ]
                    let rec = try AVAudioRecorder(url: url, settings: settings)
                    guard rec.record() else { completion(MicError.failed); return }
                    self.recorder = rec
                    self.fileURL = url
                    completion(nil)
                } catch {
                    completion(error)
                }
            }
        }
        #if os(iOS)
            AVAudioApplication.requestRecordPermission { begin($0) }
        #else
            begin(true)
        #endif
    }

    /// Stop recording and return the captured 16 kHz mono samples.
    func stop(_ completion: @escaping @Sendable ([Float]) -> Void) {
        DispatchQueue.main.async {
            self.recorder?.stop()
            self.recorder = nil
            #if os(iOS)
                try? AVAudioSession.sharedInstance().setActive(false)
            #endif
            let pcm = self.fileURL.flatMap { AudioLoader.load16kMono($0) } ?? []
            completion(pcm)
        }
    }
}
