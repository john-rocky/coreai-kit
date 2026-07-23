// AudioFile.swift — decode any audio file to the 16 kHz mono Float waveform every kit speech
// model consumes (`transcribe(samples:)` on Whisper/Qwen3-ASR/Parakeet, `attach(samples:)` on
// the audio-understanding models). Promoted from the AudioChat example so card snippets stay
// honest: the file→samples glue is a kit API, not "see the example".

import AVFoundation
import Foundation

/// Failures decoding an audio file into model input.
public enum AudioFileError: Error, LocalizedError {
    case undecodable(String)

    public var errorDescription: String? {
        switch self {
        case .undecodable(let name):
            return "Could not decode '\(name)' to 16 kHz mono PCM."
        }
    }
}

public enum AudioFile {
    /// Decode an audio file (anything AVFoundation reads: wav/m4a/mp3/…) to 16 kHz mono
    /// `[Float]`, resampling via `AVAudioConverter`. Security-scoped URLs (file importers,
    /// drag & drop) are handled.
    public static func pcm16kMono(_ url: URL) throws -> [Float] {
        // Balance the scoped-access call even when access wasn't needed (local URLs return false).
        guard url.startAccessingSecurityScopedResource() || true else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        defer { url.stopAccessingSecurityScopedResource() }
        guard let file = try? AVAudioFile(forReading: url) else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        let src = file.processingFormat
        guard
            let dst = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: src, to: dst),
            let srcBuf = AVAudioPCMBuffer(
                pcmFormat: src, frameCapacity: AVAudioFrameCount(file.length)),
            (try? file.read(into: srcBuf)) != nil
        else { throw AudioFileError.undecodable(url.lastPathComponent) }

        let outCapacity = AVAudioFrameCount(Double(srcBuf.frameLength) * 16000 / src.sampleRate) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: outCapacity) else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        var fed = false
        var err: NSError?
        converter.convert(to: dstBuf, error: &err) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return srcBuf
        }
        guard err == nil, let channel = dstBuf.floatChannelData else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        return Array(UnsafeBufferPointer(start: channel[0], count: Int(dstBuf.frameLength)))
    }

    /// Decode an audio file to **stereo** `[Float]` at `sampleRate` (default 44.1 kHz) —
    /// `[[left], [right]]`. Mono sources are duplicated into both channels. This is what the
    /// separation models consume (`KitSeparator.separate(_:)`), which need the full band, not
    /// the 16 kHz speech downmix above.
    public static func pcmStereo(_ url: URL, sampleRate: Double = 44100) throws -> [[Float]] {
        guard url.startAccessingSecurityScopedResource() || true else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        defer { url.stopAccessingSecurityScopedResource() }
        guard let file = try? AVAudioFile(forReading: url) else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        let src = file.processingFormat
        guard
            let dst = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 2,
                interleaved: false),
            let converter = AVAudioConverter(from: src, to: dst),
            let srcBuf = AVAudioPCMBuffer(
                pcmFormat: src, frameCapacity: AVAudioFrameCount(file.length)),
            (try? file.read(into: srcBuf)) != nil
        else { throw AudioFileError.undecodable(url.lastPathComponent) }

        let outCapacity = AVAudioFrameCount(Double(srcBuf.frameLength) * sampleRate / src.sampleRate) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: outCapacity) else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        var fed = false
        var err: NSError?
        converter.convert(to: dstBuf, error: &err) { _, status in
            if fed {
                status.pointee = .noDataNow
                return nil
            }
            fed = true
            status.pointee = .haveData
            return srcBuf
        }
        guard err == nil, let channels = dstBuf.floatChannelData else {
            throw AudioFileError.undecodable(url.lastPathComponent)
        }
        let n = Int(dstBuf.frameLength)
        let left = Array(UnsafeBufferPointer(start: channels[0], count: n))
        let right = dstBuf.format.channelCount > 1
            ? Array(UnsafeBufferPointer(start: channels[1], count: n)) : left
        return [left, right]
    }

    /// Deterministic white noise at 16 kHz — a demo clip so an app or CLI always has something
    /// to transcribe/ask about without shipping an audio file (models answer "hissing sound").
    public static func demoNoise(seconds: Int = 4) -> [Float] {
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> Float {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Float(Double(state >> 11) / Double(1 << 53)) * 2 - 1
        }
        return (0..<(16000 * seconds)).map { _ in next() * 0.9 }
    }
}
