// AudioFile.swift — decode any audio file to the 16 kHz mono Float waveform every kit speech
// model consumes (`transcribe(samples:)` on Whisper/Qwen3-ASR/Parakeet, `attach(samples:)` on
// the audio-understanding models), and write the `[Float]` the generative audio types return
// back out as a WAV. Promoted from the AudioChat example so card snippets stay honest: the
// file→samples glue is a kit API, not "see the example". The writer arrived the same way —
// four examples had each grown their own RIFF header.

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

    // MARK: - writing

    /// Encode mono `[Float]` as a 16-bit PCM WAV. This is the container for what every
    /// generative audio type hands back — `KokoroTTS`, `DotsTTS`, `VoxCPMTTS`, `VibeVoiceTTS`,
    /// `PocketTTS` return `[Float]` at their own `sampleRate`, with no file behind it.
    /// Samples are clamped to ±1 before rounding, so a model that overshoots clips rather
    /// than wrapping to the opposite rail.
    public static func wav16(_ samples: [Float], sampleRate: Int) -> Data {
        wav16(channels: [samples], sampleRate: sampleRate)
    }

    /// Same, from per-channel buffers — `[[left], [right]]`, what `KitSeparator` returns and
    /// what `pcmStereo(_:)` above reads back. Channels are interleaved here; shorter channels
    /// are zero-filled rather than truncating the longer ones.
    public static func wav16(channels: [[Float]], sampleRate: Int) -> Data {
        let channelCount = max(channels.count, 1)
        let frames = channels.map(\.count).max() ?? 0
        var pcm = [Int16](repeating: 0, count: frames * channelCount)
        for (c, channel) in channels.enumerated() {
            for i in 0..<channel.count {
                pcm[i * channelCount + c] = Int16((max(-1, min(1, channel[i])) * 32767).rounded())
            }
        }

        var data = Data(capacity: 44 + pcm.count * 2)
        func tag(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { data.append(contentsOf: $0) } }
        tag("RIFF"); u32(36 + pcm.count * 2); tag("WAVE")
        tag("fmt "); u32(16); u16(1); u16(channelCount)
        u32(sampleRate); u32(sampleRate * channelCount * 2); u16(channelCount * 2); u16(16)
        tag("data"); u32(pcm.count * 2)
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    /// Write mono `[Float]` to `url` as a 16-bit PCM WAV.
    public static func writeWAV(_ samples: [Float], sampleRate: Int, to url: URL) throws {
        try wav16(samples, sampleRate: sampleRate).write(to: url)
    }

    /// Write per-channel buffers to `url` as an interleaved 16-bit PCM WAV.
    public static func writeWAV(channels: [[Float]], sampleRate: Int, to url: URL) throws {
        try wav16(channels: channels, sampleRate: sampleRate).write(to: url)
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
