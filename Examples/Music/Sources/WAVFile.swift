// WAVFile.swift — minimal 16-bit PCM mono RIFF writer, shared by the GUI (playback file) and
// the CLI (`--output out.wav`). Kit hands back raw samples; the container is app territory.

import Foundation

enum WAVFile {
    /// Planar stereo (channel-major [L…R…], the VAE's layout) → interleaved 16-bit PCM.
    static func dataPlanarStereo(samples: [Float], sampleRate: Int) -> Data {
        let n = samples.count / 2
        var interleaved = [Float](repeating: 0, count: samples.count)
        for i in 0..<n {
            interleaved[2 * i] = samples[i]
            interleaved[2 * i + 1] = samples[n + i]
        }
        return data(samples: interleaved, sampleRate: sampleRate, channels: 2)
    }

    static func data(samples: [Float], sampleRate: Int, channels: Int = 1) -> Data {
        var pcm = Data(capacity: samples.count * 2)
        for s in samples {
            let v = Int16(max(-1, min(1, s)) * 32767)
            withUnsafeBytes(of: v.littleEndian) { pcm.append(contentsOf: $0) }
        }
        var data = Data()
        func chunk(_ tag: String, _ body: Data) {
            data.append(tag.data(using: .ascii)!)
            withUnsafeBytes(of: UInt32(body.count).littleEndian) { data.append(contentsOf: $0) }
            data.append(body)
        }
        var fmt = Data()
        func put<T: FixedWidthInteger>(_ v: T) {
            withUnsafeBytes(of: v.littleEndian) { fmt.append(contentsOf: $0) }
        }
        put(UInt16(1))                          // PCM
        put(UInt16(channels))
        put(UInt32(sampleRate))
        put(UInt32(sampleRate * 2 * channels))  // byte rate
        put(UInt16(2 * channels))               // block align
        put(UInt16(16))                         // bits per sample
        data.append("RIFF".data(using: .ascii)!)
        withUnsafeBytes(of: UInt32(4 + 8 + fmt.count + 8 + pcm.count).littleEndian) {
            data.append(contentsOf: $0)
        }
        data.append("WAVE".data(using: .ascii)!)
        chunk("fmt ", fmt)
        chunk("data", pcm)
        return data
    }

    static func write(samples: [Float], sampleRate: Int, channels: Int = 1, to url: URL) throws {
        if channels == 2 {
            try dataPlanarStereo(samples: samples, sampleRate: sampleRate).write(to: url)
        } else {
            try data(samples: samples, sampleRate: sampleRate).write(to: url)
        }
    }
}
