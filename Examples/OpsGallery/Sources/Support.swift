// Support.swift — sample inputs (all generated in code: the repo ships zero binary
// assets), a WAV writer for the audio results, an orientation-baking image loader,
// and the shared download-progress hub fed by `CoreAI.onDownload`.

import AVFoundation
import CoreAIOps
import CoreGraphics
import CoreImage
import Foundation
import Observation

enum Samples {
    /// One memo that gives every text op something to show: a summary, action items,
    /// PII to redact, entities to find.
    static let article = """
        Team meeting notes — Dana Suzuki (dana@example.com) asked us to ship 12 \
        AlphaWidgets to the Osaka office by Friday, and to schedule a follow-up call \
        next Tuesday to review the launch plan with the Kyoto team.
        """

    static let speakLine = "CoreAI kit turns one line of Swift into speech."
    static let composePrompt = "warm lo-fi hip hop loop, vinyl crackle, 90 BPM"

    static let searchQuery = "What should I plan for the trip?"
    static let searchDocuments = [
        "Book flights to Sapporo for the snow festival and reserve a hotel near Odori Park.",
        "The quarterly report needs the updated revenue table before Thursday's review.",
        "Try the new ramen place by the station — open until 2am on weekends.",
        "Renew the Apple Developer membership before the certificate expires.",
    ]

    /// Trend + seasonality, so the forecast has a visible shape to continue.
    static let series: [Float] = (0..<96).map { step in
        10 + 0.06 * Float(step) + 3 * sin(Float(step) / 6)
    }

    /// A generated 512² test scene (sky, sun, hills, house) — enough for caption /
    /// depth / upscale to produce something without picking a photo.
    static func image() -> CGImage? {
        let side = 512
        guard
            let ctx = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        func fill(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ rect: CGRect) {
            ctx.setFillColor(CGColor(red: r, green: g, blue: b, alpha: 1))
            ctx.fill(rect)
        }
        fill(0.55, 0.75, 0.95, CGRect(x: 0, y: 0, width: 512, height: 512))  // sky
        ctx.setFillColor(CGColor(red: 1, green: 0.85, blue: 0.3, alpha: 1))  // sun
        ctx.fillEllipse(in: CGRect(x: 380, y: 380, width: 90, height: 90))
        fill(0.35, 0.65, 0.35, CGRect(x: 0, y: 0, width: 512, height: 180))  // ground
        fill(0.75, 0.45, 0.3, CGRect(x: 120, y: 120, width: 160, height: 120))  // house
        ctx.setFillColor(CGColor(red: 0.5, green: 0.25, blue: 0.2, alpha: 1))  // roof
        ctx.beginPath()
        ctx.move(to: CGPoint(x: 100, y: 240))
        ctx.addLine(to: CGPoint(x: 200, y: 320))
        ctx.addLine(to: CGPoint(x: 300, y: 240))
        ctx.closePath()
        ctx.fillPath()
        fill(0.3, 0.5, 0.3, CGRect(x: 360, y: 120, width: 40, height: 130))  // tree
        return ctx.makeImage()
    }
}

/// Loads an image file with its EXIF orientation baked into the bitmap, so display
/// and the CGImage ops agree on what "up" is.
func uprightImage(at url: URL) throws -> CGImage {
    let loaded = try ImageFile.load(url)
    guard loaded.orientation != .up else { return loaded.cgImage }
    let oriented = CIImage(cgImage: loaded.cgImage).oriented(loaded.orientation)
    return CIContext().createCGImage(oriented, from: oriented.extent) ?? loaded.cgImage
}

/// 16-bit PCM WAV into the temp directory; `interleaved` holds `channelCount` channels.
func writeWAV(
    interleaved: [Float], sampleRate: Int, channelCount: Int, name: String
) throws -> URL {
    var data = Data()
    func tag(_ s: String) { data.append(contentsOf: Array(s.utf8)) }
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let pcm = interleaved.map { Int16((max(-1, min(1, $0)) * 32767).rounded()) }
    let byteRate = sampleRate * channelCount * 2
    tag("RIFF"); u32(UInt32(36 + pcm.count * 2)); tag("WAVE")
    tag("fmt "); u32(16); u16(1); u16(UInt16(channelCount))
    u32(UInt32(sampleRate)); u32(UInt32(byteRate)); u16(UInt16(channelCount * 2)); u16(16)
    tag("data"); u32(UInt32(pcm.count * 2))
    pcm.withUnsafeBytes { data.append(contentsOf: $0) }
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("opsgallery-\(name)-\(UUID().uuidString).wav")
    try data.write(to: url)
    return url
}

/// Same, from per-channel buffers (`[[left], [right]]` — the `Stems` layout).
func writeWAV(channels: [[Float]], sampleRate: Int, name: String) throws -> URL {
    guard let first = channels.first else {
        throw GalleryError.missingInput("No audio channels to write.")
    }
    var interleaved = [Float]()
    interleaved.reserveCapacity(first.count * channels.count)
    for i in 0..<first.count {
        for channel in channels { interleaved.append(i < channel.count ? channel[i] : 0) }
    }
    return try writeWAV(
        interleaved: interleaved, sampleRate: sampleRate, channelCount: channels.count,
        name: name)
}

/// One shared player — tapping play on another result just switches the clip.
@MainActor @Observable
final class AudioPlayer {
    static let shared = AudioPlayer()

    private var player: AVAudioPlayer?
    private(set) var playingURL: URL?

    func toggle(_ url: URL) {
        if playingURL == url, player?.isPlaying == true {
            player?.stop()
            playingURL = nil
            return
        }
        player = try? AVAudioPlayer(contentsOf: url)
        player?.play()
        playingURL = url
    }
}

/// The process-wide `CoreAI.onDownload` events, surfaced to SwiftUI. `OpsGalleryApp`
/// installs the hook once at launch.
@MainActor @Observable
final class DownloadHub {
    static let shared = DownloadHub()

    var latest: DownloadProgress?

    /// A download is on screen while the last event is short of 100%.
    var isActive: Bool { if let latest { latest.fraction < 1 } else { false } }
}
