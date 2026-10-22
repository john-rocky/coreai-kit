// WatchEffects — what an Effect rule does when it fires: an animation drawn by the app over the
// place the model names, and a short sound. The place is one of nine cells of the picture (3×3,
// numbered 1–9 left to right, top to bottom) when the launch asks the model for it (`-cells`), the
// whole picture otherwise. Three effects, each about 3 s, drawn every frame with TimelineView and
// Canvas — no image, no symbol, no words:
//
//   ripple  rings spread out from the place's middle (water)
//   sweep   a band of light runs across the place, then its edges are traced (a car)
//   rays    rays of light burst out from the place's middle (bells)
//
// Each plays its sound (Sounds/<id>.m4a: water, horn, bells) with AVAudioPlayer. The sound files are
// not in the repository: tools/fetch_examples.sh downloads and cuts them into Sounds/, which the app
// bundles; a build without one plays the system chime instead and the record says `fallback`.
//
// `-grid 1` burns the 3×3 cells and their numbers into the frame the model gets (WatchGrid), so the
// model can read the number off the picture; the screen does not show the grid.

import AVFoundation
import CoreAIKit
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import SwiftUI

/// Where the model says a rule's thing is: how the app asks (`-cells`), and the cell's place on the screen.
enum WatchCells {
    /// `-cells none|batch|ask`.
    enum Form: String {
        /// Not asked (the default): the prompt is the plain yes / no batch and an effect plays over the whole picture.
        case off = "none"
        /// In the same prompt: each line answered yes also carries the cell's number.
        case batch
        /// A second question about the same frame, for each rule that fires.
        case ask
    }

    /// The part of the picture's box an effect plays over — cell 1–9 of what the model saw, or all of it for no
    /// cell — in points and in units of the box. The camera and a video fill the box and the model gets the same
    /// crop, so the cells are the box's thirds; an example photo is fitted into the box whole (`photo` = its upright
    /// size), so they are the fitted photo's thirds.
    static func place(_ cell: Int?, box: CGSize, photo: CGSize?) -> (points: CGRect, unit: CGRect) {
        var content = CGRect(origin: .zero, size: box)
        if let photo, photo.width > 0, photo.height > 0, box.width > 0, box.height > 0 {
            let fit = min(box.width / photo.width, box.height / photo.height)
            let size = CGSize(width: photo.width * fit, height: photo.height * fit)
            content = CGRect(
                x: (box.width - size.width) / 2, y: (box.height - size.height) / 2, width: size.width, height: size.height)
        }
        var rect = content
        if let cell, (1...9).contains(cell) {
            let width = content.width / 3
            let height = content.height / 3
            rect = CGRect(
                x: content.minX + CGFloat((cell - 1) % 3) * width, y: content.minY + CGFloat((cell - 1) / 3) * height,
                width: width, height: height)
        }
        guard box.width > 0, box.height > 0 else { return (rect, CGRect(x: 0, y: 0, width: 1, height: 1)) }
        let unit = CGRect(
            x: rect.minX / box.width, y: rect.minY / box.height, width: rect.width / box.width,
            height: rect.height / box.height)
        return (rect, unit)
    }
}

/// `-grid 1`: the frame the model gets, upright, with the 3×3 cells drawn on it. Every size is a fraction of the
/// frame's width W or height H, rounded halves-to-even, so the same drawing fits any crop (and matches gridded frames
/// drawn the same way on a Mac): each inner edge at round(W·k/3) / round(H·k/3) a black band round(0.008 W) px wide
/// with a white band round(0.004 W) px wide on its middle (the vertical white bands drawn again last, over the
/// crossings); in each cell's top-left corner, round(0.012 W) px in from the bands, a black box with the cell's
/// number, white, in the system font bold at round(0.085 H) px, round(0.008 W) px of black around the digit — across,
/// from its origin to its advance, and up and down, around its ink.
enum WatchGrid {
    static let black = CGColor(srgbRed: 0, green: 0, blue: 0, alpha: 1)
    static let white = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 1)

    /// The drawing, for the record.
    static var style: [String: Any] {
        [
            "edges": "x = round(W*1/3), round(W*2/3); y = round(H*1/3), round(H*2/3)",
            "edge_band": "black round(0.008*W) px, a white round(0.004*W) px band on its middle",
            "label": "top-left of each cell, round(0.012*W) px in: a black box round(0.008*W) px around the white digit, system font bold round(0.085*H) px",
        ]
    }

    private static func round(_ value: Double) -> Int { Int(value.rounded(.toNearestOrEven)) }

    /// The image turned upright, then the cells and their numbers on it.
    static func burn(_ image: CGImage, orientation: CGImagePropertyOrientation) -> CGImage {
        let upright = image.upright(orientation)
        let w = upright.width
        let h = upright.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let context = CGContext(
                data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return upright }
        context.draw(upright, in: CGRect(x: 0, y: 0, width: w, height: h))
        let outer = round(0.008 * Double(w))
        let inner = round(0.004 * Double(w))
        let pad = round(0.012 * Double(w))
        let boxPad = round(0.008 * Double(w))
        let size = CGFloat(round(0.085 * Double(h)))
        let xs = [1, 2].map { round(Double(w * $0) / 3) }
        let ys = [1, 2].map { round(Double(h * $0) / 3) }
        // A rectangle given from the top-left (as the Mac check draws it), in CoreGraphics' bottom-up space.
        func fill(_ x: Int, _ y: Int, _ width: Int, _ height: Int, _ color: CGColor) {
            context.setFillColor(color)
            context.fill(CGRect(x: x, y: h - y - height, width: width, height: height))
        }
        for x in xs {
            fill(x - outer / 2, 0, outer, h, black)
            fill(x - inner / 2, 0, inner, h, white)
        }
        for y in ys {
            fill(0, y - outer / 2, w, outer, black)
            fill(0, y - inner / 2, w, inner, white)
        }
        for x in xs { fill(x - inner / 2, 0, inner, h, white) }
        let font = CTFontCreateUIFontForLanguage(.emphasizedSystem, size, nil)
            ?? CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): white,
        ]
        let lefts = [0] + xs.map { $0 + outer - outer / 2 }
        let tops = [0] + ys.map { $0 + outer - outer / 2 }
        for cell in 1...9 {
            let number = CTLineCreateWithAttributedString(NSAttributedString(string: "\(cell)", attributes: attributes))
            let ink = CTLineGetBoundsWithOptions(number, .useGlyphPathBounds)
            let advance = Int(CTLineGetTypographicBounds(number, nil, nil, nil).rounded())
            let inkHeight = Int(ink.height.rounded(.up))
            let x0 = lefts[(cell - 1) % 3] + pad
            let y0 = tops[(cell - 1) / 3] + pad
            fill(x0, y0, advance + 2 * boxPad, inkHeight + 2 * boxPad, black)
            // the digit's origin boxPad in from the box's left, its ink's top boxPad down from the box's top
            // (CoreGraphics' y running up)
            context.textPosition = CGPoint(x: CGFloat(x0 + boxPad), y: CGFloat(h - y0 - boxPad) - ink.maxY)
            CTLineDraw(number, context)
        }
        return context.makeImage() ?? upright
    }
}

/// One effect playing: which, where (in units of the picture's box), since when.
struct WatchEffectRun: Identifiable, Equatable {
    let id: Int
    let effect: WatchEffect
    let cell: Int?
    let place: CGRect
    let start: Date

    /// How long an effect plays, in seconds.
    static let seconds = 3.0
}

/// The effects over the picture, drawn every frame while one plays (at most 60 a second).
struct WatchEffectLayer: View {
    let runs: [WatchEffectRun]

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60, paused: runs.isEmpty)) { timeline in
            Canvas { context, size in
                for run in runs {
                    let t = timeline.date.timeIntervalSince(run.start)
                    guard t >= 0, t <= WatchEffectRun.seconds else { continue }
                    let rect = CGRect(
                        x: run.place.minX * size.width, y: run.place.minY * size.height,
                        width: run.place.width * size.width, height: run.place.height * size.height)
                    switch run.effect {
                    case .ripple: Self.ripple(context, rect, t)
                    case .sweep: Self.sweep(context, rect, t)
                    case .rays: Self.rays(context, rect, t)
                    }
                }
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    /// 0 → 1 over the first 0.12 s, then 1 until 2.6 s, then → 0 at the end (3 s).
    static func envelope(_ t: Double) -> Double {
        let rise = min(1, t / 0.12)
        let fall = t > 2.6 ? max(0, (WatchEffectRun.seconds - t) / 0.4) : 1
        return rise * fall
    }

    private static func point(_ center: CGPoint, _ radius: Double, _ angle: Double) -> CGPoint {
        CGPoint(x: center.x + radius * cos(angle), y: center.y + radius * sin(angle))
    }

    private static func circle(_ center: CGPoint, _ radius: Double) -> Path {
        Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: 2 * radius, height: 2 * radius))
    }

    /// Water: the place tinted, a ring every 0.4 s from its middle (each spreading for 1.3 s to its corners and
    /// thinning out), and a pulsing drop at the middle.
    static func ripple(_ context: GraphicsContext, _ rect: CGRect, _ t: Double) {
        let a = envelope(t)
        let water = Color(red: 0.55, green: 0.90, blue: 1.0)
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let reach = hypot(rect.width, rect.height) / 2
        context.fill(Path(roundedRect: rect.insetBy(dx: 3, dy: 3), cornerRadius: 10), with: .color(water.opacity(0.18 * a)))
        for k in 0..<5 {
            let age = t - Double(k) * 0.4
            guard age >= 0, age <= 1.3 else { continue }
            let p = age / 1.3
            let ring = circle(center, reach * (0.1 + 0.95 * (1 - pow(1 - p, 2.2))))
            let width = 2 + 7 * (1 - p)
            context.stroke(ring, with: .color(.black.opacity(0.25 * (1 - p) * a)), lineWidth: width + 3)
            context.stroke(ring, with: .color(water.opacity((1 - p) * a)), lineWidth: width)
        }
        let drop = 7 + 5 * (0.5 + 0.5 * sin(t * 2 * .pi * 1.5))
        context.fill(circle(center, drop + 2), with: .color(.black.opacity(0.25 * a)))
        context.fill(circle(center, drop), with: .color(.white.opacity(0.9 * a)))
    }

    /// A car: a band of light runs across the place from left to right twice (0–0.9 s, 0.9–1.8 s) while its edges are
    /// traced (0.05–1.25 s); the traced edges then glow until the end.
    static func sweep(_ context: GraphicsContext, _ rect: CGRect, _ t: Double) {
        let a = envelope(t)
        let light = Color(red: 1.0, green: 0.80, blue: 0.30)
        let edges = Path(roundedRect: rect.insetBy(dx: 5, dy: 5), cornerRadius: 12)
        var inside = context
        inside.clip(to: edges)
        inside.fill(edges, with: .color(light.opacity(0.14 * a)))
        if t < 1.8 {
            let pass = t.truncatingRemainder(dividingBy: 0.9) / 0.9
            let band = rect.width * 0.45
            // the band's middle runs from the place's left edge to its right edge
            let strip = CGRect(x: rect.minX - band / 2 + rect.width * pass, y: rect.minY, width: band, height: rect.height)
            let colors = [light.opacity(0), light.opacity(0.85 * a), Color.white.opacity(0.95 * a), light.opacity(0)]
            inside.fill(
                Path(strip),
                with: .linearGradient(
                    Gradient(colors: colors), startPoint: CGPoint(x: strip.minX, y: strip.midY),
                    endPoint: CGPoint(x: strip.maxX, y: strip.midY)))
        }
        let traced = min(1, max(0, (t - 0.05) / 1.2))
        guard traced > 0 else { return }
        let glow = traced < 1 ? 1 : 0.75 + 0.25 * sin((t - 1.25) * 2 * .pi * 2)
        let path = edges.trimmedPath(from: 0, to: traced)
        context.stroke(path, with: .color(.black.opacity(0.3 * a)), style: StrokeStyle(lineWidth: 10, lineCap: .round, lineJoin: .round))
        context.stroke(path, with: .color(light.opacity(glow * a)), style: StrokeStyle(lineWidth: 7, lineCap: .round, lineJoin: .round))
        context.stroke(path, with: .color(.white.opacity(0.7 * glow * a)), style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
    }

    /// Bells: twelve rays grow out from the place's middle in 0.5 s and turn slowly, flickering; a ring rings out once
    /// in the first second; the middle glows.
    static func rays(_ context: GraphicsContext, _ rect: CGRect, _ t: Double) {
        let a = envelope(t)
        let gold = Color(red: 1.0, green: 0.84, blue: 0.40)
        let center = CGPoint(x: rect.midX, y: rect.midY)
        let reach = hypot(rect.width, rect.height) / 2
        let grow = 1 - pow(1 - min(1, t / 0.5), 3)
        let inner = reach * 0.12
        for k in 0..<12 {
            let angle = t * 0.35 + Double(k) / 12 * 2 * .pi
            let length = reach * (0.35 + 0.75 * grow) * (k % 2 == 0 ? 1 : 0.72)
            let flicker = 0.65 + 0.35 * sin(t * 7 + Double(k) * 1.7)
            var ray = Path()
            ray.move(to: point(center, inner, angle - 0.04))
            ray.addLine(to: point(center, length, angle - 0.09))
            ray.addLine(to: point(center, length, angle + 0.09))
            ray.addLine(to: point(center, inner, angle + 0.04))
            ray.closeSubpath()
            context.stroke(ray, with: .color(.black.opacity(0.3 * a)), lineWidth: 1.5)
            context.fill(
                ray,
                with: .linearGradient(
                    Gradient(colors: [Color.white.opacity(flicker * a), gold.opacity(flicker * a), gold.opacity(0)]),
                    startPoint: point(center, inner, angle), endPoint: point(center, length, angle)))
        }
        if t < 1 {
            let ring = circle(center, reach * (0.2 + 0.9 * t))
            context.stroke(ring, with: .color(gold.opacity((1 - t) * a)), lineWidth: 5 * (1 - t) + 1)
        }
        let glow = reach * (0.3 + 0.06 * sin(t * 2 * .pi * 2))
        context.fill(
            circle(center, glow),
            with: .radialGradient(
                Gradient(colors: [Color.white.opacity(0.95 * a), gold.opacity(0.6 * a), gold.opacity(0)]), center: center,
                startRadius: 0, endRadius: glow))
    }
}

/// The effects' sounds, played with AVAudioPlayer from Sounds/<id>.m4a in the app bundle.
@MainActor
final class EffectSounds {
    private var players: [String: AVAudioPlayer] = [:]

    /// The file of a sound in the app, nil when this build does not have it.
    nonisolated static func url(_ id: String) -> URL? {
        Bundle.main.url(forResource: id, withExtension: "m4a", subdirectory: "Sounds")
    }

    /// The effects' sounds this build has (for the log and the record).
    nonisolated static var inBuild: [String] { WatchEffect.allCases.map(\.sound).filter { url($0) != nil } }

    /// Opens the players now, so the first fire does not wait for a file.
    func prepare(_ ids: [String]) {
        for id in ids { _ = player(id) }
    }

    /// Plays a sound from its start. nil: this build has no file for it (the caller plays the chime instead);
    /// otherwise whether it started, and its length in seconds.
    func play(_ id: String) -> (started: Bool, seconds: Double)? {
        guard let player = player(id) else { return nil }
        player.currentTime = 0
        return (player.play(), player.duration)
    }

    /// A sound is still playing.
    var playing: Bool { players.values.contains { $0.isPlaying } }

    private func player(_ id: String) -> AVAudioPlayer? {
        if let player = players[id] { return player }
        guard let url = Self.url(id), let player = try? AVAudioPlayer(contentsOf: url) else { return nil }
        player.prepareToPlay()
        players[id] = player
        return player
    }
}
