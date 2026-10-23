// WatchEffects — what an Effect rule does when it fires: the picture holds the frame the model
// answered — the crop it saw — for 3 s, with a thin border, and the app edits that frame with Core
// Image at the place the model names, while a short sound plays. The place is one of nine cells of
// the picture (3×3, numbered 1–9 left to right, top to bottom) when the launch asks the model for it
// (`-cells`), the whole picture otherwise. A video waits during the hold and then goes on from where
// it stopped; the camera's and the example photos' live picture is covered. Three edits, one picture
// rendered per screen frame (TimelineView, one Metal CIContext) — no image, no symbol, no words:
//
//   ripple     a circular wave spreads from the place's middle and leaves the frame tinted blue (water)
//   zoom       the frame zooms 1× → 2× toward the place's middle in 0.6 s and stays there (a car horn)
//   spotlight  around the place the frame goes dark and grey in 0.4 s; the place stays as it was (bells)
//
// Each plays its sound (Sounds/<id>.m4a: water, horn, bells) with AVAudioPlayer. The sound files are
// not in the repository: tools/fetch_examples.sh downloads and cuts them into Sounds/, which the app
// bundles; a build without one plays the system chime instead and the record says `fallback`.
//
// `-grid 1` burns the 3×3 cells and their numbers into the frame the model gets (WatchGrid), so the
// model can read the number off the picture; the screen shows the grid only on a held frame.

import AVFoundation
import CoreAIKit
import CoreGraphics
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreText
import Foundation
import ImageIO
import Metal
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

/// One hold: the frame the model answered, held on the picture while its edit plays at the place the model named.
struct WatchHold: Identifiable, Equatable {
    let id: Int
    let edit: WatchEffect
    /// The cell the model named; nil = the whole picture.
    let cell: Int?
    /// The frame as the model got it, and the orientation that turns it upright.
    let image: CGImage
    let orientation: CGImagePropertyOrientation
    /// An example photo is fitted into the picture's box whole; a camera or video crop fills the box.
    let fits: Bool
    let start: Date

    /// How long a hold lasts, in seconds.
    static let seconds = 3.0

    static func == (a: WatchHold, b: WatchHold) -> Bool { a.id == b.id }

    /// The frame's size once upright.
    var uprightSize: CGSize {
        let quarter = [.left, .right, .leftMirrored, .rightMirrored].contains(orientation)
        return quarter ? CGSize(width: image.height, height: image.width) : CGSize(width: image.width, height: image.height)
    }

    /// Where the held frame shows in the picture's box, in points: all of it, or the fitted photo.
    func shown(in box: CGSize) -> CGRect {
        WatchCells.place(nil, box: box, photo: fits ? uprightSize : nil).points
    }
}

/// The edits, in Core Image: a held frame turned upright and drawn at a size in pixels, and the picture `t` seconds
/// into its hold. One per hold and size. Once its edit has settled (ripple after 2.4 s, zoom 0.6 s, spotlight 0.4 s)
/// the last picture is kept.
@MainActor
final class WatchEditor {
    /// One Metal context for every hold (Core Image's default one where there is no Metal device).
    static let context: CIContext = MTLCreateSystemDefaultDevice().map { CIContext(mtlDevice: $0) } ?? CIContext()

    static let rippleSeconds = 2.4
    static let zoomSeconds = 0.6
    static let spotlightSeconds = 0.4

    let edit: WatchEffect
    /// The picture's size in pixels.
    let size: CGSize
    /// The frame upright at its own size, and drawn at `size`.
    private let full: CIImage
    private let base: CIImage
    /// The place in `size`'s pixels, Core Image's y running up: the cell, or the whole picture.
    private let place: CGRect
    private var settled: CGImage?

    convenience init(_ hold: WatchHold, size: CGSize) {
        self.init(image: hold.image, orientation: hold.orientation, edit: hold.edit, cell: hold.cell, size: size)
    }

    init(image: CGImage, orientation: CGImagePropertyOrientation, edit: WatchEffect, cell: Int?, size: CGSize) {
        self.edit = edit
        self.size = size
        let upright = CIImage(cgImage: image).oriented(orientation)
        full = upright.transformed(by: CGAffineTransform(translationX: -upright.extent.minX, y: -upright.extent.minY))
        base = full.transformed(
            by: CGAffineTransform(scaleX: size.width / full.extent.width, y: size.height / full.extent.height))
        place = Self.place(cell, in: size)
    }

    /// Cell 1–9 of a picture (left to right, top to bottom) in Core Image's coordinates; the whole picture for none.
    static func place(_ cell: Int?, in size: CGSize) -> CGRect {
        guard let cell, (1...9).contains(cell) else { return CGRect(origin: .zero, size: size) }
        let width = size.width / 3
        let height = size.height / 3
        return CGRect(
            x: CGFloat((cell - 1) % 3) * width, y: size.height - CGFloat((cell - 1) / 3 + 1) * height, width: width,
            height: height)
    }

    /// The picture t seconds into the hold.
    func picture(at t: Double) -> CGImage? {
        if let settled { return settled }
        let picture = Self.context.createCGImage(image(at: t), from: CGRect(origin: .zero, size: size))
        if t >= settles { settled = picture }
        return picture
    }

    /// When the edit stops changing, in seconds into the hold.
    var settles: Double {
        switch edit {
        case .ripple: Self.rippleSeconds
        case .zoom: Self.zoomSeconds
        case .spotlight: Self.spotlightSeconds
        }
    }

    /// The edit t seconds into the hold, as a Core Image recipe the size of the picture.
    func image(at t: Double) -> CIImage {
        let t = max(t, 0)
        let picture: CIImage
        switch edit {
        case .ripple: picture = ripple(min(t / Self.rippleSeconds, 1))
        case .zoom: picture = zoom(Self.easeOut(min(t / Self.zoomSeconds, 1)))
        case .spotlight: picture = spotlight(Self.easeOut(min(t / Self.spotlightSeconds, 1)))
        }
        return picture.cropped(to: CGRect(origin: .zero, size: size))
    }

    /// Fast, then slower to the end: 1 − (1 − x)³.
    static func easeOut(_ x: Double) -> Double { 1 - pow(1 - x, 3) }

    /// A cell's shorter side in pixels (the place's own when it is a cell).
    private var cellSide: CGFloat { min(size.width, size.height) / 3 }

    /// Water: a circular wave runs out from the place's middle (`time` 0 → 1 over 2.4 s) and leaves the frame a cold
    /// blue behind it (CIRippleTransition from the frame to a blue copy of it).
    private func ripple(_ time: Double) -> CIImage {
        let blue = base.applyingFilter(
            "CIColorMatrix",
            parameters: [
                "inputRVector": CIVector(x: 0.40, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: 0.70, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 0.80, w: 0), "inputBiasVector": CIVector(x: 0, y: 0.04, z: 0.20, w: 0),
            ])
        let wave = CIFilter.rippleTransition()
        // clamped: near the picture's edges the wave bends in pixels from beyond them (the edge's own, not clear ones)
        wave.inputImage = base.clampedToExtent()
        wave.targetImage = blue.clampedToExtent()
        wave.shadingImage = Self.shading
        wave.center = CGPoint(x: place.midX, y: place.midY)
        wave.extent = base.extent
        wave.time = Float(time)
        wave.width = Float(cellSide * 0.3)
        wave.scale = 50
        return wave.outputImage ?? blue
    }

    /// The light on the wave's slopes. The filter looks a point of the wave up in this square by the wave's slope
    /// there: where the frame is flat it reads the middle, which is clear (no light), and a slope facing the upper
    /// left reads the soft white spot there — a highlight on one side of each crest.
    private static let shading: CIImage = {
        let light = CIFilter.radialGradient()
        light.center = CGPoint(x: 34, y: 94)
        light.radius0 = 0
        light.radius1 = 34
        light.color0 = CIColor(red: 1, green: 1, blue: 1, alpha: 0.75)
        light.color1 = CIColor(red: 1, green: 1, blue: 1, alpha: 0)
        return light.outputImage?.cropped(to: CGRect(x: 0, y: 0, width: 128, height: 128)) ?? CIImage.empty()
    }()

    /// A horn: the frame zoomed by 1 + p toward the place's middle. At 2× the window is half the frame each way,
    /// centred on the place's middle, moved in when that would leave the frame; on the way the window's middle moves
    /// from the frame's to there, so the window never leaves the frame. Drawn from the frame at its own size.
    private func zoom(_ p: Double) -> CIImage {
        let width = full.extent.width
        let height = full.extent.height
        let zoom = 1 + p
        let x = min(max(place.midX * width / size.width, width / 4), width * 3 / 4)
        let y = min(max(place.midY * height / size.height, height / 4), height * 3 / 4)
        let along = 2 * (1 - 1 / zoom)  // 0 at 1×, 1 at 2×
        let window = CGRect(
            x: width / 2 + (x - width / 2) * along - width / (2 * zoom),
            y: height / 2 + (y - height / 2) * along - height / (2 * zoom), width: width / zoom, height: height / zoom)
        let transform = CGAffineTransform(translationX: -window.minX, y: -window.minY)
            .concatenating(CGAffineTransform(scaleX: size.width / window.width, y: size.height / window.height))
        return full.clampedToExtent().transformed(by: transform)
    }

    /// Bells: around the place the frame darkens (−1.6 EV) and loses its colour (saturation 0.1), the place stays as it
    /// was, behind a soft edge (the place's mask blurred by 6 % of a cell's shorter side); `a` = how far it has come
    /// (0 → 1 over 0.4 s). With no place, the middle two thirds of the frame stay lit.
    private func spotlight(_ a: Double) -> CIImage {
        let extent = base.extent
        let lit = place == extent ? extent.insetBy(dx: extent.width / 6, dy: extent.height / 6) : place
        let dark = base.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -1.6])
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.1])
        let mask = CIImage(color: .white).cropped(to: lit)
            .composited(over: CIImage(color: .black).cropped(to: extent))
            .clampedToExtent()
            .applyingGaussianBlur(sigma: Double(cellSide) * 0.06)
            .cropped(to: extent)
        let spot = base.applyingFilter(
            "CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: dark, kCIInputMaskImageKey: mask])
        guard a < 1 else { return spot }
        return base.applyingFilter("CIDissolveTransition", parameters: [kCIInputTargetImageKey: spot, kCIInputTimeKey: a])
    }
}

/// The held frame over the picture as its edit goes on (at most 30 pictures a second), with a thin border in the
/// accent colour: the frame the model answered, over the live picture.
struct WatchHoldLayer: View {
    let hold: WatchHold
    @Environment(\.displayScale) private var scale
    @State private var editors = WatchEditors()

    var body: some View {
        GeometryReader { proxy in
            let shown = hold.shown(in: proxy.size)
            let pixels = CGSize(width: (shown.width * scale).rounded(), height: (shown.height * scale).rounded())
            ZStack(alignment: .topLeading) {
                Rectangle().fill(Palette.surface)
                TimelineView(.animation(minimumInterval: 1.0 / 30, paused: false)) { timeline in
                    if let picture = editors.editor(hold, pixels).picture(at: timeline.date.timeIntervalSince(hold.start)) {
                        Image(decorative: picture, scale: scale)
                            .resizable()
                            .frame(width: shown.width, height: shown.height)
                    }
                }
                .offset(x: shown.minX, y: shown.minY)
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Palette.accent, lineWidth: 3)
                    .frame(width: shown.width, height: shown.height)
                    .offset(x: shown.minX, y: shown.minY)
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// The hold layer's editor, kept across its redraws: one per hold and size.
@MainActor
final class WatchEditors {
    private var current: (id: Int, size: CGSize, editor: WatchEditor)?

    func editor(_ hold: WatchHold, _ size: CGSize) -> WatchEditor {
        if let current, current.id == hold.id, current.size == size { return current.editor }
        let editor = WatchEditor(hold, size: size)
        current = (hold.id, size, editor)
        return editor
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
