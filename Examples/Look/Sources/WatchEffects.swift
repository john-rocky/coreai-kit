// WatchEffects — what an Effect rule does when it fires: the picture holds the frame the model
// answered — the crop it saw — for 3 s, with a thin border, and the app edits that frame with Core
// Image at the place the model names, while a short sound plays. The place is the box the model
// outputs for the rule's words (`-cells box`, the default: the model's own grounding form), or one
// of nine cells of the picture (3×3, numbered 1–9 left to right, top to bottom) with `-cells ask` /
// `batch`; with `-cells none`, or when the answer holds no box, the whole picture. A video waits
// during the hold and then goes on from where it stopped; the camera's and the example photos' live
// picture is covered. Three edits, one picture rendered per screen frame (TimelineView, one Metal
// CIContext) — no image, no symbol, no words:
//
//   ripple     a circular wave spreads from the place's middle and leaves the frame tinted blue
//   zoom       the frame zooms in 0.6 s to a window around the place (1.2–2.5×) and stays there
//   spotlight  around the place the frame goes dark and grey in 0.4 s; the place stays as it was
//
// Each plays a sound with AVAudioPlayer — its own (ripple water, zoom horn, spotlight bells) or the
// one its rule names (Sounds/<id>.m4a: water, horn, bells; or the system chime). The sound files are
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

/// Where the model says a rule's thing is: how the app asks (`-cells`), and the place on the screen.
enum WatchCells {
    /// `-cells box|none|batch|ask`.
    enum Form: String {
        /// The default: a second question about the same frame for each effect rule that fires, in the model's own
        /// grounding form (WatchPrompt.box), and the edit plays at the box it outputs.
        case box
        /// Not asked: the prompt is the plain yes / no batch and an effect plays over the whole picture.
        case off = "none"
        /// In the same prompt: each line answered yes also carries the cell's number.
        case batch
        /// A second question about the same frame, for each rule that fires: a cell, as a number or two words.
        case ask
    }

    /// Cell 1–9 as a rectangle in units of the frame the model got (x and y from the top left, 0–1): its third
    /// each way; nil for no cell.
    static func unit(cell: Int?) -> CGRect? {
        guard let cell, (1...9).contains(cell) else { return nil }
        return CGRect(x: Double((cell - 1) % 3) / 3, y: Double((cell - 1) / 3) / 3, width: 1.0 / 3, height: 1.0 / 3)
    }

    /// The model's box (left, top, right, bottom, per mille of the frame it got) as a rectangle in units of that
    /// frame; nil for no box.
    static func unit(box: [Int]?) -> CGRect? {
        guard let box, box.count == 4 else { return nil }
        return CGRect(
            x: Double(box[0]) / 1000, y: Double(box[1]) / 1000, width: Double(box[2] - box[0]) / 1000,
            height: Double(box[3] - box[1]) / 1000)
    }

    /// The part of the picture's box an effect plays over — `place`, a rectangle in units of the frame the model
    /// got, or all of it for none — in points and in units of the picture's box. The camera and a video fill the box
    /// and the model gets the same crop, so a place in units of the frame is the same in units of the box; an example
    /// photo is fitted into the box whole (`photo` = its upright size), so it is a place in the fitted photo.
    static func place(_ place: CGRect?, box: CGSize, photo: CGSize?) -> (points: CGRect, unit: CGRect) {
        var content = CGRect(origin: .zero, size: box)
        if let photo, photo.width > 0, photo.height > 0, box.width > 0, box.height > 0 {
            let fit = min(box.width / photo.width, box.height / photo.height)
            let size = CGSize(width: photo.width * fit, height: photo.height * fit)
            content = CGRect(
                x: (box.width - size.width) / 2, y: (box.height - size.height) / 2, width: size.width, height: size.height)
        }
        var rect = content
        if let place {
            rect = CGRect(
                x: content.minX + place.minX * content.width, y: content.minY + place.minY * content.height,
                width: place.width * content.width, height: place.height * content.height)
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
    /// Where the model said the thing is, in units of the frame (its box, or a cell's third); nil = the whole picture.
    let place: CGRect?
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
/// the last picture is kept. The place is a rectangle in units of the frame (the model's box, or a cell's third);
/// with none, each edit plays as it did over the whole frame before the box: a wave from the middle, a 2× zoom
/// toward the middle, the middle two thirds lit.
@MainActor
final class WatchEditor {
    /// One Metal context for every hold (Core Image's default one where there is no Metal device).
    static let context: CIContext = MTLCreateSystemDefaultDevice().map { CIContext(mtlDevice: $0) } ?? CIContext()

    static let rippleSeconds = 2.4
    static let zoomSeconds = 0.6
    static let spotlightSeconds = 0.4
    /// The zoom's range: the picture over the window around the place, at least 1.2×, at most 2.5×.
    static let zoomLeast = 1.2
    static let zoomMost = 2.5

    let edit: WatchEffect
    /// The picture's size in pixels.
    let size: CGSize
    /// The frame upright at its own size, and drawn at `size`.
    private let full: CIImage
    private let base: CIImage
    /// The place in units of the frame, y from the top; nil = the whole picture.
    private let place: CGRect?
    private var settled: CGImage?

    convenience init(_ hold: WatchHold, size: CGSize) {
        self.init(image: hold.image, orientation: hold.orientation, edit: hold.edit, place: hold.place, size: size)
    }

    init(image: CGImage, orientation: CGImagePropertyOrientation, edit: WatchEffect, place: CGRect?, size: CGSize) {
        self.edit = edit
        self.size = size
        let upright = CIImage(cgImage: image).oriented(orientation)
        full = upright.transformed(by: CGAffineTransform(translationX: -upright.extent.minX, y: -upright.extent.minY))
        base = full.transformed(
            by: CGAffineTransform(scaleX: size.width / full.extent.width, y: size.height / full.extent.height))
        self.place = place
    }

    /// A rectangle in units of the frame (y from the top) in the pixels of a picture of `size`, Core Image's y running up.
    static func pixels(_ unit: CGRect, in size: CGSize) -> CGRect {
        CGRect(
            x: unit.minX * size.width, y: (1 - unit.maxY) * size.height, width: unit.width * size.width,
            height: unit.height * size.height)
    }

    /// Where the zoom ends, in a picture of `size` (Core Image's y running up): the place widened by a quarter of its
    /// width and height on each side, then made the picture's shape; the zoom = the picture over that window, kept
    /// within 1.2–2.5×; the window's middle = the place's, moved in when the window would leave the picture. With no
    /// place, 2× toward the middle.
    static func zoomEnd(_ place: CGRect?, in size: CGSize) -> (middle: CGPoint, zoom: Double) {
        guard let place else { return (CGPoint(x: size.width / 2, y: size.height / 2), 2) }
        let box = pixels(place, in: size)
        let wide = CGSize(width: box.width * 1.5, height: box.height * 1.5)
        let aspect = size.width / size.height
        let window =
            wide.width >= wide.height * aspect
            ? CGSize(width: wide.width, height: wide.width / aspect) : CGSize(width: wide.height * aspect, height: wide.height)
        let zoom = min(max(Double(size.width / max(window.width, 1)), zoomLeast), zoomMost)
        let half = CGSize(width: size.width / (2 * zoom), height: size.height / (2 * zoom))
        let middle = CGPoint(
            x: min(max(box.midX, half.width), size.width - half.width),
            y: min(max(box.midY, half.height), size.height - half.height))
        return (middle, zoom)
    }

    /// The window the zoom shows at `zoom` (1× → its end), in a picture of `size`: its middle moves from the picture's
    /// to the end's as the zoom grows, so the window never leaves the picture.
    static func zoomWindow(_ place: CGRect?, in size: CGSize, at p: Double) -> CGRect {
        let end = zoomEnd(place, in: size)
        let zoom = 1 + (end.zoom - 1) * p
        let along = (1 - 1 / zoom) / (1 - 1 / end.zoom)  // 0 at 1×, 1 at the end
        return CGRect(
            x: size.width / 2 + (end.middle.x - size.width / 2) * along - size.width / (2 * zoom),
            y: size.height / 2 + (end.middle.y - size.height / 2) * along - size.height / (2 * zoom),
            width: size.width / zoom, height: size.height / zoom)
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

    /// The ripple's wave: its middle (the place's) and its width (0.3 of the place's shorter side, at least 8 % of the
    /// picture's; with no place a tenth of the picture's shorter side), in a picture of `size`, Core Image's y running up.
    static func rippleWave(_ place: CGRect?, in size: CGSize) -> (middle: CGPoint, width: CGFloat) {
        let short = min(size.width, size.height)
        guard let place else { return (CGPoint(x: size.width / 2, y: size.height / 2), short / 3 * 0.3) }
        let box = pixels(place, in: size)
        return (CGPoint(x: box.midX, y: box.midY), max(min(box.width, box.height) * 0.3, short * 0.08))
    }

    /// The spotlight's lit rectangle (the place widened by a tenth of its width and height on each side, inside the
    /// picture; with no place the middle two thirds) and its edge's blur (6 % of the picture's shorter side; with no
    /// place 2 %), in a picture of `size`, Core Image's y running up.
    static func spotlightLit(_ place: CGRect?, in size: CGSize) -> (lit: CGRect, sigma: Double) {
        let extent = CGRect(origin: .zero, size: size)
        let short = Double(min(size.width, size.height))
        guard let place else { return (extent.insetBy(dx: size.width / 6, dy: size.height / 6), short / 3 * 0.06) }
        let box = pixels(place, in: size)
        let lit = box.insetBy(dx: -box.width * 0.10, dy: -box.height * 0.10).intersection(extent)
        return (lit.isNull ? box : lit, short * 0.06)
    }

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
        let shape = Self.rippleWave(place, in: size)
        wave.center = shape.middle
        wave.extent = base.extent
        wave.time = Float(time)
        wave.width = Float(shape.width)
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

    /// The frame zoomed toward the place: `p` 0 → 1 = 1× → the end's zoom (zoomEnd: the window around the place,
    /// 1.2–2.5×); on the way the window's middle moves from the frame's to the end's, so the window never leaves the
    /// frame. Drawn from the frame at its own size.
    private func zoom(_ p: Double) -> CIImage {
        let window = Self.zoomWindow(place, in: full.extent.size, at: p)
        let transform = CGAffineTransform(translationX: -window.minX, y: -window.minY)
            .concatenating(CGAffineTransform(scaleX: size.width / window.width, y: size.height / window.height))
        return full.clampedToExtent().transformed(by: transform)
    }

    /// Around the place the frame darkens (−1.6 EV) and loses its colour (saturation 0.1), the place (widened by a
    /// tenth on each side) stays as it was, behind a soft edge (its mask blurred by 6 % of the picture's shorter side);
    /// `a` = how far it has come (0 → 1 over 0.4 s). With no place, the middle two thirds of the frame stay lit.
    private func spotlight(_ a: Double) -> CIImage {
        let extent = base.extent
        let (lit, sigma) = Self.spotlightLit(place, in: size)
        let dark = base.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -1.6])
            .applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.1])
        let mask = CIImage(color: .white).cropped(to: lit)
            .composited(over: CIImage(color: .black).cropped(to: extent))
            .clampedToExtent()
            .applyingGaussianBlur(sigma: sigma)
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

    /// The effects' sound files this build has (for the log and the record).
    nonisolated static var inBuild: [String] { WatchEffectSound.allCases.compactMap(\.file).filter { url($0) != nil } }

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
