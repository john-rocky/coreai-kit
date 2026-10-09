// From the model zoo's apps/D1/Sources/D1/Vision.swift (e36ad15, sha256 e75cb2045e1e), identifiers prefixed D13B for the kit.
// Vision — a picture's crop plan and its image-token run (conversion/d1/vision_host.py §1, 2, 5, 6; the provider's
// runner.py `cap_pixels` / `_image_markup` / `_request` and transformers 5.19's Lfm2VlImageProcessor /
// Lfm2VlProcessor, K/results/vision_rules.md (b)(c)(e)(h)). Sizes only: the pixels (decode, the torch uint8 bicubic
// resize, patches, the position table, unshuffle) are round 3c.
//
//   cap        w * h > 1024 * 1024 -> (max(1, int(w * s)), max(1, int(h * s))), s = sqrt(1024 * 1024 / (w * h))
//   round32    round(v / 32) * 32 (Python's round: half to even)
//   too large  max(16, round32(h)) * max(16, round32(w)) > 524,288
//   smart size h' = max(32, round32(h)), w' likewise; h' * w' > 262,144: b = sqrt(h * w / 262,144),
//              h' = max(32, floor(h / b / 32) * 32); h' * w' < 65,536: b = sqrt(65,536 / (h * w)), h' = ceil(h * b / 32) * 32
//   one crop   not too large: the picture resized to (h', w')
//   tiles      too large: (cols, rows) = the closest cols / rows to w / h among `targetRatios` (in that order; an equal
//              difference goes to the later ratio when w * h > 0.5 * 512^2 * cols * rows), tiles of 512 x 512
//              row-major, then a thumbnail = the picture resized to (h', w')
//   tokens     a crop of H x W pixels: ceil(H / 32) * ceil(W / 32) image tokens (256 per tile)
//   run        <|image_start|> + (one crop: <image> x tokens | tiles: per tile <|img_row_r_col_c|> + <image> x 256, then
//              <|img_thumbnail|> + <image> x thumbnail tokens) + <|image_end|>
//   prompt     prefix = BOS + "<|im_start|>user\n" + "<image>" per picture + the state part; the text is cut at each
//              "<image>" (code points) and the pieces encoded apart, each marker replaced by its picture's run
//   slots      the k-th <image> (124907) of a row, counted over every picture and crop in order, is sent as V + k
//              (V = 128,000, the text vocabulary) and reads row k of image_embeds

import Foundation

@available(macOS 27, iOS 27, *)
enum D13BVision {
    static let patch = 16
    static let factor = 2
    static let tile = 512
    static let maxPixels = 1024 * 1024
    static let imageToken = "<image>"
    static let imageID = 124907
    static let rowColBase = 124908
    static let thumbnailID = 125008
    static let imageStartID = 125009
    static let imageEndID = 125010
    static let extensionBase = 128000

    /// `Lfm2VlImageProcessor._target_ratios` as (cols, rows): CPython's set order of the (c, r) with 2 <= c * r <= 10,
    /// stably sorted by c * r (vision_rules.md (c); the gate compares it with vision_host.TARGET_RATIOS).
    static let targetRatios: [(cols: Int, rows: Int)] = [
        (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1), (3, 2), (2, 3), (7, 1),
        (1, 7), (4, 2), (2, 4), (1, 8), (8, 1), (1, 9), (3, 3), (9, 1), (2, 5), (5, 2), (10, 1), (1, 10),
    ]

    struct Crop: Sendable, Equatable {
        enum Kind: String, Sendable { case single, tile, thumbnail }
        let kind: Kind
        /// (H, W) pixels
        let height: Int
        let width: Int
        /// 1-based tile row / column (0 for a single crop or the thumbnail)
        let row: Int
        let col: Int
        /// (H / 16, W / 16) patches
        var grid: (h: Int, w: Int) { (height / D13BVision.patch, width / D13BVision.patch) }
        var tokens: Int {
            let g = grid
            return ((g.h + D13BVision.factor - 1) / D13BVision.factor) * ((g.w + D13BVision.factor - 1) / D13BVision.factor)
        }
    }

    struct Plan: Sendable {
        /// (h, w) of the picture the processor receives (after the cap)
        let height: Int
        let width: Int
        let rows: Int
        let cols: Int
        /// the smart size (h', w'): the single crop's or the thumbnail's
        let thumbHeight: Int
        let thumbWidth: Int
        let crops: [Crop]
        var tiled: Bool { rows > 1 || cols > 1 }
        var tokens: Int { crops.reduce(0) { $0 + $1.tokens } }
    }

    /// `runner.cap_pixels`' target size.
    static func capSize(width w: Int, height h: Int) -> (width: Int, height: Int) {
        if w * h <= maxPixels { return (w, h) }
        let s = (Double(maxPixels) / Double(w * h)).squareRoot()
        return (max(1, Int(Double(w) * s)), max(1, Int(Double(h) * s)))
    }

    /// round(v / f) * f with Python's round (half to even).
    static func roundBy(_ v: Int, _ f: Int) -> Int { Int((Double(v) / Double(f)).rounded(.toNearestOrEven)) * f }

    /// `_is_image_too_large`.
    static func isTooLarge(height h: Int, width w: Int) -> Bool {
        let total = patch * factor
        let hb = max(patch, roundBy(h, total)), wb = max(patch, roundBy(w, total))
        return Double(hb * wb) > Double(256 * patch * patch * factor * factor) * 2.0
    }

    /// `smart_resize` -> (h', w').
    static func smartSize(height h: Int, width w: Int) -> (height: Int, width: Int) {
        let total = patch * factor
        let minPx = 64 * patch * patch * factor * factor, maxPx = 256 * patch * patch * factor * factor
        var hb = max(total, roundBy(h, total)), wb = max(total, roundBy(w, total))
        if hb * wb > maxPx {
            let beta = (Double(h * w) / Double(maxPx)).squareRoot()
            hb = max(total, Int((Double(h) / beta / Double(total)).rounded(.down)) * total)
            wb = max(total, Int((Double(w) / beta / Double(total)).rounded(.down)) * total)
        } else if hb * wb < minPx {
            let beta = (Double(minPx) / Double(h * w)).squareRoot()
            hb = Int((Double(h) * beta / Double(total)).rounded(.up)) * total
            wb = Int((Double(w) * beta / Double(total)).rounded(.up)) * total
        }
        return (hb, wb)
    }

    /// `find_closest_aspect_ratio` -> (rows, cols).
    static func gridLayout(height h: Int, width w: Int) -> (rows: Int, cols: Int) {
        let aspect = Double(w) / Double(h)
        var bestDiff = Double.infinity
        var best = (cols: 1, rows: 1)
        let area = Double(w * h)
        for r in targetRatios {
            let diff = abs(aspect - Double(r.cols) / Double(r.rows))
            if diff < bestDiff {
                bestDiff = diff
                best = r
            } else if diff == bestDiff, area > 0.5 * Double(tile * tile * r.cols * r.rows) {
                best = r
            }
        }
        return (best.rows, best.cols)
    }

    /// The processor's crops for an (h, w) picture.
    static func plan(height h: Int, width w: Int) -> Plan {
        let (hb, wb) = smartSize(height: h, width: w)
        if !isTooLarge(height: h, width: w) {
            return Plan(height: h, width: w, rows: 1, cols: 1, thumbHeight: hb, thumbWidth: wb,
                        crops: [Crop(kind: .single, height: hb, width: wb, row: 0, col: 0)])
        }
        let (rows, cols) = gridLayout(height: h, width: w)
        var crops: [Crop] = []
        for r in 0..<rows {
            for c in 0..<cols { crops.append(Crop(kind: .tile, height: tile, width: tile, row: r + 1, col: c + 1)) }
        }
        if rows * cols != 1 { crops.append(Crop(kind: .thumbnail, height: hb, width: wb, row: 0, col: 0)) }
        return Plan(height: h, width: w, rows: rows, cols: cols, thumbHeight: hb, thumbWidth: wb, crops: crops)
    }

    /// A picture of (w, h) as it arrives: the cap, then the plan.
    static func plan(pictureWidth w: Int, pictureHeight h: Int) -> Plan {
        let c = capSize(width: w, height: h)
        return plan(height: c.height, width: c.width)
    }

    /// `<|img_row_r_col_c|>` (1-based r, c in 1..10).
    static func rowColID(_ r: Int, _ c: Int) -> Int { rowColBase + 10 * (r - 1) + (c - 1) }

    /// `vision_host.image_tokens`: the id run one picture expands to.
    static func imageTokens(_ p: Plan) -> [Int] {
        var ids = [imageStartID]
        if p.tiled {
            for c in p.crops {
                ids.append(c.kind == .tile ? rowColID(c.row, c.col) : thumbnailID)
                ids += [Int](repeating: imageID, count: c.tokens)
            }
        } else {
            ids += [Int](repeating: imageID, count: p.crops[0].tokens)
        }
        return ids + [imageEndID]
    }

    /// `vision_host.extension_ids`: the k-th <image> -> V + k (k from `start`), every other id unchanged.
    static func extensionIDs(_ ids: [Int], start: Int = 0) -> [Int] {
        var k = start
        return ids.map { t in
            guard t == imageID else { return t }
            defer { k += 1 }
            return extensionBase + k
        }
    }

    /// `vision_host.n_image_tokens`.
    static func imageTokenCount(_ plans: [Plan]) -> Int { plans.reduce(0) { $0 + $1.tokens } }

    /// `str.split("<image>")` on code points.
    static func splitOnMarker(_ text: String) -> [String] {
        let ns = text as NSString
        var out: [String] = []
        var pos = 0
        while pos <= ns.length {
            let r = ns.range(of: imageToken, options: .literal, range: NSRange(location: pos, length: ns.length - pos))
            if r.location == NSNotFound {
                out.append(ns.substring(from: pos))
                break
            }
            out.append(ns.substring(with: NSRange(location: pos, length: r.location - pos)))
            pos = r.location + r.length
        }
        return out
    }

    /// `vision_host.prompt_ids`: a prompt holding one "<image>" marker per plan -> ids (text pieces encoded apart).
    static func promptIDs(_ tok: D13BTokenizer, text: String, plans: [Plan]) throws -> [Int] {
        let pieces = splitOnMarker(text)
        guard pieces.count - 1 == plans.count else {
            throw D13BError.request("\(pieces.count - 1) <image> markers for \(plans.count) pictures")
        }
        var ids: [Int] = []
        for (k, piece) in pieces.enumerated() {
            if !piece.isEmpty { ids += tok.encode(piece) }
            if k < plans.count { ids += imageTokens(plans[k]) }
        }
        return ids
    }

    /// One request with pictures through the provider's `_request`: one question = the whole row; several = the trunk
    /// (prefix with the pictures) + each suffix encoded apart. -> (ids the processor makes, branches, input_tokens).
    static func requestIDs(_ tok: D13BTokenizer, _ request: D13BRequest, pictures: [(width: Int, height: Int)]) throws
        -> (text: String, ids: [Int], branches: [[Int]], inputTokens: Int, plans: [Plan])
    {
        let plans = pictures.map { plan(pictureWidth: $0.width, pictureHeight: $0.height) }
        let rows = try request.questions.map { q -> String in
            let codes = q.kind == .choice ? try tok.aliases(q.labels).map(\.code) : nil
            return D13BText.suffix(q, codes: codes)
        }
        let prefix = D13BText.prefix(request.state, images: String(repeating: imageToken, count: pictures.count))
        let text = rows.count == 1 ? prefix + rows[0] : prefix
        let ids = try promptIDs(tok, text: text, plans: plans)
        let branches = rows.count > 1 ? rows.map { tok.encode($0) } : []
        return (text, ids, branches, ids.count + branches.reduce(0) { $0 + $1.count }, plans)
    }
}
