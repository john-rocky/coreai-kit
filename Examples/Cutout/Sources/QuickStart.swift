// QuickStart.swift — the take-home core of Cutout: a photo + the words for what to cut out → each instance's box and
// its matte, one typed function, no UI. The screen (CutoutModel) makes the same calls with the model held across
// requests; this file is the part you copy into your own app, with Sources/Engine.

import CoreGraphics
import Foundation

/// One thing the model found: its box (per mille of the photo's width and height: left, top, right, bottom) and its
/// matte (the probability that each pixel belongs to it, 0…1, row-major at the photo's size).
struct CutoutPiece {
    let label: String
    let box: [Int]
    let matte: [Float]
}

/// Cut out what `words` names in the photo at `imageURL`, on this device. `modelFolder` holds the four model files
/// (README: Run it). The photo and the words go to the model in this process; nothing leaves the device.
@available(macOS 27, iOS 27, *)
func cutOut(_ words: String, inImageAt imageURL: URL, modelFolder: URL) async throws -> (pieces: [CutoutPiece], png: Data?) {
    #if os(macOS)
    let folder = try ModelFolder(root: modelFolder, preferAOT: true)
    #else
    let folder = try ModelFolder(root: modelFolder, preferAOT: false)
    #endif
    let model = try await CutoutPipeline(folder: folder)
    let file = try ImageHost.load(url: imageURL)                          // the first image and its EXIF orientation
    let photo = try ImageHost.uprightRGB8(file.image, orientation: file.orientation)
    let result = try await model.run(image: photo, words: words, task: .cutout)
    let pieces = result.instances.compactMap { inst in
        inst.logits.map { CutoutPiece(label: inst.label, box: inst.box,
                                      matte: Alpha.probabilities($0, width: photo.width, height: photo.height)) }
    }
    guard !pieces.isEmpty else { return ([], nil) }                    // the model answered "None": nothing matched
    let png = try CutoutImage.png(photo, alpha: CutoutImage.combinedAlpha(pieces.map(\.matte)))
    return (pieces, png)
}
