// ImageFile.swift — file → CGImage glue for vision-language input (the image analog of
// `AudioFile.pcm16kMono`): decode any image file and carry its EXIF orientation, so a
// QuickStart snippet can feed a camera photo without excavating ImageIO. How the caller
// OBTAINED the URL (PhotosPicker, fileImporter, a CLI flag) stays app territory.

import CoreGraphics
import Foundation
import ImageIO

public enum ImageFileError: Error, LocalizedError {
    case unreadable(URL)

    public var errorDescription: String? {
        switch self {
        case .unreadable(let url):
            return "Not a readable image file: \(url.path)"
        }
    }
}

public enum ImageFile {
    /// A decoded image plus its EXIF orientation (camera photos arrive rotated;
    /// the vision preprocessor needs the upright pixels).
    public struct Loaded: Sendable {
        public let cgImage: CGImage
        public let orientation: CGImagePropertyOrientation
    }

    /// Decodes an image file (JPEG/PNG/HEIC/…) into a `CGImage` + EXIF orientation.
    public static func load(_ url: URL) throws -> Loaded {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ImageFileError.unreadable(url) }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let raw = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        return Loaded(
            cgImage: cgImage,
            orientation: CGImagePropertyOrientation(rawValue: raw) ?? .up)
    }
}
