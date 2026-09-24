// DocumentText — the text of a file a decision can read: plain text and Markdown as they are,
// a PDF's pages joined, an image through Vision's on-device text recognizer (no download).
// Both the Checklist and the Sorter open files through this; the model only ever sees text.

import Foundation
import PDFKit
import UniformTypeIdentifiers
import Vision

enum DocumentText {
    /// What the file importers offer.
    static let readableTypes: [UTType] = [.plainText, .text, .pdf, .image]

    private static let textExtensions: Set<String> = ["txt", "md", "markdown", "text", "csv", "json", "log", "html"]

    /// The readable text of `url`, cut at `limit` characters (a decision reads the head of a
    /// document; the whole thing is a retrieval problem). Nil when nothing readable is found.
    static func read(_ url: URL, limit: Int) -> String? {
        let ext = url.pathExtension.lowercased()
        let type = UTType(filenameExtension: ext) ?? .data
        var text: String?
        if type.conforms(to: .pdf) {
            text = PDFDocument(url: url)?.string
        } else if type.conforms(to: .image) {
            text = try? recognizeText(in: url)
        } else if textExtensions.contains(ext) || type.conforms(to: .text) {
            text = try? String(contentsOf: url, encoding: .utf8)
        }
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : String(trimmed.prefix(limit))
    }

    /// Vision's text recognizer over an image file, top candidate per line.
    static func recognizeText(in url: URL) throws -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        try VNImageRequestHandler(url: url).perform([request])
        let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }
}
