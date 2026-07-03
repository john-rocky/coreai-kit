// QuickStart.swift — the take-home core of this runner: document image in → markdown out,
// one typed function, no UI. The CLI (`CLI/main.swift`) is an argument shell over exactly
// this function; the GUI drives the same `KitDocReader(catalog:)` on the image you pick.
// Want on-device document OCR in your own app? This file is the part you copy; the model
// card's 💻 snippet is the marked block below.

import CoreAIKit
import Foundation

/// OCR a document image with any doc-OCR model in the catalog
/// (`ModelCatalog.builtin.available(.ocr)`). Returns structured markdown — tables survive
/// as `<table>/<tr>/<td>` markup. First use downloads the model (progress via
/// `downloadProgress`), later runs load from the local cache.
func readDocument(
    at imageURL: URL,
    model id: String = "unlimited-ocr",
    downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil
) async throws -> String {
    // CARD-SNIPPET-BEGIN
    let reader = try await KitDocReader(catalog: id, downloadProgress: downloadProgress)
    let markdown = try await reader.read(imageAt: imageURL)
    // CARD-SNIPPET-END
    return markdown
}
