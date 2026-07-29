// QuickStart.swift — the take-home core of this example: a photo of a receipt in, a Swift value
// out. Both the CLI and the app call exactly this. Copy this file into your own project and it
// runs; nothing else here is required.

import CoreAIOps
import CoreGraphics
import FoundationModels

/// What a receipt is, as far as your app is concerned. The framework derives a schema from
/// `@Generable` and constrains generation to it, so the model cannot return a shape that fails
/// to decode — you get this type or an error, never a string to parse.
@Generable
struct Receipt: Sendable {
    @Guide(description: "Merchant or store name, as printed")
    var merchant: String
    @Guide(description: "Grand total as a number, no currency symbol")
    var total: Double
    @Guide(description: "Date in YYYY-MM-DD")
    var date: String
    @Guide(description: "One-word expense category, e.g. meals, travel, supplies")
    var category: String
}

/// Photo → typed value. Two calls: read the page, then fill the struct from what it says.
///
/// `read` returns the document's text *with its structure* — a receipt's line items stay in
/// their table rather than collapsing into a stream of words, which is what makes the total
/// findable. `extract` then fills `Receipt` from that text.
func scanReceipt(_ image: CGImage) async throws -> Receipt {
    let text = try await CoreAI.read(image)
    return try await CoreAI.extract(text, as: Receipt.self)
}
