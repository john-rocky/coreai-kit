// QuickStart.swift — the take-home core: a photo of a document in, a value of *your* type out.
// Copy this file into your own project and it runs; nothing else here is required.

import CoreAIOps
import CoreGraphics
import FoundationModels

/// A document photo, read and decoded into whatever type you asked for.
///
/// Two calls, and neither one knows what your type is about. `read` returns the page's text
/// with its structure intact — a table stays a table rather than collapsing into a stream of
/// words, which is what makes a specific field findable. `extract` then fills your type from
/// that text, constrained to the schema the framework derives from `@Generable`, so you get the
/// type or an error — never a string to parse and hope about.
///
/// The domain lives entirely in the type you pass. Swap it and this is a different app.
func scan<T: Generable>(_ image: CGImage, as type: T.Type) async throws -> T {
    let text = try await CoreAI.read(image)
    return try await CoreAI.extract(text, as: type)
}

// ── One type, as an example. Yours goes here instead. ──────────────────────────────────────

@Generable
struct Receipt: Sendable {
    @Guide(description: "Merchant or store name, as printed")
    var merchant: String
    @Guide(description: "Grand total as a number, no currency symbol")
    var total: Double
    @Guide(description: "Date in YYYY-MM-DD")
    var date: String
}
