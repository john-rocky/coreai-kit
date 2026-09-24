import XCTest

@testable import CoreAIKit
@testable import CoreAIKitCore

/// A `catalog:` initializer downloads the revision its catalog entry pins — the bytes that were
/// gated — never the model repo's `main`. These three built their `ModelID`s without the pin
/// until 2026-09-24, so they downloaded whatever `main` held.
@available(macOS 27, iOS 27, *)
final class CatalogPinTests: XCTestCase {
    private func pinnedEntry(_ id: String) throws -> (CatalogEntry, String) {
        let entry = try XCTUnwrap(ModelCatalog.builtin.entry(id: id))
        return (entry, try XCTUnwrap(entry.revision))
    }

    func testMineruDownloadsThePinnedRevision() throws {
        let (entry, pin) = try pinnedEntry("mineru2.5-pro")
        let bundles = KitMineruReader.bundles(for: entry)
        XCTAssertEqual(bundles.vision, ModelID(entry.repo, path: "vision", revision: pin))
        XCTAssertEqual(bundles.decoder, ModelID(entry.repo, path: "decoder", revision: pin))
    }

    func testGlmOcrDownloadsThePinnedRevision() throws {
        let (entry, pin) = try pinnedEntry("glm-ocr")
        let bundles = KitGlmOcrReader.bundles(for: entry)
        XCTAssertEqual(bundles.vision, ModelID(entry.repo, path: "vision", revision: pin))
        XCTAssertEqual(bundles.decoder, ModelID(entry.repo, path: "decoder", revision: pin))
    }

    /// The preset's platform path is the catalog variant's, so pinning the preset downloads
    /// exactly what the entry declares.
    func testNemotronDownloadsThePinnedRevision() throws {
        let (entry, pin) = try pinnedEntry("nemotron-3.5-asr-streaming-0.6b")
        let model = KitNemotronModel.model(for: entry)
        XCTAssertEqual(model.revision, pin)
        XCTAssertEqual(model.repo, entry.repo)
        XCTAssertEqual(model.resolvedPath, entry.variant?.path)
    }
}
