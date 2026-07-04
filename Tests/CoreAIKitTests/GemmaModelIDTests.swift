import XCTest

@testable import CoreAIKit
@testable import CoreAIKitCore

final class GemmaModelIDTests: XCTestCase {
    /// The make-or-break device invariant: an iOS build must load the AOT `…_tbl_aotc_h18p`
    /// decoder (the plain bundle's ~2 GB of graph constants crash the on-device GPU
    /// specializer), while a Mac build loads the plain JIT `…_tbl` bundle. The `#if`
    /// mirrors the preset's own platform pick, so each platform's test run checks the
    /// branch it will actually load.
    func testE2BDecoderIsPlatformPicked() {
        #if os(iOS)
        XCTAssertEqual(
            GemmaModelID.gemma4E2B.decoder.path,
            "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p")
        #else
        XCTAssertEqual(
            GemmaModelID.gemma4E2B.decoder.path,
            "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl")
        #endif
    }

    /// A QAT decode bundle pairs with the QAT PLE tables, identical on every platform.
    func testE2BPairsQatTables() {
        XCTAssertEqual(
            GemmaModelID.gemma4E2B.tables.path, "ios-frontend/gemma4_qat_gather_raw")
    }

    /// The catalog variant paths must name the same decoders the preset loads — the catalog
    /// gates platform availability (and shows the download size) for exactly what
    /// ChatSession(catalog:) will resolve through GemmaModelID.byCatalogID.
    func testCatalogVariantsMatchPreset() {
        let entry = ModelCatalog.builtin.entry(id: "gemma-4-e2b")
        XCTAssertEqual(
            entry?.variants["macos"]?.path,
            "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl")
        XCTAssertEqual(
            entry?.variants["ios"]?.path,
            "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p")
    }
}
