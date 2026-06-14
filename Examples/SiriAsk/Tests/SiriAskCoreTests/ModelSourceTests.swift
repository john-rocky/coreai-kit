// ModelSourceTests — guards the make-or-break invariant of the SIRI-GEMMA rework: the iPhone build
// must load the AOT-compiled `…_tbl_aotc_h18p` bundle (the plain bundle crashes the on-device GPU
// specializer), while the Mac build loads the plain `…_tbl` bundle. This runs on a Mac (CI / no
// device), so it checks BOTH platform branches as pure data — the device path is verified here even
// though only the macOS branch can actually load.

import Testing

@testable import SiriAskCore

struct ModelSourceTests {
    @Test func deviceUsesAotBundle() {
        let ios = GemmaSource.gemma4E2B(for: .iOS)
        // The whole point: device decoder must be the AOT (`_aotc_h18p`) variant.
        #expect(ios.decoder.path == "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p")
        #expect(ios.decoder.path?.hasSuffix("_aotc_h18p") == true)
        #expect(ios.decoder.repo == GemmaSource.repo)
    }

    @Test func macUsesPlainBundle() {
        let mac = GemmaSource.gemma4E2B(for: .macOS)
        // Mac JIT-specializes the large-constant graph; it must NOT point at the h18p AOT artifact.
        #expect(mac.decoder.path == "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl")
        #expect(mac.decoder.path?.contains("aotc") == false)
        #expect(mac.decoder.repo == GemmaSource.repo)
    }

    @Test func bothShareTheQatTables() {
        let ios = GemmaSource.gemma4E2B(for: .iOS)
        let mac = GemmaSource.gemma4E2B(for: .macOS)
        // A QAT decode bundle must pair with the QAT PLE tables, identical across platforms.
        #expect(ios.tables.path == "ios-frontend/gemma4_qat_gather_raw")
        #expect(ios.tables == mac.tables)
        #expect(ios.tables.repo == GemmaSource.repo)
    }
}
