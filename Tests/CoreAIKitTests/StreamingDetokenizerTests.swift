import XCTest

@testable import CoreAIKit

final class StreamingDetokenizerTests: XCTestCase {
    /// Byte-level fake tokenizer: each token id maps to raw UTF-8 bytes, and decoding
    /// replaces invalid sequences with U+FFFD exactly like a byte-fallback tokenizer.
    private func detokenizer(_ table: [Int: [UInt8]]) -> StreamingDetokenizer {
        StreamingDetokenizer { tokens in
            String(decoding: tokens.flatMap { table[$0] ?? [] }, as: UTF8.self)
        }
    }

    func testASCIIPassthrough() {
        var detok = detokenizer([1: Array("Hello".utf8), 2: Array(" world".utf8)])
        XCTAssertEqual(detok.consume(1), "Hello")
        XCTAssertEqual(detok.consume(2), " world")
    }

    func testMultiByteCharacterSplitAcrossTokens() {
        // "あ" = E3 81 82, one byte per token: held until the third byte lands.
        var detok = detokenizer([1: [0xE3], 2: [0x81], 3: [0x82]])
        XCTAssertEqual(detok.consume(1), "")
        XCTAssertEqual(detok.consume(2), "")
        XCTAssertEqual(detok.consume(3), "あ")
    }

    /// Regression: a stray raw-byte token must not starve the rest of the stream.
    /// The old any-U+FFFD hold kept every later delta back, and a long CJK turn
    /// ended with zero events ("Session ended without producing a response").
    func testStrayInvalidByteDoesNotStarveTheStream() {
        var detok = detokenizer([
            1: Array("こんにちは".utf8),
            2: [0x82],  // lone continuation byte — permanently invalid
            3: Array("、世界のみなさん".utf8),
            4: Array("。お元気ですか。".utf8),
        ])
        XCTAssertEqual(detok.consume(1), "こんにちは")
        XCTAssertEqual(detok.consume(2), "")  // may still be a character in progress
        // The invalid byte is now interior — emit it as U+FFFD and keep streaming.
        XCTAssertEqual(detok.consume(3), "\u{FFFD}、世界のみなさん")
        XCTAssertEqual(detok.consume(4), "。お元気ですか。")
    }

    func testGarbageRunRecoversAtNextCleanCharacter() {
        var detok = detokenizer([
            1: [0x82], 2: [0x83],  // two invalid bytes in a row
            3: Array("OK".utf8),
        ])
        XCTAssertEqual(detok.consume(1), "")
        XCTAssertEqual(detok.consume(2), "")
        let recovered = detok.consume(3)
        XCTAssertTrue(recovered.hasSuffix("OK"))
        XCTAssertFalse(recovered.isEmpty)
    }

    func testCleanTextBeforePartialCharacterIsNotHeld() {
        // One token carrying complete text plus the first byte of the next character.
        var detok = detokenizer([
            1: Array("です".utf8) + [0xE3],
            2: [0x81, 0x84],  // completes "い"
        ])
        XCTAssertEqual(detok.consume(1), "です")
        XCTAssertEqual(detok.consume(2), "い")
    }

    func testRetainedContextTokenIsNotReEmitted() {
        var detok = detokenizer([1: Array("abc".utf8), 2: Array("def".utf8)])
        XCTAssertEqual(detok.consume(1), "abc")
        XCTAssertEqual(detok.consume(2), "def")  // not "abcdef"
    }
}
