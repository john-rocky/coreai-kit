// KitDocReaderBudgetTests.swift — Unlimited-OCR's decode stops where its decoder's KV cache ends,
// without a model. The published decoder holds 2,048 positions and the image prompt takes 115, so
// the n-th generated token's decode step writes position 115 + n − 1.
//
//     swift test --filter KitDocReaderBudget

import Testing

@testable import CoreAIKit

struct KitDocReaderBudgetTests {
    @available(macOS 27, iOS 27, *)
    @Test func maxTokensInsideTheCacheIsKept() {
        for n in [0, 1, 1024, 1933] {
            #expect(KitDocReader.tokenBudget(maxTokens: n, prefixLength: 115, capacity: 2048) == n)
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func maxTokensPastTheCacheStopsAtItsLastPosition() {
        for n in [1934, 2048, 4096] {
            let budget = KitDocReader.tokenBudget(maxTokens: n, prefixLength: 115, capacity: 2048)
            #expect(budget == 1933)
            #expect(115 + budget - 1 == 2047)  // the last decode step's position
        }
    }
}
