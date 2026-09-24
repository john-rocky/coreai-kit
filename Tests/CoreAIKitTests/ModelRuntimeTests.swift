// ModelRuntimeTests.swift — which bundles must prefill one token per step, checkable without
// weights. The zoo's decode-only ports are S=1 graphs: a multi-token prefill chunk on one stops
// the process with a shape-substitution fatal, so the rule has to hold on every engine,
// `.auto` included.

import Testing

@testable import CoreAIKit

struct ModelRuntimeTests {
    @available(macOS 27, iOS 27, *)
    @Test func decodeOnlyPortsPrefillOneTokenOnEveryEngine() {
        for variant in [EngineVariant.auto, .pipelined, .sequential, .staticShape] {
            #expect(
                ModelRuntime.needsSingleTokenPrefill(
                    bundleName: "qwen3_5_2b_decode_int8hu_block32_sym", engineVariant: variant))
        }
    }

    @available(macOS 27, iOS 27, *)
    @Test func otherBundlesKeepChunkingUnlessLoadedPipelined() {
        // The official Qwen3 0.6B bundle's directory; its graph prefills in one pass.
        #expect(!ModelRuntime.needsSingleTokenPrefill(bundleName: "macos", engineVariant: .auto))
        #expect(!ModelRuntime.needsSingleTokenPrefill(bundleName: "macos", engineVariant: .sequential))
        #expect(ModelRuntime.needsSingleTokenPrefill(bundleName: "macos", engineVariant: .pipelined))
    }
}
