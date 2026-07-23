// ModelSourceTests — light sanity for the ask-anything core. The model story (loading + answering)
// needs the GPU and a real bundle, so it's covered by `swift run SiriAskGate`, not here; this just
// guards that the shared host is constructible without side effects.

import Testing

@testable import SiriAskCore

struct ModelSourceTests {
    @Test func sharedHostExists() async {
        let host = ModelHost.shared
        let ready = await host.isReady
        #expect(ready == false)  // not loaded until ensureReady/ask
    }
}
