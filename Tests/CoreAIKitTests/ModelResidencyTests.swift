// ModelResidencyTests.swift — the eviction order behind the op layer's memory budget.
//
// The thing being protected is an app that calls three ops in sequence on a phone: without
// a ceiling every op model stays resident forever and the third load is a jetsam. None of
// that needs a model or a device — what has to be right is *which* model is dropped, and
// that a model currently in use never is.

import Foundation
import Testing

@testable import CoreAIOps

struct ModelResidencyTests {
    private static let gigabyte: Int64 = 1_000_000_000

    /// Records which models were released, in order.
    private actor Releases {
        private(set) var dropped: [String] = []
        func record(_ id: String) { dropped.append(id) }
        /// A release closure for `id` — non-async, as `register` requires.
        nonisolated func release(_ id: String) -> @Sendable () -> Void {
            { Task { await self.record(id) } }
        }
        /// Releases are handed off to a `Task`, so give them a turn to land. `expecting` is
        /// how many to wait for; 0 waits out the window to prove none arrive.
        func settled(expecting: Int = 1) async -> [String] {
            for _ in 0..<20 where dropped.count < expecting {
                try? await Task.sleep(for: .milliseconds(5))
            }
            if expecting == 0 { try? await Task.sleep(for: .milliseconds(50)) }
            return dropped
        }
    }

    private func residency(available: Int64) async -> (ModelResidency, Releases) {
        let residency = ModelResidency()
        await residency.setAvailable(available)
        return (residency, Releases())
    }

    private func register(
        _ residency: ModelResidency, _ releases: Releases, _ ids: [String], bytes: Int64
    ) async {
        for id in ids {
            await residency.register(
                ResidentModel(kind: "chat", id: id), bytes: bytes, pinned: false,
                release: releases.release(id))
        }
    }

    @Test func admitDropsTheLeastRecentlyUsedFirst() async throws {
        // 2 GB free, 512 MB reserved → 1.5 GB of headroom for a 2 GB load.
        let (residency, releases) = await residency(available: 2 * Self.gigabyte)
        await register(residency, releases, ["a", "b", "c"], bytes: Self.gigabyte)
        await residency.touch(ResidentModel(kind: "chat", id: "a"))  // a is now the newest

        await residency.admit(
            ResidentModel(kind: "chat", id: "d"), bytes: 2 * Self.gigabyte)

        // One eviction is enough (1.5 GB + 1 GB ≥ 2 GB), and it is the oldest — not `a`,
        // which was just touched, and not `c`, which is newer than `b`.
        #expect(await releases.settled() == ["b"])
    }

    @Test func aModelInUseIsNeverEvicted() async throws {
        let (residency, releases) = await residency(available: 2 * Self.gigabyte)
        await register(residency, releases, ["a", "b", "c"], bytes: Self.gigabyte)
        // `a` is the least-recently-used and would be the first to go, but an op is
        // running on it.
        await residency.pin(ResidentModel(kind: "chat", id: "a"))

        await residency.admit(
            ResidentModel(kind: "chat", id: "d"), bytes: 2 * Self.gigabyte)

        // Dropping `a` would free nothing (the running op still holds it) and cost a
        // reload, so the next-oldest goes instead.
        #expect(await releases.settled() == ["b"])
    }

    @Test func aFittingLoadEvictsNothing() async throws {
        let (residency, releases) = await residency(available: 8 * Self.gigabyte)
        await register(residency, releases, ["a", "b"], bytes: Self.gigabyte)

        await residency.admit(ResidentModel(kind: "chat", id: "c"), bytes: Self.gigabyte)

        #expect(await residency.resident().count == 2)
        #expect(await releases.settled(expecting: 0).isEmpty)
    }

    @Test func dropIdleKeepsWhatIsRunning() async throws {
        let (residency, releases) = await residency(available: 8 * Self.gigabyte)
        await register(residency, releases, ["a", "b", "c"], bytes: Self.gigabyte)
        await residency.pin(ResidentModel(kind: "chat", id: "b"))

        let dropped = await residency.dropIdle()

        #expect(dropped == 2)
        #expect(await residency.resident() == ["chat:b"])
        #expect(await releases.settled(expecting: 2).sorted() == ["a", "c"])
    }

    @Test func unpinningMakesAModelEvictableAgain() async throws {
        let (residency, releases) = await residency(available: 8 * Self.gigabyte)
        await register(residency, releases, ["a"], bytes: Self.gigabyte)
        await residency.pin(ResidentModel(kind: "chat", id: "a"))
        #expect(await residency.dropIdle() == 0)

        await residency.unpin(ResidentModel(kind: "chat", id: "a"))

        #expect(await residency.dropIdle() == 1)
        #expect(await releases.settled() == ["a"])
    }

    @Test func residentModelsAreNewestFirst() async throws {
        let (residency, releases) = await residency(available: 8 * Self.gigabyte)
        await register(residency, releases, ["a", "b"], bytes: Self.gigabyte)
        await residency.touch(ResidentModel(kind: "chat", id: "a"))

        #expect(await residency.resident() == ["chat:a", "chat:b"])
    }

    /// The footprint estimate is what every admit decision is made from, so a wrong number
    /// here is a wrong decision everywhere.
    @Test func footprintComesFromTheCatalogEntry() {
        // Qwen3 0.6B: 352 MB on macOS, 454 MB on iOS.
        let chat = ModelResidency.estimatedBytes(
            for: ResidentModel(kind: ResidentKind.chat, id: "qwen3-0.6b"))
        #expect(chat == 352_000_000 || chat == 454_000_000)

        // The meeting transcriber loads the diarizer beside its ASR model, so it must not
        // be planned as if it were the ASR model alone.
        let asr = ModelResidency.estimatedBytes(
            for: ResidentModel(kind: ResidentKind.transcriber, id: "whisper-large-v3-turbo"))
        let meeting = ModelResidency.estimatedBytes(
            for: ResidentModel(kind: ResidentKind.meeting, id: "whisper-large-v3-turbo"))
        #expect(meeting > asr)

        // An id with no catalog entry still gets a footprint, so an unknown model displaces
        // something rather than stacking silently.
        #expect(
            ModelResidency.estimatedBytes(for: ResidentModel(kind: "chat", id: "nope")) > 0)
    }
}

/// The cache contract the ops rely on, unchanged by the residency work: concurrent first
/// calls share one load, and a failed load is not remembered.
struct ResidentCacheTests {
    private actor Counter {
        private(set) var count = 0
        func bump() { count += 1 }
    }

    private struct LoadFailure: Error {}

    @Test func concurrentFirstCallsShareOneLoad() async throws {
        let cache = ResidentCache<String>(kind: "test-shared")
        let loads = Counter()

        async let first = cache.value(for: "m") {
            await loads.bump()
            try? await Task.sleep(for: .milliseconds(20))
            return "loaded"
        }
        async let second = cache.value(for: "m") {
            await loads.bump()
            return "loaded"
        }

        #expect(try await first == "loaded")
        #expect(try await second == "loaded")
        #expect(await loads.count == 1)
    }

    @Test func aFailedLoadIsNotCached() async throws {
        let cache = ResidentCache<String>(kind: "test-retry")
        let attempts = Counter()

        await #expect(throws: LoadFailure.self) {
            try await cache.value(for: "m") {
                await attempts.bump()
                throw LoadFailure()
            }
        }
        // The retry must actually re-run the load rather than replay the failure.
        let value = try await cache.value(for: "m") {
            await attempts.bump()
            return "loaded"
        }

        #expect(value == "loaded")
        #expect(await attempts.count == 2)
    }

    @Test func aDroppedModelReloadsRatherThanFailing() async throws {
        let cache = ResidentCache<String>(kind: "test-drop")
        let loads = Counter()
        let load: @Sendable () async throws -> String = {
            await loads.bump()
            return "loaded"
        }

        _ = try await cache.value(for: "m", load: load)
        await cache.drop("m")
        let reloaded = try await cache.value(for: "m", load: load)

        #expect(reloaded == "loaded")
        #expect(await loads.count == 2)
    }
}
