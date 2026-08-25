// ModelResidency.swift — one resident-model budget shared by every op cache.
//
// Ops cache the model they load for the life of the process, and until this file existed
// nothing ever released one. An app that calls `transcribe` then `summarize` then
// `caption` keeps three models resident and is jetsammed on a phone — which is exactly the
// capacity planning the op layer promises the adopter will not have to do.
//
// Every op cache now admits its load through `ModelResidency`: before a new model is
// loaded, the least-recently-used models that nothing is currently running are dropped
// until the new one fits under the process's remaining allowance. A memory-pressure
// notification drops all of them.
//
// Three properties the design keeps, because breaking any of them is worse than the
// problem being solved:
//
//   • **Eviction never cancels a running op.** A model in use is pinned and skipped; and
//     even if it were dropped, ARC keeps it alive for whoever still holds a reference. The
//     worst an eviction can cost is a reload.
//   • **Re-loading after eviction is a cache miss, not an error.** The next call is slower.
//     No op ever fails because its model was evicted.
//   • **Nothing here frees memory itself.** It drops the cache's strong reference; the
//     bytes come back when the last holder releases them. Accounting is therefore
//     optimistic by exactly the length of one in-flight op.
//
// ```swift
// await CoreAI.evictModels()          // drop everything idle, e.g. entering the background
// await CoreAI.residentModels()       // ["chat:qwen3-4b", "transcriber:whisper-large-v3-turbo"]
// ```

import CoreAIKit
import Foundation

#if os(iOS)
import os
#endif

/// One cached model, identified by the cache holding it and the catalog id it was loaded
/// from. The kind is the cache's own namespace, not the catalog's `kind`: one catalog id
/// can be resident twice in different roles (an ASR model alone and inside the meeting
/// transcriber) and those are two loads with two footprints.
struct ResidentModel: Sendable, Hashable, CustomStringConvertible {
    let kind: String
    let id: String

    var description: String { "\(kind):\(id)" }
}

/// The process-wide resident-model budget. Every op cache calls `admit` before loading and
/// `register` after; ops that hold a model for the length of a turn pin it while they run.
actor ModelResidency {
    static let shared = ModelResidency()

    private struct Entry {
        var bytes: Int64
        var lastUsed: UInt64
        var pins: Int
        /// Drops the owning cache's strong reference. Called synchronously from the actor,
        /// so it must not await the cache back — the implementations hand off to a `Task`.
        var release: @Sendable () -> Void
    }

    private var entries: [ResidentModel: Entry] = [:]
    private var clock: UInt64 = 0
    private var pressureSource: (any DispatchSourceMemoryPressure)?

    /// Test seam: pretend the process has this much room left. There is no way to make a
    /// 64 GB Mac genuinely run out of memory inside a unit test, and the eviction order is
    /// the part worth testing. `nil` — always, outside tests — asks the system.
    private var availableBytesOverride: Int64?

    func setAvailable(_ bytes: Int64?) {
        availableBytesOverride = bytes
    }

    /// Bytes deliberately left unplanned. The app's own working set — its UI, the image it
    /// just decoded, the audio it is buffering — lives in the same allowance, and the
    /// footprint numbers below undercount (see `estimatedBytes`). Planning to the last byte
    /// is how a budget turns into a crash.
    private static let reserve: Int64 = 512 * 1024 * 1024

    /// Assumed footprint of a model with no catalog entry — big enough that an unknown
    /// model still displaces something rather than silently stacking.
    private static let unknownSizeMB = 1_500

    /// GLiNER2-PII, the one op model that is a pinned preset rather than a catalog entry.
    private static let gliner2SizeMB = 500

    /// The diarizer the meeting transcriber loads beside its ASR model.
    private static let diarizerSizeMB = 451

    // MARK: - The contract op caches use

    /// Makes room for a model of `bytes` before it is loaded. Drops least-recently-used
    /// unpinned models until the new one fits, or until nothing droppable is left — in
    /// which case the load proceeds anyway. Refusing to load would turn a memory budget
    /// into a functional failure, and the caller has no better answer than we do.
    func admit(_ key: ResidentModel, bytes: Int64) {
        var headroom = availableBytes() - Self.reserve
        guard bytes > headroom else { return }
        for candidate in evictionOrder(excluding: key) {
            guard let entry = entries[candidate] else { continue }
            drop(candidate)
            // The bytes are not visible to `availableBytes()` until the last holder
            // releases the model, so they are counted here rather than re-measured.
            headroom += entry.bytes
            if bytes <= headroom { break }
        }
    }

    /// Records a model as resident. `pinned` covers the window between the load starting
    /// and its value existing, so a concurrent `admit` cannot evict a model that is still
    /// being loaded — a wasted download is the one eviction that costs more than it saves.
    func register(
        _ key: ResidentModel, bytes: Int64, pinned: Bool,
        release: @escaping @Sendable () -> Void
    ) {
        clock += 1
        entries[key] = Entry(
            bytes: bytes, lastUsed: clock, pins: pinned ? 1 : 0, release: release)
        startPressureWatch()
    }

    /// Forgets a model without releasing it — for a load that failed, whose cache entry is
    /// already gone.
    func forget(_ key: ResidentModel) {
        entries[key] = nil
    }

    /// Marks a cache hit, so the model moves to the back of the eviction order.
    func touch(_ key: ResidentModel) {
        guard entries[key] != nil else { return }
        clock += 1
        entries[key]?.lastUsed = clock
    }

    /// Protects a model for the length of an op. Deliberately does not reorder: `touch` is
    /// what records use, and a cache hand-out already touched. If pinning also promoted,
    /// the least-recently-used model being busy would look like the most recent one and
    /// something genuinely idle would outlive it.
    func pin(_ key: ResidentModel) {
        guard entries[key] != nil else { return }
        entries[key]?.pins += 1
    }

    func unpin(_ key: ResidentModel) {
        guard let pins = entries[key]?.pins, pins > 0 else { return }
        entries[key]?.pins = pins - 1
    }

    /// Fire-and-forget unpin, so a `defer` can release a pin without awaiting.
    nonisolated func unpinLater(_ key: ResidentModel) {
        Task { await self.unpin(key) }
    }

    /// Drops every model nothing is currently running. What memory pressure triggers, and
    /// what `CoreAI.evictModels()` exposes.
    @discardableResult
    func dropIdle() -> Int {
        let idle = entries.filter { $0.value.pins == 0 }.map(\.key)
        for key in idle { drop(key) }
        return idle.count
    }

    /// Currently-resident models, most-recently-used first — for a memory HUD or a test.
    func resident() -> [String] {
        entries.sorted { $0.value.lastUsed > $1.value.lastUsed }.map { "\($0.key)" }
    }

    // MARK: - Internals

    private func drop(_ key: ResidentModel) {
        guard let entry = entries.removeValue(forKey: key) else { return }
        entry.release()
    }

    /// Droppable models, least-recently-used first. Pinned models are not candidates: they
    /// are in use, so dropping one frees nothing now and costs a reload later.
    private func evictionOrder(excluding key: ResidentModel) -> [ResidentModel] {
        entries
            .filter { $0.key != key && $0.value.pins == 0 }
            .sorted { $0.value.lastUsed < $1.value.lastUsed }
            .map(\.key)
    }

    /// What the process may still allocate.
    private func availableBytes() -> Int64 {
        if let availableBytesOverride { return availableBytesOverride }
        #if os(iOS)
        // The real per-process allowance, which is what jetsam actually enforces — it
        // already accounts for everything the app has allocated, models included.
        return Int64(os_proc_available_memory())
        #else
        // macOS has no per-process ceiling and swaps rather than kills, so there is no
        // equivalent number to read. Cap resident models at a share of physical memory so
        // a long-running Mac process still evicts instead of growing without bound.
        let ceiling = Int64(ProcessInfo.processInfo.physicalMemory / 2)
        let resident = entries.values.reduce(Int64(0)) { $0 + $1.bytes }
        return max(0, ceiling - resident)
        #endif
    }

    private func startPressureWatch() {
        guard pressureSource == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler {
            // Both levels drop everything idle: by the time the system says "warning" on a
            // phone, a multi-gigabyte model cache is the largest thing we can give back,
            // and the cost of being wrong is one reload.
            Task { await ModelResidency.shared.dropIdle() }
        }
        source.resume()
        pressureSource = source
    }

    /// Footprint estimate for a model about to be loaded.
    ///
    /// The catalog's `sizeMB` is the *download* size. It is the only number available —
    /// Core AI reports nothing about a loaded graph's resident footprint — and weights
    /// dominate, so it is a usable proxy. It undercounts KV cache, activations and the
    /// runtime's own allocations, which is why `reserve` is deliberately large. Read from
    /// `ModelCatalog.builtin` rather than the live catalog because this is called on the
    /// load path and must not become a network round trip.
    static func estimatedBytes(for key: ResidentModel) -> Int64 {
        var sizeMB = ModelCatalog.builtin.entry(id: key.id)?.variant?.sizeMB ?? unknownSizeMB
        switch key.kind {
        case ResidentKind.meeting:
            sizeMB += diarizerSizeMB  // the composite loads the diarizer beside the ASR model
        case ResidentKind.extractor:
            sizeMB = gliner2SizeMB  // a pinned preset, not a catalog entry
        default:
            break
        }
        return Int64(sizeMB) * 1_000_000
    }
}

/// The cache namespaces. One per distinct load, so the estimator and any memory HUD can
/// tell an ASR model loaded on its own from the same model loaded inside a composite.
enum ResidentKind {
    static let chat = "chat"
    static let transcriber = "transcriber"
    static let meeting = "meeting"
    static let audio = "audio"
    static let musician = "musician"
    static let separator = "separator"
    static let speaker = "speaker"
    static let vision = "vision"
    static let detector = "detector"
    static let reader = "reader"
    static let upscaler = "upscaler"
    static let depth = "depth"
    static let recognizer = "recognizer"
    static let embedder = "embedder"
    static let forecaster = "forecaster"
    static let extractor = "extractor"
    static let normalizer = "normalizer"
}

/// The shared body of every op cache: concurrent first calls share one load, a failed load
/// is not cached, and the loaded value is admitted to (and evictable from) the process-wide
/// residency budget.
///
/// Replaces thirteen hand-written copies of the same dictionary-of-Tasks — which is what
/// made the missing eviction a thirteen-place fix rather than a one-place one.
actor ResidentCache<Value: Sendable> {
    private let kind: String
    private var loads: [String: Task<Value, Error>] = [:]

    init(kind: String) {
        self.kind = kind
    }

    /// The cached model for `id`, loading it if necessary.
    func value(
        for id: String, load: @escaping @Sendable () async throws -> Value
    ) async throws -> Value {
        let key = ResidentModel(kind: kind, id: id)
        if let existing = loads[id] {
            await ModelResidency.shared.touch(key)
            return try await existing.value
        }

        // The load task is created and stored with no `await` in between: every suspension
        // point here is a window in which a second caller also sees an empty slot and
        // starts a duplicate download. Admission therefore happens inside the task, which
        // is also where it belongs — the eviction should land next to the allocation it is
        // making room for, not one hop earlier.
        let bytes = ModelResidency.estimatedBytes(for: key)
        let task = Task<Value, Error> {
            await ModelResidency.shared.admit(key, bytes: bytes)
            return try await load()
        }
        loads[id] = task
        // Registered pinned: the model does not exist yet, and evicting a load in flight
        // throws away a download to free memory that has not been allocated.
        await ModelResidency.shared.register(key, bytes: bytes, pinned: true) {
            // Hands off rather than awaiting: `release` is called from inside the residency
            // actor, and awaiting this cache from there is a cycle.
            Task { await ModelResidency.dropFromCache(self, id: id) }
        }
        do {
            let value = try await task.value
            await ModelResidency.shared.unpin(key)
            return value
        } catch {
            loads[id] = nil
            await ModelResidency.shared.forget(key)
            throw error
        }
    }

    /// Drops the cached load. The model itself lives until its last holder releases it.
    func drop(_ id: String) {
        loads[id] = nil
    }
}

extension ModelResidency {
    /// Bridges the non-generic release closure back to a generic cache.
    fileprivate static func dropFromCache<Value: Sendable>(
        _ cache: ResidentCache<Value>, id: String
    ) async {
        await cache.drop(id)
    }
}

/// Runs `body` with the model pinned. Ops that hold a model for a whole turn wrap the turn
/// in this: without it, a long generation is a window in which another op can evict the
/// model being generated with — which frees nothing (the turn still holds it) and costs a
/// reload on the next call.
func withPinnedModel<R: Sendable>(
    _ kind: String, _ id: String, _ body: @Sendable () async throws -> R
) async rethrows -> R {
    let key = ResidentModel(kind: kind, id: id)
    await ModelResidency.shared.pin(key)
    defer { ModelResidency.shared.unpinLater(key) }
    return try await body()
}

extension CoreAI {
    /// Drops every op model nothing is currently running, returning how many were dropped.
    ///
    /// The op layer does this on memory pressure by itself; call it directly at a moment
    /// the app knows is a good one — entering the background, closing the document the
    /// models were loaded for. Ops keep working afterwards: the next call reloads.
    @discardableResult
    public static func evictModels() async -> Int {
        await ModelResidency.shared.dropIdle()
    }

    /// The op models currently held, most-recently-used first, as `"kind:catalog-id"` —
    /// for a debug HUD, a memory test, or a log line explaining a slow call.
    public static func residentModels() async -> [String] {
        await ModelResidency.shared.resident()
    }
}
