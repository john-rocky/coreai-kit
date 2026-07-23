// ModelAsk — a process-global host for one on-device model behind FoundationModels'
// `LanguageModelSession`, shared by the app UI and the App Intent that Siri invokes. The model is
// loaded once and kept warm; every ask runs on a fresh, tool-less session (no tools, no retrieval —
// the WWDC 347 act/communicate leg is absent by construction).
//
// MODEL = a SIDE-LOADED, AOT-compiled, ANE-oriented qwen3-0.6B bundle (`Model/qwen_ane`,
// "compute-ANE" / h18p). Why this shape:
//   • GPU models (Gemma, dynamic qwen) can't run from a backgrounded Siri intent — iOS rejects GPU
//     command submission in the background (the pipelined engine then crashes/corrupts). ANE work,
//     which Apple's own background Siri model uses, is the candidate for hands-free background.
//   • AOT (pre-compiled for h18p) avoids the on-device compile that crashed the non-AOT bundle
//     (the 10-15 min ANE compile of a multi-GB .mlirb).
// `engineVariant: .auto` lets the kit pick the static-shape (Neural Engine) engine for this
// chunked-static AOT bundle. (If iOS dispatches it to GPU anyway — "ANE closed this beta" — the
// background case will still fail; that's then an OS limit, settled on device.)

import CoreAIKit
import Foundation
import FoundationModels

public actor ModelHost {
    public static let shared = ModelHost()

    /// `/no_think` keeps qwen3 (a thinking model) snappy and single-turn.
    public static let askInstructions =
        "You are a helpful assistant. Answer the question clearly and concisely in a few "
        + "sentences. /no_think"

    /// Fallback when no side-loaded bundle is present (e.g. the Mac gate): a catalog model.
    private let fallbackCatalog: ModelID = .qwen3_4B

    private var loaded: KitLanguageModel?
    private var loadTask: Task<KitLanguageModel, Error>?

    public init() {}

    public var isReady: Bool { loaded != nil }

    @discardableResult
    public func ensureReady(
        progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> Bool {
        _ = try await loadedModel(progress: progress)
        return true
    }

    private func loadedModel(
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> KitLanguageModel {
        if let loaded { return loaded }
        if let loadTask { return try await loadTask.value }
        let fallback = fallbackCatalog
        let task = Task { () throws -> KitLanguageModel in
            #if os(iOS)
            if let staged = Self.stagedSideloadedBundle() {
                // AOT, ANE-oriented bundle; `.auto` picks the static-shape (Neural Engine) engine.
                return try await KitLanguageModel(bundleAt: staged, engineVariant: .auto)
            }
            #endif
            return try await KitLanguageModel(model: fallback, downloadProgress: progress)
        }
        loadTask = task
        do {
            let m = try await task.value
            loaded = m
            loadTask = nil
            await warm(m)
            return m
        } catch {
            loadTask = nil
            throw error
        }
    }

    /// Compile the sampler graph + touch the weights so the first real ask doesn't pay warm-up cost.
    private func warm(_ model: KitLanguageModel) async {
        let session = LanguageModelSession(
            model: model, instructions: "Reply with the single word OK. /no_think")
        _ = try? await session.respond(to: "ping")
    }

    /// If the AOT bundle ships inside the app (`Model/qwen_ane`), copy it once into a writable cache
    /// and return that URL (the runtime loads from a writable location, not the read-only signed
    /// .app). Small (~0.7 GB) so the one-time local copy is quick; no network. Returns nil if not
    /// side-loaded → caller falls back to the catalog download.
    private static func stagedSideloadedBundle() -> URL? {
        let fm = FileManager.default
        let bundled = (Bundle.main.resourceURL ?? Bundle.main.bundleURL)
            .appendingPathComponent("Model/qwen_ane", isDirectory: true)
        guard fm.fileExists(atPath: bundled.appendingPathComponent("metadata.json").path) else {
            return nil
        }
        let dst = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CoreAIKit/SideloadedModel/qwen_ane", isDirectory: true)
        if fm.fileExists(atPath: dst.appendingPathComponent("metadata.json").path) {
            return dst
        }
        do {
            let parent = dst.deletingLastPathComponent()
            try fm.createDirectory(at: parent, withIntermediateDirectories: true)
            let tmp = parent.appendingPathComponent(".staging-qwen_ane", isDirectory: true)
            try? fm.removeItem(at: tmp)
            try fm.copyItem(at: bundled, to: tmp)
            try? fm.removeItem(at: dst)
            try fm.moveItem(at: tmp, to: dst)
            return dst
        } catch {
            return nil
        }
    }

    /// Answer a free-text question. Tool-less, no retrieval. The warm engine is reused across asks —
    /// the consecutive-generation bug (1st ask fast, 2nd "something went wrong") is fixed in the
    /// engine itself (v0.1.1-zoo D1: a stream break at EOS stops the engine instead of running to
    /// maxTokens, so turn 2 no longer collides with turn 1). On a genuine error the model is dropped
    /// so the next ask rebuilds fresh.
    public func ask(question: String) async throws -> String {
        let m = try await loadedModel(progress: nil)
        do {
            let session = LanguageModelSession(model: m, instructions: Self.askInstructions)
            let response = try await session.respond(to: question)
            return response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            loaded = nil
            loadTask = nil
            throw error
        }
    }
}
