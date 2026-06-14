// ModelAsk — a process-global host for one on-device Gemma 4 model behind FoundationModels'
// `LanguageModelSession`, shared by the app UI and the App Intent that Siri invokes. The model is
// loaded once and kept warm; every ask runs on a fresh, tool-less session.
//
// This is the ask-anything rework (SIRI-GEMMA): the model answers free-text questions from its own
// knowledge — no notes, no retrieval, no tools. Tool-less + no retrieval means the Lethal Trifecta
// has nothing to actuate and no untrusted data channel, so the whole prompt-injection surface the
// notes-RAG version had to defend is simply absent (WWDC 347, by construction). Fresh per turn also
// sidesteps the pipelined engine's multi-turn KV tax (GEMMA-KIT v1 re-prefills every turn).

import CoreAIKit
import Foundation
import FoundationModels

/// Which Gemma 4 E2B bundle to load on each platform. The device path is the make-or-break detail
/// of this session, so it is a pure, unit-testable function (`SiriAskCoreTests` checks both
/// branches on a Mac, where only the macOS branch can actually run).
public enum GemmaSource {
    public enum Platform: Sendable { case iOS, macOS }

    /// The Hugging Face repo holding every Gemma 4 E2B variant + the paired QAT PLE tables.
    public static let repo = "mlboydaisuke/gemma-4-E2B-CoreAI"

    /// QAT PLE tables (shared by every E2B decode bundle; a QAT bundle pairs with QAT tables).
    public static let tablesPath = "ios-frontend/gemma4_qat_gather_raw"

    /// iOS: the AOT-compiled `…_tbl_aotc_h18p` bundle. REQUIRED on device — the plain bundle's
    /// ~2 GB of graph constants crash the on-device GPU specializer (`LLVM ERROR: Failed to
    /// allocate mmap'd buffer`), so the bundle is shipped pre-compiled for the iPhone 17 Pro GPU
    /// family (h18p). Device-verified: iPhone 17 Pro decode ~30 tok/s, oracle 8/8, ~4.45 GB peak
    /// (with the increased-memory entitlement). See `ondevice/_gemma4_qat_RESULTS.md`.
    public static let iOSDecoderPath = "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl_aotc_h18p"

    /// macOS: the plain `…_tbl` bundle. The Mac runtime JIT-specializes the large-constant graph
    /// fine (there is no h18p Mac AOT artifact), and GEMMA-KIT validated this exact bundle on a Mac.
    public static let macOSDecoderPath = "gpu-pipelined/gemma4_e2b_qat_decode_int4lin_tbl"

    /// The Gemma 4 E2B model id (decoder bundle + paired tables) for a given platform.
    public static func gemma4E2B(for platform: Platform) -> GemmaModelID {
        let decoderPath = platform == .iOS ? iOSDecoderPath : macOSDecoderPath
        return GemmaModelID(
            decoder: ModelID(repo, path: decoderPath),
            tables: ModelID(repo, path: tablesPath),
            arch: .gemma4)
    }

    /// The Gemma 4 E2B model for the platform this build targets.
    public static var gemma4E2B: GemmaModelID {
        #if os(iOS)
        return gemma4E2B(for: .iOS)
        #else
        return gemma4E2B(for: .macOS)
        #endif
    }
}

/// Loads + warms a `KitGemmaModel` and answers free-text questions. An `actor` because the
/// underlying engine assumes serial use (it traps on concurrent generate calls), and because the
/// app UI and a Siri-invoked intent can both reach `shared` in the same process — the actor
/// serializes them so a Siri ask waits for an in-flight UI generation rather than colliding.
public actor ModelHost {
    /// The app's shared, warm-kept instance. Model = Gemma 4 E2B (Google's official QAT int4): a
    /// compact on-device model that runs on both iPhone 17 Pro (AOT) and Mac (JIT), behind the
    /// FoundationModels `LanguageModel` protocol via GEMMA-KIT's `KitGemmaModel`.
    public static let shared = ModelHost(model: GemmaSource.gemma4E2B)

    /// A light system instruction. Gemma 4 answers directly (thinking is off by default in the
    /// renderer); this just steers it toward concise, useful answers for the voice/read-aloud demo.
    public static let askInstructions =
        "You are Gemma, a helpful on-device assistant. Answer the user's question clearly and "
        + "concisely in a few sentences."

    public enum Phase: Sendable, Equatable {
        case idle, downloading(Double), loading, ready, error(String)
    }

    /// What to load: a hosted model (decoder + tables downloaded on demand) or local bundle dirs.
    private enum Source: Sendable {
        case hub(GemmaModelID)
        case localBundle(decoder: URL, tables: URL)
    }

    private let source: Source
    private var model: KitGemmaModel?
    private var loadTask: Task<KitGemmaModel, Error>?

    /// Load a hosted Gemma 4 model (decoder bundle + paired PLE tables) from the Hub.
    public init(model: GemmaModelID) {
        self.source = .hub(model)
    }

    /// Load a local decode-bundle directory + a local PLE-tables directory (the gate's
    /// `--decoder/--tables` flags — point at freshly exported dirs to skip the multi-GB download).
    public init(localDecoderAt decoder: URL, tablesAt tables: URL) {
        self.source = .localBundle(decoder: decoder, tables: tables)
    }

    public var isReady: Bool { model != nil }

    /// Idempotent load + warm-up. Reuses an in-flight load if one is running, so the UI's
    /// prewarm-on-launch and a racing intent share a single download/compile. `progress` reports
    /// download fraction (the multi-GB first run dominates a cold start).
    @discardableResult
    public func ensureReady(
        progress: (@Sendable (DownloadProgress) -> Void)? = nil
    ) async throws -> Bool {
        _ = try await loadedModel(progress: progress)
        return true
    }

    private func loadedModel(
        progress: (@Sendable (DownloadProgress) -> Void)?
    ) async throws -> KitGemmaModel {
        if let model { return model }
        if let loadTask { return try await loadTask.value }
        let source = self.source
        let task = Task { () throws -> KitGemmaModel in
            switch source {
            case .hub(let id):
                return try await KitGemmaModel(model: id, downloadProgress: progress)
            case .localBundle(let decoder, let tables):
                return try await KitGemmaModel(decoderBundleAt: decoder, tablesAt: tables)
            }
        }
        loadTask = task
        do {
            let loaded = try await task.value
            model = loaded
            loadTask = nil
            await warm(loaded)
            return loaded
        } catch {
            loadTask = nil
            throw error
        }
    }

    /// Compile the sampler graph and touch the weights so the first real ask doesn't pay warm-up
    /// cost — critical when the caller is Siri (a cold multi-GB model blows the intent's time
    /// budget). Done via one tiny real `respond` on the proven FM path; best-effort, so a warm-up
    /// hiccup never fails the load.
    private func warm(_ model: KitGemmaModel) async {
        let session = LanguageModelSession(model: model, instructions: "Reply with the word OK.")
        _ = try? await session.respond(to: "ping")
    }

    /// Answer a free-text question with the on-device model. Tool-less, no retrieval — the model
    /// answers from its own knowledge.
    public func ask(question: String) async throws -> String {
        let model = try await loadedModel(progress: nil)
        let session = LanguageModelSession(model: model, instructions: Self.askInstructions)
        let response = try await session.respond(to: question)
        return response.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
