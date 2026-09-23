// SystemOneMCPServer.swift — the Model Context Protocol server over stdio: typed decisions as
// the tools `decide` and `models` for a coding agent's own sessions. `systemone mcp` (and
// `decide-cli mcp` in Examples/Decide) is this class over standard input and output:
//
//   claude mcp add systemone -- /path/to/systemone mcp          # Claude Code
//   codex mcp add systemone -- /path/to/systemone mcp           # Codex
//   {"mcpServers": {"systemone": {"command": "/path/to/systemone", "args": ["mcp"]}}}   # Cursor
//
// One JSON-RPC message per line in and out (`SystemOneMCP` owns the forms, both revisions of
// the protocol); log lines go to `log` (stderr by default), never to the output. The model
// loads on the first `decide` — or at launch with `run(preload: true)` — and answers one call
// at a time (`DecisionQueue`); protocol messages are answered as they arrive. A call that
// names another catalog id replaces the resident model (one at a time: two 2B models would
// not fit a phone). `run()` returns when the input reaches end of file — the client's way of
// stopping a stdio server — after the calls already in flight have answered, so a one-shot
// pipe works too: `printf '…tools/call…\n' | systemone mcp` prints the reply and exits. A CLI
// should ignore SIGPIPE so a client that closed its end shows up as `Failure.outputClosed`
// instead of a signal.

import Foundation

public final class SystemOneMCPServer: @unchecked Sendable {
    public enum Failure: Error, LocalizedError, Equatable {
        /// A reply could not be written: the client closed the pipe.
        case outputClosed(String)

        public var errorDescription: String? {
            switch self {
            case .outputClosed(let reason): return "output closed: \(reason)"
            }
        }
    }

    /// The catalog id a `decide` without `model` uses.
    public let defaultModel: String
    public let codec: SystemOneMCP
    /// One line per event (a model loaded, each call served); stderr by default.
    public let log: @Sendable (String) -> Void
    private let store: ModelStore
    private let configuration: TypedDecisions.Configuration
    private let downloadProgress: (@Sendable (DownloadProgress) -> Void)?
    private let input: FileHandle
    private let output: FileHandle
    private let decisions = DecisionQueue()
    private let slot = ModelSlot()
    private let cancelled = CancelledIDs()
    private let lock = NSLock()
    private var outputFailure: String?
    private var finishInput: (@Sendable () -> Void)?
    private var inFlight: [Int: Task<Void, Never>] = [:]
    private var nextCall = 0

    /// - Parameters:
    ///   - defaultModel: the catalog id loaded when a call names none.
    ///   - codec: the message forms; pass `SystemOneMCP(version:)` with the binary's version.
    ///   - input, output: the wire; standard input and output for a process an MCP client spawned.
    public init(
        defaultModel: String,
        store: ModelStore = .default,
        configuration: TypedDecisions.Configuration = TypedDecisions.Configuration(),
        downloadProgress: (@Sendable (DownloadProgress) -> Void)? = nil,
        codec: SystemOneMCP = SystemOneMCP(),
        input: FileHandle = .standardInput,
        output: FileHandle = .standardOutput,
        log: @escaping @Sendable (String) -> Void = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    ) {
        self.defaultModel = defaultModel
        self.store = store
        self.configuration = configuration
        self.downloadProgress = downloadProgress
        self.codec = codec
        self.input = input
        self.output = output
        self.log = log
    }

    /// The catalog id of the resident model, once one has loaded.
    public var loadedModel: String? {
        get async { await slot.loaded }
    }

    /// Serves until the input reaches end of file, answers the calls still in flight, and
    /// returns. `preload` starts loading the default model before the first message. Throws
    /// when a reply could not be written.
    public func run(preload: Bool = false) async throws {
        if preload {
            Task {
                do {
                    _ = try await self.decisions.run { try await self.decider(for: nil) }
                } catch {
                    self.log("preload failed: \(error.localizedDescription)")
                }
            }
        }
        let input = self.input
        let lines = AsyncStream<String> { continuation in
            self.lock.withLock { self.finishInput = { continuation.finish() } }
            Thread.detachNewThread {
                var buffer: [UInt8] = []
                while true {
                    let chunk = input.availableData
                    if chunk.isEmpty { break }
                    buffer.append(contentsOf: chunk)
                    while let newline = buffer.firstIndex(of: 0x0A) {
                        continuation.yield(String(decoding: buffer[..<newline], as: UTF8.self))
                        buffer.removeSubrange(...newline)
                    }
                }
                if !buffer.isEmpty { continuation.yield(String(decoding: buffer, as: UTF8.self)) }
                continuation.finish()
            }
        }
        for await line in lines {
            guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            switch codec.route(SystemOneMCP.incoming(line)) {
            case .none:
                break
            case .cancelled(let id):
                await cancelled.insert(id)
            case .reply(let message):
                write(message)
            case .call(let tool, let arguments, let request):
                let key = lock.withLock { nextCall += 1; return nextCall }
                let task = Task {
                    defer { _ = self.lock.withLock { self.inFlight.removeValue(forKey: key) } }
                    guard let reply = await self.perform(tool: tool, arguments: arguments, request: request) else { return }
                    if await self.cancelled.remove(request.id) { return }
                    self.write(reply)
                }
                lock.withLock { inFlight[key] = task }
            }
        }
        for task in lock.withLock({ Array(inFlight.values) }) { await task.value }
        if let failure = lock.withLock({ outputFailure }) { throw Failure.outputClosed(failure) }
        log("input closed")
    }

    // MARK: - Wire

    /// Whole lines, one writer at a time. A failed write ends the run.
    private func write(_ message: JSONValue) {
        let data = Data((message.dumps() + "\n").utf8)
        lock.lock()
        defer { lock.unlock() }
        guard outputFailure == nil else { return }
        do {
            try output.write(contentsOf: data)
        } catch {
            outputFailure = error.localizedDescription
            finishInput?()
        }
    }

    // MARK: - Tools

    /// The reply to write, or nil when the client cancelled the call before it ran.
    private func perform(tool: String, arguments: JSONValue, request: SystemOneMCP.Request) async -> JSONValue? {
        do {
            switch tool {
            case "models":
                let loaded = await slot.loaded
                return codec.callResult(
                    id: request.id,
                    structured: SystemOneMCP.modelsResult(default: defaultModel, loaded: loaded),
                    modern: request.modern)
            case "decide":
                let parsed = try SystemOneMCP.decideRequest(arguments)
                let (response, modelID) = try await decisions.run {
                    if await self.cancelled.remove(request.id) { throw CallCancelled() }
                    let (decider, modelID) = try await self.decider(for: parsed.model)
                    return (try await decider.systemOne(parsed), modelID)
                }
                log("decide  \(parsed.questions.count) question(s), state \(response.stateTokens) tokens, \(Int(response.milliseconds.rounded())) ms")
                return codec.callResult(
                    id: request.id,
                    structured: SystemOne.response(model: modelID, answers: response.answers.map { ($0.id, $0.question, $0.answer) }),
                    modern: request.modern)
            default:
                return SystemOneMCP.error(id: request.id, code: -32602, message: "Unknown tool: \(tool)")
            }
        } catch is CallCancelled {
            return nil
        } catch let error as SystemOne.WireError {
            return codec.callError(id: request.id, message: error.message, modern: request.modern)
        } catch {
            return codec.callError(id: request.id, message: error.localizedDescription, modern: request.modern)
        }
    }

    /// The resident model for `requested` (the default when nil), loading or replacing it.
    /// Called inside the decision queue, so two calls never load at once.
    private func decider(for requested: String?) async throws -> (TypedDecisions, String) {
        let id = requested ?? defaultModel
        if let current = await slot.current, current.id == id { return (current.decider, id) }
        if let previous = await slot.loaded {
            log("replacing \(previous) with \(id)")
            await slot.clear()
        }
        let start = SuspendingClock.now
        let decider = try await TypedDecisions(
            catalog: id, store: store, configuration: configuration, downloadProgress: downloadProgress)
        let elapsed = SuspendingClock.now - start
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        log("loaded \(id) (\(await decider.modelName)) in \(String(format: "%.1f", seconds)) s")
        await slot.set(decider, id: id)
        return (decider, id)
    }
}

private struct CallCancelled: Error {}

/// The one resident model.
private actor ModelSlot {
    private var decider: TypedDecisions?
    private(set) var loaded: String?

    var current: (decider: TypedDecisions, id: String)? {
        guard let decider, let loaded else { return nil }
        return (decider, loaded)
    }

    func set(_ decider: TypedDecisions, id: String) {
        self.decider = decider
        loaded = id
    }

    func clear() {
        decider = nil
        loaded = nil
    }
}

/// Request ids the client cancelled: a queued call is dropped, a running one writes nothing.
private actor CancelledIDs {
    private var ids: Set<String> = []
    func insert(_ id: JSONValue) { ids.insert(id.dumps()) }
    func remove(_ id: JSONValue) -> Bool { ids.remove(id.dumps()) != nil }
}
