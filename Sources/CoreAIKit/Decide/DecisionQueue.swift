// DecisionQueue.swift — one decision at a time over a shared `TypedDecisions`.
//
// `TypedDecisions` is an actor, but each call awaits the engine twice (the rewind to the
// shared prefix, then the scoring pass) and the actor lets another caller in at every await.
// Two callers on one instance therefore drive the engine at once: six concurrent calls come
// back with no logits, or crash inside the engine (`Range requires lowerBound <= upperBound`).
// Anything that accepts work concurrently — an HTTP server, an MCP server, a task group —
// funnels its decisions through one of these and they run in arrival order.

import Foundation

/// Runs operations one after another, in the order they arrived. The caller still awaits its
/// own result; only the model is used by one operation at a time.
public actor DecisionQueue {
    private var tail: Task<Void, Never> = Task {}

    public init() {}

    /// Runs `operation` after every operation queued before it has finished.
    public func run<T: Sendable>(_ operation: @Sendable @escaping () async throws -> T) async throws -> T {
        let previous = tail
        let task = Task { () throws -> T in
            await previous.value
            return try await operation()
        }
        tail = Task { _ = try? await task.value }
        return try await task.value
    }
}
