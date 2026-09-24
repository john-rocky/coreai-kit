// DecisionBackend.swift — what answers a `/v1/systemone` request: a `TypedDecisions` (a
// catalog or local bundle, every answer read as probabilities from the logits) or a
// `FoundationModelDecisions` (Apple's on-device foundation model, every answer generated
// under a schema and reported as a one-hot distribution). `SystemOneServer` and `decide-cli`
// take either; the wire forms are the same, and a response from the second carries
// `metadata` saying what its probabilities are.

import Foundation

public protocol DecisionBackend: Sendable {
    /// What responses name as the model: a catalog id, a local bundle's directory name, or
    /// `FoundationModelDecisions.id`.
    var id: String { get }
    /// The widest choice this backend reads; a server refuses a longer list with a 422 that
    /// names it.
    var maxOptions: Int { get }
    /// What is loaded, for a log line.
    var modelName: String { get async }
    /// One question on one state.
    func decide(_ state: String, _ question: Decision.Question) async throws -> Decision.Answer
    /// A whole request: the state read once, every question decided in request order.
    func systemOne(_ request: SystemOne.Request) async throws -> SystemOne.Response
}

extension TypedDecisions: DecisionBackend {}
