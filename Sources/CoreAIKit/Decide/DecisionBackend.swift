// DecisionBackend.swift — what answers a `/v1/systemone` request: a `TypedDecisions` (a
// catalog or local bundle, every answer read as probabilities from the logits), a
// `FoundationModelDecisions` (Apple's on-device foundation model, every answer generated
// under a schema and reported as a one-hot distribution) or a `KitClefDecider` (clef-flash,
// every question read at once by a joint head, the one backend that reads an image).
// `SystemOneServer` and `decide-cli` take any of them; the wire forms are the same, and a
// response from the second or third carries `metadata` saying what its probabilities are.

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
    /// Whether a request may carry an image (`SystemOne.Request.images`); a server refuses one with a 422 that
    /// names the model when this is false. False unless the backend says otherwise (`KitClefDecider`).
    var readsImages: Bool { get }
}

extension DecisionBackend {
    public var readsImages: Bool { false }

    /// The 422 a server answers when `request` carries an image this backend cannot read; nil when it can.
    public func imageRefusal(_ request: SystemOne.Request) -> SystemOne.WireError? {
        guard !request.images.isEmpty, !readsImages else { return nil }
        return SystemOne.WireError("'\(id)' reads no images; send the request without 'images', or load a model that reads them (clef-flash)")
    }
}

@available(macOS 27, iOS 27, *)
extension TypedDecisions: DecisionBackend {}
