// OpsDemo — the two anchored ops end to end: summarize a paragraph, extract a typed
// value. No sessions, no prompts, no model names in app code — that is the point.

import CoreAIOps
import Foundation
import FoundationModels

@Generable
struct Order {
    @Guide(description: "Name of the ordered product")
    var product: String
    @Guide(description: "Number of units ordered")
    var quantity: Int
    @Guide(description: "Requested delivery city")
    var city: String
}

let article = """
    Apple's FoundationModels framework gives every app a session API for on-device
    language models. CoreAIKit plugs downloadable open models into that same API, so the
    system model and your own models are interchangeable behind one code path. On top of
    it, CoreAIOps exposes task-level operations — summarize and extract — that hide
    sessions and prompts entirely: apps state the task, the kit resolves a catalog model
    behind it.
    """

let email = """
    Hi team, please ship 12 units of the AlphaWidget to our Osaka office by Friday.
    Thanks! — Dana
    """

@main
enum OpsDemo {
    static func main() async throws {
        print("> CoreAI.summarize(article, style: .oneLine)")
        let summary = try await CoreAI.summarize(article, style: .oneLine)
        print("[summary] \(summary)\n")

        print("> CoreAI.extract(email, as: Order.self)")
        let order = try await CoreAI.extract(email, as: Order.self)
        print("[order] product=\(order.product) quantity=\(order.quantity) city=\(order.city)")
    }
}
