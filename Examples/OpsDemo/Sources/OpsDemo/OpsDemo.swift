// OpsDemo — the anchored ops end to end. Point it at a voice memo and one audio file
// becomes a transcript, a cleaned-up text, a summary, typed action items, and a
// translation — all on device, no sessions or prompts in app code. Without arguments
// the text ops run on built-in sample text instead.
//
//   swift run OpsDemo                    # text ops on built-in samples
//   swift run OpsDemo voice-memo.wav     # full speech -> text pipeline

import CoreAIOps
import Foundation
import FoundationModels

@Generable
struct ActionItems {
    @Guide(description: "One entry per distinct task, as a short imperative sentence")
    var tasks: [String]
}

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
    it, CoreAIOps exposes task-level operations — summarize, extract, translate,
    proofread, transcribe — that hide sessions and prompts entirely: apps state the task,
    the kit resolves a catalog model behind it.
    """

let email = """
    Hi team, please ship 12 units of the AlphaWidget to our Osaka office by Friday.
    Thanks! — Dana
    """

let typos = "Their going to recieve the package tommorow, weather or not its raining."

@main
enum OpsDemo {
    static func main() async throws {
        if let path = CommandLine.arguments.dropFirst().first {
            try await voiceMemoPipeline(URL(fileURLWithPath: path))
        } else {
            try await textOps()
        }
    }

    /// One voice memo in, five op results out.
    static func voiceMemoPipeline(_ memo: URL) async throws {
        print("> CoreAI.transcribe(memo)")
        let text = try await CoreAI.transcribe(memo)
        print("[transcript] \(text)\n")

        print("> CoreAI.proofread(text)")
        let clean = try await CoreAI.proofread(text)
        print("[clean] \(clean)\n")

        print("> CoreAI.summarize(clean, style: .oneLine)")
        let summary = try await CoreAI.summarize(clean, style: .oneLine)
        print("[summary] \(summary)\n")

        print("> CoreAI.extract(clean, as: ActionItems.self)")
        let items = try await CoreAI.extract(clean, as: ActionItems.self)
        for task in items.tasks { print("[task] \(task)") }
        print()

        print("> CoreAI.translate(clean, to: .japanese)")
        let japanese = try await CoreAI.translate(clean, to: .japanese)
        print("[ja] \(japanese)")
    }

    /// The text ops on built-in samples — no audio file needed.
    static func textOps() async throws {
        print("> CoreAI.summarize(article, style: .oneLine)")
        let summary = try await CoreAI.summarize(article, style: .oneLine)
        print("[summary] \(summary)\n")

        print("> CoreAI.extract(email, as: Order.self)")
        let order = try await CoreAI.extract(email, as: Order.self)
        print("[order] product=\(order.product) quantity=\(order.quantity) city=\(order.city)\n")

        print("> CoreAI.translate(email, to: .japanese)")
        let japanese = try await CoreAI.translate(email, to: .japanese)
        print("[ja] \(japanese)\n")

        print("> CoreAI.proofread(typos)")
        let clean = try await CoreAI.proofread(typos)
        print("[clean] \(clean)")
    }
}
