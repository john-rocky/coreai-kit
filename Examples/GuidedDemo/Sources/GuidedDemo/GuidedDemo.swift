import CoreAIKit
import Foundation
import FoundationModels

// What the FM path generates: the framework derives the schema from @Generable and
// parses the constrained JSON back into this type.
@Generable
struct TravelPlan {
    @Guide(description: "City to visit")
    var city: String
    @Guide(description: "Best season to go, one word")
    var season: String
    @Guide(description: "Three sight names")
    var sights: [String]
}

// What the ChatSession path generates: plain Codable + a hand-written JSON schema.
struct CityFacts: Codable {
    let name: String
    let country: String
    let population: Int
}

let cityFactsSchema = """
    {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "country": {"type": "string"},
        "population": {"type": "integer"}
      },
      "required": ["name", "country", "population"]
    }
    """

@main
enum GuidedDemo {
    static func main() async throws {
        // Usage: swift run GuidedDemo [bundle-dir] — default downloads qwen3 0.6B.
        // Guided generation needs per-step logits, so both paths load the sequential
        // engine (the default pipelined engine samples on-GPU).
        let bundleURL: URL
        if let path = CommandLine.arguments.dropFirst().first {
            bundleURL = URL(fileURLWithPath: path)
        } else {
            print("Fetching qwen3 0.6B (cached afterwards)…")
            bundleURL = try await ModelStore.default.download(.qwen3_0_6B) { progress in
                print("  \(Int(progress.fraction * 100))% \(progress.currentFile)")
            }
        }

        // 1) ChatSession: JSON schema in, typed Codable out.
        var config = ChatSession.Configuration()
        config.engineVariant = .sequential
        config.temperature = nil  // greedy: deterministic facts
        let chat = try await ChatSession(bundleAt: bundleURL, configuration: config)

        let question = "Give facts about the capital of Japan."
        print("\n> \(question)")
        let facts = try await chat.respond(
            to: question, generating: CityFacts.self, schema: cityFactsSchema)
        print("[typed] \(facts.name), \(facts.country) — population \(facts.population)")

        // 2) FoundationModels: @Generable schema, framework-parsed result.
        let model = try await KitLanguageModel(bundleAt: bundleURL, engineVariant: .sequential)
        let session = LanguageModelSession(model: model)

        let prompt = "Plan a short trip to Kyoto."
        print("\n> \(prompt)")
        let response = try await session.respond(to: prompt, generating: TravelPlan.self)
        let plan = response.content
        print("[plan] \(plan.city) in \(plan.season): \(plan.sights.joined(separator: ", "))")
    }
}
