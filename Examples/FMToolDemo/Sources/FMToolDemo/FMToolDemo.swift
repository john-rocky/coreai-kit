import CoreAIKit
import Foundation
import FoundationModels

struct WeatherTool: Tool {
    let name = "get_weather"
    let description = "Get the current weather for a city."

    @Generable
    struct Arguments {
        @Guide(description: "Name of the city, in English")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_weather(city: \(arguments.city))")
        return "Sunny, 24 degrees Celsius in \(arguments.city)."
    }
}

@main
enum FMToolDemo {
    static func main() async throws {
        // Usage: swift run FMToolDemo [bundle-dir] — default downloads qwen3 0.6B.
        let model: KitLanguageModel
        if let path = CommandLine.arguments.dropFirst().first {
            model = try await KitLanguageModel(bundleAt: URL(fileURLWithPath: path))
        } else {
            print("Fetching qwen3 0.6B (cached afterwards)…")
            model = try await KitLanguageModel(model: .qwen3_0_6B) { progress in
                print("  \(Int(progress.fraction * 100))% \(progress.currentFile)")
            }
        }

        let session = LanguageModelSession(
            model: model,
            tools: [WeatherTool()],
            instructions: "You are a helpful assistant. Use the tools when they help."
        )

        let question = "What's the weather in Sapporo right now?"
        print("\n> \(question)")
        let response = try await session.respond(to: question)
        print("\n[answer] \(response.content)")

        print("\n[transcript]")
        for entry in session.transcript {
            switch entry {
            case .instructions: print("  instructions")
            case .prompt: print("  prompt")
            case .toolCalls(let calls): print("  toolCalls: \(calls.map(\.toolName))")
            case .toolOutput: print("  toolOutput")
            case .response: print("  response")
            default: print("  (other)")
            }
        }
    }
}
