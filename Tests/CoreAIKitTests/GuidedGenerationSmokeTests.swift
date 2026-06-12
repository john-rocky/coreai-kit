import FoundationModels
import XCTest

@testable import CoreAIKit

/// End-to-end guided generation over a real local bundle. Opt-in (loads a model,
/// runs the sequential engine):
///
///     KIT_SMOKE_BUNDLE=/path/to/bundle swift test --filter GuidedGenerationSmoke
final class GuidedGenerationSmokeTests: XCTestCase {
    struct CityFacts: Codable {
        let name: String
        let country: String
    }

    static let cityFactsSchema = """
        {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "country": {"type": "string"}
          },
          "required": ["name", "country"]
        }
        """

    func testChatSessionGuidedJSON() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SMOKE_BUNDLE"] else {
            throw XCTSkip("Set KIT_SMOKE_BUNDLE to a local bundle directory to run.")
        }

        var config = ChatSession.Configuration()
        config.engineVariant = .sequential
        config.temperature = nil  // greedy: deterministic smoke
        config.maxResponseTokens = 128
        let session = try await ChatSession(
            bundleAt: URL(fileURLWithPath: path), configuration: config)

        let json = try await session.respondJSON(
            to: "Give the name and country of the capital of Japan.",
            schema: Self.cityFactsSchema)
        print("SMOKE guided JSON: \(json)")

        // The grammar guarantees this parses and carries the required keys.
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertNotNil(object["name"] as? String)
        XCTAssertNotNil(object["country"] as? String)

        // Typed path on the same session (second turn).
        let facts = try await session.respond(
            to: "Now the capital of France, same fields.",
            generating: CityFacts.self,
            schema: Self.cityFactsSchema)
        print("SMOKE guided typed: \(facts)")
        XCTAssertFalse(facts.name.isEmpty)
        XCTAssertFalse(facts.country.isEmpty)
    }

    func testPipelinedEngineThrows() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SMOKE_BUNDLE"] else {
            throw XCTSkip("Set KIT_SMOKE_BUNDLE to a local bundle directory to run.")
        }

        // Default (pipelined) engine: guided generation must fail fast and loudly.
        let session = try await ChatSession(bundleAt: URL(fileURLWithPath: path))
        do {
            _ = try await session.respondJSON(
                to: "Anything.", schema: Self.cityFactsSchema)
            XCTFail("Expected guidedGenerationUnsupported")
        } catch ChatSessionError.guidedGenerationUnsupported {
            // expected
        }
    }

    @Generable
    struct Landmark {
        @Guide(description: "Name of the landmark")
        var name: String
        @Guide(description: "City it is in")
        var city: String
    }

    func testFoundationModelsGuided() async throws {
        guard let path = ProcessInfo.processInfo.environment["KIT_SMOKE_BUNDLE"] else {
            throw XCTSkip("Set KIT_SMOKE_BUNDLE to a local bundle directory to run.")
        }

        let model = try await KitLanguageModel(
            bundleAt: URL(fileURLWithPath: path), engineVariant: .sequential)
        let session = LanguageModelSession(model: model)
        let response = try await session.respond(
            to: "Name one famous landmark in Japan.", generating: Landmark.self)
        print("SMOKE FM guided: \(response.content)")
        XCTAssertFalse(response.content.name.isEmpty)
        XCTAssertFalse(response.content.city.isEmpty)
    }
}
