// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "coreai-kit",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        // LLM: ModelStore + ChatSession + FoundationModels provider (tool calling).
        .library(name: "CoreAIKit", targets: ["CoreAIKit"]),
        // CV: generic graph runner + typed pipelines (CLIP). Links only the system
        // CoreAI framework — no LLM runtime is pulled in.
        .library(name: "CoreAIKitVision", targets: ["CoreAIKitVision"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/coreai-models", from: "0.1.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        // Shared base: model identity + Hugging Face download/cache. Foundation only.
        .target(name: "CoreAIKitCore"),
        .target(
            name: "CoreAIKit",
            dependencies: [
                "CoreAIKitCore",
                .product(name: "CoreAILM", package: "coreai-models"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "CoreAIKitVision",
            dependencies: ["CoreAIKitCore"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .testTarget(
            name: "CoreAIKitTests",
            dependencies: ["CoreAIKit", "CoreAIKitVision"]
        ),
    ]
)
