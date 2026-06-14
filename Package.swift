// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "coreai-kit",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        // LLM: ModelStore + ChatSession + FoundationModels provider (tool calling).
        .library(name: "CoreAIKit", targets: ["CoreAIKit"]),
        // CV: generic graph runner + typed pipelines (CLIP, depth, camera). Links only
        // the system CoreAI framework — no LLM runtime is pulled in.
        .library(name: "CoreAIKitVision", targets: ["CoreAIKitVision"]),
        // Text embeddings for on-device retrieval/RAG (EmbeddingGemma).
        .library(name: "CoreAIKitEmbeddings", targets: ["CoreAIKitEmbeddings"]),
        // SwiftUI components: model picker, chat transcript, stats bar.
        .library(name: "CoreAIKitUI", targets: ["CoreAIKitUI"]),
    ],
    dependencies: [
        // Patched community fork of apple/coreai-models — adds hybrid/SSM bundle support to
        // the pipelined engine so flagship hybrid models (Qwen3.5/3.6, LFM2.5, Granite 4)
        // load. See https://github.com/john-rocky/coreai-models (tag v0.1.0-zoo).
        .package(url: "https://github.com/john-rocky/coreai-models", exact: "0.1.0-zoo"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        // Shared base: model identity + Hugging Face download/cache. Foundation only.
        .target(name: "CoreAIKitCore"),
        .target(
            name: "CoreAIKit",
            dependencies: [
                "CoreAIKitCore",
                // The VL executor runs the paired vision tower through CoreAIKitVision's
                // GraphModel (stateless .aimodel runner); it also brings the CoreAI framework
                // link the vision path needs.
                "CoreAIKitVision",
                .product(name: "CoreAILM", package: "coreai-models"),
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "CoreAIKitVision",
            dependencies: ["CoreAIKitCore"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .target(
            name: "CoreAIKitEmbeddings",
            dependencies: [
                "CoreAIKitCore",
                "CoreAIKitVision",
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
        .target(
            name: "CoreAIKitUI",
            dependencies: ["CoreAIKit"]
        ),
        .testTarget(
            name: "CoreAIKitTests",
            dependencies: ["CoreAIKit", "CoreAIKitVision", "CoreAIKitEmbeddings"]
        ),
    ]
)
