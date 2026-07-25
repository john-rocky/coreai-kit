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
        // Anchored task ops (text, image, audio, video, search, forecast) over catalog
        // models — the stable task-level API. Re-exports the model layer, so this one
        // product + `import CoreAIOps` is the whole quick path.
        .library(name: "CoreAIOps", targets: ["CoreAIOps"]),
    ],
    dependencies: [
        // Patched community fork of apple/coreai-models — adds hybrid/SSM bundle support to
        // the pipelined engine so flagship hybrid models (Qwen3.5/3.6, LFM2.5, Granite 4)
        // load. See https://github.com/john-rocky/coreai-models (tag v0.1.0-zoo).
        // v0.1.1-zoo adds the D1 engine fix: a consumer that breaks the stream at EOS now stops the
        // pipelined engine (instead of running to maxTokens), so a SECOND consecutive generation no
        // longer collides with the still-running first one ("something went wrong" on turn 2).
        // v0.2.0-zoo: upstream 0.2.0 (implicit prefix caching / reset(to:) / async
        // generate / cancel API / pipeline buffer-rotation fix) merged with the zoo
        // patches (hybrid extra states, chunked prefill via "prefill" fn,
        // per-token/static inputs, D1 EOS stop), plus the sampler/stream ordering fix
        // (bogus first token per turn). For local engine work, swap in
        // .package(path: "../coreai/coreai-models") — branch zoo-0.2 matches this tag.
        // 0.2.1-zoo adds the iOS dynamic-KV capacity guard (apple/coreai-models#124, kit #5).
        .package(url: "https://github.com/john-rocky/coreai-models", exact: "0.2.1-zoo"),
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
            ],
            // Mel filterbanks for the audio frontends, each bit-exact with its HF feature
            // extractor so the mel never recomputes librosa: Whisper-large-v3 ([201,128] fp32,
            // gated cos 1.0) and Parakeet ([128,257] fp32 librosa-slaney, gated token-exact e2e).
            resources: [
                .copy("Audio/Resources/mel_filters.f32"),
                .copy("Audio/Resources/parakeet_mel_filters_128x257.f32"),
                // Raw-Metal Gemma 4 kernels + oracle refs (compiled at load by
                // Gemma4MetalEngine — .txt so nothing tries to precompile them).
                .copy("Gemma4Metal/g4msl"),
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
        .target(
            name: "CoreAIOps",
            dependencies: ["CoreAIKit", "CoreAIKitVision", "CoreAIKitEmbeddings"]
        ),
        .testTarget(
            name: "CoreAIKitTests",
            dependencies: ["CoreAIKit", "CoreAIKitVision", "CoreAIKitEmbeddings"]
        ),
    ]
)
