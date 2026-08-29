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
        // Answers "what will this app download" before someone has to answer it in a meeting.
        // An executable, so linking the libraries never drags it in.
        .executable(name: "coreai-doctor", targets: ["coreai-doctor"]),
    ],
    dependencies: [
        // Community fork of apple/coreai-models (unaffiliated with Apple). 0.2.3-zoo is
        // upstream main through #207 (2026-08-28) plus the zoo patches to the pipelined
        // engine: hybrid/SSM extra states so Qwen3.5/3.6, LFM2.5 and Granite 4 load,
        // chunked prefill via a static-chunk "prefill" function, per-token/static inputs,
        // decode-only S=1 bundles (apple/coreai-models#212), the iOS dynamic-KV capacity
        // guard (apple/coreai-models#124), EOS stop when the consumer breaks the stream,
        // and the sampler-behind-logits ordering drain. Everything else is byte-for-byte
        // upstream. See https://github.com/john-rocky/coreai-models.
        // 0.2.2-zoo and earlier predate upstream #121 (per-call sampler execution
        // descriptor; garbled text at temperature > 0 under pipelined decode).
        // For local engine work swap in .package(path: "../coreai-models") — branch
        // zoo-0.4 matches this tag.
        .package(url: "https://github.com/john-rocky/coreai-models", exact: "0.2.3-zoo"),
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
        .executableTarget(
            name: "coreai-doctor",
            dependencies: ["CoreAIOps"]
        ),
        .testTarget(
            name: "CoreAIKitTests",
            dependencies: ["CoreAIKit", "CoreAIKitVision", "CoreAIKitEmbeddings", "CoreAIOps"]
        ),
    ]
)
