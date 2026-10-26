// swift-tools-version: 6.1
import PackageDescription

// The headless shell of Cutout: `swift run cutout-cli run --models <folder> --image x.jpg --words "the dog"` cuts out on
// macOS with no Xcode and no device — the agent-verifiable door. It compiles the same Sources/Engine and
// Sources/QuickStart.swift the GUI app (project.yml / Cutout.xcodeproj) ships. The engine runs on the system CoreAI
// framework directly (the kit's pipelined engine does not hand out the decoder's hidden state the mask decoder reads),
// so the only package it needs is swift-transformers' tokenizer.
let package = Package(
    name: "Cutout",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .executableTarget(
            name: "cutout-cli",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            path: ".",
            sources: ["Sources/Engine", "Sources/QuickStart.swift", "CLI/main.swift"],
            linkerSettings: [.linkedFramework("CoreAI")]
        )
    ]
)
