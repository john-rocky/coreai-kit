// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the AudioChat runner: `swift run audiochat-cli --audio sample.wav
// --prompt "…"` answers on macOS with no Xcode and no device — the agent-verifiable door.
// It compiles the same Sources/QuickStart.swift the GUI app ships.
let package = Package(
    name: "AudioChat",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "audiochat-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            exclude: [
                "build", "AudioChat.xcodeproj", "project.yml", "README.md",
                "AudioChat.entitlements", "sample.wav",
            ],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
