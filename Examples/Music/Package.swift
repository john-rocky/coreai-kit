// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Speak runner: `swift run music-cli --text "…"` writes a WAV on
// macOS with no Xcode and no device — the agent-verifiable door. It compiles the same
// Sources/QuickStart.swift the GUI app (project.yml / Music.xcodeproj) ships.
let package = Package(
    name: "Music",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "music-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "Music.xcodeproj", "project.yml", "README.md"],
            sources: ["Sources/QuickStart.swift", "Sources/WAVFile.swift", "CLI/main.swift"]
        )
    ]
)
