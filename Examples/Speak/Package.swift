// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Speak runner: `swift run speak-cli --text "…"` writes a WAV on
// macOS with no Xcode and no device — the agent-verifiable door. It compiles the same
// Sources/QuickStart.swift the GUI app (project.yml / Speak.xcodeproj) ships.
let package = Package(
    name: "Speak",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "speak-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "Speak.xcodeproj", "project.yml", "README.md"],
            sources: ["Sources/QuickStart.swift", "Sources/WAVFile.swift", "CLI/main.swift"]
        )
    ]
)
