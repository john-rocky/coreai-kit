// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Transcribe runner: `swift run transcribe-cli --audio sample.wav`
// transcribes on macOS with no Xcode and no device — the agent-verifiable door. It compiles
// the same Sources/QuickStart.swift the GUI app (project.yml / Transcribe.xcodeproj) ships.
let package = Package(
    name: "Transcribe",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "transcribe-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
