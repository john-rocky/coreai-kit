// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Meeting runner: `swift run meeting-cli --audio clip.wav` prints a
// speaker-attributed transcript on macOS with no Xcode and no device — the agent-verifiable door.
// It compiles the same Sources/QuickStart.swift a GUI would ride.
let package = Package(
    name: "Meeting",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "meeting-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
