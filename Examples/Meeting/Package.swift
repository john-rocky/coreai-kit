// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Meeting runner: `swift run meeting-cli --audio clip.wav` prints a
// speaker-attributed transcript on macOS with no Xcode and no device — the agent-verifiable door.
// It compiles the same Sources/QuickStart.swift a GUI would ride. `diarize-gate` checks the
// diarizer alone against a reference's per-frame probabilities (agreement at 0.5, turns, wall).
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
            exclude: ["Gate"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        ),
        .executableTarget(
            name: "diarize-gate",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: "Gate"
        ),
    ]
)
