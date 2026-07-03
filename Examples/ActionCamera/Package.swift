// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the ActionCamera runner: `swift run action-cli --video clip.mp4`
// names the action on macOS with no Xcode and no device — the agent-verifiable door. It
// compiles the same Sources/QuickStart.swift the GUI app (project.yml /
// ActionCamera.xcodeproj) ships.
let package = Package(
    name: "ActionCamera",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "action-cli",
            dependencies: [.product(name: "CoreAIKitVision", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "ActionCamera.xcodeproj", "project.yml", "README.md", "sample.mp4"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
