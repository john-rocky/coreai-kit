// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the DetectCamera runner: `swift run detect-cli --image photo.jpg`
// prints detections on macOS with no Xcode and no device — the agent-verifiable door. It
// compiles the same Sources/QuickStart.swift the GUI app (DetectCamera.xcodeproj) ships.
let package = Package(
    name: "DetectCamera",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "detect-cli",
            dependencies: [.product(name: "CoreAIKitVision", package: "coreai-kit")],
            path: ".",
            exclude: [
                "build", "DetectCamera.xcodeproj", "project.yml", "README.md",
                "Models", "Resources",
            ],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
