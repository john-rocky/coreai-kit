// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the UpscaleDemo runner: `swift run upscale-cli --image small.png`
// writes the ×4 result on macOS with no Xcode and no device — the agent-verifiable door. It
// compiles the same Sources/QuickStart.swift the GUI app (UpscaleDemo.xcodeproj) ships.
let package = Package(
    name: "UpscaleDemo",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "upscale-cli",
            dependencies: [.product(name: "CoreAIKitVision", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "UpscaleDemo.xcodeproj", "project.yml", "README.md", "sample_small.png"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
