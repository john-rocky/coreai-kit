// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the DepthCamera runner: `swift run depth-cli --image photo.jpg` writes
// a depth map on macOS with no Xcode and no device — the agent-verifiable door. It compiles the
// same Sources/QuickStart.swift the GUI app (project.yml / DepthCamera.xcodeproj) ships.
let package = Package(
    name: "DepthCamera",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "depth-cli",
            dependencies: [.product(name: "CoreAIKitVision", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "DepthCamera.xcodeproj", "project.yml", "README.md", "sample.jpg"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
