// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the VLChat runner: `swift run vlchat-cli --image photo.jpg --prompt "…"`
// answers on macOS with no Xcode and no device — the agent-verifiable door. It compiles the same
// Sources/QuickStart.swift the GUI app (project.yml / VLChat.xcodeproj) ships.
let package = Package(
    name: "VLChat",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "vlchat-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "VLChat.xcodeproj", "project.yml", "README.md", "sample.jpg", "screenshot.png"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
