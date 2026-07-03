// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the DiffuseChat runner: `swift run diffuse-cli --prompt "…"` prints
// a denoised reply on macOS with no Xcode and no device — the agent-verifiable door. It
// compiles the same Sources/QuickStart.swift the GUI app (project.yml /
// DiffuseChat.xcodeproj) ships.
let package = Package(
    name: "DiffuseChat",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "diffuse-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "DiffuseChat.xcodeproj", "project.yml", "README.md"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
