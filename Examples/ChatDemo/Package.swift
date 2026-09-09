// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the ChatDemo runner: `swift run chat-cli --prompt "…"` answers on
// macOS with no Xcode and no device — the agent-verifiable door. It compiles the same
// Sources/QuickStart.swift the GUI app (project.yml / ChatDemo.xcodeproj) ships.
let package = Package(
    name: "ChatDemo",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(url: "https://github.com/john-rocky/coreai-kit", exact: "0.4.1")
    ],
    targets: [
        .executableTarget(
            name: "chat-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            // build_ios: device-build DerivedData Xcode recreates locally — without the
            // exclude, SPM globs its PrivacyInfo.xcprivacy copies into the target.
            exclude: ["build", "build_ios", "ChatDemo.xcodeproj", "project.yml", "README.md", "ReleaseCheck"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        ),
        .executableTarget(
            name: "entry-check",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: "ReleaseCheck"
        )
    ]
)
