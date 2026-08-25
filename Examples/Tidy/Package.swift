// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the Tidy runner: `swift run tidy-cli --text "…"` rewrites a raw
// dictation transcript on macOS with no Xcode and no device — the agent-verifiable door. It
// compiles the same Sources/QuickStart.swift the GUI app (project.yml / Tidy.xcodeproj) ships.
let package = Package(
    name: "Tidy",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "tidy-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
