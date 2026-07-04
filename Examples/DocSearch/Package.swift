// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the DocSearch runner: `swift run docsearch-cli --query "..."` ranks
// the bundled sample pages on macOS with no Xcode and no device — the agent-verifiable door.
// It compiles the same Sources/QuickStart.swift the GUI app (project.yml / DocSearch.xcodeproj)
// ships.
let package = Package(
    name: "DocSearch",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "docsearch-cli",
            dependencies: [.product(name: "CoreAIKitEmbeddings", package: "coreai-kit")],
            path: ".",
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
