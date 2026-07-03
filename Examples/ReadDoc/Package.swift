// swift-tools-version: 6.0
import PackageDescription

// The headless shell of the ReadDoc runner: `swift run readdoc-cli --image sample.png` prints
// a document's markdown on macOS with no Xcode and no device — the agent-verifiable door. It
// compiles the same Sources/QuickStart.swift the GUI app (project.yml / ReadDoc.xcodeproj)
// ships.
let package = Package(
    name: "ReadDoc",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "readdoc-cli",
            dependencies: [.product(name: "CoreAIKit", package: "coreai-kit")],
            path: ".",
            exclude: ["build", "ReadDoc.xcodeproj", "project.yml", "README.md", "sample.png"],
            sources: ["Sources/QuickStart.swift", "CLI/main.swift"]
        )
    ]
)
