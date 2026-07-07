// swift-tools-version: 6.0
import PackageDescription

// Headless runner for `InformationExtractor`: `swift run infoextract-cli --gate` reproduces the
// GLiNER2-PII demo suite on macOS with no Xcode and no device — the agent-verifiable door. It
// drives the same kit class the GUI would ship.
let package = Package(
    name: "InfoExtract",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "infoextract-cli",
            dependencies: [.product(name: "CoreAIKitEmbeddings", package: "coreai-kit")],
            path: ".",
            sources: ["CLI/main.swift"]
        )
    ]
)
