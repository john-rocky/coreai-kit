// swift-tools-version: 6.0
import PackageDescription

// Headless runner for `TextClassifier`: `swift run textclassify-cli --gate <fixtures>` checks the kit's
// collator and decisions against the zoo's gliner2 oracle fixtures on macOS with no Xcode and no
// device — the agent-verifiable door. It drives the same kit class an app would ship.
let package = Package(
    name: "TextClassify",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "textclassify-cli",
            dependencies: [.product(name: "CoreAIKitEmbeddings", package: "coreai-kit")],
            path: ".",
            sources: ["CLI/main.swift"]
        )
    ]
)
