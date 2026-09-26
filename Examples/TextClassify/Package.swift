// swift-tools-version: 6.0
import PackageDescription

// Headless runner for `TextClassifier`: `swift run textclassify-cli --gate <fixtures>` checks the kit's
// collator and decisions against the zoo's gliner2 oracle fixtures on macOS with no Xcode and no
// device — the agent-verifiable door. It drives the same kit class an app would ship.
//
// `InboxCore` is the inbox demo's core — the synthetic support inbox and the sorter that asks every
// message three questions. `swift run textclassify-cli --inbox <count>` runs it headless; the SwiftUI
// app in App/ compiles the same sources into itself, so both report numbers from one code path.
let package = Package(
    name: "TextClassify",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .target(
            name: "InboxCore",
            dependencies: [.product(name: "CoreAIKitEmbeddings", package: "coreai-kit")]
        ),
        .executableTarget(
            name: "textclassify-cli",
            dependencies: ["InboxCore", .product(name: "CoreAIKitEmbeddings", package: "coreai-kit")],
            path: "CLI"
        ),
    ]
)
