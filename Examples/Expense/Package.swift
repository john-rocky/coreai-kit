// swift-tools-version: 6.0
import PackageDescription

// Headless runner for the receipt flow: `swift run expense-cli --image receipt.jpg` reads a
// photo and prints the typed value it produced. Same two calls the GUI makes, no Xcode and no
// device — so the example can be checked by anyone, including an agent.
let package = Package(
    name: "Expense",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "expense-cli",
            dependencies: [.product(name: "CoreAIOps", package: "coreai-kit")],
            path: ".",
            sources: ["CLI/main.swift", "Sources/QuickStart.swift"]
        )
    ]
)
